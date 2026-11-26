"""Shared training helpers: env wrapper, eval, logging, model loading, sweep configs and analyze()."""
from __future__ import annotations

import csv
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torchrl.envs import GymWrapper, RewardSum, TransformedEnv
from torchrl.modules import MLP

from environment.custom_env import N_ACTIONS, OBS_DIM, DeliveryDroneEnv

os.environ.setdefault("RL_WARNINGS", "False")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TOTAL_FRAMES = 150_000
EVAL_EVERY = 10_000
EVAL_EPISODES = 30
SPLITS = ("train", "heldout", "stress")
OUTCOMES = ("delivered", "damaged", "parcel_lost", "crashed", "battery_empty", "left_geofence", "timeout")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int = 0) -> TransformedEnv:
    """env wrapped for torchrl (train layouts only)"""
    env = TransformedEnv(GymWrapper(DeliveryDroneEnv(split="train"), categorical_action_encoding=True), RewardSum())
    env.set_seed(seed)
    return env


def build_mlp(out_features: int, hidden: list[int] | tuple[int, ...], activation: str = "tanh") -> MLP:
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    return MLP(in_features=OBS_DIM, out_features=out_features, num_cells=list(hidden), activation_class=act)


# ---- evaluation ----
def evaluate(act_fn: Callable[[np.ndarray], int], split: str = "train", n_episodes: int = EVAL_EPISODES,
             seed: int = 10_000) -> dict:
    """split: 'train' (training layouts), 'heldout' (layouts never trained on), 'stress' (6 obstacles, 2x wind)."""
    env = DeliveryDroneEnv(split=split)
    rets, steps, batt, counts = [], [], [], {o: 0 for o in OUTCOMES}
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done, R = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(int(act_fn(obs)))
            R += r
            done = term or trunc
        rets.append(R)
        steps.append(info["steps"])
        counts[info["outcome"]] += 1
        if info["outcome"] == "delivered":
            batt.append(info["battery"])
    n = float(n_episodes)
    return {
        "return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
        "success_rate": counts["delivered"] / n, "crash_rate": counts["crashed"] / n,
        "steps": float(np.mean(steps)), "battery_left": float(np.mean(batt)) if batt else 0.0,
        "outcomes": {k: v / n for k, v in counts.items()},
    }


def baseline_eval(policy_factory, n_episodes: int = 100) -> dict:
    """eval for the random / scripted baselines on all 3 splits"""
    out = {}
    for s in SPLITS:
        env = DeliveryDroneEnv(split=s)
        act = policy_factory(env)
        out[s] = evaluate(act, s, n_episodes)
    return out


def full_evaluation(act_fn: Callable[[np.ndarray], int], n_episodes: int = 100) -> dict:
    return {s: evaluate(act_fn, s, n_episodes) for s in SPLITS}


# ---- logging ----
class RunLogger:
    """one csv per run with the curve, losses, entropy and eval results"""

    FIELDS = ["frames", "train_return", "eval_return", "eval_success", "loss", "loss_objective",
              "loss_critic", "entropy", "epsilon", "q_mean", "time_s"]

    def __init__(self, algo: str, run_id: str, logdir: Path | None = None):
        self.algo, self.run_id = algo, run_id
        self.dir = (logdir or ROOT / "logs") / "curves"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{algo}_{run_id}.csv"
        self.t0 = time.time()
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()

    def log(self, **kw) -> None:
        row = {k: "" for k in self.FIELDS}
        row.update({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in kw.items() if k in row})
        row["time_s"] = f"{time.time() - self.t0:.1f}"
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def save_summary(summary: dict, logdir: Path | None = None) -> None:
    """append the run's results to logs/runs.jsonl"""
    path = (logdir or ROOT / "logs") / "runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(summary) + "\n")


class EpisodeTracker:
    """keeps track of the returns of finished training episodes"""

    def __init__(self, window: int = 50):
        self.rets: list[float] = []
        self.window = window

    def update(self, batch) -> None:
        done = batch["next", "done"].reshape(-1)
        if done.any():
            ep = batch["next", "episode_reward"].reshape(-1)[done] if ("next", "episode_reward") in batch.keys(True) else None
            if ep is not None:
                self.rets.extend(ep.tolist())

    @property
    def mean(self) -> float:
        return float(np.mean(self.rets[-self.window:])) if self.rets else float("nan")


# ---- loading saved agents ----
class Policy:
    def __init__(self, net: nn.Module, algo: str):
        self.net, self.algo = net, algo

    def __call__(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(self.net(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())

    def scores(self, obs: np.ndarray) -> list[float]:
        with torch.no_grad():
            return self.net(torch.as_tensor(obs, dtype=torch.float32)).tolist()


def load_policy(path: str | Path) -> Policy:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    net = build_mlp(N_ACTIONS, ckpt["hidden"], ckpt["activation"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return Policy(net, ckpt["algo"])


def best_model_path(algo: str | None = None) -> Path:
    """which model to load: whatever models/best.json says, otherwise the newest .pt file"""
    best = ROOT / "models" / "best.json"
    if best.exists():
        info = json.loads(best.read_text())
        return ROOT / info["models"][algo if algo else info["overall"]]["path"]
    pattern = f"{algo}_*.pt" if algo else "*.pt"
    found = sorted((p for p in (ROOT / "models").rglob(pattern)), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"No trained {algo or 'agent'} checkpoint in models/. Train one first: "
                                f"uv run main.py train --algo {algo or 'dqn'}")
    return found[-1]


# ---- hyperparameter sweeps ----
SWEEP_FRAMES = 300_000
SWEEPS: dict[str, list[dict]] = {
    "dqn": [
        dict(note="baseline: DQN, eps decay 20k"),
        dict(double_dqn=True, eps_anneal_frames=100_000, note="Double DQN, eps decay 100k"),
        dict(double_dqn=True, eps_anneal_frames=50_000, note="Double DQN, eps decay 50k"),
        dict(double_dqn=True, eps_anneal_frames=100_000, lr=3e-4, note="Double DQN, lr 3e-4"),
        dict(double_dqn=True, eps_anneal_frames=100_000, gamma=0.95, note="Double DQN, gamma 0.95"),
        dict(double_dqn=True, eps_anneal_frames=100_000, target="hard", hard_interval=500, note="Double DQN, hard target"),
        dict(double_dqn=True, eps_anneal_frames=100_000, buffer_size=100_000, note="Double DQN, buffer 100k"),
        dict(double_dqn=True, eps_anneal_frames=100_000, batch_size=128, note="Double DQN, batch 128"),
        dict(double_dqn=True, eps_anneal_frames=100_000, hidden=[256, 256], note="Double DQN, net 256x2"),
        dict(double_dqn=True, eps_anneal_frames=100_000, eps_end=0.1, note="Double DQN, final eps 0.10"),
    ],
    "reinforce": [
        dict(note="baseline: batch-mean baseline, 512 frames/batch"),
        dict(baseline="value", note="learned value baseline"),
        dict(baseline="value", lr=3e-4, note="value baseline, lr 3e-4"),
        dict(baseline="none", note="no baseline"),
        dict(frames_per_batch=1024, note="batch 1024 frames"),
        dict(baseline="value", frames_per_batch=2048, note="value baseline, batch 2048"),
        dict(entropy_coef=0.05, note="entropy coef 0.05"),
        dict(lr=3e-3, note="lr 3e-3"),
        dict(baseline="value", gamma=0.98, note="value baseline, gamma 0.98"),
        dict(baseline="value", frames_per_batch=2048, lr=5e-4, entropy_coef=0.02, note="value, batch 2048, lr 5e-4, ent 0.02"),
        # 11-13 added after 01-10 failed: smaller batches = more updates for reinforce
        dict(baseline="value", frames_per_batch=128, note="value baseline, small batch 128 (more updates)"),
        dict(frames_per_batch=256, lr=3e-3, note="batch 256, lr 3e-3"),
        dict(baseline="value", frames_per_batch=256, gamma=0.98, note="value baseline, batch 256, gamma 0.98"),
    ],
    "ppo": [
        dict(note="baseline: clip 0.2, 4 epochs, ent 0.01"),
        dict(lr=1e-3, note="lr 1e-3"),
        dict(lr=1e-4, note="lr 1e-4"),
        dict(clip_epsilon=0.1, note="clip 0.1"),
        dict(clip_epsilon=0.3, note="clip 0.3"),
        dict(epochs=8, note="8 epochs per batch"),
        dict(entropy_coef=0.05, note="entropy coef 0.05"),
        dict(entropy_coef=0.0, note="entropy coef 0"),
        dict(lmbda=0.9, gamma=0.98, note="GAE lambda 0.9, gamma 0.98"),
        dict(frames_per_batch=2048, note="batch 2048 frames"),
    ],
    "a2c": [
        dict(note="baseline: lr 1e-3, 256 frames/batch"),
        dict(lr=3e-4, note="lr 3e-4"),
        dict(lr=3e-3, note="lr 3e-3"),
        dict(frames_per_batch=128, note="rollout 128 frames"),
        dict(frames_per_batch=512, note="rollout 512 frames"),
        dict(entropy_coef=0.05, note="entropy coef 0.05"),
        dict(entropy_coef=0.0, note="entropy coef 0"),
        dict(critic_coef=1.0, note="critic coef 1.0"),
        dict(lmbda=0.9, gamma=0.98, note="GAE lambda 0.9, gamma 0.98"),
        dict(hidden=[128, 128], note="net 128x2"),
    ],
}


def run_config(job: tuple) -> dict:
    """train one sweep config, job = (algo, index, frames, seed). runs in a worker process"""
    algo, idx, frames, seed = job
    cfg = dict(SWEEPS[algo][idx])
    run_id = f"cfg{idx + 1:02d}"
    if algo == "dqn":
        from training.dqn_training import train_dqn
        s = train_dqn(cfg, run_id, seed=seed, total_frames=frames)
    else:
        from training.pg_training import train_pg
        s = train_pg(algo, cfg, run_id, seed=seed, total_frames=frames)
    return s


# ---- tables + plots ----
ALGOS = ("dqn", "reinforce", "ppo", "a2c")
ALGO_LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
ALGO_COLOR = {"dqn": "#0072B2", "reinforce": "#D55E00", "ppo": "#009E73", "a2c": "#CC79A7"}
CONVERGENCE_RETURN = 5.0


def load_sweep_runs() -> dict[str, list[dict]]:
    """latest result for each cfgNN run, per algo"""
    latest: dict[tuple, dict] = {}
    path = ROOT / "logs" / "runs.jsonl"
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if str(r["run_id"]).startswith("cfg"):
            latest[(r["algo"], r["run_id"])] = r
    return {a: sorted((r for (al, _), r in latest.items() if al == a), key=lambda r: r["run_id"]) for a in ALGOS}


def _curve(algo: str, run_id: str):
    import pandas as pd
    return pd.read_csv(ROOT / "logs" / "curves" / f"{algo}_{run_id}.csv")


def frames_to_converge(algo: str, run_id: str) -> float:
    c = _curve(algo, run_id).dropna(subset=["eval_return"])
    hits = c["frames"].to_numpy()[c["eval_return"].to_numpy() >= CONVERGENCE_RETURN]
    return float(hits[0]) if len(hits) else float("nan")


def _hp_text(r: dict) -> str:
    c = r["config"]
    keys = {"dqn": ["lr", "gamma", "double_dqn", "eps_anneal_frames", "eps_end", "target", "buffer_size", "batch_size", "hidden"],
            "reinforce": ["lr", "gamma", "baseline", "frames_per_batch", "entropy_coef", "hidden"],
            "ppo": ["lr", "gamma", "lmbda", "clip_epsilon", "epochs", "entropy_coef", "frames_per_batch", "hidden"],
            "a2c": ["lr", "gamma", "lmbda", "frames_per_batch", "entropy_coef", "critic_coef", "hidden"]}[r["algo"]]
    return ", ".join(f"{k}={c[k]}" for k in keys if k in c)


def summary_table() -> str:
    with open(ROOT / "logs" / "tables" / "sweep_all.csv") as f:
        all_rows = list(csv.DictReader(f))
    lines = ["Algorithm  Best run  Train return  Delivered train / held-out / stress  Crashed (train)"]
    for a in ALGOS:
        rows = [r for r in all_rows if r["algo"] == a]
        if rows:
            b = max(rows, key=lambda r: float(r["train_return"]))
            v = {k: float(b[k]) for k in ("train_return", "train_delivered", "heldout_delivered", "stress_delivered", "train_crashed")}
            lines.append(f"{ALGO_LABEL[a]:<10} {b['run']:<9} {v['train_return']:+12.2f}  {v['train_delivered']:>15.0%} / "
                         f"{v['heldout_delivered']:.0%} / {v['stress_delivered']:.0%} {v['train_crashed']:>14.0%}")
    return "\n".join(lines)


def analyze() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    runs = load_sweep_runs()
    tdir, adir = ROOT / "logs" / "tables", ROOT / "assets"
    tdir.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)
    rows, best = [], {}
    for a in ALGOS:
        for r in runs[a]:
            e = r["final_eval"]
            rows.append({"algo": a, "run": r["run_id"], "note": r["config"].get("note", ""), "hyperparameters": _hp_text(r),
                         "frames": r["frames"], "train_return": e["train"]["return_mean"],
                         "train_delivered": e["train"]["success_rate"], "train_crashed": e["train"]["crash_rate"],
                         "heldout_return": e["heldout"]["return_mean"], "heldout_delivered": e["heldout"]["success_rate"],
                         "stress_return": e["stress"]["return_mean"], "stress_delivered": e["stress"]["success_rate"],
                         "frames_to_converge": frames_to_converge(a, r["run_id"]), "train_time_min": r["train_time_s"] / 60,
                         "model": r.get("model_path", "")})
        if runs[a]:
            best[a] = max((x for x in rows if x["algo"] == a), key=lambda x: x["train_return"])
    df = pd.DataFrame(rows)
    df.to_csv(tdir / "sweep_all.csv", index=False)
    overall = max(best, key=lambda a: best[a]["train_return"])
    (ROOT / "models" / "best.json").write_text(json.dumps(
        {"overall": overall, "models": {a: {"path": best[a]["model"], "run": best[a]["run"]} for a in best}}, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25})

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a in best:
        c = _curve(a, best[a]["run"])
        ev = c.dropna(subset=["eval_return"])
        ax[0].plot(ev["frames"], ev["eval_return"], color=ALGO_COLOR[a], lw=2, label=f"{ALGO_LABEL[a]} ({best[a]['run']})")
        tr = c.dropna(subset=["train_return"])
        ax[1].plot(tr["frames"], tr["train_return"], color=ALGO_COLOR[a], lw=2, label=ALGO_LABEL[a])
    ax[0].set(title="Greedy evaluation return (training layouts)", xlabel="environment frames", ylabel="mean return")
    ax[1].set(title="Rolling training-episode return (last 50 episodes)", xlabel="environment frames", ylabel="mean return")
    ax[0].legend(); fig.tight_layout(); fig.savefig(adir / "fig_reward_comparison.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True, sharey=True)
    for ax, a in zip(axs.flat, ALGOS):
        if a not in best:
            continue
        c = _curve(a, best[a]["run"])
        tr, ev = c.dropna(subset=["train_return"]), c.dropna(subset=["eval_return"])
        ax.plot(tr["frames"], tr["train_return"], color=ALGO_COLOR[a], lw=1.2, alpha=0.7, label="training episodes (mean of last 50)")
        ax.plot(ev["frames"], ev["eval_return"], color="black", lw=1.8, label="greedy evaluation (30 episodes)")
        ax.axhline(CONVERGENCE_RETURN, color="grey", ls="--", lw=1)
        ax.set(title=f"{ALGO_LABEL[a]} ({best[a]['run']})")
    for ax in axs[1]:
        ax.set_xlabel("environment frames")
    for ax in axs[:, 0]:
        ax.set_ylabel("episode return")
    axs[0, 0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(adir / "fig_cumulative_rewards.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in runs["dqn"]:
        c = _curve("dqn", r["run_id"]).dropna(subset=["loss"])
        ax.plot(c["frames"], c["loss"], lw=1.6, alpha=0.9, label=r["run_id"])
    ax.set(title="DQN objective: smooth-L1 TD loss, all configurations", xlabel="environment frames",
           ylabel="loss (mean of last 200 updates)", yscale="log")
    ax.legend(ncol=2, fontsize=8); fig.tight_layout(); fig.savefig(adir / "fig_dqn_loss.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for k, a in enumerate(("reinforce", "ppo", "a2c")):
        for r in runs[a]:
            c = _curve(a, r["run_id"]).dropna(subset=["entropy"])
            is_best = a in best and r["run_id"] == best[a]["run"]
            ax[k].plot(c["frames"], c["entropy"], color=ALGO_COLOR[a], lw=2.4 if is_best else 0.9, alpha=1 if is_best else 0.45)
        ax[k].set(title=f"{ALGO_LABEL[a]} (best in bold)", xlabel="environment frames")
    ax[0].set_ylabel("policy entropy (nats; max = ln 8 = 2.08)")
    fig.tight_layout(); fig.savefig(adir / "fig_pg_entropy.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.2
    nmax = int(df.groupby('algo').size().to_numpy().max())
    for k, a in enumerate(ALGOS):
        d = df[df.algo == a]
        vals = d.frames_to_converge.fillna(0).to_numpy() / 1000
        ax.bar(np.arange(len(vals)) + (k - 1.5) * w, vals, w, color=ALGO_COLOR[a], label=ALGO_LABEL[a])
        for i, v in enumerate(d.frames_to_converge.to_numpy()):
            if np.isnan(v):
                ax.text(i + (k - 1.5) * w, 2, "x", ha="center", fontsize=8, color=ALGO_COLOR[a])
    ax.set(title=f"Frames until greedy evaluation return first reaches {CONVERGENCE_RETURN:+.0f}  (x = never)",
           xlabel="configuration number", ylabel="thousand frames", xticks=range(nmax), xticklabels=[f"{i:02d}" for i in range(1, nmax + 1)])
    ax.legend(ncol=4); fig.tight_layout(); fig.savefig(adir / "fig_convergence.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    splits = [("train_delivered", "train layouts"), ("heldout_delivered", "held-out layouts"), ("stress_delivered", "stress (6 obstacles, 2x wind)")]
    for k, a in enumerate(best):
        ax.bar(np.arange(3) + (k - 1.5) * 0.2, [best[a][s] * 100 for s, _ in splits], 0.2, color=ALGO_COLOR[a], label=ALGO_LABEL[a])
    ax.set(title="Generalization: parcels delivered by the best agent of each algorithm", ylabel="% of 100 episodes",
           xticks=range(3), xticklabels=[n for _, n in splits], ylim=(0, 112))
    ax.legend(ncol=4, fontsize=8, loc="upper center"); fig.tight_layout(); fig.savefig(adir / "fig_generalization.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for k, a in enumerate(ALGOS):
        d = df[df.algo == a]
        ax[k].bar(range(len(d)), d.train_return, color=ALGO_COLOR[a])
        ax[k].set(title=ALGO_LABEL[a], xticks=range(len(d)), xticklabels=[r[-2:] for r in d.run], xlabel="configuration")
    ax[0].set_ylabel("final greedy return (train layouts)")
    fig.tight_layout(); fig.savefig(adir / "fig_sweep_returns.png", dpi=150); plt.close(fig)
    print(summary_table())
    print(f"Overall best: {ALGO_LABEL[overall]} {best[overall]['run']}. Tables -> logs/tables/, figures -> assets/fig_*.png, models/best.json updated.")
