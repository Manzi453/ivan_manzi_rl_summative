"""Shared helpers for all four TorchRL training scripts.

Every algorithm uses exactly the same environment, the same layout splits (train / held-out / stress),
the same total frame budget and the same evaluation protocol, so the DQN / REINFORCE / PPO / A2C
comparison is fair.
"""
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
from torchrl.envs import Compose, GymWrapper, RewardSum, TransformedEnv
from torchrl.modules import MLP

from environment.custom_env import N_ACTIONS, OBS_DIM, DeliveryDroneEnv

os.environ.setdefault("RL_WARNINGS", "False")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TOTAL_FRAMES = 150_000         # identical environment-interaction budget for every run
EVAL_EVERY = 10_000            # frames between greedy evaluations on the training layouts
EVAL_EPISODES = 30
SPLITS = ("train", "heldout", "stress")
OUTCOMES = ("delivered", "damaged", "parcel_lost", "crashed", "battery_empty", "left_geofence", "timeout")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int = 0) -> TransformedEnv:
    """TorchRL-wrapped training environment (training layouts only)."""
    env = TransformedEnv(GymWrapper(DeliveryDroneEnv(split="train"), categorical_action_encoding=True),
                         Compose(RewardSum()))
    env.set_seed(seed)
    return env


def build_mlp(out_features: int, hidden: list[int] | tuple[int, ...], activation: str = "tanh") -> MLP:
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    return MLP(in_features=OBS_DIM, out_features=out_features, num_cells=list(hidden), activation_class=act)


# --------------------------------------------------------------------------- #
# Evaluation (greedy / deterministic policy on the raw Gymnasium environment)
# --------------------------------------------------------------------------- #
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
    """Evaluate a policy factory(env) -> act on all three splits (used for random / scripted rows)."""
    out = {}
    for s in SPLITS:
        env = DeliveryDroneEnv(split=s)
        act = policy_factory(env)
        out[s] = evaluate(act, s, n_episodes)
    return out


def full_evaluation(act_fn: Callable[[np.ndarray], int], n_episodes: int = 100) -> dict:
    return {s: evaluate(act_fn, s, n_episodes) for s in SPLITS}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class RunLogger:
    """Writes one CSV per run (training curve, losses, entropy, eval) + final summary row."""

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
    """Append one JSON line per finished run to logs/runs.jsonl."""
    path = (logdir or ROOT / "logs") / "runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(summary) + "\n")


class EpisodeTracker:
    """Rolling mean of completed training-episode returns, extracted from collector batches."""

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
