"""Policy-gradient methods with TorchRL: REINFORCE, PPO and A2C.

TorchRL building blocks: GymWrapper env, ProbabilisticActor (Categorical policy),
ValueOperator (critic), Collector, GAE, ClipPPOLoss, A2CLoss and
`reward2go` for the Monte-Carlo returns used by REINFORCE.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.value.functional import reward2go

from environment.custom_env import N_ACTIONS
from training import (EVAL_EVERY, ROOT, TOTAL_FRAMES, EpisodeTracker, RunLogger, build_mlp,
                             evaluate, full_evaluation, make_env, save_summary, seed_everything)

DEFAULTS = {
    "reinforce": dict(lr=1e-3, gamma=0.99, hidden=(64, 64), activation="tanh", entropy_coef=0.01,
                      frames_per_batch=512, baseline="batch_mean", normalize_returns=True, max_grad_norm=1.0,
                      critic_lr=1e-3),
    "ppo": dict(lr=3e-4, gamma=0.99, lmbda=0.95, hidden=(64, 64), activation="tanh", entropy_coef=0.01,
                critic_coef=1.0, clip_epsilon=0.2, epochs=4, minibatch=128, frames_per_batch=1024,
                max_grad_norm=0.5),
    "a2c": dict(lr=1e-3, gamma=0.99, lmbda=0.95, hidden=(64, 64), activation="tanh", entropy_coef=0.01,
                critic_coef=0.5, frames_per_batch=256, max_grad_norm=0.5),
}


def make_policy_act(actor_net: torch.nn.Module):
    """Deterministic (arg-max) policy used for evaluation and the demo."""
    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(actor_net(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())
    return act


def _entropy(lv, actor, td) -> float:
    """Policy entropy; TorchRL omits it from the loss output when the entropy coefficient is 0."""
    if "entropy" in lv.keys():
        return float(lv["entropy"].detach())
    with torch.no_grad():
        return float(actor.get_dist(td).entropy().mean())


def _build_actor_critic(env, c, need_critic: bool):
    actor_net = build_mlp(N_ACTIONS, c["hidden"], c["activation"])
    actor = ProbabilisticActor(
        TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"]),
        spec=env.action_spec, in_keys=["logits"], distribution_class=Categorical, return_log_prob=True,
    )
    critic = ValueOperator(build_mlp(1, c["hidden"], c["activation"]), in_keys=["observation"]) if need_critic else None
    return actor_net, actor, critic


def train_pg(algo: str, cfg: dict, run_id: str, seed: int = 0, total_frames: int = TOTAL_FRAMES,
             save: bool = True, verbose: bool = False) -> dict:
    assert algo in DEFAULTS, algo
    c = {**DEFAULTS[algo], **cfg}
    torch.set_num_threads(1)
    seed_everything(seed)
    env = make_env(seed)
    use_value_baseline = algo == "reinforce" and c["baseline"] == "value"
    actor_net, actor, critic = _build_actor_critic(env, c, need_critic=(algo != "reinforce" or use_value_baseline))

    if algo == "ppo":
        adv_mod = GAE(gamma=c["gamma"], lmbda=c["lmbda"], value_network=critic, average_gae=True)
        loss_mod = ClipPPOLoss(actor, critic, clip_epsilon=c["clip_epsilon"], entropy_coeff=c["entropy_coef"],
                               critic_coeff=c["critic_coef"], loss_critic_type="smooth_l1", normalize_advantage=False)
        params = loss_mod.parameters()
    elif algo == "a2c":
        adv_mod = GAE(gamma=c["gamma"], lmbda=c["lmbda"], value_network=critic, average_gae=True)
        loss_mod = A2CLoss(actor, critic, entropy_coeff=c["entropy_coef"], critic_coeff=c["critic_coef"],
                           loss_critic_type="smooth_l1")
        params = loss_mod.parameters()
    else:  # reinforce: policy only, Monte-Carlo reward-to-go
        adv_mod = loss_mod = None
        params = actor.parameters()
    optim = torch.optim.Adam(params, lr=c["lr"])
    critic_optim = torch.optim.Adam(critic.parameters(), lr=c["critic_lr"]) if use_value_baseline else None

    collector = Collector(lambda: make_env(seed), actor, frames_per_batch=c["frames_per_batch"],
                          total_frames=total_frames)
    logger = RunLogger(algo, run_id)
    tracker = EpisodeTracker()
    frames, next_eval, t0 = 0, EVAL_EVERY, time.time()
    stats = {"loss": [], "obj": [], "crit": [], "ent": []}

    def recent(k):
        return float(np.mean(stats[k][-30:])) if stats[k] else float("nan")

    for batch in collector:
        frames += batch.numel()
        tracker.update(batch)

        if algo == "reinforce":
            done = batch["next", "done"]
            idx = done.reshape(-1).nonzero()
            if len(idx) == 0:
                continue
            last = int(idx[-1]) + 1                      # drop the trailing, unfinished episode
            b = batch[:last]
            with torch.no_grad():
                g_raw = reward2go(b["next", "reward"], b["next", "done"], c["gamma"], time_dim=-2).reshape(-1)
                if c["baseline"] == "value":
                    g = g_raw - critic(b.clone(False))["state_value"].reshape(-1)
                elif c["baseline"] == "batch_mean":
                    g = g_raw - g_raw.mean()
                else:
                    g = g_raw
                if c["normalize_returns"]:
                    g = g / (g.std() + 1e-8)
            dist = actor.get_dist(b)
            logp = dist.log_prob(b["action"])
            entropy = dist.entropy().mean()
            loss_obj = -(logp * g).mean()
            loss = loss_obj - c["entropy_coef"] * entropy
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), c["max_grad_norm"])
            optim.step()
            crit_val = float("nan")
            if use_value_baseline:
                v = critic(b.clone(False))["state_value"].reshape(-1)
                critic_loss = torch.nn.functional.mse_loss(v, g_raw)
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()
                crit_val = float(critic_loss.detach())
            stats["loss"].append(float(loss.detach())); stats["obj"].append(float(loss_obj.detach()))
            stats["crit"].append(crit_val); stats["ent"].append(float(entropy.detach()))
        elif algo == "a2c":
            with torch.no_grad():
                adv_mod(batch)
            lv = loss_mod(batch)
            loss = sum(v for k, v in lv.items() if k.startswith("loss_"))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), c["max_grad_norm"])
            optim.step()
            stats["loss"].append(float(loss.detach())); stats["obj"].append(float(lv["loss_objective"].detach()))
            stats["crit"].append(float(lv["loss_critic"].detach())); stats["ent"].append(_entropy(lv, actor, batch))
        else:  # ppo
            with torch.no_grad():
                adv_mod(batch)
            data = batch.reshape(-1)
            for _ in range(c["epochs"]):
                perm = torch.randperm(data.numel())
                for chunk in perm.split(c["minibatch"]):
                    lv = loss_mod(data[chunk])
                    loss = sum(v for k, v in lv.items() if k.startswith("loss_"))
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), c["max_grad_norm"])
                    optim.step()
                    stats["loss"].append(float(loss.detach())); stats["obj"].append(float(lv["loss_objective"].detach()))
                    stats["crit"].append(float(lv["loss_critic"].detach())); stats["ent"].append(_entropy(lv, actor, data[chunk]))
        collector.update_policy_weights_()

        if frames >= next_eval or frames >= total_frames:
            ev = evaluate(make_policy_act(actor_net), "train")
            next_eval += EVAL_EVERY
            logger.log(frames=frames, train_return=tracker.mean, eval_return=ev["return_mean"],
                       eval_success=ev["success_rate"], loss=recent("loss"), loss_objective=recent("obj"),
                       loss_critic=recent("crit"), entropy=recent("ent"))
            if verbose:
                print(f"[{algo} {run_id}] frames={frames} train={tracker.mean:.2f} eval={ev['return_mean']:.2f} "
                      f"succ={ev['success_rate']:.2f} ent={recent('ent'):.3f} loss={recent('loss'):.4f}")
        else:
            logger.log(frames=frames, train_return=tracker.mean, loss=recent("loss"), loss_objective=recent("obj"),
                       loss_critic=recent("crit"), entropy=recent("ent"))
        if frames >= total_frames:
            break
    collector.shutdown()
    logger.close()
    act = make_policy_act(actor_net)
    final = full_evaluation(act)
    summary = {"algo": algo, "run_id": run_id, "seed": seed, "config": {**c, "hidden": list(c["hidden"])},
               "frames": frames, "train_time_s": time.time() - t0, "final_train_return": tracker.mean,
               "final_eval": final}
    if save:
        path = ROOT / "models" / "pg" / f"{algo}_{run_id}.pt"
        torch.save({"algo": algo, "hidden": list(c["hidden"]), "activation": c["activation"],
                    "state_dict": actor_net.state_dict()}, path)
        summary["model_path"] = str(path.relative_to(ROOT))
        save_summary(summary)
    return summary


if __name__ == "__main__":
    import json
    import sys
    algo = sys.argv[1] if len(sys.argv) > 1 else "ppo"
    s = train_pg(algo, {}, "debug", save=False, verbose=True, total_frames=int(sys.argv[2]) if len(sys.argv) > 2 else 12_000)
    print(json.dumps({k: (round(v["return_mean"], 2), v["success_rate"]) for k, v in s["final_eval"].items()}))
    print("time", round(s["train_time_s"], 1))
