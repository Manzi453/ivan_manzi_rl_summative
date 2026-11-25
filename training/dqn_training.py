"""Value-based method: Deep Q-Network (DQN) with TorchRL.

TorchRL building blocks: GymWrapper env, QValueActor, EGreedyModule, Collector,
ReplayBuffer + LazyTensorStorage, DQNLoss, SoftUpdate / HardUpdate.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from environment.custom_env import N_ACTIONS
from training import (EVAL_EVERY, ROOT, TOTAL_FRAMES, EpisodeTracker, RunLogger, build_mlp,
                             evaluate, full_evaluation, make_env, save_summary, seed_everything)

DEFAULTS = dict(
    lr=1e-3, gamma=0.99, hidden=(128, 128), activation="relu",
    eps_init=1.0, eps_end=0.05, eps_anneal_frames=20_000,   # exploration schedule
    buffer_size=50_000, batch_size=64, frames_per_batch=32, utd=4,   # replay + updates per collected batch
    target="soft", tau=0.01, hard_interval=500,             # target-network update rule
    double_dqn=False, loss_fn="smooth_l1", init_random_frames=1_000, max_grad_norm=10.0,
)


def make_policy_act(qnet: torch.nn.Module):
    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(qnet(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())
    return act


def train_dqn(cfg: dict, run_id: str, seed: int = 0, total_frames: int = TOTAL_FRAMES, save: bool = True,
              verbose: bool = False) -> dict:
    c = {**DEFAULTS, **cfg}
    torch.set_num_threads(1)
    seed_everything(seed)
    env = make_env(seed)
    qnet = build_mlp(N_ACTIONS, c["hidden"], c["activation"])
    qvalue = QValueActor(qnet, in_keys=["observation"], spec=env.action_spec)
    egreedy = EGreedyModule(env.action_spec, eps_init=c["eps_init"], eps_end=c["eps_end"],
                            annealing_num_steps=c["eps_anneal_frames"])
    explore_policy = TensorDictSequential(qvalue, egreedy)

    loss_mod = DQNLoss(qvalue, action_space=env.action_spec, loss_function=c["loss_fn"],
                       double_dqn=c["double_dqn"], delay_value=True)
    loss_mod.make_value_estimator(gamma=c["gamma"])
    if c["target"] == "soft":
        target_updater = SoftUpdate(loss_mod, tau=c["tau"])
    else:
        target_updater = HardUpdate(loss_mod, value_network_update_interval=c["hard_interval"])
    optim = torch.optim.Adam(loss_mod.parameters(), lr=c["lr"])
    rb = ReplayBuffer(storage=LazyTensorStorage(c["buffer_size"]))

    collector = Collector(lambda: make_env(seed), explore_policy, frames_per_batch=c["frames_per_batch"],
                          total_frames=total_frames, init_random_frames=c["init_random_frames"])
    logger = RunLogger("dqn", run_id)
    tracker = EpisodeTracker()
    frames, next_eval = 0, EVAL_EVERY
    losses, qmeans = [], []
    eval_ret, eval_succ = float("nan"), float("nan")
    t0 = time.time()
    for batch in collector:
        n = batch.numel()
        frames += n
        tracker.update(batch)
        rb.extend(batch.reshape(-1).cpu())
        if frames >= c["init_random_frames"]:
            for _ in range(c["utd"]):
                sample = rb.sample(c["batch_size"])
                lv = loss_mod(sample)
                optim.zero_grad()
                lv["loss"].backward()
                torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), c["max_grad_norm"])
                optim.step()
                target_updater.step()
                losses.append(float(lv["loss"].detach()))
            with torch.no_grad():
                qmeans.append(float(qnet(sample["observation"]).max(-1).values.mean()))
        egreedy.step(n)
        collector.update_policy_weights_()
        if frames >= next_eval or frames >= total_frames:
            ev = evaluate(make_policy_act(qnet), "train")
            eval_ret, eval_succ = ev["return_mean"], ev["success_rate"]
            next_eval += EVAL_EVERY
            logger.log(frames=frames, train_return=tracker.mean, eval_return=eval_ret, eval_success=eval_succ,
                       loss=float(np.mean(losses[-200:])) if losses else float("nan"),
                       loss_objective=float(np.mean(losses[-200:])) if losses else float("nan"),
                       epsilon=float(egreedy.eps), q_mean=float(np.mean(qmeans[-50:])) if qmeans else float("nan"))
            if verbose:
                print(f"[dqn {run_id}] frames={frames} train={tracker.mean:.2f} eval={eval_ret:.2f} succ={eval_succ:.2f} "
                      f"loss={np.mean(losses[-200:]):.4f} eps={float(egreedy.eps):.2f}")
        elif frames % (EVAL_EVERY // 6) < n:
            logger.log(frames=frames, train_return=tracker.mean,
                       loss=float(np.mean(losses[-200:])) if losses else float("nan"),
                       loss_objective=float(np.mean(losses[-200:])) if losses else float("nan"),
                       epsilon=float(egreedy.eps), q_mean=float(np.mean(qmeans[-50:])) if qmeans else float("nan"))
        if frames >= total_frames:
            break
    collector.shutdown()
    logger.close()
    act = make_policy_act(qnet)
    final = full_evaluation(act)
    summary = {"algo": "dqn", "run_id": run_id, "seed": seed, "config": {**c, "hidden": list(c["hidden"])},
               "frames": frames, "train_time_s": time.time() - t0, "final_train_return": tracker.mean,
               "final_eval": final}
    if save:
        path = ROOT / "models" / "dqn" / f"dqn_{run_id}.pt"
        torch.save({"algo": "dqn", "hidden": list(c["hidden"]), "activation": c["activation"],
                    "state_dict": qnet.state_dict()}, path)
        summary["model_path"] = str(path.relative_to(ROOT))
        save_summary(summary)
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(train_dqn({}, "debug", save=False, verbose=True, total_frames=12_000)["final_eval"], indent=1))
