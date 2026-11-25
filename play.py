"""Replay an agent in the MuJoCo drone world.

    uv run mjpython play.py --algo scripted            # live MuJoCo window (macOS needs mjpython) + terminal log
    uv run play.py --algo scripted --headless          # no window: renders off-screen and saves a GIF in assets/
    uv run play.py --split heldout --episodes 5        # layouts never seen in training
    uv run play.py --split stress                      # 6 obstacles + stronger wind

--algo: best (default) | dqn | reinforce | ppo | a2c | scripted | random
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment.baselines import random_policy, scripted_policy  # noqa: E402
from environment.custom_env import ACTION_NAMES, DeliveryDroneEnv  # noqa: E402


def load_agent(algo: str, env):
    if algo == "scripted":
        return scripted_policy(), "scripted"
    if algo == "random":
        return random_policy(env), "random"
    from training.policy_io import best_model_path, load_policy      # trained TorchRL agents

    path = best_model_path(None if algo == "best" else algo)
    act = load_policy(path)
    return act, act.algo


def play(algo: str = "best", episodes: int = 3, headless: bool = False, split: str = "train", seed: int = 2024,
         fps: int = 10, follow: bool = False, snapshot: bool = False) -> None:
    mode = "rgb_array" if headless else "human"
    env = DeliveryDroneEnv(render_mode=mode, split=split, width=720, height=480, follow=follow,
                           camera="chase" if follow and headless else "overview")
    act, name = load_agent(algo, env)
    print(f"Agent: {name.upper()} | split: {split} | mode: {'headless (GIF)' if headless else 'live MuJoCo window'}")
    frames, returns, wins = [], [], 0
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            print(f"\n=== Episode {ep + 1}: layout {info['layout_id']}, hub->clinic distance {info['pad_distance']:.1f} m, "
                  f"wind {sum(v * v for v in env.wind) ** 0.5:.2f} m/s^2 ===")
            if headless:
                frames.append(env.render(status=True))
            else:
                env.render()
                time.sleep(1.0)
            done, total = False, 0.0
            while not done:
                t0 = time.time()
                a = int(act(obs))
                obs, r, term, trunc, info = env.step(a)      # env.step renders the new state
                total += r
                done = term or trunc
                print(f"step {info['steps']:>3} | {ACTION_NAMES[a]:<10} | reward {r:+6.2f} | battery {info['battery']:5.1f}% | "
                      f"height {info['height']:4.2f} m | speed {info['speed']:4.2f} m/s | to pad {info['pad_distance']:5.2f} m")
                if headless:
                    if info["steps"] % 2 == 0 or done:
                        frames.append(env.render(status=True))
                else:
                    if not env._renderer.viewer_running():
                        print("window closed - stopping")
                        return
                    time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))
            wins += info["outcome"] == "delivered"
            returns.append(total)
            print(f"--> outcome: {info['outcome'].upper()} | return {total:+.2f} | steps {info['steps']} | battery left {info['battery']:.0f}%")
            if not headless:
                time.sleep(2.0)
            else:
                frames.extend([frames[-1]] * 6)
        print(f"\nSummary: {wins}/{episodes} delivered | mean return {sum(returns) / len(returns):+.2f}")
        if headless:
            import imageio.v2 as imageio

            out = ROOT / "assets" / f"demo_{name}_{split}.gif"
            out.parent.mkdir(exist_ok=True)
            imageio.mimsave(out, frames, duration=1000 / 10, loop=0)
            print(f"Saved replay GIF -> {out.relative_to(ROOT)}")
            if snapshot:
                import imageio.v2 as imageio2

                imageio2.imwrite(ROOT / "assets" / "env_preview.png", frames[len(frames) // 2])
    except RuntimeError as e:
        if "mjpython" in str(e):
            sys.exit("\nOn macOS the live MuJoCo window must be started with:\n    uv run mjpython play.py ...\n"
                     "(or add --headless to save a GIF instead).")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="best", choices=["best", "dqn", "reinforce", "ppo", "a2c", "scripted", "random"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--split", default="train", choices=["train", "heldout", "stress"])
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--fps", type=int, default=10, help="live replay speed (lower = easier to narrate)")
    p.add_argument("--follow", action="store_true", help="camera follows the drone")
    p.add_argument("--snapshot", action="store_true", help="headless: also save assets/env_preview.png")
    a = p.parse_args()
    play(a.algo, a.episodes, a.headless, a.split, a.seed, a.fps, a.follow, a.snapshot)
