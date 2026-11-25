"""Entry point for the medical-delivery-drone RL project.

    uv run main.py train --algo dqn|reinforce|ppo|a2c [--frames 150000] [--seed 0] [--run-id NAME]
    uv run main.py play  [--algo best|dqn|reinforce|ppo|a2c|scripted|random] [--episodes 3] [--split train|heldout|stress]
                         [--headless] [--fps 10] [--follow] [--snapshot]

`play` opens a live MuJoCo window (on macOS start it with `uv run mjpython main.py play ...`) and prints a
step-by-step log. With --headless it renders off-screen and saves a GIF in assets/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment.custom_env import (ACTION_NAMES, DeliveryDroneEnv, random_policy,  # noqa: E402
                                    scripted_policy)


def load_agent(algo: str, env):
    if algo == "scripted":
        return scripted_policy(), "scripted"
    if algo == "random":
        return random_policy(env), "random"
    from training import best_model_path, load_policy      # trained TorchRL agents

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
            sys.exit("\nOn macOS the live MuJoCo window must be started with:\n    uv run mjpython main.py play ...\n"
                     "(or add --headless to save a GIF instead).")
        raise
    finally:
        env.close()


def train(algo: str, frames: int | None, seed: int, run_id: str | None) -> None:
    from training import TOTAL_FRAMES
    frames = frames or TOTAL_FRAMES
    run_id = run_id or f"manual{seed}"
    if algo == "dqn":
        from training.dqn_training import train_dqn
        summary = train_dqn({}, run_id, seed=seed, total_frames=frames, verbose=True)
    else:
        from training.pg_training import train_pg
        summary = train_pg(algo, {}, run_id, seed=seed, total_frames=frames, verbose=True)
    print(f"\nSaved model: {summary.get('model_path')}   curve: logs/curves/{algo}_{run_id}.csv")
    for split, e in summary["final_eval"].items():
        print(f"{split:<8} return {e['return_mean']:+.2f}  delivered {e['success_rate']:.0%}  crashed {e['crash_rate']:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Medical delivery drone RL")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train one agent with TorchRL and save it to models/")
    tr.add_argument("--algo", required=True, choices=["dqn", "reinforce", "ppo", "a2c"])
    tr.add_argument("--frames", type=int, default=None, help="environment-interaction budget (default 150000)")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--run-id", default=None)

    pl = sub.add_parser("play", help="replay an agent (live window or GIF)")
    pl.add_argument("--algo", default="best", choices=["best", "dqn", "reinforce", "ppo", "a2c", "scripted", "random"])
    pl.add_argument("--episodes", type=int, default=3)
    pl.add_argument("--headless", action="store_true")
    pl.add_argument("--split", default="train", choices=["train", "heldout", "stress"])
    pl.add_argument("--seed", type=int, default=2024)
    pl.add_argument("--fps", type=int, default=10, help="live replay speed (lower = easier to narrate)")
    pl.add_argument("--follow", action="store_true", help="camera follows the drone")
    pl.add_argument("--snapshot", action="store_true", help="headless: also save assets/env_preview.png")
    a = ap.parse_args()

    if a.cmd == "train":
        train(a.algo, a.frames, a.seed, a.run_id)
    else:
        play(a.algo, a.episodes, a.headless, a.split, a.seed, a.fps, a.follow, a.snapshot)


if __name__ == "__main__":
    main()
