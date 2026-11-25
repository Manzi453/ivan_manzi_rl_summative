# Medical Delivery Drone: Reinforcement Learning (MuJoCo + TorchRL)

A drone takes off from a hub carrying a medical parcel (blood or vaccines). It must release the parcel
over a rural clinic's landing pad (low, slow and inside the pad) without crashing into obstacles,
leaving the geofence or running out of battery. Rwanda already runs this kind of drone delivery.

Four agents are trained on the same custom Gymnasium environment with TorchRL: **DQN** (value-based) and
**REINFORCE, PPO, A2C** (policy-gradient).

## Project structure

```
.
├── pyproject.toml          # dependencies (managed with uv)
├── uv.lock
├── README.md
├── main.py                 # entry point: train an agent / replay an agent
├── environment/
│   ├── __init__.py         # registers DeliveryDrone-v0
│   ├── custom_env.py       # DeliveryDroneEnv + random / scripted reference policies
│   └── rendering.py        # MuJoCo scene (MJCF), off-screen + live-window rendering
├── training/
│   ├── __init__.py         # shared helpers: env factory, evaluation, CSV logging, model loading
│   ├── dqn_training.py     # DQN (QValueActor, EGreedyModule, replay buffer, DQNLoss)
│   └── pg_training.py     # REINFORCE, PPO, A2C (ProbabilisticActor, GAE, ClipPPOLoss, A2CLoss)
├── models/
│   ├── dqn/                # saved DQN checkpoints
│   └── pg/                 # saved REINFORCE / PPO / A2C checkpoints
├── logs/                   # training curves (logs/curves/*.csv) and run summaries
├── assets/                 # environment preview and demo GIF
└── tests/                  # pytest suite for the environment
```

## Setup

Requires Python 3.10 to 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest            # 13 environment tests
```

## Usage

```bash
# Train one agent (writes models/<dqn|pg>/ and logs/curves/<algo>_<run-id>.csv)
uv run main.py train --algo dqn --frames 150000 --seed 0
uv run main.py train --algo ppo            # dqn | reinforce | ppo | a2c

# Replay an agent
uv run mjpython main.py play --algo scripted     # live MuJoCo window (macOS needs mjpython)
uv run main.py play --algo scripted --headless   # no window: saves a GIF in assets/
uv run main.py play --algo ppo --split heldout   # layouts never seen in training
uv run main.py play --split stress               # 6 obstacles + stronger wind
```

`--algo` for `play` is `best` (default, the newest checkpoint or the one named in `models/best.json`),
`dqn`, `reinforce`, `ppo`, `a2c`, `scripted` or `random`. Trained agents need a checkpoint in `models/`;
`scripted` and `random` work out of the box.

## Environment: `DeliveryDrone-v0`

Physics is real MuJoCo (gravity, inertia, collisions). The agent picks a high-level discrete command every
0.2 s; a small velocity controller turns the current commands into forces. Episodes last at most 120 steps.

**Actions (Discrete 8):** `HOLD`, `CLIMB`, `DESCEND`, `SPEED_UP`, `SLOW_DOWN`, `TURN_LEFT`, `TURN_RIGHT`, `RELEASE`.
`RELEASE` is interlocked: farther than 2 m from the clinic it does nothing and costs -0.05.

**Observation (Box, 19 values in [-1, 1]):** pad position in the drone frame (forward, left), pad distance and
bearing, height, velocity (forward, lateral, vertical), speed command, battery, five horizontal range-finders
(-90, -45, 0, +45, +90 degrees), a downward range-finder, wind (forward, left) and steps left.

**Start state:** drone on the hub at 0.5 m height, parcel attached, full battery, random wind. The layout
(hub, pad and obstacle positions) is fixed by a layout id, so seeds are reproducible.

**Terminal states:**

| Outcome | Reward |
|---|---|
| `delivered`: released inside the pad (0.6 m), below 1 m and slower than 0.6 m/s | +10, plus up to +2 for battery left |
| `damaged`: released over the pad but too high or too fast | +2 |
| `parcel_lost`, `crashed`, `battery_empty`, `left_geofence` | -5 |
| `timeout` | -2 |

**Reward shaping:** -0.02 per step, plus potential-based terms for progress toward the pad, facing the pad,
and being low and slow near it. Potential-based shaping leaves the optimal policy unchanged (Ng et al., 1999).

**Layout splits:** `train` (400 layouts, 4 obstacles), `heldout` (100 layouts never trained on) and `stress`
(100 layouts, 6 obstacles, double wind).

## Fair comparison

Every algorithm uses the same environment, the same budget (150,000 frames), the same evaluation protocol
(greedy policy, 30 episodes on the training layouts every 10,000 frames, and 100 episodes on each split at the
end). Hyperparameter defaults are in `DEFAULTS` at the top of `training/dqn_training.py` and
`training/pg_training.py`. Training curves from earlier runs are in `logs/curves/`.
