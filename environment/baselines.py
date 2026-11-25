"""Reference policies used to sanity-check the environment and as comparison rows in the report.

* ``random_policy``  - uniform random actions (lower bound).
* ``scripted_policy`` - a hand-written "fly high, steer, slow down, descend, release" controller.
  It only sees the observation vector (same information as the RL agents), so it shows the task is
  solvable and gives a human-engineered reference for the learned agents.
"""
from __future__ import annotations

import math

from environment.custom_env import (CEILING, CLIMB, DESCEND, HOLD, RELEASE, SLOW_DOWN, SPEED_UP, TURN_LEFT,
                                    TURN_RIGHT)


def random_policy(env):
    return lambda obs: env.action_space.sample()


def scripted_policy(cruise_height: float = 3.9):
    """Returns act(obs) -> action using only the 19-dim observation."""
    def act(obs):
        dist = obs[2] * 10.0
        bearing = obs[3] * math.pi
        height = obs[4] * CEILING
        speed = math.hypot(obs[5] * 2.5, obs[6] * 2.5)
        cmd = obs[8] * 2.0
        if dist < 0.55 and height <= 0.9 and speed < 0.5:
            return RELEASE
        if abs(bearing) > 0.3 and dist > 0.55:                      # steer first (slow down to turn)
            if cmd > 1.0:
                return SLOW_DOWN
            return TURN_LEFT if bearing > 0 else TURN_RIGHT
        if dist > 3.5:
            if height < cruise_height - 0.3:
                return CLIMB
            return SPEED_UP if cmd < 2.0 else HOLD
        target = 1.0 if dist > 1.5 else 0.5  # approach: brake in stages
        if cmd > target:
            return SLOW_DOWN
        if height > 0.8:
            return DESCEND
        return SPEED_UP if cmd < target else HOLD
    return act


def evaluate(policy_factory, env, episodes: int = 50, seed: int = 0):
    """Run a policy for N episodes; returns mean return and outcome counts."""
    from collections import Counter

    outcomes, returns, steps = Counter(), [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        act = policy_factory(env)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            total += r
            done = term or trunc
        returns.append(total)
        steps.append(info["steps"])
        outcomes[info["outcome"]] += 1
    return {"return": sum(returns) / len(returns), "steps": sum(steps) / len(steps), "outcomes": dict(outcomes)}
