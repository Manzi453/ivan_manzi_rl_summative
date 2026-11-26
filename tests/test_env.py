import numpy as np
import pytest
from gymnasium.spaces import Discrete

from environment.custom_env import (ACTION_NAMES, CLIMB, DESCEND, HOLD, OBS_DIM, PAD_RADIUS, RELEASE, SPLITS, DeliveryDroneEnv,
                                    evaluate_policy, make_layout, scripted_policy)


def test_spaces_and_obs_range():
    env = DeliveryDroneEnv()
    assert isinstance(env.action_space, Discrete) and env.action_space.n == 8 == len(ACTION_NAMES)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    for _ in range(30):
        obs, r, te, tr, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if te or tr:
            obs, _ = env.reset()


def test_start_state():
    env = DeliveryDroneEnv()
    _, info = env.reset(seed=1)
    assert info["battery"] == 100.0 and info["speed"] == 0.0 and info["steps"] == 0
    assert info["outcome"] == "flying" and env.parcel_attached
    assert 0.4 <= info["height"] <= 0.6


def test_determinism_same_seed():
    def run(seed):
        env = DeliveryDroneEnv()
        env.reset(seed=seed)
        out = []
        for a in [1, 1, 3, 3, 5, 0, 3, 6, 1, 0]:
            obs, r, *_ = env.step(a)
            out.append((obs.copy(), r))
        return out
    a, b = run(7), run(7)
    for (o1, r1), (o2, r2) in zip(a, b):
        assert np.allclose(o1, o2) and r1 == r2


def test_actions_change_state():
    env = DeliveryDroneEnv()
    env.reset(seed=2)
    z0 = env.pos[2]
    for _ in range(5):
        env.step(CLIMB)
    assert env.pos[2] > z0 + 0.3
    yaw0 = env.yaw
    env.step(5)
    assert env.yaw != yaw0
    env.step(3)
    assert env.fwd_cmd == 0.5


def test_release_interlock_far_from_pad():
    env = DeliveryDroneEnv()
    env.reset(seed=3)
    _, r, te, tr, info = env.step(RELEASE)
    assert not te and info["outcome"] == "flying" and r < 0


def test_parcel_lost_near_but_outside_pad():
    env = DeliveryDroneEnv()
    env.reset(seed=3)
    pad = env.layout["pad"]
    env.data.qpos[:3] = (pad[0] + 1.5, pad[1], 0.6)
    _, r, te, _, info = env.step(RELEASE)
    assert te and info["outcome"] == "parcel_lost" and r < -4


def test_soft_delivery_and_damaged():
    env = DeliveryDroneEnv()
    env.reset(seed=4)
    pad = env.layout["pad"]
    env.data.qpos[:3] = (pad[0], pad[1], 0.6)
    env.data.qvel[:] = 0
    _, r, te, _, info = env.step(RELEASE)
    assert te and info["outcome"] == "delivered" and r > 9
    env.reset(seed=4)
    env.data.qpos[:3] = (pad[0], pad[1], 3.0)
    _, r, te, _, info = env.step(RELEASE)
    assert te and info["outcome"] == "damaged" and 0 < r < 5


def test_crash_into_obstacle():
    env = DeliveryDroneEnv()
    env.reset(seed=5)
    ox, oy, rad, h = env.layout["obstacles"][0]
    env.data.qpos[:3] = (ox - rad - 0.35, oy, min(h / 2, 1.0))
    env.data.qpos[3] = 0.0
    env.fwd_cmd = 2.0
    out, r = None, 0.0
    for _ in range(20):
        _, r, te, tr, info = env.step(HOLD)
        if te:
            out = info["outcome"]
            break
    assert out == "crashed" and r == -5.0


def test_flying_into_ground_is_a_crash():
    env = DeliveryDroneEnv()
    env.reset(seed=5)
    info = {}
    for _ in range(10):
        _, r, te, _, info = env.step(DESCEND)
        if te:
            break
    assert info["outcome"] == "crashed" and env.pos[2] > -0.02


def test_battery_geofence_timeout():
    env = DeliveryDroneEnv()
    env.reset(seed=6)
    env.battery = 0.1
    _, r, te, _, info = env.step(HOLD)
    assert te and info["outcome"] == "battery_empty"
    env.reset(seed=6)
    env.data.qpos[0] = 6.45
    env.fwd_cmd = 2.0
    env.data.qpos[3] = 0.0
    out = None
    for _ in range(10):
        _, r, te, _, info = env.step(HOLD)
        if te:
            out = info["outcome"]
            break
    assert out == "left_geofence"
    env2 = DeliveryDroneEnv(max_steps=5)
    env2.reset(seed=6)
    tr, info = False, {}
    for _ in range(5):
        _, r, te, tr, info = env2.step(HOLD)
    assert tr and info["outcome"] == "timeout"


def test_splits_do_not_overlap_and_layouts_valid():
    assert SPLITS["train"][1] <= SPLITS["heldout"][0]
    for lid in list(range(0, 400, 40)) + [10_000, 20_000]:
        lay = make_layout(lid, 6 if lid >= 20_000 else 4)
        assert np.linalg.norm(lay["pad"] - lay["hub"]) >= 6.0
        for x, y, r, h in lay["obstacles"]:
            assert np.hypot(x - lay["pad"][0], y - lay["pad"][1]) >= 2.0 - 1e-6
            assert np.hypot(x - lay["hub"][0], y - lay["hub"][1]) >= 2.2 - 1e-6
    assert DeliveryDroneEnv(split="heldout").reset(seed=0)[1]["layout_id"] >= 10_000


def test_scripted_baseline_solves_task():
    res = evaluate_policy(lambda e: scripted_policy(), DeliveryDroneEnv(), episodes=20, seed=100)
    assert res["outcomes"].get("delivered", 0) >= 14


def test_to_json():
    env = DeliveryDroneEnv()
    env.reset(seed=0)
    env.step(CLIMB)
    j = env.to_json()
    assert {"drone", "battery", "pad", "obstacles", "wind", "path", "outcome"} <= set(j)
    import json
    json.dumps(j)


def test_render_rgb_array():
    env = DeliveryDroneEnv(render_mode="rgb_array", width=320, height=240)
    env.reset(seed=0)
    f = env.render()
    assert f is not None
    assert f.shape == (240, 320, 3) and f.dtype == np.uint8 and f.std() > 5
    env.close()
