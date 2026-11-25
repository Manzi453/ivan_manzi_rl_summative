"""DeliveryDrone-v0: a MuJoCo medical-delivery drone (Gymnasium environment).

Mission: a drone takes off from a hub carrying a medical parcel (blood / vaccines) and must
release it over a rural clinic's landing pad - low, slow and inside the pad - without crashing
into obstacles, leaving the geofence or running out of battery. Rwanda already runs this kind of
drone delivery, which is what makes the problem realistic.

Physics is real MuJoCo (gravity, inertia, collisions). The agent chooses *high-level* discrete
commands; a small velocity controller turns the current commands into forces on the drone.
Everything below (actions, observations, rewards, start / terminal states) is documented in README.md.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from environment.rendering import ARENA, MAX_OBSTACLES, SceneRenderer, annotate, build_xml

# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
HOLD, CLIMB, DESCEND, SPEED_UP, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, RELEASE = range(8)
ACTION_NAMES = ["HOLD", "CLIMB", "DESCEND", "SPEED_UP", "SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT", "RELEASE"]

OBS_NAMES = [
    "pad_forward", "pad_left", "pad_distance", "pad_bearing",
    "height", "v_forward", "v_lateral", "v_vertical", "speed_command", "battery",
    "range_-90", "range_-45", "range_0", "range_+45", "range_+90", "range_down",
    "wind_forward", "wind_left", "steps_left",
]
OBS_DIM = len(OBS_NAMES)
N_ACTIONS = len(ACTION_NAMES)

# --------------------------------------------------------------------------- #
# Physical / task constants
# --------------------------------------------------------------------------- #
FRAME_SKIP = 20                 # 20 physics steps x 0.01 s  = 0.2 s per decision
DT = 0.2
MAX_STEPS = 120
CRUISE_STEP = 0.5               # SPEED_UP / SLOW_DOWN change the speed command by 0.5 m/s
MAX_SPEED_CMD = 2.0
CLIMB_RATE = 1.0                # m/s while CLIMB / DESCEND is chosen
TURN_RATE = 1.5                 # rad/s while TURN_* is chosen
KP_V, KP_VZ, KP_W = 4.0, 6.0, 8.0
A_MAX = 3.5                     # m/s^2 control-acceleration limit (gives real inertia)
MASS, IZ, G = 1.2, 0.02, 9.81
CEILING = 4.5                   # legal flight ceiling (m)
RANGE_MAX = 4.0                 # range-finder reach (m)
PAD_RADIUS = 0.6
SOFT_HEIGHT = 1.0               # release must be below this height ...
SOFT_SPEED = 0.6                # ... and slower than this ground speed (m/s)
START_Z = 0.5
RELEASE_LOCK = 2.0              # drop mechanism is locked farther than this from the clinic (safety interlock)
R_INVALID = -0.05               # RELEASE while locked

# Rewards
R_STEP = -0.02
R_PROGRESS = 0.5                # per metre closer to the pad          \
W_ALIGN = 0.4                   # facing the pad                        |  potential-based shaping terms:
W_HEIGHT = 0.25                 # low over the pad (per metre above 0.8)|  reward = phi(s') - phi(s), which leaves the
W_SPEED = 0.5                   # slow over the pad (per m/s)           /  optimal policy unchanged (Ng et al., 1999)
NEAR_RADIUS = 3.0               # height / speed shaping fades in inside this distance
R_DELIVERED = 10.0
R_BATTERY_BONUS = 2.0           # x battery fraction left on a soft delivery
R_DAMAGED = 2.0                 # released over the pad but too high / too fast
R_FAIL = -5.0                   # crash, battery empty, geofence, parcel lost
R_TIMEOUT = -2.0

SPLITS = {            # layout-id pools. "train" / "heldout" never overlap; "stress" is harder.
    "train": (0, 400),
    "heldout": (10_000, 10_100),
    "stress": (20_000, 20_100),
}
N_OBSTACLES = {"train": 4, "heldout": 4, "stress": 6}
WIND_MAX = {"train": 1.5, "heldout": 1.5, "stress": 3.0}   # m/s^2 disturbance


def make_layout(layout_id: int, n_obstacles: int) -> dict[str, Any]:
    """Deterministic world geometry for one layout id (hub, pad, obstacles)."""
    rng = np.random.default_rng(layout_id)
    for _ in range(200):
        hub = rng.uniform(-4.8, 4.8, 2)
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(6.0, 9.5)
        pad = hub + dist * np.array([math.cos(ang), math.sin(ang)])
        if np.all(np.abs(pad) <= 4.8):
            break
    obstacles: list[tuple[float, float, float, float]] = []   # x, y, radius, height
    tries = 0
    while len(obstacles) < n_obstacles and tries < 500:
        tries += 1
        if not obstacles:                                     # first one blocks the straight line
            t = rng.uniform(0.35, 0.65)
            p = hub + t * (pad - hub)
            nrm = np.array([-(pad - hub)[1], (pad - hub)[0]]) / np.linalg.norm(pad - hub)
            p = p + nrm * rng.uniform(-0.5, 0.5)
        else:
            p = rng.uniform(-5.5, 5.5, 2)
        r = rng.uniform(0.4, 0.8)
        h = rng.choice([rng.uniform(1.0, 2.0), rng.uniform(2.5, 3.5)])
        if np.linalg.norm(p - hub) < 2.2 or np.linalg.norm(p - pad) < 2.0:
            continue
        if any(np.hypot(p[0] - o[0], p[1] - o[1]) < r + o[2] + 1.2 for o in obstacles):
            continue
        obstacles.append((float(p[0]), float(p[1]), float(r), float(h)))
    return {"hub": hub, "pad": pad, "obstacles": obstacles}


class DeliveryDroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None, split: str = "train", max_steps: int = MAX_STEPS,
                 camera: str = "overview", width: int = 640, height: int = 480, follow: bool = False):
        assert split in SPLITS, f"split must be one of {list(SPLITS)}"
        self.render_mode, self.split, self.max_steps = render_mode, split, max_steps
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._drone_id = m.body("drone").id
        self._drone_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == self._drone_id}
        self._floor_geom = m.geom("floor").id
        self._obs_geoms = [m.geom(f"obs{i}_g").id for i in range(MAX_OBSTACLES)]
        self._obs_mocap = [m.body_mocapid[m.body(f"obs{i}").id] for i in range(MAX_OBSTACLES)]
        self._hub_mocap = m.body_mocapid[m.body("hub").id]
        self._clinic_mocap = m.body_mocapid[m.body("clinic").id]
        self._parcel_qadr = m.jnt_qposadr[m.joint("pfree").id]
        self._parcel_geom = m.geom("parcel_g").id
        self._ray_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # solid world only
        self._renderer: SceneRenderer | None = None
        self._cam = (camera, width, height)
        self._follow = follow
        self._reset_state()

    # ------------------------------------------------------------------ #
    # state helpers
    # ------------------------------------------------------------------ #
    def _reset_state(self):
        self.steps = 0
        self.done = False
        self.battery = 100.0
        self.fwd_cmd = 0.0
        self.parcel_attached = True
        self.outcome = "flying"
        self.layout_id = -1
        self.layout: dict[str, Any] = {"hub": np.zeros(2), "pad": np.zeros(2), "obstacles": []}
        self.wind = np.zeros(2)
        self.last_action = None
        self.last_reward = 0.0
        self.path: list[tuple[float, float, float]] = []

    @property
    def pos(self) -> np.ndarray:
        return self.data.qpos[:3]

    @property
    def yaw(self) -> float:
        return float(self.data.qpos[3])

    def _pad_vec_world(self) -> np.ndarray:
        return self.layout["pad"] - self.pos[:2]

    def _pad_distance(self) -> float:
        return float(np.linalg.norm(self._pad_vec_world()))

    def _phi(self) -> float:
        """Potential used for reward shaping (progress, alignment, low + slow near the pad)."""
        pv = self._pad_vec_world()
        dist = float(np.hypot(pv[0], pv[1]))
        bearing = abs(math.atan2(*self._to_body(pv[0], pv[1])[::-1]))
        near = min(1.0, max(0.0, (NEAR_RADIUS - dist) / NEAR_RADIUS))
        return (-R_PROGRESS * dist - W_ALIGN * bearing / math.pi
                - W_HEIGHT * near * max(0.0, float(self.pos[2]) - 0.8) - W_SPEED * near * self._ground_speed())

    def _ground_speed(self) -> float:
        return float(np.linalg.norm(self.data.qvel[:2]))

    def _to_body(self, vx: float, vy: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * vx + s * vy, -s * vx + c * vy

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        options = options or {}
        lo, hi = SPLITS[self.split]
        self.layout_id = int(options.get("layout_id", self.np_random.integers(lo, hi)))
        self.layout = make_layout(self.layout_id, N_OBSTACLES[self.split])

        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        hub, pad = self.layout["hub"], self.layout["pad"]
        d.mocap_pos[self._hub_mocap] = (hub[0], hub[1], 0.0)
        d.mocap_pos[self._clinic_mocap] = (pad[0], pad[1], 0.0)
        for i in range(MAX_OBSTACLES):
            gid, mid = self._obs_geoms[i], self._obs_mocap[i]
            if i < len(self.layout["obstacles"]):
                x, y, r, h = self.layout["obstacles"][i]
                m.geom_size[gid] = (r, h / 2, 0)
                d.mocap_pos[mid] = (x, y, h / 2)
                m.geom_rgba[gid] = (0.55, 0.55, 0.62, 1) if h > 2.3 else (0.18, 0.5, 0.22, 1)
            else:
                m.geom_size[gid] = (0.5, 0.5, 0)
                d.mocap_pos[mid] = (60 + 3 * i, 60, 0.5)

        heading = float(self.np_random.uniform(-math.pi, math.pi))
        d.qpos[:4] = (hub[0], hub[1], START_Z, heading)
        w_ang = float(self.np_random.uniform(0, 2 * math.pi))
        w_mag = float(self.np_random.uniform(0, WIND_MAX[self.split]))
        self.wind = np.array([w_mag * math.cos(w_ang), w_mag * math.sin(w_ang)])
        m.geom_contype[self._parcel_geom] = 0
        m.geom_conaffinity[self._parcel_geom] = 0
        m.geom_rgba[self._parcel_geom] = (0.95, 0.15, 0.15, 1)
        mujoco.mj_forward(m, d)
        self._attach_parcel()
        self._prev_phi = self._phi()
        self.path = [tuple(self.pos)]
        obs = self._obs()
        return obs, self._info()

    def step(self, action):
        assert not self.done, "episode finished - call reset()"
        a = int(action)
        self.last_action = a
        self.steps += 1
        reward = R_STEP
        vz_cmd, w_cmd = 0.0, 0.0

        if a == CLIMB:
            vz_cmd = CLIMB_RATE
        elif a == DESCEND:
            vz_cmd = -CLIMB_RATE
        elif a == SPEED_UP:
            self.fwd_cmd = min(MAX_SPEED_CMD, self.fwd_cmd + CRUISE_STEP)
        elif a == SLOW_DOWN:
            self.fwd_cmd = max(0.0, self.fwd_cmd - CRUISE_STEP)
        elif a == TURN_LEFT:
            w_cmd = TURN_RATE
        elif a == TURN_RIGHT:
            w_cmd = -TURN_RATE

        terminated = truncated = False
        crashed = False

        locked = a == RELEASE and self._pad_distance() > RELEASE_LOCK
        if locked:
            reward += R_INVALID
            a = HOLD                      # interlock: nothing is dropped, the drone just holds
        if a == RELEASE:
            reward += self._release()
            terminated = True
        else:
            crashed = self._simulate(vz_cmd, w_cmd)
            vh = self._ground_speed()
            vz = float(self.data.qvel[2])
            self.battery -= DT * (3.0 + 2.5 * vh * vh + 4.0 * max(vz, 0.0))
            phi = self._phi()
            reward += phi - self._prev_phi
            self._prev_phi = phi
            x, y, z = self.pos
            if crashed:
                reward, terminated, self.outcome = R_FAIL, True, "crashed"
            elif self.battery <= 0:
                reward, terminated, self.outcome = R_FAIL, True, "battery_empty"
            elif abs(x) > ARENA or abs(y) > ARENA or z > CEILING:
                reward, terminated, self.outcome = R_FAIL, True, "left_geofence"
            elif self.steps >= self.max_steps:
                reward, truncated, self.outcome = R_TIMEOUT, True, "timeout"

        self.path.append(tuple(self.pos))
        self.done = terminated or truncated
        self.last_reward = reward
        if self.render_mode is not None:
            self.render()
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------ #
    # physics
    # ------------------------------------------------------------------ #
    def _simulate(self, vz_cmd: float, w_cmd: float) -> bool:
        d, m = self.data, self.model
        for _ in range(FRAME_SKIP):
            c, s = math.cos(d.qpos[3]), math.sin(d.qpos[3])
            tvx, tvy = self.fwd_cmd * c, self.fwd_cmd * s
            ax = max(-A_MAX, min(A_MAX, KP_V * (tvx - d.qvel[0])))
            ay = max(-A_MAX, min(A_MAX, KP_V * (tvy - d.qvel[1])))
            az = max(-A_MAX, min(A_MAX, KP_VZ * (vz_cmd - d.qvel[2])))
            aw = max(-12.0, min(12.0, KP_W * (w_cmd - d.qvel[3])))
            d.qfrc_applied[0] = MASS * (ax + self.wind[0])      # wind is a disturbance the controller can't see
            d.qfrc_applied[1] = MASS * (ay + self.wind[1])
            d.qfrc_applied[2] = MASS * (G + az)                 # hover thrust + commanded vertical accel
            d.qfrc_applied[3] = IZ * aw
            mujoco.mj_step(m, d)
            for k in range(d.ncon):
                g1, g2 = d.contact[k].geom1, d.contact[k].geom2
                if g1 in self._drone_geoms or g2 in self._drone_geoms:
                    other = g2 if g1 in self._drone_geoms else g1
                    if other == self._floor_geom or other in self._obs_geoms:
                        self._attach_parcel()
                        return True
        self._attach_parcel()
        return False

    def _attach_parcel(self):
        if not self.parcel_attached:
            return
        q = self.data.qpos
        a = self._parcel_qadr
        q[a:a + 3] = (q[0], q[1], q[2] - 0.15)
        q[a + 3:a + 7] = (1, 0, 0, 0)
        self.data.qvel[4:10] = 0
        mujoco.mj_kinematics(self.model, self.data)

    def _release(self) -> float:
        """RELEASE: drop the parcel. Outcome depends on where / how fast the drone is."""
        dist, height, speed = self._pad_distance(), float(self.pos[2]), self._ground_speed()
        if dist <= PAD_RADIUS and height <= SOFT_HEIGHT and speed <= SOFT_SPEED:
            self.outcome = "delivered"
            r = R_DELIVERED + R_BATTERY_BONUS * max(self.battery, 0.0) / 100.0
        elif dist <= PAD_RADIUS:
            self.outcome = "damaged"        # over the pad but too high / too fast
            r = R_DAMAGED
        else:
            self.outcome = "parcel_lost"
            r = R_FAIL
        self._drop_animation()
        return r

    def _drop_animation(self):
        """Cosmetic only: let the parcel fall for 1 s so the replay shows the drop."""
        d, m = self.data, self.model
        self.parcel_attached = False
        m.geom_contype[self._parcel_geom] = 1
        m.geom_conaffinity[self._parcel_geom] = 1
        d.qvel[4:10] = 0
        for _ in range(100):
            d.qfrc_applied[2] = MASS * G
            d.qfrc_applied[:2] = 0
            mujoco.mj_step(m, d)

    # ------------------------------------------------------------------ #
    # observations
    # ------------------------------------------------------------------ #
    def _ray(self, angle: float, down: bool = False) -> float:
        d, m = self.data, self.model
        origin = np.array([d.qpos[0], d.qpos[1], d.qpos[2]], dtype=np.float64)
        if down:
            vec = np.array([0.0, 0.0, -1.0])
        else:
            a = d.qpos[3] + angle
            vec = np.array([math.cos(a), math.sin(a), 0.0])
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(m, d, origin, vec, self._ray_mask, 1, self._drone_id, gid)
        return RANGE_MAX if (dist < 0 or dist > RANGE_MAX) else float(dist)

    def _obs(self) -> np.ndarray:
        d = self.data
        pv = self._pad_vec_world()
        fwd, left = self._to_body(pv[0], pv[1])
        dist = math.hypot(fwd, left)
        bearing = math.atan2(left, fwd)
        vf, vl = self._to_body(d.qvel[0], d.qvel[1])
        rng = self.np_random
        rays = [self._ray(a) for a in (-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2)]
        down = self._ray(0.0, down=True)
        wf, wl = self._to_body(*(self.wind + rng.normal(0, 0.1, 2)))   # noisy anemometer
        o = np.array([
            fwd / 10.0, left / 10.0, dist / 10.0, bearing / math.pi,
            d.qpos[2] / CEILING, vf / 2.5, vl / 2.5, d.qvel[2] / 1.5, self.fwd_cmd / MAX_SPEED_CMD,
            max(self.battery, 0.0) / 100.0,
            *[(r + rng.normal(0, 0.02)) / RANGE_MAX for r in rays],
            down / RANGE_MAX,
            wf / WIND_MAX["stress"], wl / WIND_MAX["stress"],
            1.0 - self.steps / self.max_steps,
        ], dtype=np.float32)
        return np.clip(o, -1.0, 1.0)

    def _info(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "success": self.outcome == "delivered",
            "steps": self.steps,
            "layout_id": self.layout_id,
            "pad_distance": self._pad_distance(),
            "height": float(self.pos[2]),
            "speed": self._ground_speed(),
            "battery": max(self.battery, 0.0),
            "speed_command": self.fwd_cmd,
        }

    # ------------------------------------------------------------------ #
    # integration: serialise the world for a web / mobile front end
    # ------------------------------------------------------------------ #
    def to_json(self) -> dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "drone": {"x": float(self.pos[0]), "y": float(self.pos[1]), "z": float(self.pos[2]),
                      "yaw": self.yaw, "speed": self._ground_speed(), "speed_command": self.fwd_cmd},
            "battery": max(self.battery, 0.0),
            "hub": [float(v) for v in self.layout["hub"]],
            "pad": [float(v) for v in self.layout["pad"]],
            "pad_radius": PAD_RADIUS,
            "obstacles": [{"x": o[0], "y": o[1], "radius": o[2], "height": o[3]} for o in self.layout["obstacles"]],
            "wind": [float(v) for v in self.wind],
            "arena": ARENA,
            "ceiling": CEILING,
            "step": self.steps,
            "outcome": self.outcome,
            "done": self.done,
            "path": [[float(v) for v in p] for p in self.path],
        }

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def render(self, status: bool = False):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            cam, w, h = self._cam
            self._renderer = SceneRenderer(self.model, self.data, self.render_mode, cam, w, h, self._follow)
        frame = self._renderer.render()
        if frame is not None and status:
            name = ACTION_NAMES[self.last_action] if self.last_action is not None else "-"
            frame = annotate(frame, [
                f"step {self.steps:>3}  action {name:<10}  reward {self.last_reward:+.2f}  outcome: {self.outcome}",
                f"battery {max(self.battery, 0):5.1f}%  height {self.pos[2]:.2f} m  speed {self._ground_speed():.2f} m/s  "
                f"distance to pad {self._pad_distance():.2f} m",
            ])
        return frame

    def set_camera(self, name: str):
        self._cam = (name, self._cam[1], self._cam[2])
        if self._renderer is not None:
            self._renderer.camera = name

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# --------------------------------------------------------------------------- #
# Reference policies (sanity checks for the environment and comparison rows in the report)
#   random_policy  - uniform random actions (lower bound)
#   scripted_policy - hand-written "fly high, steer, slow down, descend, release" controller that only
#                     sees the observation vector (same information as the RL agents)
# --------------------------------------------------------------------------- #

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


def evaluate_policy(policy_factory, env, episodes: int = 50, seed: int = 0):
    """Run a policy for N episodes; returns mean return and outcome counts."""
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
