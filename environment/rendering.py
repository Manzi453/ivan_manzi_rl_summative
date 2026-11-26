"""MuJoCo scene (MJCF) and rendering; rays only hit geom group 0 (floor + obstacles)."""
from __future__ import annotations

import time

import numpy as np

ARENA = 6.5
MAX_OBSTACLES = 6
TIMESTEP = 0.01


def _obstacle_xml() -> str:
    out = []
    for i in range(MAX_OBSTACLES):
        out.append(
            f'<body name="obs{i}" mocap="true" pos="{50 + 3 * i} 50 1">'
            f'<geom name="obs{i}_g" type="cylinder" size="0.5 1.0" rgba="0.5 0.5 0.55 1" '
            f'contype="1" conaffinity="1" group="0"/></body>'
        )
    return "\n    ".join(out)


def build_xml() -> str:
    a = ARENA
    return f"""
<mujoco model="delivery_drone">
  <option timestep="{TIMESTEP}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0 0 0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.68 0.95" rgb2="0.92 0.96 1.0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.40 0.60 0.36" rgb2="0.34 0.54 0.31"
             width="300" height="300" mark="edge" markrgb="0.25 0.42 0.22"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
  </asset>
  <contact><exclude body1="drone" body2="parcel"/></contact>
  <worldbody>
    <light pos="0 -4 12" dir="0 0.3 -1" directional="true" diffuse="0.7 0.7 0.7"/>
    <body name="center" pos="0 0 0.5"/>
    <camera name="overview" mode="targetbody" target="center" pos="0 -11.5 9.5"/>
    <geom name="floor" type="plane" size="14 14 0.1" material="grid" group="0" contype="1" conaffinity="1"/>

    <geom type="box" pos="0 {a} 0.01" size="{a} 0.04 0.01" rgba="0.9 0.2 0.2 0.9" contype="0" conaffinity="0" group="2"/>
    <geom type="box" pos="0 {-a} 0.01" size="{a} 0.04 0.01" rgba="0.9 0.2 0.2 0.9" contype="0" conaffinity="0" group="2"/>
    <geom type="box" pos="{a} 0 0.01" size="0.04 {a} 0.01" rgba="0.9 0.2 0.2 0.9" contype="0" conaffinity="0" group="2"/>
    <geom type="box" pos="{-a} 0 0.01" size="0.04 {a} 0.01" rgba="0.9 0.2 0.2 0.9" contype="0" conaffinity="0" group="2"/>

    <body name="hub" mocap="true" pos="-3 -3 0">
      <geom type="cylinder" size="0.7 0.01" pos="0 0 0.01" rgba="0.15 0.35 0.9 1" contype="0" conaffinity="0" group="2"/>
      <geom type="box" size="0.35 0.07 0.012" pos="0 0 0.025" rgba="1 1 1 1" contype="0" conaffinity="0" group="2"/>
      <geom type="box" size="0.07 0.35 0.012" pos="0 0 0.025" rgba="1 1 1 1" contype="0" conaffinity="0" group="2"/>
    </body>

    <body name="clinic" mocap="true" pos="3 3 0">
      <geom type="cylinder" size="0.6 0.01" pos="0 0 0.01" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0" group="2"/>
      <geom type="box" size="0.30 0.07 0.012" pos="0 0 0.025" rgba="0.85 0.1 0.1 1" contype="0" conaffinity="0" group="2"/>
      <geom type="box" size="0.07 0.30 0.012" pos="0 0 0.025" rgba="0.85 0.1 0.1 1" contype="0" conaffinity="0" group="2"/>
      <geom type="box" size="0.55 0.40 0.35" pos="0 1.45 0.35" rgba="0.93 0.93 0.88 1" contype="0" conaffinity="0" group="2"/>
      <geom type="box" size="0.58 0.43 0.04" pos="0 1.45 0.72" rgba="0.75 0.2 0.2 1" contype="0" conaffinity="0" group="2"/>
    </body>

    {_obstacle_xml()}

    <body name="drone" pos="0 0 0.5">
      <joint name="jx" type="slide" axis="1 0 0" damping="0.05"/>
      <joint name="jy" type="slide" axis="0 1 0" damping="0.05"/>
      <joint name="jz" type="slide" axis="0 0 1" damping="0.05"/>
      <joint name="jyaw" type="hinge" axis="0 0 1" damping="0.01"/>
      <inertial pos="0 0 0" mass="1.2" diaginertia="0.01 0.01 0.02"/>
      <geom name="d_body" type="box" size="0.10 0.10 0.03" rgba="0.12 0.12 0.18 1" group="1" contype="1" conaffinity="1"/>
      <geom name="d_arm1" type="capsule" fromto="-0.2 -0.2 0.02 0.2 0.2 0.02" size="0.012" rgba="0.2 0.2 0.25 1" group="1" contype="1" conaffinity="1"/>
      <geom name="d_arm2" type="capsule" fromto="-0.2 0.2 0.02 0.2 -0.2 0.02" size="0.012" rgba="0.2 0.2 0.25 1" group="1" contype="1" conaffinity="1"/>
      <geom name="d_r_fl" type="cylinder" pos="0.2 0.2 0.04" size="0.11 0.004" rgba="1.0 0.35 0.2 0.55" group="1" contype="1" conaffinity="1"/>
      <geom name="d_r_fr" type="cylinder" pos="0.2 -0.2 0.04" size="0.11 0.004" rgba="1.0 0.35 0.2 0.55" group="1" contype="1" conaffinity="1"/>
      <geom name="d_r_bl" type="cylinder" pos="-0.2 0.2 0.04" size="0.11 0.004" rgba="0.8 0.8 0.85 0.5" group="1" contype="1" conaffinity="1"/>
      <geom name="d_r_br" type="cylinder" pos="-0.2 -0.2 0.04" size="0.11 0.004" rgba="0.8 0.8 0.85 0.5" group="1" contype="1" conaffinity="1"/>
      <geom name="d_nose" type="box" pos="0.13 0 0.05" size="0.04 0.02 0.015" rgba="1 0.1 0.1 1" group="1" contype="0" conaffinity="0"/>
    </body>

    <body name="parcel" pos="0 0 0.3">
      <freejoint name="pfree"/>
      <geom name="parcel_g" type="box" size="0.07 0.07 0.07" mass="0.3" rgba="0.95 0.15 0.15 1" group="2" contype="0" conaffinity="0"/>
      <geom type="box" size="0.072 0.012 0.072" mass="0.0001" rgba="1 1 1 1" group="2" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class SceneRenderer:
    """either the offscreen renderer or the live viewer"""

    def __init__(self, model, data, mode: str = "rgb_array", camera: str = "overview",
                 width: int = 640, height: int = 480, follow: bool = False):
        self.model, self.data, self.mode, self.camera = model, data, mode, camera
        self.follow = follow
        self._chase = None
        self.width, self.height = width, height
        self._renderer = None
        self._viewer = None

    def render(self) -> np.ndarray | None:
        import mujoco

        if self.mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                cam = self._viewer.cam
                cam.lookat[:] = (0.0, 0.0, 0.8)
                cam.distance, cam.azimuth, cam.elevation = 17.0, -90.0, -38.0
            if self._viewer.is_running():
                if self.follow:
                    self._viewer.cam.lookat[:] = self.data.qpos[:3]
                self._viewer.sync()
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        if self.camera == "chase":
            if self._chase is None:
                self._chase = mujoco.MjvCamera()
                self._chase.distance, self._chase.elevation = 4.2, -30.0
            self._chase.lookat[:] = self.data.qpos[:3]
            self._chase.azimuth = float(np.degrees(self.data.qpos[3]))
            self._renderer.update_scene(self.data, camera=self._chase)
        else:
            self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render().copy()

    def viewer_running(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """status bar text on top of a frame (for the gifs)"""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    h = 8 + 15 * len(lines)
    d.rectangle([0, 0, img.size[0], h], fill=(0, 0, 0, 150))
    for i, t in enumerate(lines):
        d.text((8, 5 + 15 * i), t, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


def sleep_to_fps(t0: float, fps: float) -> None:
    dt = 1.0 / fps - (time.time() - t0)
    if dt > 0:
        time.sleep(dt)
