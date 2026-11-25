import gymnasium as gym

from environment.custom_env import ACTION_NAMES, OBS_NAMES, DeliveryDroneEnv  # noqa: F401

if "DeliveryDrone-v0" not in gym.registry:
    gym.register(id="DeliveryDrone-v0", entry_point="environment.custom_env:DeliveryDroneEnv")
