"""Load a saved agent (DQN Q-network or policy-gradient actor) as a plain `obs -> action` function."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from environment.custom_env import N_ACTIONS
from training.common import ROOT, build_mlp


def load_policy(path: str | Path):
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    net = build_mlp(N_ACTIONS, ckpt["hidden"], ckpt["activation"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())

    def scores(obs: np.ndarray) -> list[float]:
        with torch.no_grad():
            return net(torch.as_tensor(obs, dtype=torch.float32)).tolist()

    act.scores = scores           # Q-values (DQN) or logits (policy gradient)
    act.algo = ckpt["algo"]
    return act


def best_model_path(algo: str | None = None) -> Path:
    """Path of the best trained agent (overall, or for one algorithm) as chosen by training/analysis.py."""
    info = json.loads((ROOT / "models" / "best.json").read_text())
    key = algo if algo else info["overall"]
    return ROOT / info["models"][key]["path"]
