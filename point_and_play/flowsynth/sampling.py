"""Euler integration of the learned velocity field from Gaussian noise (u = 0) to data (u = 1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


@torch.no_grad()
def sample_euler(model: nn.Module, *, x0: Tensor, code: Tensor, pitch: Tensor, velocity: Tensor,
                 steps: int = 50) -> Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    if code.shape[0] != x0.shape[0] or pitch.shape != (x0.shape[0],) or velocity.shape != (x0.shape[0],):
        raise ValueError("conditioning batch dimensions must match x0")
    latent, dt = x0, 1.0 / steps
    for index in range(steps):
        timestep = torch.tensor(index * dt, device=x0.device, dtype=torch.float32).expand(x0.shape[0])
        latent = latent + dt * model(latent, timestep, code, pitch, velocity)
    return latent
