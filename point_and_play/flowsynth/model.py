"""RoPE / AdaLN diffusion transformer predicting flow-matching velocities in VAE latent space."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig


def timestep_embedding(timestep: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(-math.log(10_000.0)
                            * torch.arange(half, device=timestep.device, dtype=torch.float32)
                            / max(half - 1, 1))
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0) * 1_000.0
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def apply_rotary(tensor: Tensor) -> Tensor:
    """1-D rotary position embedding on [B, H, T, D] queries or keys."""
    _, _, tokens, head_dim = tensor.shape
    position = torch.arange(tokens, device=tensor.device, dtype=torch.float32)
    inverse_frequency = torch.exp(-math.log(10_000.0)
                                  * torch.arange(0, head_dim, 2, device=tensor.device,
                                                 dtype=torch.float32) / head_dim)
    angles = torch.outer(position, inverse_frequency)
    cosine = angles.cos().to(dtype=tensor.dtype).view(1, 1, tokens, head_dim // 2)
    sine = angles.sin().to(dtype=tensor.dtype).view(1, 1, tokens, head_dim // 2)
    even, odd = tensor[..., 0::2], tensor[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RotarySelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, value: Tensor) -> Tensor:
        batch, tokens, width = value.shape
        qkv = self.qkv(value).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, values = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attended = F.scaled_dot_product_attention(apply_rotary(query), apply_rotary(key), values)
        return self.output(attended.transpose(1, 2).reshape(batch, tokens, width))


class AdaLNBlock(nn.Module):
    def __init__(self, width: int, heads: int, condition_dim: int) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = RotarySelfAttention(width, heads)
        self.norm_mlp = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * width, width))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 6 * width))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        a_shift, a_scale, a_gate, m_shift, m_scale, m_gate = self.adaln(condition).chunk(6, dim=-1)
        value = value + a_gate.unsqueeze(1) * self.attention(
            _modulate(self.norm_attention(value), a_shift, a_scale))
        return value + m_gate.unsqueeze(1) * self.mlp(_modulate(self.norm_mlp(value), m_shift, m_scale))


class FinalAdaLN(nn.Module):
    def __init__(self, width: int, condition_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * width))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.adaln(condition).chunk(2, dim=-1)
        return _modulate(self.norm(value), shift, scale)


class FlowSynthDiT(nn.Module):
    """F_phi(y_u, u, c) with c = W_z z + E_p(p) + E_v(v) (Eq. 4), injected through AdaLN."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(config.latent_channels, config.width)
        self.text_projection = nn.Linear(config.cond_dim, config.width)  # W_z (timbre code)
        self.pitch_embedding = nn.Embedding(128, config.width)
        self.velocity_embedding = nn.Embedding(128, config.width)
        self.time_mlp = nn.Sequential(nn.Linear(config.width, config.width), nn.SiLU(),
                                      nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList([AdaLNBlock(config.width, config.heads, config.width)
                                     for _ in range(config.depth)])
        self.final_adaln = FinalAdaLN(config.width, config.width)
        self.mu_head = nn.Linear(config.width, config.latent_channels)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)

    def condition(self, code: Tensor, pitch: Tensor, velocity: Tensor) -> Tensor:
        if code.ndim != 2 or code.shape[1] != self.config.cond_dim:
            raise ValueError(f"timbre code must have shape [B,{self.config.cond_dim}]")
        if pitch.shape != (code.shape[0],) or velocity.shape != (code.shape[0],):
            raise ValueError("pitch and velocity must each have shape [B]")
        if torch.any((pitch < 0) | (pitch > 127)) or torch.any((velocity < 1) | (velocity > 127)):
            raise ValueError("pitch must be in 0..127 and velocity in 1..127")
        return self.text_projection(code) + self.pitch_embedding(pitch) + self.velocity_embedding(velocity)

    def forward(self, latent: Tensor, timestep: Tensor, code: Tensor, pitch: Tensor,
                velocity: Tensor) -> Tensor:
        """latent: [B, C, T]; timestep: [B] in [0, 1]; code: [B, 512] -> velocity [B, C, T]."""
        expected = (code.shape[0], self.config.latent_channels, self.config.latent_frames)
        if latent.shape != expected or timestep.shape != (latent.shape[0],):
            raise ValueError(f"latent must have shape {expected} and timestep shape [B]")
        condition = self.condition(code, pitch, velocity) + self.time_mlp(
            timestep_embedding(timestep, self.config.width).to(dtype=code.dtype))
        hidden = self.input_projection(latent.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.mu_head(self.final_adaln(hidden, condition)).transpose(1, 2)
