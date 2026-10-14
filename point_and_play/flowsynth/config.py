"""Decoder configuration (paper recipe)."""

from __future__ import annotations

from dataclasses import dataclass

LATENT_SR = 44100
LATENT_SECONDS = 4.0
VAE_CHANNELS = 64  # first 64 of the 128 DiffRhythm-VAE encoder channels (the posterior mean)

# Mean / std over all latents of the official NSynth-train split; written into every latent
# cache and every decoder checkpoint, and used to (un)normalize latents.
NSYNTH_LATENT_MEAN = -0.04388732327064043
NSYNTH_LATENT_STD = 0.9117347436443132


@dataclass(frozen=True)
class ModelConfig:
    depth: int = 24
    width: int = 1024
    heads: int = 16
    latent_channels: int = VAE_CHANNELS
    latent_frames: int = 86
    cond_dim: int = 512

    def validate(self) -> None:
        if min(self.depth, self.width, self.heads, self.latent_channels, self.latent_frames) < 1:
            raise ValueError("model dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must split into an even rotary head dimension")


@dataclass(frozen=True)
class TrainConfig:
    """150k steps, AdamW 1e-4, 2k warmup + cosine, 32 x 8 GPUs, condition dropout 0.1."""

    steps: int = 150_000
    batch_per_gpu: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 2_000
    training_timesteps: int = 1_000
    gradient_clip: float = 1.0
    ema_decay: float = 0.9999
    condition_dropout: float = 0.1
    seed: int = 20260901
