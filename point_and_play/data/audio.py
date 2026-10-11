"""Small mono-audio helpers shared by the datasets."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def fix_length(wav: torch.Tensor, n: int) -> torch.Tensor:
    return wav[:n] if wav.numel() >= n else F.pad(wav, (0, n - wav.numel()))


def load_mono(path, n: int) -> torch.Tensor:
    """First ``n`` samples of a file (zero-padded), mixed down to mono."""
    wav, _ = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(-1)
    return fix_length(torch.from_numpy(np.ascontiguousarray(wav)), n)


def read_window(path, start: int, n: int) -> torch.Tensor:
    """``n`` mono samples from ``start``; zeros if the file is shorter or unreadable."""
    try:
        wav, _ = sf.read(path, start=max(0, start), frames=n, dtype="float32")
    except Exception:
        return torch.zeros(n)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if wav.size == 0:
        return torch.zeros(n)
    return fix_length(torch.from_numpy(np.ascontiguousarray(wav)), n)


def sum_window(paths, start: int, n: int) -> torch.Tensor:
    out = torch.zeros(n)
    for path in paths:
        out = out + read_window(path, start, n)
    return out


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().clamp_min(1e-12).sqrt()


def snr_gain(running_mix: torch.Tensor, interferer: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Gain placing ``interferer`` at ``snr_db`` below the current (running) mixture."""
    return rms(running_mix) / (rms(interferer) + 1e-8) * (10.0 ** (-snr_db / 20.0))
