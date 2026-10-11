"""Flow-matching training step, EMA and exactly resumable checkpoints."""

from __future__ import annotations

import copy
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TrainState:
    step: int
    sampler_epoch: int
    sampler_offset: int
    world_size: int
    config_hash: str
    cache_hashes: Mapping[str, str]
    resume_history: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StepMetrics:
    loss: float
    gradient_norm: float
    skipped: bool


class GlobalBatchStream:
    """Seeded shuffled global batches split across ranks; ``(epoch, offset)`` resumes exactly."""

    def __init__(self, *, size: int, batch_per_rank: int, rank: int, world_size: int, seed: int,
                 epoch: int = 0, offset: int = 0) -> None:
        if size < batch_per_rank * world_size:
            raise ValueError("dataset must contain at least one global batch")
        self.size, self.batch_per_rank = size, batch_per_rank
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch, self.offset = epoch, offset

    @property
    def position(self) -> tuple[int, int]:
        return self.epoch, self.offset

    def next_indices(self) -> list[int]:
        global_batch = self.batch_per_rank * self.world_size
        if self.offset + global_batch > self.size:
            self.epoch += 1
            self.offset = 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(self.size, generator=generator)
        start = self.offset + self.rank * self.batch_per_rank
        self.offset += global_batch
        return permutation[start:start + self.batch_per_rank].tolist()


def _gradient_norm(parameters) -> float:
    norms = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    return float(torch.stack(norms).norm(2).cpu()) if norms else 0.0


def train_step(*, model: nn.Module, x1: Tensor, code: Tensor, pitch: Tensor, velocity: Tensor,
               optimizer: torch.optim.Optimizer, training_timesteps: int = 1000,
               gradient_clip: float = 1.0) -> StepMetrics:
    """One conditional flow-matching update (Eq. 4) on normalized latents ``x1``.

    y_u = (1 - u) eps + u y with u on a uniform grid of ``training_timesteps`` points; the
    model regresses the velocity y - eps under bf16 autocast on CUDA.
    """
    optimizer.zero_grad(set_to_none=True)
    x0 = torch.randn_like(x1)
    time_index = torch.randint(0, training_timesteps, (x1.shape[0],), device=x1.device)
    timestep = time_index.to(dtype=torch.float32) / float(training_timesteps - 1)
    weight = timestep.view(-1, *([1] * (x1.ndim - 1)))
    x_t = (1.0 - weight) * x0 + weight * x1
    target = x1 - x0
    with torch.autocast(device_type=x1.device.type, dtype=torch.bfloat16,
                        enabled=x1.device.type == "cuda"):
        prediction = model(x_t, timestep, code, pitch, velocity)
        loss = F.mse_loss(prediction.float(), target.float())
    finite = torch.isfinite(loss).to(dtype=torch.int32)
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(finite, op=distributed.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    gradient_norm = _gradient_norm(model.parameters())
    skipped = not math.isfinite(gradient_norm)
    if skipped:
        # A bf16 overflow produced a non-finite gradient: drop this update instead of applying NaNs.
        optimizer.zero_grad(set_to_none=True)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
    return StepMetrics(loss=float(loss.detach().cpu()), gradient_norm=gradient_norm, skipped=skipped)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def clone_as_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(unwrap(model)).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(ema: nn.Module, online: nn.Module, *, decay: float) -> None:
    online_state = unwrap(online).state_dict()
    for name, value in unwrap(ema).state_dict().items():
        if torch.is_floating_point(value):
            value.mul_(decay).add_(online_state[name], alpha=1.0 - decay)
        else:
            value.copy_(online_state[name])


def _distributed_context() -> tuple[int, int]:
    if distributed.is_available() and distributed.is_initialized():
        return distributed.get_rank(), distributed.get_world_size()
    return 0, 1


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count differs from runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, *, state: TrainState, model: nn.Module, ema: nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, extra: Mapping[str, Any] | None = None):
    """Atomically save weights, EMA, optimizer, scheduler and every rank's RNG state (rank 0 writes)."""
    rank, world_size = _distributed_context()
    if world_size != state.world_size:
        raise ValueError("world size differs from the training state")
    rng_states: list = [None] * world_size
    if world_size > 1:
        distributed.all_gather_object(rng_states, _rng_state())
        distributed.barrier()
    else:
        rng_states[0] = _rng_state()
    if rank == 0:
        payload = {"schema_version": 1, "train_state": asdict(state),
                   "model": unwrap(model).state_dict(), "ema": unwrap(ema).state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "rng_states": rng_states, **(extra or {})}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if world_size > 1:
        distributed.barrier()


def load_checkpoint(path: Path, *, model: nn.Module, ema: nn.Module, optimizer, scheduler,
                    world_size: int, config_hash: str, cache_hashes: Mapping[str, str]) -> TrainState:
    """Restore a checkpoint written by ``save_checkpoint`` for an identical run configuration."""
    rank, runtime_world_size = _distributed_context()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = TrainState(**{**payload["train_state"],
                          "resume_history": tuple(payload["train_state"].get("resume_history", ()))})
    if state.world_size != world_size or runtime_world_size != world_size:
        raise ValueError("resume requires the same number of GPUs as the original run")
    if state.config_hash != config_hash:
        raise ValueError("resume configuration differs from the checkpoint")
    if dict(state.cache_hashes) != dict(cache_hashes):
        raise ValueError("latent or condition cache differs from the checkpoint")
    unwrap(model).load_state_dict(payload["model"])
    unwrap(ema).load_state_dict(payload["ema"])
    optimizer.load_state_dict(payload["optimizer"])
    for parameter, slots in optimizer.state.items():
        for key, value in slots.items():
            if isinstance(value, torch.Tensor):
                slots[key] = value.to(parameter.device)
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng_state(payload["rng_states"][rank])
    if world_size > 1:
        distributed.barrier()
    return state
