"""Train the conditional note synthesizer (FM, 150k steps) on a latent cache + condition cache.

The paper recipe uses 8 GPUs x 32 notes (global batch 256):

    torchrun --standalone --nproc-per-node 8 -m point_and_play.flowsynth.train \
        --latents caches/nsynth-train-latents --conditions caches/matched.npy --out runs/matched-decoder

Checkpoints (``checkpoints/latest.pt`` every 5k steps, ``checkpoints/final.pt``) store model, EMA,
optimizer, scheduler, sampler cursor and per-rank RNG; ``--resume`` continues bit-exactly with
the same command and GPU count. Inference uses the EMA weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from ..assets import sha256_file
from .config import TINY_MODEL, ModelConfig, TrainConfig
from .latents import LatentDataset, cache_fingerprint
from .model import FlowSynthDiT
from .training import (GlobalBatchStream, TrainState, clone_as_ema, ema_update, load_checkpoint,
                       save_checkpoint, train_step)


def parse_args(argv=None):
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", type=Path, required=True, help="latent cache directory")
    p.add_argument("--conditions", type=Path, required=True, help="[N,512] condition cache (.npy)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="continue from <out>/checkpoints/latest.pt")
    p.add_argument("--steps", type=int, default=defaults.steps)
    p.add_argument("--batch-per-gpu", type=int, default=defaults.batch_per_gpu)
    p.add_argument("--condition-dropout", type=float, default=defaults.condition_dropout,
                   help="probability of zeroing a timbre code (classifier-free dropout)")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--tiny-model", action="store_true", help="1-layer model for smoke tests")
    args = p.parse_args(argv)
    if args.steps < 1 or args.batch_per_gpu < 1 or not 0.0 <= args.condition_dropout < 1.0:
        p.error("invalid --steps, --batch-per-gpu or --condition-dropout")
    return args


def _canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _setup(device_name: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if device_name != "cuda":
            raise SystemExit("distributed training requires CUDA")
        distributed.init_process_group(backend="nccl")
    if device_name == "cuda":
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, torch.device("cuda", local_rank)
    return rank, world_size, local_rank, torch.device("cpu")


def _mean_across_ranks(values, device, world_size):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
        tensor /= world_size
    return tensor.cpu().tolist()


def main(argv=None) -> None:
    args = parse_args(argv)
    defaults = TrainConfig()
    rank, world_size, local_rank, device = _setup(args.device)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    dataset = LatentDataset(args.latents)
    conditions = np.load(args.conditions, mmap_mode="r")
    model_config = TINY_MODEL if args.tiny_model else ModelConfig()
    if conditions.shape != (len(dataset), model_config.cond_dim):
        raise SystemExit(f"condition cache must have shape [{len(dataset)}, {model_config.cond_dim}]")
    cache_hashes = {"latents": cache_fingerprint(args.latents), "conditions": sha256_file(args.conditions)}
    run_config = {"model": asdict(model_config), "steps": args.steps, "batch_per_gpu": args.batch_per_gpu,
                  "world_size": world_size, "learning_rate": defaults.learning_rate,
                  "warmup_steps": defaults.warmup_steps, "training_timesteps": defaults.training_timesteps,
                  "gradient_clip": defaults.gradient_clip, "ema_decay": defaults.ema_decay,
                  "condition_dropout": args.condition_dropout, "seed": args.seed,
                  "latent_normalization": {"mean": dataset.mean, "std": dataset.std}}
    config_hash = _canonical_hash(run_config)
    checkpoint_dir = args.out / "checkpoints"
    if rank == 0:
        if args.resume:
            recorded = json.loads((args.out / "config.json").read_text())
            if _canonical_hash(recorded) != config_hash:
                raise SystemExit("--resume requires the original command and GPU count")
        else:
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")
    if world_size > 1:
        distributed.barrier()

    model = FlowSynthDiT(model_config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    ema = clone_as_ema(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=defaults.learning_rate)
    warmup = min(defaults.warmup_steps, max(args.steps - 1, 0))

    def lr_factor(index: int) -> float:
        current = index + 1  # index counts completed scheduler steps
        if current <= warmup:
            return current / max(1, warmup)
        return 0.5 * (1.0 + math.cos(math.pi * (current - warmup) / max(1, args.steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_step, epoch, offset, history = 0, 0, 0, ()
    if args.resume:
        resume_from = checkpoint_dir / "latest.pt"
        state = load_checkpoint(resume_from, model=model, ema=ema, optimizer=optimizer,
                                scheduler=scheduler, world_size=world_size,
                                config_hash=config_hash, cache_hashes=cache_hashes)
        start_step, epoch, offset = state.step, state.sampler_epoch, state.sampler_offset
        history = tuple(state.resume_history) + ({"checkpoint": str(resume_from), "step": state.step},)
        if rank == 0:
            print(f"[decoder] resumed at step {start_step}", flush=True)

    stream = GlobalBatchStream(size=len(dataset), batch_per_rank=args.batch_per_gpu, rank=rank,
                               world_size=world_size, seed=args.seed, epoch=epoch, offset=offset)
    extra = {"model_config": asdict(model_config),
             "latent_normalization": {"mean": dataset.mean, "std": dataset.std}}

    def state_at(step):
        return TrainState(step=step, sampler_epoch=stream.position[0], sampler_offset=stream.position[1],
                          world_size=world_size, config_hash=config_hash, cache_hashes=cache_hashes,
                          resume_history=history)

    started = interval_started = time.monotonic()
    steps_in_interval = 0
    for step in range(start_step + 1, args.steps + 1):
        indices = stream.next_indices()
        x1, pitch, velocity = dataset.batch(indices)
        x1, pitch, velocity = x1.to(device), pitch.to(device), velocity.to(device)
        code = torch.from_numpy(np.array(conditions[indices], dtype=np.float32, copy=True)).to(device)
        if args.condition_dropout > 0.0:
            # Zeroed codes map to the bias of W_z, i.e. the learned null condition.
            keep = (torch.rand(code.shape[0], device=device) >= args.condition_dropout).to(code.dtype)
            code = code * keep.unsqueeze(1)
        metrics = train_step(model=model, x1=x1, code=code, pitch=pitch, velocity=velocity,
                             optimizer=optimizer, training_timesteps=defaults.training_timesteps,
                             gradient_clip=defaults.gradient_clip)
        scheduler.step()
        ema_update(ema, model, decay=defaults.ema_decay)
        steps_in_interval += 1

        if step == 1 or step == args.steps or step % args.log_every == 0:
            loss, grad_norm = _mean_across_ranks([metrics.loss, metrics.gradient_norm], device, world_size)
            if rank == 0:
                elapsed = max(time.monotonic() - interval_started, 1e-9)
                record = {"step": step, "loss": loss, "gradient_norm": grad_norm,
                          "learning_rate": scheduler.get_last_lr()[0],
                          "steps_per_second": steps_in_interval / elapsed,
                          "wall_seconds": time.monotonic() - started}
                with (args.out / "train.jsonl").open("a") as log:
                    log.write(json.dumps(record, sort_keys=True) + "\n")
                print(f"[decoder] {json.dumps(record, sort_keys=True)}", flush=True)
            interval_started, steps_in_interval = time.monotonic(), 0
        if step % args.checkpoint_every == 0 and step < args.steps:
            save_checkpoint(checkpoint_dir / "latest.pt", state=state_at(step), model=model, ema=ema,
                            optimizer=optimizer, scheduler=scheduler, extra=extra)

    save_checkpoint(checkpoint_dir / "final.pt", state=state_at(args.steps), model=model, ema=ema,
                    optimizer=optimizer, scheduler=scheduler, extra=extra)
    if world_size > 1:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
