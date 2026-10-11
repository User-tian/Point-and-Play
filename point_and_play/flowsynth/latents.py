"""DiffRhythm-VAE latent cache of NSynth-train notes (decoder training targets).

Row ``i`` of every cache and every condition cache is the ``i``-th entry of
``sorted(nsynth-train/examples.json.items())``. Build once (multi-GPU via torchrun):

    torchrun --nproc-per-node 8 -m point_and_play.flowsynth.latents \
        --nsynth-root /data/nsynth --out caches/nsynth-train-latents
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..assets import default_vae, require_file, sha256_file
from .config import LATENT_SECONDS, LATENT_SR, VAE_CHANNELS

NSYNTH_TRAIN_ROWS = 289_205
_ARRAYS = ("lat", "pitch", "vel", "row_id")


def nsynth_rows(nsynth_root, split: str = "train"):
    """Sorted (name, metadata) pairs of an official NSynth split; defines the cache row order."""
    with open(Path(nsynth_root) / f"nsynth-{split}" / "examples.json") as stream:
        return sorted(json.load(stream).items())


class LatentDataset:
    """Memory-mapped latent cache returning normalized latents with pitch / velocity."""

    def __init__(self, root) -> None:
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        self.latents = np.load(root / "lat.npy", mmap_mode="r")
        self.pitch = np.load(root / "pitch.npy", mmap_mode="r")
        self.velocity = np.load(root / "vel.npy", mmap_mode="r")
        self.mean = float(meta["normalization"]["mean"])
        self.std = float(meta["normalization"]["std"])
        count = int(meta["count"])
        if self.latents.shape != (count, int(meta["C"]), int(meta["T"])):
            raise ValueError("latent array does not match meta.json")
        if len(self.pitch) != count or len(self.velocity) != count:
            raise ValueError("pitch / velocity arrays do not match meta.json")
        if not (np.isfinite(self.mean) and np.isfinite(self.std) and self.std > 0):
            raise ValueError("invalid latent normalization")

    def __len__(self) -> int:
        return len(self.pitch)

    def batch(self, indices):
        latents = np.stack([np.asarray(self.latents[i], dtype=np.float32) for i in indices])
        if not np.isfinite(latents).all():
            raise ValueError("latent cache contains non-finite values")
        latents = (latents - self.mean) / self.std
        pitch = torch.tensor([int(self.pitch[i]) for i in indices], dtype=torch.long)
        velocity = torch.tensor([int(self.velocity[i]) for i in indices], dtype=torch.long)
        return torch.from_numpy(latents), pitch, velocity


def cache_fingerprint(root) -> str:
    digest = hashlib.sha256()
    for name in ("lat.npy", "pitch.npy", "vel.npy", "row_id.npy", "meta.json"):
        digest.update(name.encode())
        digest.update(sha256_file(Path(root) / name).encode())
    return digest.hexdigest()


def write_cache(root: Path, arrays: dict, dependency_hashes: dict) -> Path:
    latents = np.asarray(arrays["lat"])
    if not np.isfinite(latents).all():
        raise ValueError("non-finite latents")
    as_double = latents.astype(np.float64, copy=False)
    std = float(np.std(as_double))
    meta = {"schema_version": 1, "split": "train", "count": int(latents.shape[0]),
            "C": int(latents.shape[1]), "T": int(latents.shape[2]), "dtype": str(latents.dtype),
            "normalization": {"mean": float(np.mean(as_double)), "std": std if std > 0 else 1.0},
            "dependency_hashes": dependency_hashes}
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for name in _ARRAYS:
            np.save(temporary / f"{name}.npy", np.asarray(arrays[name]))
        (temporary / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return root


def _load_audio(path, samples: int):
    import soundfile as sf
    import torch.nn.functional as F
    import torchaudio.functional as AF

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    wav = torch.as_tensor(audio)
    if wav.ndim == 2:
        wav = wav.mean(dim=1)
    if sample_rate != LATENT_SR:
        wav = AF.resample(wav, sample_rate, LATENT_SR)
    wav = wav[:samples]
    return F.pad(wav, (0, samples - wav.numel())) if wav.numel() < samples else wav


def main(argv=None) -> None:
    import torch.distributed as distributed

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nsynth-root", type=Path, required=True, help="directory with nsynth-train/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vae", type=Path, default=default_vae())
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="encode only the first N rows (smoke test)")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists")
    require_file(args.vae, "DiffRhythm VAE")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    rows = nsynth_rows(args.nsynth_root)
    if args.limit:
        if args.limit < world_size * args.batch_size:
            raise SystemExit("--limit must cover at least one batch per process")
        rows = rows[:args.limit]
    elif len(rows) != NSYNTH_TRAIN_ROWS:
        raise SystemExit(f"expected {NSYNTH_TRAIN_ROWS} NSynth-train rows, found {len(rows)}")
    audio_dir = args.nsynth_root / "nsynth-train" / "audio"
    local = [(i, rows[i]) for i in range(rank, len(rows), world_size)]
    vae = torch.jit.load(str(args.vae), map_location=device).eval()
    samples = int(LATENT_SECONDS * LATENT_SR)
    chunks, pitch, velocity, row_id = [], [], [], []
    for start in range(0, len(local), args.batch_size):
        part = local[start:start + args.batch_size]
        audio = torch.stack([_load_audio(audio_dir / f"{name}.wav", samples) for _, (name, _) in part])
        stereo = torch.stack((audio, audio), dim=1).to(device)
        with torch.inference_mode():
            chunks.append(vae.encode_export(stereo)[:, :VAE_CHANNELS].to(torch.float16).cpu().numpy())
        for index, (_, meta) in part:
            row_id.append(index)
            pitch.append(int(meta["pitch"]))
            velocity.append(int(meta["velocity"]))
        if (start // args.batch_size) % 200 == 0:
            print(f"[latents] rank {rank}: {start + len(part)}/{len(local)}", flush=True)

    shard_dir = Path(f"{args.out}.shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez(shard_dir / f"rank-{rank:03d}.npz", lat=np.concatenate(chunks),
             pitch=np.asarray(pitch, np.int16), vel=np.asarray(velocity, np.int16),
             row_id=np.asarray(row_id, np.int64))
    if world_size > 1:
        distributed.barrier()
    if rank == 0:
        shards = [np.load(shard_dir / f"rank-{r:03d}.npz") for r in range(world_size)]
        merged = {name: np.concatenate([s[name] for s in shards]) for name in _ARRAYS}
        order = np.argsort(merged["row_id"])
        merged = {name: value[order] for name, value in merged.items()}
        meta_path = args.nsynth_root / "nsynth-train" / "examples.json"
        write_cache(args.out, merged, {"metadata": sha256_file(meta_path), "vae": sha256_file(args.vae)})
        shutil.rmtree(shard_dir)
        print(f"[latents] wrote {args.out} ({len(order)} rows)", flush=True)
    if world_size > 1:
        distributed.barrier()
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
