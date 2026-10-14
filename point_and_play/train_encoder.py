"""Train the Point-and-Play timbre extractor (Stage 1, Sec. 3.1-3.2).

Defaults reproduce the paper recipe: LoRA (r=8, alpha=16) on music-CLAP qkv/fc1/fc2, CGSP head,
NSynth/NLakh/Slakh/MoisesDB sampled 0.30/0.30/0.10/0.30, 4-s mixtures, batch 16, 18k AdamW
steps (500 warmup to 3e-4, cosine to 3e-5), InfoNCE temperature 0.1 and
L_E = L_inv + L_algn^a + L_algn^t + 0.25 L_sel^a.

Every ``--eval-every`` steps the text-cued code is scored on NSynth-valid mixtures (mean cosine
to the isolated-note CLAP embedding); the best checkpoint is saved as ``extractor_best.pt``
(used for the paper's decoders) and the last one as ``extractor.pt``.

    python -m point_and_play.train_encoder --data-root /data --out runs/encoder
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import losses
from .assets import default_music_clap, default_nlakh_map_dir, require_file
from .data.mixtures import (DistinctKeySampler, MultitrackMixtures, NLakhMixtures,
                            NSynthMixtures, collate_mixtures)
from .data.multitrack import index_moisesdb, index_slakh, moisesdb_training_songs
from .encoder import PointAndPlayEncoder
from .text_cues import TextCueBank

DOMAINS = ("nsynth", "nlakh", "slakh", "moises")


def dataset_paths(args, domains) -> dict[str, Path]:
    """Roots of the active corpora (NSynth is always needed for validation); all must exist."""
    root = Path(args.data_root) if args.data_root else None
    layout = {"nsynth": (args.nsynth_root, "nsynth"),
              "nlakh": (args.nlakh_root, "nlakh/rendered_single_inst"),
              "slakh": (args.slakh_root, "slakh2100_redux_16k"),
              "moises": (args.moisesdb_root, "moisesdb/moisesdb_v0.1")}
    paths = {}
    for name in {"nsynth", *domains}:
        override, default = layout[name]
        if not override and root is None:
            raise SystemExit(f"pass --data-root or the {name} root")
        paths[name] = Path(override) if override else root / default
        if not paths[name].is_dir():
            raise SystemExit(f"{name} data not found at {paths[name]}")
    return paths


def build_domain(name, args, paths):
    """Return (dataset, sampler keys, identity labels) for one training corpus."""
    common = dict(seconds=args.seconds, n_sources=args.n_sources, seed=args.seed, cue_mode="class")
    if name == "nsynth":
        ds = NSynthMixtures(paths["nsynth"] / "nsynth-train/audio", n_interf=args.n_interf,
                            note_split="train", **common)
        return ds, [identity for _, identity in ds.files], list(ds.ids)
    if name == "nlakh":
        ds = NLakhMixtures(paths["nlakh"] / "train", args.nlakh_map_dir, split="train",
                           n_interf=args.n_interf, **common)
        return ds, [identity for _, identity in ds.files], list(ds.ids)
    if name == "slakh":
        songs = index_slakh(paths["slakh"] / "train")
    elif name == "moises":
        songs = moisesdb_training_songs(index_moisesdb(paths["moises"]))
    else:
        raise ValueError(name)
    ds = MultitrackMixtures(songs, items_per_song=args.items_per_song, domain=name, **common)
    return ds, ds.song_keys(), ds.labels()


def infinite_batches(ds, keys, args):
    sampler = DistinctKeySampler(keys, batch_size=args.batch, steps=args.steps + 16, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_mixtures, persistent_workers=args.workers > 0)
    while True:
        yield from loader


def validation_set(args, paths):
    """NSynth-valid mixtures with a same-instrument audio cue (fixed across evaluations)."""
    ds = NSynthMixtures(paths["nsynth"] / "nsynth-valid/audio", seconds=args.seconds,
                        n_interf=args.n_interf, n_sources=1, note_split=None, seed=123,
                        cue_mode="self")
    if len(ds.files) > args.eval_max_items:
        ds.files = random.Random(123).sample(ds.files, args.eval_max_items)
    return ds


@torch.no_grad()
def evaluate(model, ds, bank, args):
    """Cosine of audio-, text- and wrong-text-cued codes to the isolated-note CLAP embedding."""
    model.set_train(False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_mixtures)
    codes = {"audio": [], "text": [], "wrong": []}
    targets, count = [], 0
    for batch in loader:
        labels = [row[0] for row in batch["labels"]]
        _, tokens = model.mixture_tokens(batch["mix"], ds.src_sr)
        codes["audio"].append(model.pool(tokens, model.audio_cue(batch["cues"][:, 0], ds.src_sr)).cpu())
        codes["text"].append(model.pool(tokens, bank.cues(labels)).cpu())
        codes["wrong"].append(model.pool(tokens, bank.wrong_cues(labels)).cpu())
        targets.append(model.teacher(batch["isolated"][:, 0], ds.src_sr).cpu())
        count += len(labels)
    target = F.normalize(torch.cat(targets), dim=-1)
    z = {name: F.normalize(torch.cat(chunks), dim=-1) for name, chunks in codes.items()}
    cosine = {name: float((code * target).sum(-1).mean()) for name, code in z.items()}
    recall = {name: float(((z[name] @ target.t()).argmax(1) == torch.arange(count)).float().mean())
              for name in ("audio", "text")}
    model.set_train(True)
    return {"n": count, "audio_clapcos": cosine["audio"], "text_clapcos": cosine["text"],
            "wrong_clapcos": cosine["wrong"], "audio_r1": recall["audio"], "text_r1": recall["text"],
            "selectivity": cosine["text"] - cosine["wrong"]}


def learning_rate(step, warmup, total, peak, floor):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * min(1.0, progress)))


def atomic_save(payload, path: Path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path, help="run directory (must not exist)")
    p.add_argument("--data-root", type=Path, help="root holding nsynth/, nlakh/, slakh2100_redux_16k/, moisesdb/")
    p.add_argument("--nsynth-root", type=Path, help="directory with nsynth-train/ and nsynth-valid/")
    p.add_argument("--nlakh-root", type=Path, help="NLakh rendered_single_inst/ directory")
    p.add_argument("--slakh-root", type=Path, help="slakh2100_redux_16k/ directory")
    p.add_argument("--moisesdb-root", type=Path, help="directory holding the MoisesDB song folders")
    p.add_argument("--nlakh-map-dir", type=Path, default=default_nlakh_map_dir())
    p.add_argument("--clap-checkpoint", type=Path, default=default_music_clap())
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--weights", default="0.30,0.30,0.10,0.30")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--n-interf", type=int, default=3, help="interfering instruments per note mixture")
    p.add_argument("--n-sources", type=int, default=3, help="supervised sources per mixture (K)")
    p.add_argument("--items-per-song", type=int, default=8)
    p.add_argument("--steps", type=int, default=18000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=3e-5)
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--lambda-inv", type=float, default=1.0, help="L_inv weight")
    p.add_argument("--lambda-audio", type=float, default=1.0, help="L_algn^a weight")
    p.add_argument("--lambda-text", type=float, default=1.0, help="L_algn^t weight")
    p.add_argument("--lambda-sel", type=float, default=0.25, help="L_sel^a weight (0 disables)")
    p.add_argument("--sel-margin", type=float, default=0.0, help="hinge margin m of L_sel^a")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-targets", default="qkv,fc1,fc2")
    p.add_argument("--head-dim", type=int, default=512, help="CGSP projection width d")
    p.add_argument("--ckpt-steps", default="3000,6000,9000,12000,15000")
    p.add_argument("--eval-every", type=int, default=3000)
    p.add_argument("--eval-max-items", type=int, default=600)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.eval_max_items < 1 or args.eval_every < 1 or args.log_every < 1:
        p.error("--eval-max-items, --eval-every and --log-every must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    domains = [d for d in args.domains.split(",") if d]
    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != len(domains) or any(d not in DOMAINS for d in domains):
        raise SystemExit(f"--domains must be drawn from {DOMAINS} and match --weights")
    weights = [w / sum(weights) for w in weights]
    paths = dataset_paths(args, domains)
    require_file(args.clap_checkpoint, "music-CLAP checkpoint")
    if "nlakh" in domains:
        require_file(args.nlakh_map_dir / "idx_to_fam_inst_train.json", "NLakh identity map")
    args.out.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda"
    # Recorded with every checkpoint; `load_encoder` rebuilds the architecture from it.
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update(ext_head="gatestats", ext_dim=args.head_dim, clap_space="music",
                  amodel="HTSAT-base", domains=domains, weights=weights)
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    model = PointAndPlayEncoder(args.clap_checkpoint, device=device, lora_r=args.lora_r,
                                lora_alpha=args.lora_alpha,
                                lora_targets=tuple(t for t in args.lora_targets.split(",") if t),
                                head_dim=args.head_dim)
    model.set_train(True)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    trainable = sum(p.numel() for p in model.trainable_parameters()) / 1e6
    print(f"[encoder] {model.n_lora} LoRA layers, {trainable:.2f}M trainable parameters", flush=True)

    loaders, banks, rates = {}, {}, {}
    for name in domains:
        ds, keys, labels = build_domain(name, args, paths)
        loaders[name] = infinite_batches(ds, keys, args)
        banks[name] = TextCueBank(model, labels, device=device)
        rates[name] = ds.src_sr
        print(f"[encoder] {name}: {len(ds)} items @ {ds.src_sr} Hz, "
              f"{len(banks[name].families)} categories", flush=True)
    val_ds = validation_set(args, paths)
    val_bank = TextCueBank(model, val_ds.ids, device=device)

    checkpoint_steps = {int(s) for s in args.ckpt_steps.split(",") if s}
    domain_rng = random.Random(args.seed + 12345)
    best = {"metric": -1.0, "step": -1}

    def run_eval(step):
        result = evaluate(model, val_ds, val_bank, args)
        if not math.isfinite(result["text_clapcos"]):
            raise FloatingPointError(f"non-finite validation score at step {step}")
        if result["text_clapcos"] > best["metric"]:
            best.update(metric=result["text_clapcos"], step=step)
            atomic_save(model.checkpoint(config, step, best_text_clapcos=result["text_clapcos"]),
                        args.out / "extractor_best.pt")
        print(f"[encoder] step {step} EVAL text_cos={result['text_clapcos']:.4f} "
              f"audio_cos={result['audio_clapcos']:.4f} selectivity={result['selectivity']:.4f} "
              f"(best {best['metric']:.4f} @ {best['step']})", flush=True)

    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.warmup, args.steps, args.lr, args.lr_min)
        name = domain_rng.choices(domains, weights)[0]
        batch, sr = next(loaders[name]), rates[name]
        B, K, T = batch["cues"].shape
        valid, labels = batch["valid"], batch["labels"]

        g_a, tokens = model.mixture_tokens(batch["mix"], sr)
        g_b = model.global_embedding(batch["mixB"], sr)
        audio_cues = model.audio_cue(batch["cues"].reshape(B * K, T), sr).reshape(B, K, -1)
        text_cues = banks[name].batch_cues(labels, valid)
        teacher = model.teacher(batch["isolated"].reshape(B * K, T), sr).reshape(B, K, -1)

        loss_inv = losses.info_nce(g_a, g_b, args.temp)
        loss_audio = losses.alignment_loss(model.pool_multi(tokens, audio_cues), teacher, valid,
                                           labels, args.temp)
        loss_text = losses.alignment_loss(model.pool_multi(tokens, text_cues), teacher, valid,
                                          labels, args.temp)
        loss = args.lambda_inv * loss_inv + args.lambda_audio * loss_audio + args.lambda_text * loss_text
        loss_sel = torch.zeros((), device=device)
        if args.lambda_sel > 0:
            wrong_valid = losses.wrong_audio_cue_mask(valid, labels)
            wrong_codes = model.pool_multi(tokens, torch.roll(audio_cues, 1, dims=0))
            loss_sel = losses.selectivity_loss(wrong_codes, teacher, wrong_valid, args.sel_margin)
            loss = loss + args.lambda_sel * loss_sel
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=float("inf"),
                                       error_if_nonfinite=True)
        optimizer.step()

        if step % args.log_every == 0:
            print(f"[encoder] step {step} {name} lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"loss={loss.item():.4f} inv={loss_inv.item():.4f} audio={loss_audio.item():.4f} "
                  f"text={loss_text.item():.4f} sel={loss_sel.item():.4f}", flush=True)
        if step > 0 and (step % args.eval_every == 0 or step in checkpoint_steps):
            run_eval(step)
        if step > 0 and step in checkpoint_steps:
            atomic_save(model.checkpoint(config, step), args.out / f"ckpt_{step}.pt")

    run_eval(args.steps)
    atomic_save(model.checkpoint(config, args.steps), args.out / "extractor.pt")
    print(f"[encoder] done; best text_cos={best['metric']:.4f} @ step {best['step']}", flush=True)


if __name__ == "__main__":
    main()
