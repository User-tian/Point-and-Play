"""Build the [N, 512] condition cache that conditions the note synthesizer.

Row i pairs with row i of the latent cache (NSynth-train note i: its latent, pitch and velocity).
With a per-row seeded generator, half of the rows (Bernoulli 0.5) use a 4-s NLakh phrase window
of the same instrument identity and the rest use the NSynth note itself:

* ``--kind matched`` (Point-and-Play decoder): the source is mixed with 3 other instruments of
  the same corpus at random SNRs in [-5, 15] dB and encoded by a trained extractor with the
  category text cue. These are the "conditioning codes extracted by the final encoder" (Sec. 4.1).
* ``--kind neutral`` (decoder for the oracle / cue / separate-then-clone baselines and the loss
  ablation): the clean source is embedded by the frozen music-CLAP audio encoder.

The random choices depend only on (seed, row), so every encoder sees the same mixtures.

    python -m point_and_play.flowsynth.conditions --kind matched --encoder runs/encoder/extractor_best.pt \
        --nsynth-root /data/nsynth --nlakh-root /data/nlakh/rendered_single_inst --out caches/matched.npy
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from ..assets import default_music_clap, default_nlakh_map_dir, require_file, sha256_file
from ..data.catalog import nlakh_identity_map
from .latents import NSYNTH_TRAIN_ROWS, nsynth_rows

SR = 16000
# 144,527 NSynth rows and 144,678 NLakh rows for the full train split with the default seed.
EXPECTED_DOMAIN_COUNTS = (144_527, 144_678)


def _fix(wav: np.ndarray, n: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if wav.shape[0] >= n:
        return np.ascontiguousarray(wav[:n], dtype=np.float32)
    return np.pad(wav.astype(np.float32), (0, n - wav.shape[0]))


def load_note(path, n: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    return _fix(wav, n)


def load_phrase_window(path, n: int, rng: random.Random) -> np.ndarray:
    total = sf.info(path).frames
    if total > n:
        wav, sr = sf.read(path, dtype="float32", start=rng.randrange(0, total - n + 1), frames=n)
    else:
        wav, sr = sf.read(path, dtype="float32")
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    return _fix(wav, n)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def mix_at_snr(mix: np.ndarray, interferer: np.ndarray, snr_db: float) -> np.ndarray:
    return mix + _rms(mix) / (_rms(interferer) + 1e-8) * (10.0 ** (-snr_db / 20.0)) * interferer


def row_rng(seed: int, split: str, row: int) -> random.Random:
    digest = hashlib.blake2b(f"{seed}|{split}|note|{row}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "little"))


class ConditionSources:
    """Deterministic per-row source audio for the condition caches."""

    def __init__(self, nsynth_root, nlakh_root, nlakh_map_dir, split="train", seconds=4.0,
                 n_interf=3, snr=(-5.0, 15.0), nlakh_fraction=0.5, seed=1234):
        split_dir = Path(nsynth_root) / f"nsynth-{split}"
        rows = nsynth_rows(nsynth_root, split)
        self.identities = [meta["instrument_str"] for _, meta in rows]
        self.paths = [str(split_dir / "audio" / f"{name}.wav") for name, _ in rows]
        self.by_identity: dict[str, list[int]] = {}
        for index, identity in enumerate(self.identities):
            self.by_identity.setdefault(identity, []).append(index)
        self.identity_list = sorted(self.by_identity)
        nlakh_split = Path(nlakh_root) / split
        self.phrases: dict[str, list[str]] = {}
        for directory, identity in nlakh_identity_map(nlakh_split, nlakh_map_dir, split).items():
            clips = sorted(glob.glob(str(nlakh_split / directory / "*.wav")))
            if clips:
                self.phrases[identity] = clips
        self.phrase_identities = sorted(self.phrases)
        self.split, self.n, self.n_interf = split, int(seconds * SR), n_interf
        self.snr, self.nlakh_fraction, self.seed = tuple(snr), nlakh_fraction, seed

    def __len__(self) -> int:
        return len(self.identities)

    def source(self, row: int, mixed: bool):
        """(16-kHz waveform, domain) for ``row``; domain is ``nsynth``, ``nlakh`` or ``nlakh_missing``."""
        identity = self.identities[row]
        rng = row_rng(self.seed, self.split, row)
        wants_phrase = rng.random() < self.nlakh_fraction
        clips = self.phrases.get(identity)
        if wants_phrase and clips:
            wav = load_phrase_window(clips[rng.randrange(len(clips))], self.n, rng).copy()
            if mixed:
                others = [i for i in self.phrase_identities if i != identity]
                for other in rng.sample(others, min(self.n_interf, len(others))):
                    other_clips = self.phrases[other]
                    path = other_clips[rng.randrange(len(other_clips))]
                    wav = mix_at_snr(wav, load_phrase_window(path, self.n, rng), rng.uniform(*self.snr))
            return wav, "nlakh"
        wav = load_note(self.paths[row], self.n).copy()
        if mixed:
            others = [i for i in self.identity_list if i != identity]
            for other in rng.sample(others, min(self.n_interf, len(others))):
                notes = self.by_identity[other]
                wav = mix_at_snr(wav, load_note(self.paths[notes[rng.randrange(len(notes))]], self.n),
                                 rng.uniform(*self.snr))
        return wav, "nlakh_missing" if wants_phrase else "nsynth"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=("matched", "neutral"), required=True)
    p.add_argument("--encoder", type=Path, help="extractor checkpoint (required for --kind matched)")
    p.add_argument("--nsynth-root", type=Path, required=True)
    p.add_argument("--nlakh-root", type=Path, required=True, help="NLakh rendered_single_inst/ directory")
    p.add_argument("--nlakh-map-dir", type=Path, default=default_nlakh_map_dir())
    p.add_argument("--clap-checkpoint", type=Path, default=default_music_clap())
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    args = p.parse_args(argv)
    if args.kind == "matched" and args.encoder is None:
        p.error("--kind matched requires --encoder")
    sidecar_path = Path(f"{args.out}.json")
    if args.out.exists() or sidecar_path.exists():
        raise SystemExit(f"{args.out} already exists")
    require_file(args.clap_checkpoint, "music-CLAP checkpoint")
    # TF32 changes CLAP embeddings noticeably between batch layouts; keep full precision.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    device = "cuda"

    sources = ConditionSources(args.nsynth_root, args.nlakh_root, args.nlakh_map_dir, seed=args.seed)
    missing = sorted(set(sources.identity_list) - set(sources.phrases))
    if missing and not args.limit:
        raise SystemExit(f"{len(missing)} NSynth instruments have no NLakh phrases (e.g. {missing[0]}); "
                         "check the NLakh download")
    count = min(args.limit, len(sources)) if args.limit else len(sources)
    if args.kind == "matched":
        from ..encoder import load_encoder
        from ..text_cues import TextCueBank

        encoder = load_encoder(args.encoder, args.clap_checkpoint, device=device)
        bank = TextCueBank(encoder, sources.identity_list, device=device)

        def embed(wav, identities):
            return encoder.extract(wav, bank.cues(identities), SR)
    else:
        from ..clap import ClapAudio

        clap = ClapAudio(args.clap_checkpoint, device=device)
        clap.eval()

        def embed(wav, identities):
            return clap.embed(wav, SR)

    codes = np.zeros((count, 512), dtype=np.float32)
    domains = {"nsynth": 0, "nlakh": 0, "nlakh_missing": 0}
    pending: list[tuple[int, np.ndarray]] = []

    def flush():
        if not pending:
            return
        wav = torch.from_numpy(np.stack([w for _, w in pending])).to(device)
        with torch.no_grad():
            z = F.normalize(embed(wav, [sources.identities[i] for i, _ in pending]).float(), dim=-1)
        codes[[i for i, _ in pending]] = z.cpu().numpy()
        pending.clear()

    for row in range(count):
        wav, domain = sources.source(row, mixed=args.kind == "matched")
        domains[domain] += 1
        pending.append((row, wav))
        if len(pending) >= args.batch_size:
            flush()
            if (row + 1) % (50 * args.batch_size) == 0:
                print(f"[conditions] {row + 1}/{count}", flush=True)
    flush()

    if not np.all(np.linalg.norm(codes, axis=1) > 0.9):
        raise RuntimeError("degenerate condition rows")
    if not args.limit:
        observed = (domains["nsynth"] + domains["nlakh_missing"], domains["nlakh"])
        if count != NSYNTH_TRAIN_ROWS or observed != EXPECTED_DOMAIN_COUNTS or domains["nlakh_missing"]:
            raise RuntimeError(f"unexpected row/domain counts {count} {domains}; check the NSynth "
                               "and NLakh downloads (every identity needs its NLakh phrases)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{args.out}.tmp.npy")
    np.save(temporary, codes)
    sidecar = {"kind": args.kind, "rows": count, "dim": 512, "seed": args.seed, "domains": domains,
               "encoder_sha256": sha256_file(args.encoder) if args.encoder else None,
               "music_clap_sha256": sha256_file(args.clap_checkpoint),
               "sha256": sha256_file(temporary)}
    temporary_sidecar = Path(f"{sidecar_path}.tmp")
    temporary_sidecar.write_text(json.dumps(sidecar, indent=2) + "\n")
    os.replace(temporary, args.out)
    os.replace(temporary_sidecar, sidecar_path)
    print(f"[conditions] wrote {args.out}: {domains}", flush=True)


if __name__ == "__main__":
    main()
