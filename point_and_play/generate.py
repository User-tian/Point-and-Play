"""Clone a cued instrument from a mixture and play it at new pitches (inference, Sec. 3.3).

    python -m point_and_play.generate --encoder runs/encoder/extractor_best.pt \
        --decoder runs/matched-decoder/checkpoints/final.pt --mixture song.wav --text guitar \
        --pitches 48 55 60 67 --velocity 100 --out-dir samples/

Use ``--cue-audio clip.wav`` instead of ``--text`` for an audio cue. For each pitch, ``--candidates``
latents are integrated with ``--steps`` Euler steps; the candidate whose music-CLAP embedding is
closest to the timbre code is written as a 44.1-kHz WAV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from .assets import default_music_clap, default_vae, require_file
from .encoder import load_encoder
from .flowsynth.config import (LATENT_SECONDS, LATENT_SR, NSYNTH_LATENT_MEAN, NSYNTH_LATENT_STD,
                               ModelConfig)
from .flowsynth.model import FlowSynthDiT
from .flowsynth.sampling import sample_euler

SELECTION_SR = 16000


def load_audio(path, offset: float, seconds: float):
    """Mono window [offset, offset + seconds), zero-padded like the training clips."""
    info = sf.info(str(path))
    frames = int(seconds * info.samplerate)
    wav, sr = sf.read(str(path), dtype="float32", start=int(offset * info.samplerate),
                      frames=frames, always_2d=True)
    if wav.shape[0] == 0:
        raise ValueError(f"{path}: no audio at offset {offset} s")
    wav = torch.from_numpy(np.ascontiguousarray(wav.mean(axis=1)))
    return F.pad(wav, (0, frames - wav.numel())), sr


def load_decoder(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["model_config"]) if "model_config" in payload else ModelConfig()
    model = FlowSynthDiT(config)
    model.load_state_dict(payload["ema"])
    norm = payload.get("latent_normalization", {"mean": NSYNTH_LATENT_MEAN, "std": NSYNTH_LATENT_STD})
    return model.to(device).eval(), float(norm["mean"]), float(norm["std"])


@torch.no_grad()
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True, help="decoder checkpoint (EMA weights are used)")
    p.add_argument("--mixture", type=Path, required=True)
    cue = p.add_mutually_exclusive_group(required=True)
    cue.add_argument("--text", help="instrument category name, e.g. guitar")
    cue.add_argument("--cue-audio", type=Path, help="recording of the target instrument category")
    p.add_argument("--offset", type=float, default=0.0, help="mixture start time in seconds")
    p.add_argument("--seconds", type=float, default=4.0, help="mixture context (training used 4 s)")
    p.add_argument("--pitches", type=int, nargs="+", default=[60])
    p.add_argument("--velocity", type=int, default=100)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("samples"))
    p.add_argument("--clap-checkpoint", type=Path, default=default_music_clap())
    p.add_argument("--vae", type=Path, default=default_vae())
    args = p.parse_args(argv)
    if any(not 0 <= pitch <= 127 for pitch in args.pitches) or not 1 <= args.velocity <= 127:
        p.error("pitches must be in 0..127 and velocity in 1..127")
    if args.offset < 0 or args.seconds <= 0 or args.steps < 1 or args.candidates < 1:
        p.error("--offset must be >= 0 and --seconds, --steps, --candidates positive")
    if args.text is not None and not args.text.strip():
        p.error("--text must be a non-empty category name")
    require_file(args.clap_checkpoint, "music-CLAP checkpoint")
    require_file(args.vae, "DiffRhythm VAE")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"

    encoder = load_encoder(args.encoder, args.clap_checkpoint, device=device)
    mixture, mixture_sr = load_audio(args.mixture, args.offset, args.seconds)
    if args.text:
        cue_embedding = encoder.text_cue([args.text])
    else:
        cue_wav, cue_sr = load_audio(args.cue_audio, 0.0, args.seconds)
        cue_embedding = encoder.audio_cue(cue_wav[None], cue_sr)
    code = F.normalize(encoder.extract(mixture[None], cue_embedding, mixture_sr).float(), dim=-1)

    decoder, mean, std = load_decoder(args.decoder, device)
    vae = torch.jit.load(str(args.vae), map_location=device).eval()
    config = decoder.config
    n = args.candidates
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for pitch in args.pitches:
        generator = torch.Generator(device=device).manual_seed(args.seed * 1000 + pitch)
        x0 = torch.randn(n, config.latent_channels, config.latent_frames, device=device, generator=generator)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = sample_euler(decoder, x0=x0, code=code.repeat(n, 1),
                                  pitch=torch.full((n,), pitch, device=device, dtype=torch.long),
                                  velocity=torch.full((n,), args.velocity, device=device, dtype=torch.long),
                                  steps=args.steps)
        audio = vae.decode_export((latent.float() * std + mean).contiguous()).mean(1)
        # Rank candidates on the full 4-s note at 16 kHz, as in the paper's evaluation.
        selection_audio = AF.resample(audio, LATENT_SR, SELECTION_SR)[:, :int(LATENT_SECONDS * SELECTION_SR)]
        similarity = (F.normalize(encoder.teacher(selection_audio, SELECTION_SR), dim=-1) * code).sum(-1)
        best = int(similarity.argmax())
        path = args.out_dir / f"pitch{pitch:03d}_vel{args.velocity:03d}.wav"
        sf.write(path, audio[best].float().cpu().numpy(), LATENT_SR)
        report.append({"pitch": pitch, "file": str(path), "selected": best,
                       "clap_similarity": [round(float(s), 4) for s in similarity]})
        print(f"pitch {pitch}: candidate {best} (cos {float(similarity[best]):.3f}) -> {path}", flush=True)
    (args.out_dir / "generation.json").write_text(json.dumps(
        {"mixture": str(args.mixture), "offset": args.offset, "seconds": args.seconds,
         "cue": args.text or str(args.cue_audio), "velocity": args.velocity, "steps": args.steps,
         "notes": report}, indent=2))


if __name__ == "__main__":
    main()
