#!/usr/bin/env bash
# Train the full Point-and-Play system: extractor, matched decoder and neutral decoder.
#
#   DATA_ROOT=/path/to/data bash scripts/reproduce.sh
#
# Stage 1 uses one GPU (~3 h on an A100). The latent cache, condition caches and decoders use
# NUM_GPUS GPUs; the paper decoders were trained with 8 x A100 (~7 h each).
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT (see README: Data)}"
NUM_GPUS="${NUM_GPUS:-8}"
RUNS="${RUNS:-runs}"
CACHES="${CACHES:-caches}"
NSYNTH="$DATA_ROOT/nsynth"
NLAKH="$DATA_ROOT/nlakh/rendered_single_inst"

# Stage 1: timbre extractor (LoRA music-CLAP + CGSP).
python -m point_and_play.train_encoder --data-root "$DATA_ROOT" --out "$RUNS/encoder"
ENCODER="$RUNS/encoder/extractor_best.pt"

# Stage 2a: DiffRhythm-VAE latents of NSynth-train (shared by both decoders).
torchrun --standalone --nproc-per-node "$NUM_GPUS" -m point_and_play.flowsynth.latents \
    --nsynth-root "$NSYNTH" --out "$CACHES/nsynth-train-latents"

# Stage 2b: matched decoder, conditioned on codes extracted by the trained encoder.
python -m point_and_play.flowsynth.conditions --kind matched --encoder "$ENCODER" \
    --nsynth-root "$NSYNTH" --nlakh-root "$NLAKH" --out "$CACHES/matched-conditions.npy"
torchrun --standalone --nproc-per-node "$NUM_GPUS" -m point_and_play.flowsynth.train \
    --latents "$CACHES/nsynth-train-latents" --conditions "$CACHES/matched-conditions.npy" \
    --out "$RUNS/matched-decoder"

# Stage 2c: neutral decoder, conditioned on clean music-CLAP embeddings (baselines / ablations).
python -m point_and_play.flowsynth.conditions --kind neutral \
    --nsynth-root "$NSYNTH" --nlakh-root "$NLAKH" --out "$CACHES/neutral-conditions.npy"
torchrun --standalone --nproc-per-node "$NUM_GPUS" -m point_and_play.flowsynth.train \
    --latents "$CACHES/nsynth-train-latents" --conditions "$CACHES/neutral-conditions.npy" \
    --out "$RUNS/neutral-decoder"
