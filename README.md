# Point-and-Play: Instrument Timbre Cloning from Music Mixtures

Official implementation of **"Point-and-Play: Instrument Timbre Cloning from Music Mixtures."**

Point-and-Play clones the timbre of a *cued* instrument directly from a polyphonic mixture and
plays it at new pitches and velocities. A LoRA-adapted music-CLAP encodes the mixture, a frozen
CLAP encodes an audio or text cue, and **Cue-Gated Statistics Pooling (CGSP)** turns both into a
512-d timbre code. A separately trained flow-matching note synthesizer (a FlowSynth-style DiT)
renders that code, a MIDI pitch and a velocity into DiffRhythm-VAE latents, which the frozen VAE
decodes to 44.1 kHz audio.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Links

- 🎧 [**Demo page**](https://user-tian.github.io/PAP_demo/)
- 📄 [**Paper**](TODO-arxiv-link)
- 💾 [**Checkpoints**](https://huggingface.co/Usertian1/Point-and-Play)

## Checkpoints

Pre-trained checkpoints are hosted at the link above. Download and place them under
`checkpoints/` using the paths shown below (any other location works too — just pass the paths
explicitly).

| Model | Description | Default path |
| --- | --- | --- |
| Timbre extractor | LoRA music-CLAP + CGSP, 18k steps | `checkpoints/encoder.pt` |
| Matched decoder | FlowSynth DiT conditioned on extractor codes | `checkpoints/matched_decoder.pt` |
| Neutral decoder | Same DiT conditioned on clean music-CLAP embeddings (baselines/ablations) | `checkpoints/neutral_decoder.pt` |

These checkpoints were retrained with this repository's recipe. On the 27 held-out NSynth
instruments they reach a CLAP score of 0.717 (text cue), 0.670 (same-category audio cue) and
0.663 (target-instrument audio cue), with FAD 1.67 / 1.86 / 2.13. Over four training seeds the
CLAP scores are 0.703 ± 0.010, 0.663 ± 0.007 and 0.663 ± 0.005.

The third-party assets (music-CLAP checkpoint, DiffRhythm VAE, NLakh identity maps) are fetched
separately by `scripts/download_assets.py` — see below.

## Setup

We use [uv](https://github.com/astral-sh/uv) to manage the Python 3.10 environment. A CUDA GPU is
required for training and inference.

```bash
# clone project
git clone TODO-repo-url
cd point-and-play

# create a Python 3.10 virtual environment with uv (install uv first)
uv venv .venv --python 3.10
source .venv/bin/activate

# install the PyTorch stack first (CUDA wheels):
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

# install the package and the remaining dependencies
uv pip install -e ".[test]"

# download pinned third-party assets (music-CLAP, DiffRhythm VAE, NLakh maps) into ./assets
python scripts/download_assets.py
```

If you are not on CUDA 12.4, keep the same version trio from `pyproject.toml` and change only the
PyTorch wheel index URL to the one matching your system from the official
[PyTorch install selector](https://pytorch.org/get-started/locally/).

`download_assets.py` fetches each file from a pinned revision and checks its SHA-256. Set
`POINTPLAY_ASSET_DIR` to keep the assets elsewhere. On first use, LAION-CLAP also downloads its
RoBERTa text tower from the Hugging Face Hub. Unit tests run on CPU: `pytest`.

## How to run

### Inference

Clone a cued instrument from a mixture and play it at new pitches:

```bash
python -m point_and_play.generate \
    --encoder checkpoints/encoder.pt --decoder checkpoints/matched_decoder.pt \
    --mixture song.wav --offset 30 --text guitar \
    --pitches 48 55 60 67 --velocity 100 --out-dir samples/
```

The encoder sees a 4-s mixture window (`--offset`, `--seconds`). The cue is either a category
name (`--text`; training used the NSynth categories bass, brass, flute, guitar, keyboard, mallet,
organ, reed, string, synth and vocal) or a recording of the instrument category
(`--cue-audio`). For every pitch, 8 candidates are integrated with 50 Euler steps. The candidate
whose music-CLAP embedding is closest to the timbre code is written as a 44.1-kHz WAV, and all
scores go to `generation.json`.

### Training

Training happens in two stages; `scripts/reproduce.sh` runs every step in order
(`DATA_ROOT=/path bash scripts/reproduce.sh`). See [Datasets](#datasets) for the expected data
layout.

**Stage 1: timbre extractor** (about 3 h on one A100):

```bash
python -m point_and_play.train_encoder --data-root $DATA_ROOT --out runs/encoder
```

The defaults are the paper recipe: rank-8 LoRA (α = 16) on every `qkv`, `fc1` and `fc2` layer of
the music-CLAP HTSAT-base audio tower, plus a CGSP head; corpus weights 0.30 / 0.30 / 0.10 /
0.30; 4-s mixtures, batch 16; 18k AdamW steps with 500 warm-up steps to 3e-4 and cosine decay to
3e-5; InfoNCE temperature 0.1; `L_E = L_inv + L_algn^a + L_algn^t + 0.25 L_sel^a`. Each loss
weight is a flag (`--lambda-inv`, `--lambda-audio`, `--lambda-text`, `--lambda-sel`); setting one
to 0 gives the corresponding leave-one-out ablation. Every 3k steps the text-cued code is scored
on 600 fixed NSynth-valid mixtures; the best checkpoint is saved as `extractor_best.pt`, the
final one as `extractor.pt`.

**Stage 2: note synthesizer** (about 7 h per decoder on 8 × A100 80 GB):

```bash
NSYNTH=$DATA_ROOT/nsynth; NLAKH=$DATA_ROOT/nlakh/rendered_single_inst

# 2a. VAE latents of every NSynth-train note (shared by both decoders)
torchrun --standalone --nproc-per-node 8 -m point_and_play.flowsynth.latents \
    --nsynth-root $NSYNTH --out caches/nsynth-train-latents

# 2b. matched decoder: conditioned on codes extracted by the trained encoder
python -m point_and_play.flowsynth.conditions --kind matched --encoder runs/encoder/extractor_best.pt \
    --nsynth-root $NSYNTH --nlakh-root $NLAKH --out caches/matched-conditions.npy
torchrun --standalone --nproc-per-node 8 -m point_and_play.flowsynth.train \
    --latents caches/nsynth-train-latents --conditions caches/matched-conditions.npy --out runs/matched-decoder

# 2c. neutral decoder: conditioned on clean music-CLAP embeddings (baselines/ablations)
python -m point_and_play.flowsynth.conditions --kind neutral \
    --nsynth-root $NSYNTH --nlakh-root $NLAKH --out caches/neutral-conditions.npy
torchrun --standalone --nproc-per-node 8 -m point_and_play.flowsynth.train \
    --latents caches/nsynth-train-latents --conditions caches/neutral-conditions.npy --out runs/neutral-decoder
```

Row *i* of a condition cache pairs with NSynth-train note *i*; a seeded coin flip selects the
note itself or a 4-s NLakh phrase of the same instrument as the conditioning source, and the
random draws depend only on the row, so different encoders see identical mixtures. The decoder is
a 24-layer, 1024-wide, 16-head DiT with RoPE and AdaLN, trained with conditional flow matching
for 150k steps (AdamW 1e-4, 2k warm-up + cosine decay, gradient clipping 1.0, bf16 autocast, EMA
0.9999 used at inference, condition dropout 0.1, global batch 32 × 8 GPUs). `checkpoints/latest.pt`
is written every 5k steps; re-running the same command with `--resume` continues bit-exactly on
the same number of GPUs.

## Method overview

Point-and-Play is trained in two stages:

1. **Timbre extractor.** A LoRA-adapted music-CLAP encodes a 4-s polyphonic mixture into a token
   sequence, and a frozen music-CLAP encodes the cue (a category name, or a recording of the
   instrument category). Cue-Gated Statistics Pooling gates the mixture tokens by their
   similarity to the cue and reduces the gated tokens to a 512-d timbre code. Training mixes
   NSynth, NLakh, Slakh2100 and MoisesDB, and combines an invariance loss between paired
   mixtures, audio- and text-cued alignment losses against the isolated-source CLAP embedding,
   and a selectivity loss that pushes mismatched cues away from the target.

2. **Note synthesizer.** A FlowSynth-style DiT is trained with conditional flow matching to
   generate DiffRhythm-VAE latents of a note from three conditions: the timbre code, a MIDI
   pitch, and a velocity. The frozen VAE decoder renders the latents to 44.1 kHz audio. The
   *matched* decoder is conditioned on codes produced by the trained extractor from mixtures, so
   its training distribution matches inference.

## Datasets

Download [NSynth](https://magenta.tensorflow.org/datasets/nsynth) (train and valid),
[NLakh](https://github.com/minju0821/musical_instrument_retrieval) (`rendered_single_inst`),
[Slakh2100-redux at 16 kHz](http://www.slakh.com/) and
[MoisesDB](https://github.com/moises-ai/moises-db), and arrange them as

```
$DATA_ROOT/
├── nsynth/nsynth-{train,valid}/{examples.json,audio/*.wav}
├── nlakh/rendered_single_inst/{train/001..953,valid/01..53}/*.wav
├── slakh2100_redux_16k/train/Track*/{mix.flac,metadata.yaml,stems/*.flac}
└── moisesdb/moisesdb_v0.1/<song>/{data.json,<stem>/<track>.wav}
```

or pass `--nsynth-root`, `--nlakh-root`, `--slakh-root` and `--moisesdb-root` to the encoder
trainer. How the splits are used:

- **Extractor training:** NSynth-train (80% of the notes of every instrument, chosen by a hash of
  the file name), NLakh-train, the Slakh2100 train split, and MoisesDB minus every 10th song.
- **Decoder training:** all 289,205 NSynth-train notes are the synthesis targets. Half of the
  conditioning sources are NLakh phrases of the same instrument.
- **Evaluation:** the 53 NSynth-valid instruments in sorted order. The paper's 27 held-out test
  instruments are the last 27.

## Reproducibility

This repository is a cleaned-up version of the research code behind the paper. On the same
hardware it reproduces that code bitwise in every stage we checked: extractor training data and
training steps, condition caches, VAE latents, decoder training state, and Euler-sampled latents.
A seed fixes the training stream; each DataLoader worker holds its own copy of the dataset's
random generator, so the stream also depends on `--workers` (the paper used 4).

## Citation

```bibtex
@misc{tian2026pointandplay,
  title  = {Point-and-Play: Instrument Timbre Cloning from Music Mixtures},
  author = {Tian, Baotong and Duan, Zhiyao},
  year   = {2026},
  note   = {Under review}
}
```

## Acknowledgments

This work builds on [LAION-CLAP](https://github.com/LAION-AI/CLAP) (music checkpoint
`music_audioset_epoch_15_esc_90.14.pt`) and the
[DiffRhythm VAE](https://huggingface.co/ASLP-lab/DiffRhythm-vae), whose license applies to its
weights. The note synthesizer re-implements the FlowSynth decoder of Yang et al. (ICASSP 2026).
The official NLakh identity maps come from
[minju0821/musical_instrument_retrieval](https://github.com/minju0821/musical_instrument_retrieval)
(MIT).

## License

The code is released under the [MIT License](LICENSE). The assets fetched by
`scripts/download_assets.py` and the datasets keep their own licenses.
