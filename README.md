# Point-and-Play: Instrument Timbre Cloning from Music Mixtures

Baotong Tian, Zhiyao Duan — University of Rochester

[Demo page](https://user-tian.github.io/PAP_demo/)

Point-and-Play clones the timbre of a *cued* instrument directly from a polyphonic mixture
and plays it at new pitches and velocities. A LoRA-adapted music-CLAP encodes the mixture, a
frozen CLAP encodes an audio or text cue, and **Cue-Gated Statistics Pooling (CGSP)** turns
both into a 512-d timbre code. A separately trained flow-matching note synthesizer
(modelled on FlowSynth) renders that code, a MIDI pitch and a velocity into DiffRhythm-VAE
latents, which the frozen VAE decodes to audio.

This repository contains everything needed to train and run the shipped system:

| Stage | What | Entry point |
| --- | --- | --- |
| 1 | Timbre extractor (LoRA music-CLAP + CGSP, 4 losses) | `point_and_play.train_encoder` |
| 2a | DiffRhythm-VAE latents of NSynth-train | `point_and_play.flowsynth.latents` |
| 2b | **Matched decoder**: conditioned on codes extracted by the trained encoder | `point_and_play.flowsynth.conditions --kind matched` + `point_and_play.flowsynth.train` |
| 2c | **Neutral decoder**: conditioned on clean music-CLAP embeddings (used for the oracle, cue-only and separate-then-clone baselines and the loss ablation) | `point_and_play.flowsynth.conditions --kind neutral` + `point_and_play.flowsynth.train` |
| — | Inference: mixture + cue → notes | `point_and_play.generate` |

## Installation

Python 3.10 and a CUDA GPU are required.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python scripts/download_assets.py      # music-CLAP, DiffRhythm VAE, NLakh identity maps -> ./assets
```

`download_assets.py` fetches each file from a pinned revision and checks its SHA-256. Set
`POINTPLAY_ASSET_DIR` to keep the assets elsewhere. On first use, LAION-CLAP also downloads
its RoBERTa text tower from the Hugging Face Hub. Unit tests run on CPU: `pytest`.

## Data

Download [NSynth](https://magenta.tensorflow.org/datasets/nsynth) (train and valid),
[NLakh](https://github.com/minju0821/musical_instrument_retrieval) (`rendered_single_inst`),
[Slakh2100-redux at 16 kHz](http://www.slakh.com/) and [MoisesDB](https://github.com/moises-ai/moises-db),
and arrange them as

```
$DATA_ROOT/
├── nsynth/nsynth-{train,valid}/{examples.json,audio/*.wav}
├── nlakh/rendered_single_inst/{train/001..953,valid/01..53}/*.wav
├── slakh2100_redux_16k/train/Track*/{mix.flac,metadata.yaml,stems/*.flac}
└── moisesdb/moisesdb_v0.1/<song>/{data.json,<stem>/<track>.wav}
```

or pass `--nsynth-root`, `--nlakh-root`, `--slakh-root` and `--moisesdb-root` to the encoder trainer.

How the splits are used:

* **Extractor training:** NSynth-train (80% of the notes of every instrument, chosen by a hash of the
  file name), NLakh-train, the Slakh2100 train split, and MoisesDB minus every 10th song.
* **Decoder training:** all 289,205 NSynth-train notes are the synthesis targets. Half of the
  conditioning sources are NLakh phrases of the same instrument.
* **Evaluation:** the 53 NSynth-valid instruments in sorted order. The paper's 27 held-out test
  instruments are the last 27.

## Training

`scripts/reproduce.sh` runs every step below in order (`DATA_ROOT=/path bash scripts/reproduce.sh`).

### Stage 1: timbre extractor (about 3 h on one A100)

```bash
python -m point_and_play.train_encoder --data-root $DATA_ROOT --out runs/encoder
```

The defaults are the paper recipe:

* rank-8 LoRA (α = 16) on every `qkv`, `fc1` and `fc2` layer of the music-CLAP HTSAT-base audio tower, plus a CGSP head
* corpus weights 0.30 / 0.30 / 0.10 / 0.30, 4-s mixtures, batch 16
* 18k AdamW steps: 500 warm-up steps to 3e-4, then cosine decay to 3e-5
* InfoNCE temperature 0.1
* `L_E = L_inv + L_algn^a + L_algn^t + 0.25 L_sel^a`

Each loss weight is a flag (`--lambda-inv`, `--lambda-audio`, `--lambda-text`, `--lambda-sel`).
Setting one to 0 gives the corresponding leave-one-out ablation in Table 2.

Every 3k steps the text-cued code is scored on 600 fixed NSynth-valid mixtures: the mean cosine to the
music-CLAP embedding of the isolated note. The run directory contains:

* `extractor_best.pt`: highest validation score. This checkpoint conditioned the paper's matched decoder.
* `extractor.pt`: final step.
* `ckpt_<step>.pt`: intermediate checkpoints.
* `result.json`: validation history.

The validation mixtures come from all 53 NSynth-valid instruments, so they include the 27
evaluation instruments. Use `extractor.pt` if you want model selection that never touches them.

### Stage 2: note synthesizer (about 7 h per decoder on 8 × A100 80 GB)

```bash
NSYNTH=$DATA_ROOT/nsynth; NLAKH=$DATA_ROOT/nlakh/rendered_single_inst

# 2a. VAE latents of every NSynth-train note (shared by both decoders)
torchrun --standalone --nproc-per-node 8 -m point_and_play.flowsynth.latents \
    --nsynth-root $NSYNTH --out caches/nsynth-train-latents

# 2b. matched decoder
python -m point_and_play.flowsynth.conditions --kind matched --encoder runs/encoder/extractor_best.pt \
    --nsynth-root $NSYNTH --nlakh-root $NLAKH --out caches/matched-conditions.npy
torchrun --standalone --nproc-per-node 8 -m point_and_play.flowsynth.train \
    --latents caches/nsynth-train-latents --conditions caches/matched-conditions.npy --out runs/matched-decoder

# 2c. neutral decoder
python -m point_and_play.flowsynth.conditions --kind neutral \
    --nsynth-root $NSYNTH --nlakh-root $NLAKH --out caches/neutral-conditions.npy
torchrun --standalone --nproc-per-node 8 -m point_and_play.flowsynth.train \
    --latents caches/nsynth-train-latents --conditions caches/neutral-conditions.npy --out runs/neutral-decoder
```

**Condition caches.** Row *i* of a condition cache pairs with NSynth-train note *i*. For each row, a seeded
coin flip selects one of two sources: the note itself, or a 4-s NLakh phrase of the same instrument
(144,527 and 144,678 rows respectively).

* **Matched decoder:** that source is mixed with three other instruments at −5 to 15 dB and encoded by the
  extractor with the category text cue.
* **Neutral decoder:** the clean source is embedded by the frozen music-CLAP.

The random draws depend only on the row, so different encoders see identical mixtures.

**Decoder.** A 24-layer, 1024-wide, 16-head DiT with RoPE and AdaLN. It is trained with conditional flow
matching for 150k steps:

* AdamW 1e-4 with 2k warm-up steps and cosine decay
* gradient clipping at 1.0, bf16 autocast
* EMA 0.9999, used at inference
* condition dropout 0.1
* 32 notes per GPU on 8 GPUs, a global batch of 256

Another GPU count changes the global batch and the sample order. `checkpoints/latest.pt` is
written every 5k steps. Re-running the same command with `--resume` continues bit-exactly
on the same number of GPUs.

## Inference

```bash
python -m point_and_play.generate \
    --encoder runs/encoder/extractor_best.pt --decoder runs/matched-decoder/checkpoints/final.pt \
    --mixture song.wav --offset 30 --text guitar \
    --pitches 48 55 60 67 --velocity 100 --out-dir samples/
```

The encoder sees a 4-s mixture window (`--offset`, `--seconds`). The cue is either a category name
(`--text`; training used the NSynth categories bass, brass, flute, guitar, keyboard, mallet, organ,
reed, string, synth and vocal) or a recording of the instrument category (`--cue-audio`). For every
pitch, 8 candidates are integrated with 50 Euler steps. The candidate whose music-CLAP embedding is
closest to the timbre code is written as a 44.1-kHz WAV, and all scores go to `generation.json`.

## Reproducibility

This repository is a cleaned-up version of the research code behind the paper. On the same
hardware it reproduces that code bitwise in every stage we checked:

* **Extractor training data:** items from all four corpora and the validation mixtures.
* **Extractor training:** 101 steps across all corpora, with identical losses and LoRA/CGSP weights.
* **Condition caches:** matched and neutral rows.
* **VAE latents:** with 8 processes. Other process counts differ only at float16 rounding.
* **Decoder training:** model, EMA and optimizer state.
* **Sampling:** the Euler-sampled latents.

A seed fixes the training stream. Each DataLoader worker holds its own copy of the dataset's
random generator, so the stream also depends on `--workers`; the paper used 4.

## Citation

```bibtex
@misc{tian2026pointandplay,
  title  = {Point-and-Play: Instrument Timbre Cloning from Music Mixtures},
  author = {Tian, Baotong and Duan, Zhiyao},
  year   = {2026},
  note   = {Under review}
}
```

## License

The code is released under the [MIT License](LICENSE). The assets fetched by
`scripts/download_assets.py` and the datasets keep their own licenses.

## Acknowledgements

This work builds on [LAION-CLAP](https://github.com/LAION-AI/CLAP) (music checkpoint
`music_audioset_epoch_15_esc_90.14.pt`) and the [DiffRhythm VAE](https://huggingface.co/ASLP-lab/DiffRhythm-vae),
whose license applies to its weights. The note synthesizer re-implements the FlowSynth decoder of
Yang et al. (ICASSP 2026). The official NLakh identity maps come from
[minju0821/musical_instrument_retrieval](https://github.com/minju0821/musical_instrument_retrieval) (MIT).
