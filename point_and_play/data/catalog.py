"""NSynth indexing and the official NLakh <-> NSynth instrument identity map."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

NLAKH_INSTRUMENTS = {"train": 953, "valid": 53}


def parse_nsynth_name(path) -> tuple[str, int, int]:
    """``<family>_<source>_<NNN>-<pitch>-<velocity>.wav`` -> (identity, pitch, velocity)."""
    identity, pitch, velocity = os.path.basename(path)[:-4].split("-")
    return identity, int(pitch), int(velocity)


def _note_bucket(path, buckets: int = 10) -> int:
    return int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16) % buckets


def index_nsynth(audio_dir, note_split: str | None = None, test_fraction: float = 0.2):
    """{identity: [note paths]} for an NSynth ``audio`` directory.

    ``note_split="train"`` keeps the 80% of notes whose filename hash falls outside the
    held-out buckets (the extractor never trains on the other 20%); ``None`` keeps all notes.
    """
    held_out = max(1, round(test_fraction * 10))
    by_identity: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)):
        if note_split is not None and (note_split == "test") != (_note_bucket(path) < held_out):
            continue
        by_identity.setdefault(parse_nsynth_name(path)[0], []).append(path)
    return by_identity


def normalize_preset(preset: str) -> str:
    """``<family>_<source>_<N>`` -> NSynth filename identity ``<family>_<source>_<NNN>``."""
    prefix, number = preset.rsplit("_", 1)
    return f"{prefix}_{int(number):03d}"


def nlakh_identity_map(nlakh_split_dir, map_dir, split: str) -> dict[str, str]:
    """{NLakh directory name: NSynth identity} for ``split`` (``train`` or ``valid``).

    NLakh renders one directory per NSynth preset, named ``001..953`` (train) / ``01..53``
    (valid). Directory ``d`` corresponds to entry ``int(d) - 1`` of the official
    ``idx_to_fam_inst_<split>.json`` released with the NLakh paper.
    """
    with open(Path(map_dir) / f"idx_to_fam_inst_{split}.json") as stream:
        index_to_preset = {int(i): normalize_preset(p) for p, i in json.load(stream).items()}
    if sorted(index_to_preset) != list(range(NLAKH_INSTRUMENTS[split])):
        raise ValueError(f"unexpected NLakh identity map for split {split}")
    directories = sorted(d for d in os.listdir(nlakh_split_dir)
                         if os.path.isdir(os.path.join(nlakh_split_dir, d)))
    return {d: index_to_preset[int(d) - 1] for d in directories}
