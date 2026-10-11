"""Paired-mixture datasets for extractor training (Sec. 4.1).

Every item is one primary mixture ``mix`` plus a paired mixture ``mixB`` with the same
instruments but different musical content (the ``L_inv`` positive). Up to ``n_sources``
sources of the mixture expose a dry isolated recording (aligned with ``mix``; the CLAP
teacher target) and a dry audio cue. With ``cue_mode="class"`` the audio cue comes from a
*different* instrument of the same category, as in the paper.

All randomness comes from a per-dataset ``random.Random(seed)``. The order of random draws
follows the reference implementation used for the paper, so a given seed reproduces the
same sample stream. Note that each DataLoader worker holds its own copy of that generator,
so the stream also depends on ``num_workers`` (the paper used 4).
"""

from __future__ import annotations

import glob
import os
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .audio import fix_length, read_window, snr_gain, sum_window
from .catalog import index_nsynth, nlakh_identity_map, parse_nsynth_name

CUE_MODES = ("self", "class")


class _NoteMixtures(Dataset):
    """NSynth notes / NLakh phrases: one target plus ``n_interf`` other identities at random SNR.

    Subclasses define ``self.by`` ({identity: [clip paths]}), ``src_sr`` and ``_load``.
    """

    domain = "note"
    src_sr = 16000

    def _setup(self, seconds, n_interf, n_sources, snr, seed, cue_mode):
        if cue_mode not in CUE_MODES:
            raise ValueError(f"cue_mode must be one of {CUE_MODES}")
        self.ids = sorted(self.by)
        self.files = [(path, identity) for identity in self.ids for path in self.by[identity]]
        self.n = int(seconds * self.src_sr)
        self.n_interf = int(n_interf)
        self.n_sources = max(1, int(n_sources))
        self.snr = snr
        self.rng = random.Random(seed)
        self.cue_mode = cue_mode
        self.family_ids: dict[str, list[str]] = {}
        for identity in self.ids:
            self.family_ids.setdefault(identity.split("_")[0], []).append(identity)

    def __len__(self):
        return len(self.files)

    def _other_clip(self, identity, *exclude):
        exclude = set(exclude)
        pool = [f for f in self.by[identity] if f not in exclude] or list(self.by[identity])
        return pool[self.rng.randrange(len(pool))]

    def _cue_clip(self, identity, *exclude):
        if self.cue_mode == "class":
            siblings = [i for i in self.family_ids.get(identity.split("_")[0], ()) if i != identity]
            if siblings:
                sibling = siblings[self.rng.randrange(len(siblings))]
                pool = self.by[sibling]
                return pool[self.rng.randrange(len(pool))]
        return self._other_clip(identity, *exclude)

    def __getitem__(self, index):
        path, identity = self.files[index]
        path_b = self._other_clip(identity, path)
        path_cue = self._cue_clip(identity, path, path_b)
        interferers = self.rng.sample([i for i in self.ids if i != identity],
                                      min(self.n_interf, len(self.ids) - 1))
        target = self._load(path)
        mix = target.clone()
        mix_b = self._load(path_b)
        sources = [(identity, target, self._load(path_cue))]
        for other in interferers:
            other_a = self.by[other][self.rng.randrange(len(self.by[other]))]
            other_b = self._other_clip(other, other_a) if len(self.by[other]) > 1 else other_a
            wav_a = self._load(other_a)
            mix = mix + snr_gain(mix, wav_a, self.rng.uniform(*self.snr)) * wav_a
            wav_b = self._load(other_b)
            mix_b = mix_b + snr_gain(mix_b, wav_b, self.rng.uniform(*self.snr)) * wav_b
            if len(sources) < self.n_sources:
                sources.append((other, wav_a, self._load(self._cue_clip(other, other_a, other_b))))
        return {"mix": mix, "mixB": mix_b,
                "cues": [cue for _, _, cue in sources],
                "isolated": [iso for _, iso, _ in sources],
                "labels": [label for label, _, _ in sources],
                "domain": self.domain}


class NSynthMixtures(_NoteMixtures):
    """Mixtures of single NSynth notes; positives and cues use notes of a different pitch."""

    domain = "nsynth"

    def __init__(self, audio_dir, seconds=4.0, n_interf=3, n_sources=3, note_split="train",
                 snr=(-5.0, 15.0), seed=0, cue_mode="class"):
        by = index_nsynth(audio_dir, note_split=note_split)
        self.by = {identity: paths for identity, paths in by.items() if len(paths) >= 3}
        self._setup(seconds, n_interf, n_sources, snr, seed, cue_mode)

    def _load(self, path):
        wav, _ = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(-1)
        return fix_length(torch.from_numpy(np.ascontiguousarray(wav)), self.n)

    def _other_clip(self, identity, *exclude):
        exclude = set(exclude)
        candidates = [f for f in self.by[identity] if f not in exclude] or list(self.by[identity])
        used_pitches = {parse_nsynth_name(e)[1] for e in exclude}
        pool = [f for f in candidates if parse_nsynth_name(f)[1] not in used_pitches] or candidates
        return pool[self.rng.randrange(len(pool))]


class NLakhMixtures(_NoteMixtures):
    """Mixtures of NLakh single-instrument phrases (random window per clip), keyed by NSynth identity."""

    domain = "nlakh"

    def __init__(self, split_dir, map_dir, split="train", seconds=4.0, n_interf=3, n_sources=3,
                 snr=(-5.0, 15.0), seed=0, cue_mode="class", min_clips=3):
        dir_to_identity = nlakh_identity_map(split_dir, map_dir, split)
        self.by = {}
        for directory in sorted(d for d in os.listdir(split_dir)
                                if os.path.isdir(os.path.join(split_dir, d))):
            clips = sorted(glob.glob(os.path.join(split_dir, directory, "*.wav")))
            if len(clips) >= min_clips and directory in dir_to_identity:
                self.by[dir_to_identity[directory]] = clips
        self._setup(seconds, n_interf, n_sources, snr, seed, cue_mode)

    def _load(self, path):
        try:
            total = sf.info(path).frames
        except Exception:
            total = 0
        if total > self.n:
            offset = self.rng.randrange(0, total - self.n + 1)
            wav, _ = sf.read(path, dtype="float32", start=offset, frames=self.n)
        else:
            wav, _ = sf.read(path, dtype="float32")
        wav = torch.from_numpy(np.ascontiguousarray(wav))
        if wav.dim() > 1:
            wav = wav.mean(-1)
        return fix_length(wav, self.n)


class MultitrackMixtures(Dataset):
    """Slakh2100 / MoisesDB: ``mix`` and ``mixB`` are two random windows of the same song.

    Sources are song-local stem classes; the isolated target is the stem window aligned with
    ``mix``. With ``cue_mode="class"`` the cue is a window of the same stem class from another
    song (falling back to another window of the same stem if the class occurs in one song only).
    """

    def __init__(self, songs, seconds=4.0, items_per_song=8, n_sources=3, seed=0,
                 domain="slakh", cue_mode="class"):
        if cue_mode not in CUE_MODES:
            raise ValueError(f"cue_mode must be one of {CUE_MODES}")
        self.songs = [s for s in songs if s["nframes"] > int(seconds * s["sr"]) * 2]
        if not self.songs:
            raise ValueError("no usable songs")
        self.src_sr = self.songs[0]["sr"]
        self.seconds = seconds
        self.n_sources = max(1, int(n_sources))
        self.index = [(song, j) for song in range(len(self.songs)) for j in range(items_per_song)]
        self.rng = random.Random(seed)
        self.domain = domain
        self.cue_mode = cue_mode
        self.class_songs: dict[str, list[int]] = {}
        for song_index, song in enumerate(self.songs):
            for label in song["tracks"]:
                self.class_songs.setdefault(label, []).append(song_index)

    def __len__(self):
        return len(self.index)

    def song_keys(self):
        return [song for song, _ in self.index]

    def labels(self):
        return sorted({label for song in self.songs for label in song["tracks"]})

    def _mix_window(self, song, start, n):
        if song["mix"] is not None:
            return read_window(song["mix"], start, n)
        return sum_window([p for paths in song["tracks"].values() for p in paths], start, n)

    @staticmethod
    def _stem_frames(paths, fallback):
        try:
            return min(sf.info(p).frames for p in paths)
        except Exception:
            return fallback

    def _cue_window(self, song_index, label, stem_paths, start, n):
        if self.cue_mode == "class":
            others = [i for i in self.class_songs.get(label, ()) if i != song_index]
            if others:
                other = self.songs[others[self.rng.randrange(len(others))]]
                paths = other["tracks"][label]
                high = min(self._stem_frames(paths, other["nframes"]), other["nframes"]) - n - 1
                return sum_window(paths, self.rng.randint(0, high) if high > 0 else 0, n)
        return sum_window(stem_paths, start, n)

    def __getitem__(self, index):
        song_index, _ = self.index[index]
        song = self.songs[song_index]
        n = int(self.seconds * song["sr"])
        stems = list(song["tracks"].keys())
        self.rng.shuffle(stems)
        max_start = song["nframes"] - n - 1
        start_a = self.rng.randint(0, max(1, max_start))
        start_b = self.rng.randint(0, max(1, max_start))
        item = {"mix": self._mix_window(song, start_a, n), "mixB": self._mix_window(song, start_b, n),
                "cues": [], "isolated": [], "labels": [], "domain": self.domain}
        for label in stems[: self.n_sources]:
            stem_paths = song["tracks"][label]
            high = min(self._stem_frames(stem_paths, song["nframes"]), song["nframes"]) - n - 1
            # Same-song cue window; drawn in every mode so the random stream does not depend on it.
            cue_start = self.rng.randint(0, high) if high > 0 else 0
            item["isolated"].append(sum_window(stem_paths, start_a, n))
            item["cues"].append(self._cue_window(song_index, label, stem_paths, cue_start, n))
            item["labels"].append(label)
        return item


def collate_mixtures(batch):
    """Pad items to K = max sources in the batch."""
    size = len(batch)
    slots = max(len(item["cues"]) for item in batch)
    length = batch[0]["mix"].numel()
    cues = torch.zeros(size, slots, length)
    isolated = torch.zeros(size, slots, length)
    valid = torch.zeros(size, slots, dtype=torch.bool)
    labels = []
    for b, item in enumerate(batch):
        row = []
        for k in range(slots):
            if k < len(item["cues"]):
                cues[b, k] = item["cues"][k]
                isolated[b, k] = item["isolated"][k]
                valid[b, k] = True
                row.append(item["labels"][k])
            else:
                row.append(None)
        labels.append(row)
    return {"mix": torch.stack([item["mix"] for item in batch]),
            "mixB": torch.stack([item["mixB"] for item in batch]),
            "cues": cues, "isolated": isolated, "valid": valid, "labels": labels,
            "domain": batch[0]["domain"]}


class DistinctKeySampler:
    """``steps`` batches whose items have pairwise distinct keys (instrument or song)."""

    def __init__(self, keys, batch_size=16, steps=18000, seed=0):
        self.by_key: dict = {}
        for index, key in enumerate(keys):
            self.by_key.setdefault(key, []).append(index)
        self.keys = list(self.by_key)
        self.batch_size = batch_size
        self.steps = steps
        self.rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.steps):
            keys = self.rng.sample(self.keys, min(self.batch_size, len(self.keys)))
            yield [self.rng.choice(self.by_key[key]) for key in keys]

    def __len__(self):
        return self.steps
