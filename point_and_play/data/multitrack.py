"""Song indexing for Slakh2100 and MoisesDB."""

from __future__ import annotations

import glob
import json
import os

import soundfile as sf
import yaml


def index_slakh(split_dir, min_labels: int = 2):
    """Slakh2100 songs with at least ``min_labels`` non-drum stem classes.

    Each song: {"mix": path, "tracks": {class: [stem paths]}, "nframes": int, "sr": int}.
    """
    songs = []
    for track in sorted(glob.glob(os.path.join(split_dir, "Track*"))):
        if os.path.basename(track).startswith("._"):
            continue
        mix_path = os.path.join(track, "mix.flac")
        meta_path = os.path.join(track, "metadata.yaml")
        if not (os.path.exists(mix_path) and os.path.exists(meta_path)):
            continue
        try:
            with open(meta_path, "r", errors="ignore") as stream:
                meta = yaml.safe_load(stream)
        except Exception:
            continue
        tracks = {}
        for stem_id, stem in (meta.get("stems") or {}).items():
            if stem.get("is_drum") or not stem.get("audio_rendered", True):
                continue
            wav = os.path.join(track, "stems", f"{stem_id}.flac")
            if os.path.exists(wav):
                tracks.setdefault(stem.get("inst_class") or "unknown", []).append(wav)
        if len(tracks) < min_labels:
            continue
        info = sf.info(mix_path)
        songs.append({"mix": mix_path, "tracks": tracks, "nframes": info.frames,
                      "sr": info.samplerate})
    return songs


def index_moisesdb(root, min_labels: int = 2):
    """MoisesDB songs with at least ``min_labels`` stem names; the mixture is the stem sum."""
    songs = []
    for song_dir in sorted(glob.glob(os.path.join(root, "*"))):
        meta_path = os.path.join(song_dir, "data.json")
        if not os.path.isdir(song_dir) or not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as stream:
                meta = json.load(stream)
        except Exception:
            continue
        tracks = {}
        for stem in meta.get("stems", []):
            name = stem.get("stemName", "unknown")
            for track in stem.get("tracks", []):
                wav = os.path.join(song_dir, name, f"{track['id']}.wav")
                if os.path.exists(wav):
                    tracks.setdefault(name, []).append(wav)
        if len(tracks) < min_labels:
            continue
        info = sf.info(next(iter(next(iter(tracks.values())))))
        songs.append({"mix": None, "tracks": tracks, "nframes": info.frames,
                      "sr": info.samplerate})
    return songs


def moisesdb_training_songs(songs):
    """Every 10th song (in sorted order) is held out from extractor training."""
    return [song for i, song in enumerate(songs) if i % 10 != 0]
