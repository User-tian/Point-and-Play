"""Pinned third-party assets and their default local paths.

Every asset is downloaded from a fixed revision and checked against its SHA-256
by ``scripts/download_assets.py``. Set ``POINTPLAY_ASSET_DIR`` to store them
somewhere other than ``./assets``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    relative_path: str
    url: str
    sha256: str


MUSIC_CLAP = Asset(
    "music_audioset_epoch_15_esc_90.14.pt",
    "https://huggingface.co/lukewys/laion_clap/resolve/"
    "b3708341862f581175dba5c356a4ebf74a9b6651/music_audioset_epoch_15_esc_90.14.pt",
    "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd",
)
DIFFRHYTHM_VAE = Asset(
    "diffrhythm_vae/vae_model.pt",
    "https://huggingface.co/ASLP-lab/DiffRhythm-vae/resolve/"
    "74e2afacfd91dd1b96662c96dcef763c1258768b/vae_model.pt",
    "712693f27299937c6ccf1a6d6f1d9b45c7c8c11210d3b0cbb0f36181465ba29f",
)
_NLAKH_COMMIT = "375e28f4d0cea8ef99faf2aa841b6a55e83dce3b"
NLAKH_MAPS = {
    split: Asset(
        f"nlakh_maps/idx_to_fam_inst_{split}.json",
        "https://raw.githubusercontent.com/minju0821/musical_instrument_retrieval/"
        f"{_NLAKH_COMMIT}/evaluation/idx_to_fam_inst_{split}.json",
        sha,
    )
    for split, sha in (
        ("train", "676158f091fea92c341d13e44382e8bc4ee3bb9c23bdbb653615d6d328ff3fbd"),
        ("valid", "ddfb5d084c37cd750ca042e8a162e59f31cffd823765351b677f7a4cd7f61708"),
    )
}
ALL_ASSETS = (MUSIC_CLAP, DIFFRHYTHM_VAE, *NLAKH_MAPS.values())


def asset_dir() -> Path:
    return Path(os.environ.get("POINTPLAY_ASSET_DIR", "assets"))


def asset_path(asset: Asset) -> Path:
    return asset_dir() / asset.relative_path


def default_music_clap() -> Path:
    return asset_path(MUSIC_CLAP)


def default_vae() -> Path:
    return asset_path(DIFFRHYTHM_VAE)


def default_nlakh_map_dir() -> Path:
    return asset_path(NLAKH_MAPS["train"]).parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, what: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{what} not found at {path}; run `python scripts/download_assets.py` "
            "or pass the path explicitly"
        )
    return path
