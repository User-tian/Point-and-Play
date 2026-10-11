"""Download the pinned third-party assets into ``$POINTPLAY_ASSET_DIR`` (default ``./assets``).

* music-CLAP checkpoint (LAION-CLAP ``music_audioset_epoch_15_esc_90.14.pt``)
* DiffRhythm VAE (TorchScript ``vae_model.pt``)
* official NLakh <-> NSynth identity maps (``idx_to_fam_inst_{train,valid}.json``)

Each file is fetched from a fixed revision and verified against its SHA-256.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from point_and_play.assets import ALL_ASSETS, asset_path, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()
    failures = []
    for asset in ALL_ASSETS:
        path = asset_path(asset)
        if path.exists() and not args.force:
            if sha256_file(path) == asset.sha256:
                print(f"{path}: present, ok")
            else:
                print(f"{path}: SHA-256 MISMATCH (re-run with --force)")
                failures.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        print(f"downloading {asset.url}", flush=True)
        urllib.request.urlretrieve(asset.url, temporary)
        if sha256_file(temporary) != asset.sha256:
            temporary.unlink()
            raise SystemExit(f"SHA-256 mismatch for {asset.url}")
        temporary.replace(path)
        print(f"{path}: ok")
    if failures:
        raise SystemExit(f"{len(failures)} asset(s) failed verification")


if __name__ == "__main__":
    main()
