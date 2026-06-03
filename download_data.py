#!/usr/bin/env python
"""Download and extract vcaml datasets from Google Drive.

File IDs are read from the `gdrive` section of config.yaml. Each entry is a
Google Drive file ID pointing to a zip archive of the corresponding dataset.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --path /data/taveesh/vca
    uv run python scripts/download_data.py --dataset in_lab_data
"""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import gdown
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config():
    with open(_PROJECT_ROOT / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('data_root', '/data/taveesh/vca'), cfg.get('gdrive', {})


def _download_and_extract(file_id: str, dest: Path, name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {name} (id={file_id}) → {dest}")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / f"{name}.zip"
        gdown.download(url, str(zip_path), quiet=False)
        print(f"Extracting {zip_path.name} → {dest}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest.parent)
    print(f"Done: {dest}")


def main():
    parser = argparse.ArgumentParser(description='Download vcaml datasets from Google Drive')
    parser.add_argument(
        '--path', default=None,
        help='Destination root directory (default: data_root from config.yaml)')
    parser.add_argument(
        '--dataset', choices=['in_lab_data', 'real_world_data', 'all'], default='all',
        help='Which dataset to download (default: all)')
    args = parser.parse_args()

    data_root, gdrive_ids = _load_config()
    dest_root = Path(args.path) if args.path else Path(data_root)

    datasets = ['in_lab_data', 'real_world_data'] if args.dataset == 'all' else [args.dataset]

    missing = [ds for ds in datasets if not gdrive_ids.get(ds)]
    if missing:
        print(
            f"ERROR: No Google Drive file ID configured for: {', '.join(missing)}\n"
            f"Set the IDs under `gdrive` in config.yaml and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    for ds in datasets:
        _download_and_extract(gdrive_ids[ds], dest_root / ds, ds)


if __name__ == '__main__':
    main()
