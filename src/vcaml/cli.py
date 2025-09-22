from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from vcaml.features.feature_extractor import extract_features_folder
from vcaml.io.pcap_to_parquet import (
    convert_dir_to_parquet,
    pcap_to_parquet,
)

# from vcaml.heuristic.predict import predict_heuristic

app = typer.Typer(help='PCAP → Parquet → features → (heuristic) QoE')

# ---------- INGEST ----------
ingest = typer.Typer(help='Parse PCAPs into Parquet')
app.add_typer(ingest, name='ingest')


@ingest.command('file')
def ingest_file(
    pcap: Path = typer.Argument(..., help='Path to a .pcap/.pcapng'),
    out: Path = typer.Option(..., '--out', '-o', help='Output .parquet path'),
    include_rtp: bool = typer.Option(False, '--include-rtp', help='Decode UDP ports as RTP'),
):
    pcap_to_parquet(pcap, out, include_rtp=include_rtp)
    typer.echo(f'Wrote {out}')


@ingest.command('dir')
def ingest_dir(
    in_dir: Path = typer.Argument('data/raw', help='Folder with pcaps'),
    out_dir: Path = typer.Option('data/interim', '--out', '-o', help='Output folder'),
    include_rtp: bool = typer.Option(False, '--include-rtp'),
    recursive: bool = typer.Option(False, '--recursive', '-R'),
    overwrite: bool = typer.Option(False, '--overwrite'),
):
    written = list(
        convert_dir_to_parquet(
            in_dir, out_dir, include_rtp=include_rtp, overwrite=overwrite, recursive=recursive
        )
    )
    typer.echo(f'Wrote {len(written)} parquet file(s) to {out_dir}')


# ---------- FEATURES ----------
@app.command('features')
def features_adv_all(
    src: Path = typer.Argument(..., help='Packet parquet file or folder'),
    out: Path = typer.Option(..., '--out', '-o', help='Output file or folder'),
    config_path: Path = typer.Option('configs/features.yaml', '--config'),
    dataset: str = typer.Option('lab', '--dataset', help='lab|real'),
    feature_subset_csv: str = typer.Option('SIZE,IAT,LSTATS,TSTATS,RTP', '--feature-subset'),
):
    cfg = (
        yaml.safe_load(config_path.read_text())
        if config_path.suffix.lower() in {'.yml', '.yaml'}
        else json.loads(config_path.read_text())
    )
    feature_subset = [s.strip() for s in feature_subset_csv.split(',') if s.strip()]
    extract_features_folder(src, out, cfg, dataset, feature_subset)
    typer.echo(f'Wrote advanced features under {out}')


# ---------- PREDICT (heuristic) ----------
@app.command('predict')
def predict(
    src: Path = typer.Argument('data/features', help='Feature file or folder'),
    out: Path = typer.Option('data/predictions', '--out', '-o', help='Output file or folder'),
):
    src = Path(src)
    out = Path(out)
    if src.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        predict_heuristic(src, out)
        typer.echo(f'Wrote {out}')
    else:
        out.mkdir(parents=True, exist_ok=True)
        for p in src.glob('**/*.parquet'):
            rel = p.relative_to(src)
            dst = (out / rel).with_suffix('.parquet')
            dst.parent.mkdir(parents=True, exist_ok=True)
            predict_heuristic(p, dst)
        typer.echo(f'Wrote predictions under {out}')


if __name__ == '__main__':
    app()
