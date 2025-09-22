from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import polars as pl

# Base fields we always extract (UDP over IP)
TSHARK_FIELDS_BASE: list[str] = [
    'frame.time_epoch',  # float seconds since epoch
    'ip.src',
    'ip.dst',
    'udp.srcport',
    'udp.dstport',
    'frame.len',
]

# Optional RTP fields (only emitted if include_rtp=True and traffic actually decodes as RTP)
TSHARK_FIELDS_RTP: list[str] = [
    'rtp.ssrc',
    'rtp.timestamp',
    'rtp.seq',
    'rtp.p_type',
    'rtp.marker',
]


def _tshark_to_separated(
    pcap: Path,
    tmp_path: Path,
    include_rtp: bool,
    sep: str = '\t',
) -> None:
    """
    Run tshark to extract fields into a headered, separator-delimited text file (TSV by default).
    Streams stdout directly to a file to avoid buffering huge outputs in memory.
    """
    fields: list[str] = TSHARK_FIELDS_BASE + (TSHARK_FIELDS_RTP if include_rtp else [])
    cmd: list[str] = [
        'tshark',
        '-r',
        str(pcap),
        '-T',
        'fields',
        '-E',
        f'separator={sep}',  # <-- tab-separated output
        '-E',
        'header=y',
    ]
    for f in fields:
        cmd.extend(['-e', f])

    display_filter = 'ip && udp'
    cmd.extend(['-Y', display_filter])

    if include_rtp:
        cmd.extend(['-d', 'udp.port==1024-49152,rtp'])

    with tmp_path.open('w') as fh:
        subprocess.run(cmd, stdout=fh, check=True, text=True)


def _separated_to_parquet(
    in_path: Path,
    out_parquet: Path,
    sep: str = '\t',
) -> None:
    """
    Load TSV (or other sep-delimited) with Polars, normalize column names/types,
    add timestamp_ns, and write Parquet.
    """
    df = pl.read_csv(
        in_path,
        separator=sep,
        null_values=[''],  # empty fields → null
        ignore_errors=True,  # tolerate occasional bad rows
        infer_schema_length=1000,  # safer inference for mixed captures
    )

    rename_map = {
        'frame.time_epoch': 'time_epoch',
        'ip.src': 'src_ip',
        'ip.dst': 'dst_ip',
        'udp.srcport': 'src_port',
        'udp.dstport': 'dst_port',
        'frame.len': 'length',
        # RTP (optional)
        'rtp.ssrc': 'rtp_ssrc',
        'rtp.timestamp': 'rtp_timestamp',
        'rtp.seq': 'rtp_seq',
        'rtp.p_type': 'rtp_ptype',
        'rtp.marker': 'rtp_marker',
    }
    existing = {c: rename_map[c] for c in df.columns if c in rename_map}
    df = df.rename(existing)

    if 'time_epoch' in df.columns:
        df = df.with_columns(pl.col('time_epoch').cast(pl.Float64, strict=False)).with_columns(
            (pl.col('time_epoch') * 1_000_000_000).cast(pl.Int64).alias('timestamp_ns')
        )

    for c in ('src_port', 'dst_port', 'length'):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Int64, strict=False))

    for c, typ in [
        ('rtp_ssrc', pl.Utf8),
        ('rtp_timestamp', pl.Int64),
        ('rtp_seq', pl.Int64),
        ('rtp_ptype', pl.Int64),
        ('rtp_marker', pl.Int64),
    ]:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(typ, strict=False))

    core_cols = ['timestamp_ns', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'length']
    rtp_cols = [
        c
        for c in ['rtp_ssrc', 'rtp_timestamp', 'rtp_seq', 'rtp_ptype', 'rtp_marker']
        if c in df.columns
    ]
    select_cols = [c for c in core_cols if c in df.columns] + rtp_cols

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.select(select_cols).write_parquet(out_parquet)


def pcap_to_parquet(
    pcap: str | Path,
    out_parquet: str | Path,
    include_rtp: bool = False,
    sep: str = '\t',
) -> None:
    """
    One-file conversion: PCAP → TSV (temp) → Parquet.
    """
    pcap_path = Path(pcap)
    out_parquet_path = Path(out_parquet)
    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=True) as tmp:
        tmp_path = Path(tmp.name)
        _tshark_to_separated(pcap_path, tmp_path, include_rtp=include_rtp, sep=sep)
        _separated_to_parquet(tmp_path, out_parquet_path, sep=sep)


def convert_dir_to_parquet(
    in_dir: str | Path,
    out_dir: str | Path | None = None,
    include_rtp: bool = False,
    overwrite: bool = False,
    recursive: bool = False,
    sep: str = '\t',
) -> Iterable[Path]:
    """
    Convert all .pcap/.pcapng under in_dir to .parquet.
    Yields written Parquet paths.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir) if out_dir else in_dir

    def pcaps_iter() -> Iterable[Path]:
        if recursive:
            yield from (p for p in in_dir.rglob('*') if p.suffix in {'.pcap', '.pcapng'})
        else:
            yield from (p for p in in_dir.iterdir() if p.suffix in {'.pcap', '.pcapng'})

    for pcap in pcaps_iter():
        rel = pcap.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix('.parquet')
        if out_path.exists() and not overwrite:
            yield out_path
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'{pcap.name} -> {out_path.name}')
        try:
            pcap_to_parquet(pcap, out_path, include_rtp=include_rtp, sep=sep)
        except Exception as e:
            print(f'Error processing {pcap}: {e}')
            continue
        yield out_path
