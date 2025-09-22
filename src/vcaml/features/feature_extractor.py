from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# ------------------------- helpers (unchanged ideas) -------------------------


def _canon_flow(df: pl.DataFrame) -> pl.DataFrame:
    if not {'timestamp_ns', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'length'}.issubset(
        df.columns
    ):
        missing = {'timestamp_ns', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'length'} - set(
            df.columns
        )
        raise ValueError(f'Missing columns: {missing}')

    out = (
        df.with_columns(
            [
                pl.when(pl.col('src_ip') <= pl.col('dst_ip'))
                .then(pl.struct(['src_ip', 'src_port', 'dst_ip', 'dst_port']))
                .otherwise(pl.struct(['dst_ip', 'dst_port', 'src_ip', 'src_port']))
                .alias('canon')
            ]
        )
        .with_columns(
            [
                pl.col('canon').struct.field('src_ip').alias('ip_a'),
                pl.col('canon').struct.field('src_port').alias('port_a'),
                pl.col('canon').struct.field('dst_ip').alias('ip_b'),
                pl.col('canon').struct.field('dst_port').alias('port_b'),
                (pl.col('timestamp_ns') / 1_000_000_000).alias('time_s'),
            ]
        )
        .drop('canon')
        .with_columns(
            [
                pl.concat_str(
                    [
                        pl.col('ip_a'),
                        pl.lit(':'),
                        pl.col('port_a').cast(pl.Utf8),
                        pl.lit('->'),
                        pl.col('ip_b'),
                        pl.lit(':'),
                        pl.col('port_b').cast(pl.Utf8),
                    ]
                ).alias('flow_id')
            ]
        )
    )

    out = (
        out.with_columns([pl.min('time_s').over('flow_id').alias('_t0')])
        .with_columns([(pl.col('time_s') - pl.col('_t0')).alias('time_normed')])
        .drop('_t0')
    )
    return out


def _assign_windows(df: pl.DataFrame, prediction_window: float) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col('time_normed') / prediction_window).floor().cast(pl.Int64).alias('win_idx'),
        ]
    )


def _compute_iat_ms(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(['flow_id', 'timestamp_ns'])
    dt_ns = pl.col('timestamp_ns') - pl.col('timestamp_ns').shift(1)
    same_flow = pl.col('flow_id') == pl.col('flow_id').shift(1)
    base_iat_ms = (dt_ns / 1_000_000).cast(pl.Float64)
    first_iat_ms = pl.col('time_normed') * 1000.0
    return df.with_columns(
        [pl.when(same_flow).then(base_iat_ms).otherwise(first_iat_ms).alias('iat_ms')]
    )


def _pad_truncate(seq: list[float], n: int, pad_value: float = 0.0) -> list[float]:
    return seq[:n] + [pad_value] * max(0, n - len(seq))


def _stats_from_list(x: list[float]) -> dict[str, float]:
    if not x:
        return dict(mean=0.0, std=0.0, min=0.0, max=0.0, q1=0.0, q2=0.0, q3=0.0)
    a = np.array(x, dtype=float)
    q1, q2, q3 = np.quantile(a, [0.25, 0.5, 0.75])
    return dict(
        mean=float(a.mean()),
        std=float(a.std(ddof=0)),
        min=float(a.min()),
        max=float(a.max()),
        q1=float(q1),
        q2=float(q2),
        q3=float(q3),
    )


def _num_rtx_from_sizes(size_list: list[int]) -> int:
    d = {}
    for s in size_list:
        d[s] = d.get(s, 0) + 1
    return sum(d.get(x + 2, 0) for x in d)


def _burst_count(iats_ms: list[float], thresh_ms: float = 30.0) -> int:
    if len(iats_ms) <= 1:
        return 0
    a = np.array(iats_ms, dtype=float)
    return int((a >= thresh_ms).sum())


def _has_rtp_cols(df: pl.DataFrame) -> bool:
    return {'rtp.timestamp', 'rtp.seq', 'rtp.p_type', 'rtp.marker'}.issubset(set(df.columns))


def _rtp_features_for_window(
    pdf: pd.DataFrame,
    video_ptypes: list[str],
    rtx_ptypes: list[str | None],
) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        'vid_ts_unique': 0,
        'rtx_ts_unique': 0,
        'common_vid_rtx_ts_unique': 0,
        'union_ts_unique': 0,
        'vid_marker_sum': 0,
        'rtx_marker_sum': 0,
        'ooo_seqno_vid': 0,
        'buffer_time_mean': 0.0,
        'buffer_time_std': 0.0,
        'buffer_time_min': 0.0,
        'buffer_time_max': 0.0,
        'buffer_time_q1': 0.0,
        'buffer_time_q2': 0.0,
        'buffer_time_q3': 0.0,
        'rtp_lag_mean': 0.0,
        'rtp_lag_std': 0.0,
        'rtp_lag_min': 0.0,
        'rtp_lag_max': 0.0,
        'rtp_lag_q1': 0.0,
        'rtp_lag_q2': 0.0,
        'rtp_lag_q3': 0.0,
    }
    if not {'rtp.timestamp', 'rtp.p_type'}.issubset(pdf.columns):
        return out

    df = pdf.copy()
    df['rtp.timestamp'] = pd.to_numeric(df['rtp.timestamp'], errors='coerce')
    df['rtp.seq'] = pd.to_numeric(df.get('rtp.seq'), errors='coerce').astype('Int64')
    df['rtp.p_type'] = df['rtp.p_type'].astype(str)
    df['rtp.marker'] = pd.to_numeric(df.get('rtp.marker'), errors='coerce').fillna(0).astype(int)

    vid_mask = df['rtp.p_type'].isin(video_ptypes)
    rtx_vals = [x for x in rtx_ptypes if x is not None]
    rtx_mask = df['rtp.p_type'].isin(rtx_vals)

    vid_ts = set(df.loc[vid_mask, 'rtp.timestamp'].dropna().to_list())
    rtx_ts = set(df.loc[rtx_mask, 'rtp.timestamp'].dropna().to_list())
    out['vid_ts_unique'] = len(vid_ts)
    out['rtx_ts_unique'] = len(rtx_ts)
    out['common_vid_rtx_ts_unique'] = len(vid_ts.intersection(rtx_ts))
    out['union_ts_unique'] = len(vid_ts.union(rtx_ts))
    out['vid_marker_sum'] = int(df.loc[vid_mask, 'rtp.marker'].sum())
    out['rtx_marker_sum'] = int(df.loc[rtx_mask, 'rtp.marker'].sum())

    # OOO seq for video
    ooo = 0
    df_vid = df.loc[vid_mask].sort_values(['rtp.timestamp', 'rtp.seq'])
    prev_seq: int | None = None
    for _, r in df_vid.iterrows():
        s = r['rtp.seq']
        if pd.notna(s):
            s = int(s)
            if prev_seq is not None and s - prev_seq != 1:
                ooo += 1
            prev_seq = s
    out['ooo_seqno_vid'] = int(ooo)

    # buffer_time = 90 / diff(sorted union ts)
    union_ts = sorted(list(vid_ts.union(rtx_ts)))
    if len(union_ts) >= 2:
        btime = 90.0 / np.diff(np.array(union_ts, dtype=float))
        out.update({f'buffer_time_{k}': v for k, v in _stats_from_list(list(btime)).items()})

    # rtp_lag stats
    try:
        df_nonnull = df.dropna(subset=['rtp.timestamp']).copy()
        if len(df_nonnull) >= 1:
            rtp0 = float(df_nonnull['rtp.timestamp'].min())
            t0 = float(df_nonnull.groupby('rtp.timestamp')['time_normed'].max().min())
            lags = []
            for ts, g in df_nonnull.groupby('rtp.timestamp'):
                actual_dur = float(g['time_normed'].max() - t0)
                expected_dur = (float(ts) - rtp0) / 90000.0
                lags.append(actual_dur - expected_dur)
            out.update({f'rtp_lag_{k}': v for k, v in _stats_from_list(lags).items()})
    except Exception:
        pass

    return out


# ------------------------- main (all VCAs) -------------------------


def extract_features_one_file(
    in_parquet: str | Path,
    out_parquet: str | Path,
    config: dict[str, Any],
    dataset: str,
    feature_subset: list[str] = ('SIZE', 'IAT', 'LSTATS', 'TSTATS', 'RTP'),
) -> None:
    """
    Compute features for all VCAs at once.
      - generic SIZE/IAT/LSTATS/TSTATS columns (not VCA-specific)
      - RTP features computed per VCA with prefixed column names: <vca>_<feature>
    """
    pw = float(config['prediction_window'])

    # Determine sequence lengths: use MAX across all VCAs in config
    n_size = int(max(config['n_features_size'].values()))
    n_iat = int(max(config['n_features_iat'].values()))

    df = pl.read_parquet(in_parquet)
    df = _canon_flow(df)
    df = _compute_iat_ms(df)
    df = _assign_windows(df, pw)
    has_rtp = _has_rtp_cols(df) if 'RTP' in feature_subset else False

    # group packets by (flow_id, win_idx)
    grp = (
        df.sort(['flow_id', 'win_idx', 'timestamp_ns'])
        .group_by(['flow_id', 'win_idx'])
        .agg(
            [
                pl.col('length').alias('sizes_list'),
                pl.col('iat_ms').alias('iats_list'),
                pl.col('time_s').max().alias('window_end_s'),
                pl.col('time_s').min().alias('window_start_s'),
                pl.len().alias('_count'),
                pl.sum('length').alias('_bytes'),
                pl.col('length').n_unique().alias('_n_unique'),
                *(
                    [
                        pl.col('rtp.timestamp'),
                        pl.col('rtp.seq'),
                        pl.col('rtp.p_type'),
                        pl.col('rtp.marker'),
                    ]
                    if has_rtp
                    else []
                ),
            ]
        )
    )
    pdf = grp.to_pandas()

    # Collect VCA list from config keys
    vcas: list[str] = sorted(
        set(config.get('n_features_size', {}).keys())
        | set(config.get('n_features_iat', {}).keys())
        | set(config.get('video_ptype', {}).get(dataset, {}).keys())
        | set(config.get('rtx_ptype', {}).get(dataset, {}).keys())
    )

    rows: list[dict[str, Any]] = []

    # Pre-build RTP ptype maps per VCA (may be absent)
    vca_video_ptypes = {
        v: list(map(str, config.get('video_ptype', {}).get(dataset, {}).get(v, []))) for v in vcas
    }
    vca_rtx_ptypes = {v: config.get('rtx_ptype', {}).get(dataset, {}).get(v, []) for v in vcas}

    for _, r in pdf.iterrows():
        sizes: list[int] = list(r['sizes_list'])
        iats: list[float] = list(r['iats_list'])

        row: dict[str, Any] = dict(
            flow_id=r['flow_id'],
            window_start_s=float(r['window_start_s']),
            window_end_s=float(r['window_end_s']),
        )

        # --- SIZE sequence (generic) ---
        if 'SIZE' in feature_subset:
            for i, v in enumerate(_pad_truncate(sizes, n_size, 0.0), 1):
                row[f'size_{i}'] = float(v)

        # --- IAT sequence (generic) ---
        if 'IAT' in feature_subset:
            for i, v in enumerate(_pad_truncate(iats, n_iat, 0.0), 1):
                row[f'iat_{i}'] = float(v)

        # --- LSTATS (generic) ---
        if 'LSTATS' in feature_subset:
            stats = _stats_from_list(sizes)
            row.update({f'l_{k}': v for k, v in stats.items()})
            row['l_num_pkts'] = int(r['_count'])
            row['l_num_bytes'] = int(r['_bytes'])
            row['l_num_unique'] = int(r['_n_unique'])
            row['l_num_rtx'] = int(_num_rtx_from_sizes(sizes))

        # --- TSTATS (generic) ---
        if 'TSTATS' in feature_subset:
            stats = _stats_from_list(iats)
            row.update({f't_{k}': v for k, v in stats.items()})
            row['t_burst_count'] = int(_burst_count(iats, 30.0))

        # --- RTP per-VCA (prefixed) ---
        if has_rtp and 'RTP' in feature_subset:
            # Build a minimal per-window packet view from the aggregated lists (approx.)
            lens = len(sizes)
            if lens > 0:
                # Time scaffolding across the window (coarse but sufficient for these stats)
                tn = np.linspace(
                    0, row['window_end_s'] - row['window_start_s'], num=lens, endpoint=True
                )
                tt = np.linspace(
                    row['window_start_s'], row['window_end_s'], num=lens, endpoint=True
                )
                data = {'time_normed': tn, 'time': tt, 'length': sizes}
                for c in ['rtp.timestamp', 'rtp.seq', 'rtp.p_type', 'rtp.marker']:
                    if c in pdf.columns:
                        data[c] = r[c]
                win_df = pd.DataFrame(data)

                for v in vcas:
                    feats = _rtp_features_for_window(
                        win_df,
                        vca_video_ptypes.get(v, []),
                        vca_rtx_ptypes.get(v, []),
                    )
                    # prefix columns by VCA
                    row.update({f'{v}_{k}': v2 for k, v2 in feats.items()})

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(['flow_id', 'window_start_s']).reset_index(drop=True)
    out.to_parquet(out_parquet, index=False)


def extract_features_folder(
    in_path: str | Path,
    out_path: str | Path,
    config: dict[str, Any],
    dataset: str,
    feature_subset: list[str] = ('SIZE', 'IAT', 'LSTATS', 'TSTATS', 'RTP'),
) -> None:
    in_path, out_path = Path(in_path), Path(out_path)
    if in_path.is_file():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        extract_features_one_file(in_path, out_path, config, dataset, feature_subset)
        return
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_files = list(in_path.rglob('*.parquet'))
    for p in tqdm(parquet_files, desc='Extracting features', total=len(parquet_files), unit='file'):
        rel = p.relative_to(in_path)
        dst = (out_path / rel).with_suffix('.parquet')
        dst.parent.mkdir(parents=True, exist_ok=True)
        extract_features_one_file(p, dst, config, dataset, feature_subset)
