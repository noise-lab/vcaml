"""Load vcaml experiment results from MLflow.

Replaces the inline load_results() / load_results_for_dataset() functions
that previously read .pkl files from *_intermediates/ directories.

Artifacts are cached under ~/.cache/vcaml/mlflow_artifacts/<run_id>/ so
repeated load_results() calls (e.g. re-running a notebook cell) never
re-download the same file. Runs are immutable once logged, so the cache
is always valid for a given run_id.
"""
import json
import pickle
from pathlib import Path

import mlflow
import pandas as pd

from vcaml.config import data_root, mlflow_tracking_uri as _DEFAULT_TRACKING_URI

_CACHE_ROOT = data_root / '.cache' / 'vcaml' / 'mlflow_artifacts'


def _cached_artifact(run_id: str, artifact_name: str) -> Path:
    """Return local path for an artifact, downloading it only if not cached."""
    local = _CACHE_ROOT / run_id / artifact_name
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_name,
            dst_path=str(local.parent),
        )
    return local

METHOD_TAG = {
    'IP/UDP ML':        'ip-udp-ml',
    'IP/UDP Heuristic': 'ip-udp-heuristic',
    'RTP ML':           'rtp-ml',
    'RTP Heuristic':    'rtp-heuristic',
}


def _find_child_run(client, experiment_id, metric, method_key, feature_tag, cv_index):
    filter_str = (
        f"params.metric = '{metric}' and "
        f"params.estimation_method = '{method_key}' and "
        f"params.feature_subset = '{feature_tag}' and "
        f"params.cv_index = '{cv_index}'"
    )
    runs = client.search_runs(experiment_ids=[experiment_id], filter_string=filter_str)
    return runs[0] if runs else None


def load_results(experiment_name, metrics, methods, vcas,
                 k_folds=5, feature_tag='LSTATS-TSTATS', tracking_uri=_DEFAULT_TRACKING_URI):
    """Load prediction results and feature importances from MLflow.

    Returns (df, f_imp) where:
      - df has columns: Prediction, Ground Truth, ts, csv_file, method, VCA,
        metric, cross_val, deviation, abs_deviation
      - f_imp[metric][cv][vca] = {feature_name: importance}
    """
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f'WARNING: no MLflow experiment named "{experiment_name}"')
        return pd.DataFrame(), {}

    exp_id = experiment.experiment_id
    all_preds = []
    f_imp = {m: {cv: {v: {} for v in vcas} for cv in range(1, k_folds + 1)}
             for m in metrics}

    for metric in metrics:
        for method_display in methods:
            method_key = METHOD_TAG[method_display]
            if metric == 'frameHeight' and 'heuristic' in method_key:
                continue
            for cv in range(1, k_folds + 1):
                run = _find_child_run(
                    client, exp_id, metric, method_key, feature_tag, cv)
                if run is None:
                    continue

                run_id = run.info.run_id
                artifact_paths = {a.path for a in client.list_artifacts(run_id)}

                for vca in vcas:
                    fi_name = f'feature_importances_{vca}.json'
                    if fi_name in artifact_paths:
                        with open(_cached_artifact(run_id, fi_name)) as fd:
                            f_imp[metric][cv][vca] = json.load(fd)

                for vca in vcas:
                    pred_name = f'predictions_{vca}.pkl'
                    if pred_name not in artifact_paths:
                        continue
                    with open(_cached_artifact(run_id, pred_name), 'rb') as fd:
                        df = pickle.load(fd)
                    pred_col = f'{metric}_{method_key}'
                    gt_col = f'{metric}_gt'
                    if pred_col not in df.columns or gt_col not in df.columns:
                        continue
                    df = df[[pred_col, gt_col, 'timestamp', 'file', 'dataset']].copy()
                    df = df.rename(columns={
                        pred_col:    'Prediction',
                        gt_col:      'Ground Truth',
                        'timestamp': 'ts',
                        'file':      'csv_file',
                    })
                    df['method']        = method_display
                    df['VCA']           = vca
                    df['metric']        = metric
                    df['cross_val']     = cv
                    df['deviation']     = df['Prediction'] - df['Ground Truth']
                    df['abs_deviation'] = df['deviation'].abs()
                    all_preds.append(df)

    if not all_preds:
        print(f'WARNING: no results found in MLflow experiment "{experiment_name}"')
        return pd.DataFrame(), f_imp
    return pd.concat(all_preds, axis=0, ignore_index=True), f_imp


def load_results_for_dataset(dataset_name, metrics, methods, vcas,
                              k_folds=1, feature_tag='LSTATS-TSTATS', tracking_uri=_DEFAULT_TRACKING_URI):
    """Load prediction results for a single sensitivity dataset from MLflow.

    Returns df with an additional sens_dataset column.
    """
    df, _ = load_results(
        dataset_name, metrics, methods, vcas,
        k_folds=k_folds, feature_tag=feature_tag, tracking_uri=tracking_uri,
    )
    if not df.empty:
        df['sens_dataset'] = dataset_name
    return df
