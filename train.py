#!/usr/bin/env python
"""Entry point for vcaml model training and evaluation.

Run from the project root:
    uv run python train.py                                  # in-lab dataset
    uv run python train.py --dataset data/real_world_data
    uv run python train.py --metrics framesReceivedPerSecond bitrate
    uv run python train.py --methods ip-udp-ml rtp-ml
"""
import argparse
import json
import logging
import os
import pickle
import sys
import tempfile
from itertools import product
from pathlib import Path

from vcaml.logging_setup import setup_logging
setup_logging()

import mlflow
import pandas as pd
import yaml

from vcaml.config import data_root, mlflow_tracking_uri, project_config
from vcaml.models.run_model import ModelRunner
from vcaml.pipeline.data_splitter import KfoldCVOverFiles
from vcaml.pipeline.file_processor import FileProcessor

logger = logging.getLogger('vcaml')


def _parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate vcaml QoE models')
    parser.add_argument(
        '--dataset', default='data/in_lab_data',
        help='Path to dataset directory (default: data/in_lab_data)')
    parser.add_argument(
        '--metrics', nargs='+', default=None,
        help='Metrics to predict — overrides config.yaml training.metrics')
    parser.add_argument(
        '--methods', nargs='+', default=None,
        help='Estimation methods — overrides config.yaml training.estimation_methods')
    return parser.parse_args()


def _load_training_config(args):
    """Resolve CLI args against config.yaml and return the merged training config."""
    dataDir = str(Path(args.dataset).resolve())
    datasetName = os.path.basename(dataDir)

    with open(Path(__file__).parent / 'config.yaml') as f:
        trainingCfg = yaml.safe_load(f).get('training', {})

    metrics = args.metrics or trainingCfg.get('metrics', ['framesReceivedPerSecond'])
    estimationMethods = args.methods or trainingCfg.get('estimation_methods', ['ip-udp-ml'])
    featureSubsets = trainingCfg.get('feature_subsets', [['LSTATS', 'TSTATS']])
    kFolds = trainingCfg.get('k_folds', 5)
    return dataDir, datasetName, metrics, estimationMethods, featureSubsets, kFolds


def _build_file_splits(dataDir, kFolds):
    """Discover linked CSV/JSON file pairs and compute k-fold CV splits."""
    datasetName = os.path.basename(dataDir)
    fileProcessor = FileProcessor(dataDirectory=dataDir)
    fileDict = fileProcessor.get_linked_files()
    if not fileDict:
        logger.error('No files found in %s — check directory structure', dataDir)
        sys.exit(1)
    return KfoldCVOverFiles(kFolds, fileDict, project_config, datasetName).split()


def _log_fold_artifacts(vcaModels, predictions):
    """Stage models, feature importances, and predictions as MLflow artifacts.

    Must be called inside an active MLflow run context.
    """
    with tempfile.TemporaryDirectory() as tmpDir:
        tmpPath = Path(tmpDir)

        for vca, model in vcaModels.items():
            modelPath = tmpPath / f'model_{vca}.pkl'
            with open(modelPath, 'wb') as fd:
                pickle.dump(model, fd)
            mlflow.log_artifact(str(modelPath))

            fi = getattr(model, 'featureImportances', {})
            if fi:
                fiPath = tmpPath / f'feature_importances_{vca}.json'
                with open(fiPath, 'w') as fd:
                    json.dump(fi, fd)
                mlflow.log_artifact(str(fiPath))

        for vca, predsList in predictions.items():
            validPreds = [p for p in predsList if p is not None]
            if validPreds:
                predPath = tmpPath / f'predictions_{vca}.pkl'
                with open(predPath, 'wb') as fd:
                    pickle.dump(pd.concat(validPreds, axis=0), fd)
                mlflow.log_artifact(str(predPath))


def _run_fold(metric, estimationMethod, featureSubset, dataDir, cvIdx, foldSplit):
    """Train and evaluate one CV fold, logging metrics and artifacts to the active MLflow run.

    Must be called inside an active nested MLflow run context.
    Returns (foldMaes, foldAccs) dicts keyed by VCA for parent-run aggregation.
    """
    featureTag = '-'.join(featureSubset) if featureSubset else 'none'
    datasetName = os.path.basename(dataDir)

    mlflow.log_params({
        'metric': metric,
        'estimation_method': estimationMethod,
        'feature_subset': featureTag,
        'dataset': datasetName,
        'cv_index': cvIdx,
    })
    for vca, split in foldSplit.items():
        mlflow.log_param(f'n_train_{vca}', len(split['train']))
        mlflow.log_param(f'n_test_{vca}', len(split['test']))

    runner = ModelRunner(metric, estimationMethod, featureSubset, dataDir, cvIdx)
    vcaModels = runner.trainModel(foldSplit)
    predictions, maes, accuracies = runner.getTestSetPredictions(foldSplit, vcaModels)

    foldMaes, foldAccs = {}, {}
    for vca in foldSplit:
        if maes.get(vca):
            m = sum(maes[vca]) / len(maes[vca])
            mlflow.log_metric(f'mae_{vca}', round(m, 4))
            foldMaes[vca] = m
        if accuracies.get(vca):
            a = 100 * sum(accuracies[vca]) / len(accuracies[vca])
            mlflow.log_metric(f'accuracy_{vca}', round(a, 2))
            foldAccs[vca] = a

    _log_fold_artifacts(vcaModels, predictions)
    return foldMaes, foldAccs


def main():
    args = _parse_args()
    dataDir, datasetName, metrics, estimationMethods, featureSubsets, kFolds = \
        _load_training_config(args)

    logger.info('Dataset:  %s', dataDir)
    logger.info('Metrics:  %s', metrics)
    logger.info('Methods:  %s', estimationMethods)
    logger.info('Features: %s', featureSubsets)
    logger.info('K-folds:  %d', kFolds)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(datasetName) is None:
        client.create_experiment(
            datasetName,
            artifact_location=str(data_root / 'mlartifacts' / datasetName),
        )
    mlflow.set_experiment(datasetName)

    fileSplits = _build_file_splits(dataDir, kFolds)

    for metric, estimationMethod, featureSubset in product(
            metrics, estimationMethods, featureSubsets):
        if metric == 'frameHeight' and 'heuristic' in estimationMethod:
            logger.debug('Skipping frameHeight with heuristic method')
            continue

        featureTag = '-'.join(featureSubset) if featureSubset else 'none'

        with mlflow.start_run(run_name=f'{metric}_{estimationMethod}_{featureTag}'):
            mlflow.log_params({
                'metric': metric,
                'estimation_method': estimationMethod,
                'feature_subset': featureTag,
                'dataset': datasetName,
                'k_folds': kFolds,
            })

            allFoldMaes, allFoldAccs = {}, {}
            for cvIdx, foldSplit in enumerate(fileSplits, 1):
                with mlflow.start_run(run_name=f'cv_{cvIdx}', nested=True):
                    foldMaes, foldAccs = _run_fold(
                        metric, estimationMethod, featureSubset, dataDir, cvIdx, foldSplit)
                for vca, mae in foldMaes.items():
                    allFoldMaes.setdefault(vca, []).append(mae)
                for vca, acc in foldAccs.items():
                    allFoldAccs.setdefault(vca, []).append(acc)

            for vca, foldMaes in allFoldMaes.items():
                mlflow.log_metric(f'mean_mae_{vca}',
                                  round(sum(foldMaes) / len(foldMaes), 4))
            for vca, foldAccs in allFoldAccs.items():
                mlflow.log_metric(f'mean_accuracy_{vca}',
                                  round(sum(foldAccs) / len(foldAccs), 2))


if __name__ == '__main__':
    main()
