#!/usr/bin/env python
"""Entry point for vcaml model training and evaluation.

Run from the project root:
    uv run python train.py                                  # in-lab dataset
    uv run python train.py --dataset data/real_world_data
    uv run python train.py --metrics framesReceivedPerSecond bitrate
    uv run python train.py --methods ip-udp-ml rtp-ml
"""
import argparse
import logging
import os
import pickle
import sys
from itertools import product
from pathlib import Path

# Make src/ and src/models/ importable from the project root
_srcDir = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(_srcDir))
sys.path.insert(0, str(_srcDir / 'models'))

from logging_setup import setup_logging
setup_logging()

import pandas as pd
import yaml

from config import project_config
from models.run_model import ModelRunner
from util.data_splitter import KfoldCVOverFiles
from util.file_processor import FileProcessor

logger = logging.getLogger('vcaml')


def main():
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
    args = parser.parse_args()

    dataDir = str(Path(args.dataset).resolve())

    with open(Path(__file__).parent / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    trainingCfg = cfg.get('training', {})

    metrics = args.metrics or trainingCfg.get('metrics', ['framesReceivedPerSecond'])
    estimationMethods = args.methods or trainingCfg.get('estimation_methods', ['ip-udp-ml'])
    featureSubsets = trainingCfg.get('feature_subsets', [['LSTATS', 'TSTATS']])
    kFolds = trainingCfg.get('k_folds', 5)

    intermediatesDir = Path(f'{dataDir}_intermediates')
    intermediatesDir.mkdir(exist_ok=True, parents=True)

    datasetName = os.path.basename(dataDir)
    logger.info('Dataset:  %s', dataDir)
    logger.info('Metrics:  %s', metrics)
    logger.info('Methods:  %s', estimationMethods)
    logger.info('Features: %s', featureSubsets)
    logger.info('K-folds:  %d', kFolds)

    fileProcessor = FileProcessor(dataDirectory=dataDir)
    fileDict = fileProcessor.get_linked_files()
    if not fileDict:
        logger.error('No files found in %s — check directory structure', dataDir)
        sys.exit(1)

    kcv = KfoldCVOverFiles(kFolds, fileDict, project_config, datasetName)
    fileSplits = kcv.split()

    with open(intermediatesDir / 'cv_splits.pkl', 'wb') as fd:
        pickle.dump(fileSplits, fd)

    for metric, estimationMethod, featureSubset in product(
            metrics, estimationMethods, featureSubsets):
        if metric == 'frameHeight' and 'heuristic' in estimationMethod:
            logger.debug('Skipping frameHeight with heuristic method')
            continue
        for cvIdx, foldSplit in enumerate(fileSplits, 1):
            runner = ModelRunner(metric, estimationMethod, featureSubset, dataDir, cvIdx)
            trialDir = intermediatesDir / runner.trialId
            trialDir.mkdir(exist_ok=True, parents=True)

            vcaModels = runner.trainModel(foldSplit)
            predictions = runner.getTestSetPredictions(foldSplit, vcaModels)

            with open(trialDir / 'model.pkl', 'wb') as fd:
                pickle.dump(vcaModels, fd)
            for vca, predsList in predictions.items():
                validPreds = [p for p in predsList if p is not None]
                if validPreds:
                    df_preds = pd.concat(validPreds, axis=0)
                    with open(trialDir / f'predictions_{vca}.pkl', 'wb') as fd:
                        pickle.dump(df_preds, fd)


if __name__ == '__main__':
    main()
