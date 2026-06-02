import logging
import time
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from vcaml.features.feature_extraction import FeatureExtractor
from vcaml.util.helper_functions import _FPS_METRICS, mergeWithWebrtc

logger = logging.getLogger(__name__)


class BaseMLEstimator(ABC):

    def __init__(self, vca, featureSubset, estimator, config, metric, dataset):
        self.vca = vca
        self.featureSubset = featureSubset
        self.estimator = estimator
        self.config = config
        self.metric = metric
        self.featureImportances = {}
        self.featureMatrix = None
        self.targetValues = None
        self.dataset = dataset

    @abstractmethod
    def _loadAndPrepareFile(self, csvFile):
        """Load, filter, rename, and select columns. Returns DataFrame or None."""

    @abstractmethod
    def _extractFeatures(self, df_net, featureExtractor):
        """Extract features from prepared df_net. Returns feature DataFrame."""

    @property
    @abstractmethod
    def modelTag(self):
        """String suffix used in output column names (e.g. 'ip-udp-ml')."""

    def train(self, fileList):
        featureExtractor = FeatureExtractor(self.featureSubset, self.config, self.vca, self.dataset)
        fsLabel = '-'.join(self.featureSubset)
        logger.info('Extracting features  vca=%s  model=%s  features=%s  metric=%s',
                    self.vca, self.estimator.__class__.__name__, fsLabel, self.metric)
        t0 = time.time()
        trainData = []
        totalFiles = len(fileList)
        for fileIdx, fileTuple in enumerate(fileList, 1):
            csvFile, webrtcFile = fileTuple[0], fileTuple[1]
            logger.debug('Extracting features for file %d/%d: %s', fileIdx, totalFiles, csvFile)
            df_net = self._loadAndPrepareFile(csvFile)
            if df_net is None:
                logger.warning('Faulty trace, skipping: %s', csvFile)
                continue
            df_features = self._extractFeatures(df_net, featureExtractor)
            df_merged = mergeWithWebrtc(df_features, webrtcFile, self.dataset, self.metric)
            trainData.append(df_merged)

        logger.info('Feature extraction took %.2fs', time.time() - t0)
        logger.info('Fitting model...')
        X = pd.concat(trainData, axis=0).dropna()
        logger.debug('Training matrix shape: %s', X.shape)
        y = X[self.metric]
        X = X[X.columns.difference([self.metric, 'et', 'ts', 'file'])]
        self.featureMatrix = X.copy()
        self.targetValues = y.copy()
        if self.metric in _FPS_METRICS:
            y = y.apply(round)
        t0 = time.time()
        self.estimator.fit(X, y)
        logger.info('Model training took %.2fs', time.time() - t0)
        if isinstance(self.estimator, (RandomForestRegressor, RandomForestClassifier,
                                       DecisionTreeRegressor)):
            for i, col in enumerate(X.columns):
                self.featureImportances[col] = self.estimator.feature_importances_[i]

    def estimate(self, fileTuple):
        csvFile, webrtcFile = fileTuple[0], fileTuple[1]
        logger.debug('Estimating: %s', csvFile)
        featureExtractor = FeatureExtractor(
            featureSubset=self.featureSubset, config=self.config,
            vca=self.vca, dataset=self.dataset)
        df_net = self._loadAndPrepareFile(csvFile)
        if df_net is None:
            logger.warning('Faulty trace, skipping: %s', csvFile)
            return None
        df_features = self._extractFeatures(df_net, featureExtractor)
        X = mergeWithWebrtc(df_features, webrtcFile, self.dataset, self.metric).dropna()
        timestamps = X['ts']
        yTest = X[self.metric]
        X = X[X.columns.difference([self.metric, 'et', 'ts', 'file'])]
        if X.shape[0] == 0:
            return None
        yPred = self.estimator.predict(X)
        if self.metric in _FPS_METRICS:
            yPred = list(map(round, yPred))
            yTest = yTest.apply(round)
        tag = self.modelTag
        X[self.metric + '_' + tag] = yPred
        X[self.metric + '_gt'] = yTest
        X['timestamp'] = timestamps
        X['file'] = csvFile
        X['dataset'] = self.dataset
        return X[[self.metric + '_' + tag, self.metric + '_gt',
                  'timestamp', 'file', 'dataset']]
