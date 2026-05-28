import logging
import sys
import time
import warnings
from os.path import abspath, dirname

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)

d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

from features.feature_extraction import FeatureExtractor
from util.webrtc_reader import WebRTCReader

logger = logging.getLogger(__name__)

_FPS_METRICS = frozenset({
    'framesPerSecond', 'framesRendered', 'framesReceived', 'framesReceivedPerSecond',
})


class IP_UDP_ML:

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
            df_net = pd.read_csv(csvFile)
            if df_net['ip.proto'].dtype == object:
                df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
            df_net = df_net[~df_net['ip.proto'].isna()]
            df_net['ip.proto'] = df_net['ip.proto'].astype(int)
            df_net = df_net[df_net['ip.proto'] == 17]
            try:
                dstIp = (df_net.groupby('ip.dst')
                         .agg({'udp.length': 'sum', 'rtp.p_type': 'count'})
                         .reset_index()
                         .sort_values(by='udp.length', ascending=False)
                         .head(1)['ip.dst'].iloc[0])
                df_net = df_net[df_net['ip.dst'] == dstIp]
            except IndexError:
                logger.warning('Faulty trace, skipping: %s', csvFile)
                continue
            df_net = df_net[df_net['udp.length'] > 306]
            df_net = df_net.rename(columns={
                'udp.length': 'length',
                'frame.time_epoch': 'time',
                'frame.time_relative': 'time_normed',
            })
            df_net = df_net.sort_values('time_normed')[['length', 'time', 'time_normed']]
            df_features = featureExtractor.extract_features(df_net=df_net)
            webrtcReader = WebRTCReader(webrtcFile, self.dataset)
            df_webrtc = webrtcReader.get_webrtc()[[self.metric, 'ts']]
            df_merged = pd.merge(df_features, df_webrtc, left_on='et', right_on='ts')
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
        df_net = pd.read_csv(csvFile)
        if df_net['ip.proto'].dtype == object:
            df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
        df_net = df_net[~df_net['ip.proto'].isna()]
        df_net['ip.proto'] = df_net['ip.proto'].astype(int)
        df_net = df_net[df_net['ip.proto'] == 17]
        try:
            dstIp = (df_net.groupby('ip.dst')
                     .agg({'udp.length': 'sum', 'rtp.p_type': 'count'})
                     .reset_index()
                     .sort_values(by='udp.length', ascending=False)
                     .head(1)['ip.dst'].iloc[0])
            df_net = df_net[df_net['ip.dst'] == dstIp]
        except IndexError:
            logger.warning('Faulty trace, skipping: %s', csvFile)
            return None
        df_net = df_net[df_net['udp.length'] > 306]
        df_net = df_net.rename(columns={
            'udp.length': 'length',
            'frame.time_epoch': 'time',
            'frame.time_relative': 'time_normed',
        })
        df_net = df_net.sort_values('time_normed')[['length', 'time', 'time_normed']]
        df_features = featureExtractor.extract_features(df_net=df_net)
        webrtcReader = WebRTCReader(webrtcFile, self.dataset)
        df_webrtc = webrtcReader.get_webrtc()[[self.metric, 'ts']]
        X = pd.merge(df_features, df_webrtc, left_on='et', right_on='ts').dropna()
        timestamps = X['ts']
        yTest = X[self.metric]
        X = X[X.columns.difference([self.metric, 'et', 'ts', 'file'])]
        if X.shape[0] == 0:
            return None
        yPred = self.estimator.predict(X)
        if self.metric in _FPS_METRICS:
            yPred = list(map(round, yPred))
            yTest = yTest.apply(round)
        X[self.metric + '_ip-udp-ml'] = yPred
        X[self.metric + '_gt'] = yTest
        X['timestamp'] = timestamps
        X['file'] = csvFile
        X['dataset'] = self.dataset
        return X[[self.metric + '_ip-udp-ml', self.metric + '_gt',
                  'timestamp', 'file', 'dataset']]
