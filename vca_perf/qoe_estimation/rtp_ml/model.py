from util.webrtc_reader import WebRTCReader
from util.helper_functions import read_net_file, filter_video_frames_rtp, get_net_stats
import sys
import pandas as pd
import numpy as np
from os.path import dirname, abspath
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

from ml.feature_extraction import FeatureExtractor

class RTPMLModel:
    
    def __init__(self, vca, feature_subset, estimator, data_partition_criteria, config, metric, dataset):
        self.vca = vca
        self.feature_subset = feature_subset
        self.estimator = estimator 
        self.data_partition_criteria = data_partition_criteria
        self.config = config
        self.metric = metric 
        self.feature_importances = {}
        self.feature_matrix = None
        self.target_vals = None
        self.dataset = dataset
        self.net_columns = ['frame.time_relative','frame.time_epoch','ip.src','ip.dst','ip.proto','ip.len','udp.srcport','udp.dstport','udp.length','rtp.ssrc','rtp.timestamp','rtp.seq','rtp.p_type', 'rtp.marker']

    def train(self, list_of_files):

        fs_label = '-'.join(self.feature_subset)
        print(f'\nExtracting features on training set...\nVCA: {self.vca}\nModel: {self.estimator.__class__.__name__}\nFeature Subset: {fs_label}\nData Partition Criteria: {self.data_partition_criteria}\nMetric: {self.metric}\n')

        t1 = time.time()

        train_data = []
        idx = 1
        total = len(list_of_files)
        for file_tuple in list_of_files:
            csv_file = file_tuple[1]
            webrtc_file = file_tuple[2]
            feature_file = csv_file[:-4]+f'_rtp_ml_{self.vca}_{self.dataset}.csv'
            feature_matrix = pd.read_csv(feature_file, index_col=False)
            feature_matrix = feature_matrix[self.config['features_list']['rtp_ml'] + ['et', 'ts', self.metric]]
            """
            csv_file = file_tuple[1]
            webrtc_file = file_tuple[2]
            print(f'Extracting features for file # : {idx} of {total}')
            df_net = read_net_file(csv_file)
            df_net = df_net[(df_net['rtp.p_type'] == self.video_ptype[self.vca]) | ((df_net['rtp.p_type'] == self.rtx_ptype[self.vca]) & (df_net['udp.length'] > 306))]
            df_net = df_net.rename(columns={'udp.length':'length', 'frame.time_epoch': 'time', 'frame.time_relative': 'time_normed'})
            df_net = df_net.sort_values(by=['time_normed'])
            df_net = df_net[['length', 'time', 'time_normed', 'rtp.timestamp', 'rtp.seq', 'rtp.marker', 'rtp.p_type']]
            df_netml = feature_extractor.extract_features(df_net=df_net)
            df_rtp = feature_extractor.extract_rtp_features(df_net=df_net)
            df = pd.merge(df_netml, df_rtp, on='et')
            webrtc_reader = WebRTCReader(webrtc_file)
            df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
            df_merged = pd.merge(df, df_webrtc,
                                 left_on='et', right_on="ts")
            """
            train_data.append(feature_matrix)
            idx += 1

        dur = round(time.time() - t1, 2)
        print(f'\nFeature extraction took {dur} seconds.\n')
        print('\nFitting the model...\n')
        X = pd.concat(train_data, axis=0)
        X = X.dropna()

        y = X[self.metric]
        X = X[X.columns.difference([self.metric, 'et', 'ts'])]
        print(list(X.columns))

        self.feature_matrix = X.copy()
        self.target_vals = y.copy()
        if self.metric == 'framesPerSecond' or self.metric == 'framesRendered' or self.metric == 'framesReceived' or self.metric == 'framesReceivedPerSecond':
            y = y.apply(lambda x: round(x))
        t1 = time.time()
        self.estimator.fit(X, y)
        dur = round(time.time() - t1, 2)
        print(f'\nModel training took {dur} seconds.\n')
        
        if isinstance(self.estimator, RandomForestRegressor) or isinstance(self.estimator, RandomForestClassifier) or isinstance(self.estimator, DecisionTreeRegressor):
            print('\nCalculating feature importance...\n')
            for idx, col in enumerate(X.columns):
                self.feature_importances[col] = self.estimator.feature_importances_[
                    idx]
    def estimate(self, file_tuple):
        """
        pcap_file = file_tuple[0]
        csv_file = file_tuple[1]
        webrtc_file = file_tuple[2]
        feature_extractor = FeatureExtractor(
            feature_subset=self.feature_subset, config=self.config, vca=self.vca)
        print(csv_file)
        df_net = read_net_file(csv_file)
        df_net = df_net[(df_net['rtp.p_type'] == self.video_ptype[self.vca]) | ((df_net['rtp.p_type'] == self.rtx_ptype[self.vca]) & (df_net['udp.length'] > 306))]
        df_net = df_net.rename(columns={'udp.length':'length', 'frame.time_epoch': 'time', 'frame.time_relative': 'time_normed'})
        df_net = df_net.sort_values(by=['time_normed'])
        df_net = df_net[['length', 'time', 'time_normed', 'rtp.timestamp', 'rtp.seq', 'rtp.marker', 'rtp.p_type']]
        df_netml = feature_extractor.extract_features(df_net=df_net)
        df_rtp = feature_extractor.extract_rtp_features(df_net=df_net)
        df = pd.merge(df_netml, df_rtp, on='et')
        webrtc_reader = WebRTCReader(webrtc_file)
        df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
        X = pd.merge(df, df_webrtc, left_on='et', right_on="ts")
        """
        pcap_file = file_tuple[0]
        csv_file = file_tuple[1]
        webrtc_file = file_tuple[2]
        feature_file = csv_file[:-4]+f'_rtp_ml_{self.vca}_{self.dataset}.csv'
        feature_matrix = pd.read_csv(feature_file, index_col=False)
        X = feature_matrix[self.config['features_list']['rtp_ml'] + ['et', 'ts', self.metric]]
        # X = pd.merge(df_rtp, df_netml, left_on='timestamp', right_on="et")
        # df_fps = pd.read_csv(file_tuple[5], index_col=0)
        # df_fps['timestamp'] = df_fps['timestamp'].apply(lambda x: int(x))
        # df_fps = df_fps.rename(columns={'fps': 'framesRendered', 'timestamp': 'rec_ts'})
        # X = pd.merge(df_fps, df_netml, left_on='rec_ts', right_on='et')
        X = X.dropna()
        # timestamps = X['rec_ts']
        timestamps = X['ts']
        # y_test = X[self.metric+'_rtp']
        y_test = X[self.metric]
        # X = X[X.columns.difference([self.metric, 'et', 'rec_ts', 'file'])]
        X = X[X.columns.difference([self.metric, 'et', 'ts', 'file'])]
        if X.shape[0] == 0:
            return None
        y_pred = self.estimator.predict(X)
        if self.metric == 'framesPerSecond' or self.metric == 'framesRendered' or self.metric == 'framesReceived' or self.metric == 'framesReceivedPerSecond':
            y_pred = list(map(lambda x: round(x), y_pred))
            y_test = y_test.apply(lambda x: round(x))
        X[self.metric+'_rtp_ml'] = y_pred
        # X[self.metric+'_rtp'] = y_test
        X[self.metric+'_gt'] = y_test
        X['timestamp'] = timestamps
        X['file'] = pcap_file
        X['dataset'] = self.dataset
        return X[[self.metric+'_rtp_ml', self.metric+'_gt', 'timestamp', 'file', 'dataset']]        