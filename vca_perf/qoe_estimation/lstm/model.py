from util.webrtc_reader import WebRTCReader
import time
import numpy as np
import sys
from os.path import dirname, abspath
from .feature_extraction import FeatureExtractor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
d = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(d)
from util.helper_functions import filter_ptype
from qoe_estimation.rtp.model import RTPModel
from .trainer import SequenceDataset, ShallowRegressionLSTM, LSTMTrainer
import torch
from torch import nn
from torch.utils.data import DataLoader

class LSTM_Model:

    def __init__(self, vca, feature_subset, estimator, data_partition_criteria, config, metric):
        self.vca = vca
        self.feature_subset = feature_subset
        self.estimator = estimator 
        self.data_partition_criteria = data_partition_criteria
        self.config = config
        self.metric = metric 
        self.feature_importances = {}
        self.feature_matrix = None
        self.target_vals = None
        self.net_columns = ['frame.time_relative','frame.time_epoch','ip.src','ip.dst','ip.proto','ip.len','udp.srcport','udp.dstport', 'udp.length','rtp.ssrc','rtp.timestamp','rtp.seq','rtp.p_type', 'rtp.marker']

    def train(self, list_of_files):

        fs_label = '-'.join(self.feature_subset)
        print(
            f'\nExtracting features on training set...\nVCA: {self.vca}\nModel: {self.estimator.__class__.__name__}\nFeature Subset: {fs_label}\nData Partition Criteria: {self.data_partition_criteria}\nMetric: {self.metric}\n')

        feature_extractor = FeatureExtractor(
            feature_subset=self.feature_subset, config=self.config, vca=self.vca)

        t1 = time.time()

        train_X = []
        train_Y = []
        timestamps = []
        idx = 1
        total = len(list_of_files)
        for file_tuple in list_of_files:
            csv_file = file_tuple[1]
            webrtc_file = file_tuple[2]
            print(f'Extracting features for file # : {idx} of {total}')
            df_net = pd.read_csv(csv_file, header=None, sep='\t', names=self.net_columns, lineterminator='\n', encoding='ascii')
            if df_net['ip.proto'].dtype == object:
                df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
            df_net = df_net[~df_net['ip.proto'].isna()]
            df_net['ip.proto'] = df_net['ip.proto'].astype(int)
            df_net = df_net[(df_net['ip.proto'] == 17) & (df_net['ip.dst'] == self.config['destination_ip'])]
            df_net['rtp.p_type'] = df_net['rtp.p_type'].apply(filter_ptype)
            df_net = df_net[df_net['udp.length'] > 306]
            # print(df_net['rtp.p_type'].unique())
            # print(df_net[df_net['rtp.p_type'].isna()])
            df_net = df_net.rename(columns={'udp.length':'length', 'frame.time_epoch': 'time', 'frame.time_relative': 'time_normed'})
            df_net = df_net.sort_values(by=['time_normed'])
            df_net = df_net[['length', 'time', 'time_normed']]
            df_netml = feature_extractor.extract_features(df_net=df_net)
            # df_rtp = rtp_model.estimate(file_tuple)[[self.metric+'_rtp', 'timestamp']]
            # webrtc_reader = WebRTCReader(webrtc_file)
            # df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
            # df_merged = pd.merge(df_webrtc, df_netml, left_on='ts', right_on="et")
            # df_merged = pd.merge(df_rtp, df_netml, left_on='timestamp', right_on="et")
            df_fps = pd.read_csv(file_tuple[5], index_col=0)
            df_fps['timestamp'] = df_fps['timestamp'].apply(lambda x: int(x))
            df_fps = df_fps.rename(columns={'fps': 'framesRendered', 'timestamp': 'rec_ts'})
            df_merged = pd.merge(df_fps, df_netml, left_on='rec_ts', right_on='et')
            timestamps.append(df_merged['rec_ts'].to_numpy()) 
            cols = [f'f_{i}' for i in range(1, (self.config['n_features_bps'][self.vca])+1)]
            df = df_merged[cols]
            isz = df.to_numpy()
            isz = isz.reshape((isz.shape[0], isz.shape[1] // 2, 2))
            train_X.append(isz)
            train_Y.append(df_merged[self.metric].to_numpy())
            idx += 1
        dur = round(time.time() - t1, 2)
        print(f'\nFeature extraction took {dur} seconds.\n')
        print('\nGenerating input for LSTM...\n')
        features = np.concatenate(train_X, axis=0)
        targets = np.concatenate(train_Y, axis=0)
        # y = X[self.metric+'_rtp']
        t1 = time.time()
        train_stats = []
        sz = features[:, :, 0]
        iat = features[:, :, 1]
        sz = sz[sz >= 0]
        iat = iat[iat >= 0]
        train_stats.append(sz.mean())
        train_stats.append(sz.std())
        train_stats.append(iat.mean())
        train_stats.append(iat.std())
        mask1 = features[:, :, 0] >= 0
        mask2 = features[:, :, 1] >= 0 
        features[:, :, 0][mask1] = (features[:, :, 0][mask1] - train_stats[0]) / train_stats[1]
        features[:, :, 1][mask2] = (features[:, :, 1][mask2] - train_stats[2]) / train_stats[3]
        self.train_stats = train_stats
        # X = X[X.columns.difference([self.metric, 'et', 'ts', 'file'])]    
        self.feature_matrix = features.copy()
        self.target_vals = targets.copy()

        train_dataset = SequenceDataset(features, targets, 2, 300)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

        learning_rate = 0.01
        num_hidden_units = 100
        model = ShallowRegressionLSTM(num_sensors = 2, hidden_units=num_hidden_units)

        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        trainer = LSTMTrainer()

        for ix_epoch in range(10):
            print(f"Epoch {ix_epoch}\n---------")
            trainer.train_model(train_loader, model, loss_function, optimizer)
        
        self.estimator = model

        dur = round(time.time() - t1, 2)
        print(f'\nModel training took {dur} seconds.\n')
        
    def estimate(self, file_tuple):
        # rtp_model = RTPModel(self.vca, self.metric)
        pcap_file = file_tuple[0]
        csv_file = file_tuple[1]
        webrtc_file = file_tuple[2]
        feature_extractor = FeatureExtractor(
            feature_subset=self.feature_subset, config=self.config, vca=self.vca)
        print(csv_file)
        df_net = pd.read_csv(csv_file, header=None, sep='\t', names=self.net_columns, lineterminator='\n', encoding='ascii')
        if df_net['ip.proto'].dtype == object:
            df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
        df_net = df_net[~df_net['ip.proto'].isna()]
        df_net['ip.proto'] = df_net['ip.proto'].astype(int)
        df_net = df_net[(df_net['ip.proto'] == 17) & (
            df_net['ip.dst'] == self.config['destination_ip'])]
        df_net = df_net[df_net['udp.length'] > 306]
        df_net = df_net.rename(columns={'udp.length':'length', 'frame.time_epoch': 'time', 'frame.time_relative': 'time_normed'})
        df_net = df_net.sort_values(by=['time_normed'])
        df_net = df_net[['length', 'time', 'time_normed']]
        df_netml = feature_extractor.extract_features(df_net=df_net)
                # df_rtp = rtp_model.estimate(file_tuple)[[self.metric+'_rtp', 'timestamp']]
        # webrtc_reader = WebRTCReader(webrtc_file)
        # df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
        # X = pd.merge(df_webrtc, df_netml, left_on='ts', right_on="et")
        # X = pd.merge(df_rtp, df_netml, left_on='timestamp', right_on="et")
        df_fps = pd.read_csv(file_tuple[5], index_col=0)
        df_fps['timestamp'] = df_fps['timestamp'].apply(lambda x: int(x))
        df_fps = df_fps.rename(columns={'fps': 'framesRendered', 'timestamp': 'rec_ts'})
        df_merged = pd.merge(df_fps, df_netml, left_on='rec_ts', right_on='et')
        cols = [f'f_{i}' for i in range(1, (self.config['n_features_bps'][self.vca])+1)]
        df = df_merged[cols]
        features = df.to_numpy()
        features = features.reshape((features.shape[0], features.shape[1] // 2, 2))
        targets = df_merged[self.metric]
        gt = df_merged[self.metric]
        timestamps = df_merged['et']
        
        mask1 = features[:, :, 0] >= 0
        mask2 = features[:, :, 1] >= 0
        features[:, :, 0][mask1] = (features[:, :, 0][mask1] - self.train_stats[0]) / self.train_stats[1]
        features[:, :, 1][mask2] = (features[:, :, 1][mask2] - self.train_stats[2]) / self.train_stats[3]

        test_dataset = SequenceDataset(features, targets, 2, 300)
            
        trainer = LSTMTrainer()
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        # X = X[X.columns.difference([self.metric+'_rtp', 'et', 'timestamp', 'ts', 'file'])]
        y_pred = trainer.predict(test_loader, self.estimator).numpy()
        if self.metric == 'framesPerSecond' or self.metric == 'framesRendered' or self.metric == 'framesReceived':
            y_pred = list(map(lambda x: round(x), y_pred))
        df_merged[self.metric+'_lstm'] = y_pred
        # X[self.metric+'_rtp'] = y_test
        df_merged[self.metric+'_gt'] = gt
        df_merged['timestamp'] = timestamps
        df_merged['file'] = pcap_file
        return df_merged[[self.metric+'_lstm', self.metric+'_gt', 'timestamp', 'file']]
