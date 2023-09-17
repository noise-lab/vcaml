from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import time
import sys
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)
from features.feature_extraction import FeatureExtractor
from util.webrtc_reader import WebRTCReader

class IP_UDP_ML:

    def __init__(self, vca, feature_subset, estimator, config, metric, dataset):
        self.vca = vca
        self.feature_subset = feature_subset
        self.estimator = estimator
        self.config = config
        self.metric = metric
        self.feature_importances = {}
        self.feature_matrix = None
        self.target_vals = None
        self.dataset = dataset
        self.net_columns = ['frame.time_relative', 'frame.time_epoch', 'ip.src', 'ip.dst', 'ip.proto', 'ip.len',
                            'udp.srcport', 'udp.dstport', 'udp.length', 'rtp.ssrc', 'rtp.timestamp', 'rtp.seq', 'rtp.p_type', 'rtp.marker']

    def train(self, list_of_files):

        fs_label = '-'.join(self.feature_subset)
        feature_extractor = FeatureExtractor(self.feature_subset, self.config, self.vca, self.dataset)
        print(
            f'\nExtracting features on training set...\nVCA: {self.vca}\nModel: {self.estimator.__class__.__name__}\nFeature Subset: {fs_label}\nMetric: {self.metric}\n')

        t1 = time.time()

        train_data = []
        idx = 1
        total = len(list_of_files)
        for file_tuple in list_of_files:
            csv_file = file_tuple[0]
            webrtc_file = file_tuple[1]
            print(f'Extracting features for file # {idx} of {total}')
            df_net = pd.read_csv(csv_file)
            if df_net['ip.proto'].dtype == object:
                df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
            df_net = df_net[~df_net['ip.proto'].isna()]
            df_net['ip.proto'] = df_net['ip.proto'].astype(int)
            df_net = df_net[(df_net['ip.proto'] == 17)]
            try:
                dst = df_net.groupby('ip.dst').agg({'udp.length': sum, 'rtp.p_type': 'count'}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
                df_net = df_net[df_net['ip.dst'] == dst]
            except IndexError:
                print('Faulty trace. Continuing..')
                idx += 1
                continue
            df_net = df_net[df_net['udp.length'] > 306]
            df_net = df_net.rename(columns={
                                   'udp.length': 'length', 'frame.time_epoch': 'time', 'frame.time_relative': 'time_normed'})
            df_net = df_net.sort_values(by=['time_normed'])
            df_net = df_net[['length', 'time', 'time_normed']]
            df_netml = feature_extractor.extract_features(df_net=df_net)
            webrtc_reader = WebRTCReader(webrtc_file, self.dataset)
            df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
            df_merged = pd.merge(df_netml, df_webrtc,
                                 left_on='et', right_on="ts")
            train_data.append(df_merged)
            idx += 1

        dur = round(time.time() - t1, 2)
        print(f'\nFeature extraction took {dur} seconds.\n')
        print('\nFitting the model...\n')
        X = pd.concat(train_data, axis=0)
        X = X.dropna()
        print(X.shape)
        y = X[self.metric]
        X = X[X.columns.difference([self.metric, 'et', 'ts', 'file'])]
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
        csv_file = file_tuple[0]
        webrtc_file = file_tuple[1]
        feature_extractor = FeatureExtractor(
            feature_subset=self.feature_subset, config=self.config, vca=self.vca, dataset=self.dataset)
        print(csv_file)
        df_net = pd.read_csv(csv_file)
        if df_net['ip.proto'].dtype == object:
            df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
        df_net = df_net[~df_net['ip.proto'].isna()]
        df_net['ip.proto'] = df_net['ip.proto'].astype(int)
        df_net = df_net[(df_net['ip.proto'] == 17)]
        try:
            dst = df_net.groupby('ip.dst').agg({'udp.length': sum, 'rtp.p_type': 'count'}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
            df_net = df_net[df_net['ip.dst'] == dst]
        except IndexError:
            print('Faulty trace. Continuing..')
            return None
        df_net = df_net[df_net['udp.length'] > 306]
        df_net = df_net.rename(columns={
                               'udp.length': 'length', 'frame.time_epoch': 'time', 'frame.time_relative': 'time_normed'})
        df_net = df_net.sort_values(by=['time_normed'])
        df_net = df_net[['length', 'time', 'time_normed']]
        df_netml = feature_extractor.extract_features(df_net=df_net)
        webrtc_reader = WebRTCReader(webrtc_file, self.dataset)
        df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
        X = pd.merge(df_netml, df_webrtc, left_on='et', right_on="ts")
        X = X.dropna()
        timestamps = X['ts']
        y_test = X[self.metric]
        X = X[X.columns.difference([self.metric, 'et', 'ts', 'file'])]
        if X.shape[0] == 0:
            return None
        y_pred = self.estimator.predict(X)
        if self.metric == 'framesPerSecond' or self.metric == 'framesRendered' or self.metric == 'framesReceived' or self.metric == 'framesReceivedPerSecond':
            y_pred = list(map(lambda x: round(x), y_pred))
            y_test = y_test.apply(lambda x: round(x))
        X[self.metric+'_ip-udp-ml'] = y_pred
        X[self.metric+'_gt'] = y_test
        X['timestamp'] = timestamps
        X['file'] = csv_file
        X['dataset'] = self.dataset
        return X[[self.metric+'_ip-udp-ml', self.metric+'_gt', 'timestamp', 'file', 'dataset']]
