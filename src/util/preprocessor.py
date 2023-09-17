import sys
import os
from util.file_processor import FileProcessor
from util.webrtc_reader import WebRTCReader
from util.helper_functions import *
from config import project_config
from qoe_estimation.ml.feature_extraction import FeatureExtractor
from os.path import dirname, abspath, basename
project_root = dirname(dirname(abspath(__file__)))
sys.path.append(project_root)


class Preprocessor:

    def __init__(self, vca, feature_subset, dataset):
        self.vca = vca
        self.feature_subset = feature_subset
        self.config = project_config
        self.dataset = dataset
        self.net_columns = ['frame.time_relative', 'frame.time_epoch', 'ip.src', 'ip.dst', 'ip.proto', 'ip.len',
                            'udp.srcport', 'udp.dstport', 'udp.length', 'rtp.ssrc', 'rtp.timestamp', 'rtp.seq', 'rtp.p_type', 'rtp.marker']

    def process_input(self, file_tuples):
        n = len(file_tuples)
        idx = 1
        feature_extractor = FeatureExtractor(
            self.feature_subset, self.config, self.vca, self.dataset)
        for ftuple in file_tuples:
            pcap_file = ftuple[0]
            csv_file = ftuple[1]
            # if os.path.exists(csv_file[:-4]+f'_rtp_ml_{self.vca}_{self.dataset}.csv'):
            #     print('Already exists')
            #     continue
            webrtc_file = ftuple[2]
            print(f'Extracting features for {idx} of {n}...')
            df_net = pd.read_csv(csv_file)
            if df_net is None or len(df_net) == 0:
                idx += 1
                continue
            df_net = df_net[~df_net['ip.proto'].isna()]
            df_net['ip.proto'] = df_net['ip.proto'].astype(str)
            df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
            df_net['ip.proto'] = df_net['ip.proto'].apply(
                lambda x: int(float(x)))
            ip_addr = df_net.groupby('ip.dst').agg({'udp.length': sum}).reset_index(
            ).sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
            print(ip_addr)
            df_net = df_net[(df_net['ip.proto'] == 17) &
                            (df_net['ip.dst'] == ip_addr)]
            df_net = df_net[df_net['udp.length'] > 306]
            df_net = df_net.rename(columns={
                                   'udp.length': 'length', 'frame.time_epoch': 'time', 'frame.time_relative': 'time_normed'})
            df_net = df_net.sort_values(by=['time_normed'])
            df_net['iat'] = df_net['time_normed'].diff().shift(-1)
            # cutoff_time = df_net[df_net['iat'] > 3].sort_values(by='time_normed', ascending=False)
            # if cutoff_time.shape[0] > 0:
            #     cutoff_time = cutoff_time.iloc[0]['time_normed']
            #     df_net = df_net[df_net['time_normed'] > cutoff_time]
            if df_net.shape[0] == 0:
                idx += 1
                continue
            src_df = df_net.groupby('ip.src').agg({'length': sum, 'rtp.p_type': 'count'}).reset_index(
            ).sort_values(by='length', ascending=False).head(1)['ip.src']
            if len(src_df) == 0:
                idx += 1
                continue
            src = src_df.iloc[0]
            df_net = df_net[df_net['ip.src'] == src]
            print('ML', df_net.shape)
            df_netml = feature_extractor.extract_features(df_net=df_net)
            webrtc_reader = WebRTCReader(webrtc_file, self.dataset)
            df_webrtc = webrtc_reader.get_webrtc()
            if df_webrtc is None or len(df_webrtc) == 0:
                idx += 1
                continue
            df_merged = pd.merge(df_netml, df_webrtc,
                                 left_on='et', right_on="ts")
            feature_file = csv_file[:-4]+f'_ml_{self.vca}_{self.dataset}.csv'
            print(feature_file)
            df_merged.to_csv(feature_file, index=False)

            df_net = read_net_file(self.dataset, csv_file)
            if df_net is None:
                idx += 1
                continue
            df_net = df_net[(df_net['rtp.p_type'].isin(self.config['video_ptype'][self.dataset][self.vca])) | (
                (df_net['rtp.p_type'].isin(self.config['rtx_ptype'][self.dataset][self.vca])) & (df_net['udp.length'] > 306))]
            df_net = df_net.rename(columns={
                                   'udp.length': 'length', 'frame.time_epoch': 'time', 'frame.time_relative': 'time_normed'})
            df_net = df_net.sort_values(by=['time_normed'])
            src_df = df_net.groupby('ip.src').agg({'length': sum, 'rtp.p_type': 'count'}).reset_index(
            ).sort_values(by='length', ascending=False).head(1)['ip.src']
            if len(src_df) == 0:
                idx += 1
                continue
            src = src_df.iloc[0]
            df_net = df_net[df_net['ip.src'] == src]

            print('RTP-ML', df_net.shape)
            df_netml = feature_extractor.extract_features(df_net=df_net)
            df_rtp = feature_extractor.extract_rtp_features(df_net=df_net)
            df = pd.merge(df_netml, df_rtp, on='et')
            webrtc_reader = WebRTCReader(webrtc_file, self.dataset)
            df_webrtc = webrtc_reader.get_webrtc()
            df_merged = pd.merge(df, df_webrtc,
                                 left_on='et', right_on="ts")
            print(df_merged.head(5)[
                  ['framesReceivedPerSecond', 'bitrate', 'frame_jitter', 'et']])
            feature_file = csv_file[:-4] + \
                f'_rtp_ml_{self.vca}_{self.dataset}.csv'
            df_merged.to_csv(feature_file, index=False)
            idx += 1