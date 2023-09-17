import time
import sys
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)
from util.webrtc_reader import WebRTCReader
from util.helper_functions import read_net_file, filter_video_frames_rtp, get_net_stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.metrics import mean_absolute_error
import random
from collections import defaultdict
import numpy as np

class RTP_Heuristic:
    def __init__(self, vca, metric, config, dataset):
        self.metric = metric
        self.vca=vca
        self.config = config
        self.dataset = dataset

    def estimate(self, file_tuple):
        csv_file = file_tuple[0]
        webrtc_file = file_tuple[1]
        df = read_net_file(self.dataset, csv_file)
        if df is None:
            return None
        df = df.sort_values(by=['frame.time_relative'])
        df['frame.time_relative'] = df['frame.time_relative'].astype(float)
        df = df[(df['rtp.p_type'].isin([*self.config['video_ptype'][self.dataset][self.vca], *self.config['rtx_ptype'][self.dataset][self.vca]]))]
        try:
            dst = df.groupby('ip.dst').agg({'udp.length': sum, 'rtp.p_type': 'count'}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
            df = df[df['ip.dst'] == dst]
        except IndexError:
            print('Faulty trace. Continuing..')
            return None
        df = df[df['udp.length'] > 306]
        df['udp.length'] = df['udp.length'] - 12
        
        df_grp = df.groupby("rtp.timestamp").agg({"frame.time_epoch": list, "udp.length": list, "rtp.seq": list, "rtp.marker": list}).reset_index()

        df_grp['is_valid'] = df_grp['rtp.marker'].apply(lambda x: int(sum(x)))
        df_grp = df_grp[df_grp['is_valid'] == 1]
        df_grp["frame_st"] = df_grp["frame.time_epoch"].apply(lambda x: min(x))
        df_grp["frame_et"] = df_grp["frame.time_epoch"].apply(lambda x: max(x))
        df_grp["frame_size"] = df_grp["udp.length"].apply(lambda x: sum(x))
        df_grp = get_net_stats(df_video=df_grp)
        webrtc_reader = WebRTCReader(webrtc_file, self.dataset)
        df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
        col = "frame_et_int"
        df_merge = pd.merge(df_grp, df_webrtc, left_on=col, right_on="ts")
        metric_col = f'{self.metric}_rtp-heuristic'
        webrtc_col = f'{self.metric}_gt'
        df_merge = df_merge.rename(columns={f'predicted_{self.metric}': metric_col, self.metric: webrtc_col, col: 'timestamp'})
        df_merge['file'] = csv_file
        df_merge['dataset'] = self.dataset
        df_merge = df_merge[[webrtc_col, metric_col, 'timestamp', 'file', 'dataset']]
        
        df_merge = df_merge.dropna()
        if self.metric == 'framesReceivedPerSecond':
            df_merge[webrtc_col] = df_merge[webrtc_col].apply(lambda x: round(x))
        return df_merge
