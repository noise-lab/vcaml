import numpy as np
from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import time
import sys
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)
from util.helper_functions import read_net_file, filter_video_frames, is_freeze, get_freeze_dur
from util.webrtc_reader import WebRTCReader

class IP_UDP_Heuristic:
    def __init__(self, vca, metric, config, dataset):
        self.intra = 2
        self.config = config
        self.vca = vca
        self.metric = metric
        self.net_columns = ['frame.time_relative', 'frame.time_epoch', 'ip.src', 'ip.dst', 'ip.proto', 'ip.len',
                            'udp.srcport', 'udp.dstport', 'udp.length', 'rtp.ssrc', 'rtp.timestamp', 'rtp.seq', 'rtp.p_type', 'rtp.marker']
        self.max_lookback = {'meet': 3, 'teams': 2, 'webex': 1}
        self.dataset = dataset

    def assign(self, df, vca):
        l = self.max_lookback[vca]
        frame_id_assignment = [-1 for _ in range(df.shape[0])]
        frame_id = 0
        for i in range(df.shape[0]):
            found = False
            s = df.iloc[i]['udp.length']
            for j in range(i-1, max(0, i-l-1), -1):
                if abs(df.iloc[j]['udp.length'] - s) <= self.intra:
                    frame_id_assignment[i] = frame_id
                    found = True
                    break
            if not found:
                frame_id += 1
                frame_id_assignment[i] = frame_id
        return frame_id_assignment

    def estimate(self, file_tuple):
        csv_file = file_tuple[0]
        webrtc_file = file_tuple[1]
        df = pd.read_csv(csv_file)
        df = df[~df['ip.proto'].isna()]
        df['ip.proto'] = df['ip.proto'].astype(str)
        df = df[df['ip.proto'].str.contains(',') == False]
        df['ip.proto'] = df['ip.proto'].apply(lambda x: int(float(x)))
        ip_addr = df.groupby('ip.dst').agg({'udp.length': sum}).reset_index(
            ).sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
        df = df[df["ip.dst"] == ip_addr]
        df = df[(df['ip.proto'] == 17) & (df['ip.dst'] == ip_addr)]
        try:
            dst = df.groupby('ip.dst').agg({'udp.length': sum, 'rtp.p_type': 'count'}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
            df = df[df['ip.dst'] == dst]
        except IndexError:
            print('Faulty trace. Continuing..')
            return None
        df = df[(df['udp.length'] > 306)]
        df = df.sort_values(by=['frame.time_relative'])
        frame_id_assignment = self.assign(df, self.vca)
        df["frame_num"] = frame_id_assignment
        df['udp.length'] = df['udp.length'] - 12
        df_grp_udp = df.groupby("frame_num").agg(
            {"frame.time_epoch": list, "udp.length": list}).reset_index()
        df_grp_udp["frame_st"] = df_grp_udp["frame.time_epoch"].apply(
            lambda x: min(x))
        df_grp_udp["frame_et"] = df_grp_udp["frame.time_epoch"].apply(
            lambda x: max(x))
        df_grp_udp["frame_size"] = df_grp_udp["udp.length"].apply(
            lambda x: sum(x))
        df_grp_udp["ft_end"] = df_grp_udp['frame_et'].apply(lambda x: int(x)+1)

        df_grp_udp["frame_dur"] = df_grp_udp["frame_et"].diff()
        df_grp_udp["avg_frame_dur"] = df_grp_udp["frame_dur"].rolling(
            30).mean()
        df_grp_udp = df_grp_udp.fillna(0)
        idx = df_grp_udp.index[df_grp_udp['frame_dur'] >= 8].tolist()
        if len(idx) > 0:
            idx = idx[0]+1
        else:
            idx = 0
        df_grp_udp = df_grp_udp.iloc[idx:]
        # freeze calculation
        df_grp_udp["is_freeze"] = df_grp_udp.apply(is_freeze, axis=1)
        df_grp_udp["freeze_dur"] = df_grp_udp.apply(get_freeze_dur, axis=1)

        df_grp_udp = df_grp_udp.groupby("ft_end").agg({"frame_size": ["count", "sum"], "is_freeze": "sum",
                                                       "freeze_dur": "sum",
                                                       "frame_dur": "std"}).reset_index()

        # rename columns
        df_grp_udp.columns = ['_'.join(col).strip('_')
                              for col in df_grp_udp.columns.values]
        df_grp_udp = df_grp_udp.rename(columns={'frame_size_count': 'predicted_framesReceivedPerSecond',
                                                'is_freeze_sum': 'freeze_count',
                                                'frame_size_sum': 'predicted_bitrate',
                                                'freeze_dur_sum': 'freeze_dur',
                                                'frame_dur_std': 'predicted_frame_jitter'
                                                })
        df_grp_udp['predicted_bitrate'] = df_grp_udp['predicted_bitrate']*8
        df_grp_udp['predicted_frame_jitter'] *= 1000
        webrtc_reader = WebRTCReader(webrtc_file, self.dataset)
        df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
        col = "ft_end"
        df_merge = pd.merge(df_grp_udp, df_webrtc, left_on=col, right_on="ts")
        metric_col = f'{self.metric}_ip-udp-heuristic'
        webrtc_col = f'{self.metric}_gt'
        df_merge = df_merge.rename(columns={
                                   f'predicted_{self.metric}': metric_col, self.metric: webrtc_col, 'ts': 'timestamp'})
        df_merge['file'] = csv_file
        df_merge['dataset'] = self.dataset
        df_merge = df_merge[[webrtc_col, metric_col,
                             'timestamp', 'file', 'dataset']]
        df_merge = df_merge.dropna()
        if df_merge.shape[0] == 0:
            return None
        if self.metric == 'framesReceivedPerSecond':
            df_merge[webrtc_col] = df_merge[webrtc_col].apply(
                lambda x: round(x))
        return df_merge
