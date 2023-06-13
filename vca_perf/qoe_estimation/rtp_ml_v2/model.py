from util.webrtc_reader import WebRTCReader
from util.helper_functions import read_net_file, filter_video_frames_rtp, get_net_stats
import sys
import pandas as pd
from os.path import dirname, abspath
from sklearn.metrics import mean_absolute_error
import random

d = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(d)


class RTPModel:
    def __init__(self, vca, metric):
        self.metric = metric
        self.vca=vca
        self.video_ptype = {'meet': '98', 'teams': '102', 'webex': '102'}

    def estimate(self, file_tuple):
        pcap_file = file_tuple[0]
        csv_file = file_tuple[1]
        webrtc_file = file_tuple[2]
        df = read_net_file(csv_file)
        if df is None:
            return None
        df = df.sort_values(by=['frame.time_relative'])
        df['frame.time_relative'] = df['frame.time_relative'].astype(float)
        df = filter_video_frames_rtp(df, self.vca)
        df = df[df['rtp.p_type'] == self.video_ptype[self.vca]]
        df['rtp.seq'] = df['rtp.seq'].apply(int)
        min_seq_no = df['rtp.seq'].min()-1
        df_grp = df.groupby("rtp.timestamp").agg({"frame.time_epoch": list, "udp.length": list, "rtp.seq": list}).reset_index()
        
        to_discard = set()
        for i in range(df_grp.shape[0]):
            seq_no_list = sorted(df_grp.iloc[i]['rtp.seq'])
            discard = False
            for j in range(len(seq_no_list)):
                if seq_no_list[j] != min_seq_no + 1:
                    to_discard.add(i)
                min_seq_no = seq_no_list[j]

        df_grp = df_grp.drop(to_discard)

        df_grp["frame_st"] = df_grp["frame.time_epoch"].apply(lambda x: min(x))
        df_grp["frame_et"] = df_grp["frame.time_epoch"].apply(lambda x: max(x))
        df_grp["frame_size"] = df_grp["udp.length"].apply(lambda x: sum(x))
        df_grp = get_net_stats(df_video=df_grp)
        # webrtc_reader = WebRTCReader(webrtc_file)
        # df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
        col = "frame_et_int"
        # df_test = pd.merge(df_grp, df_webrtc, left_on=col, right_on="ts")
        # mae = round(mean_absolute_error(df_test['fps'], df_test['framesReceived']), 2)
        # print(f'MAE = {mae}')

        metric_col = f'{self.metric}_rtp'
        # webrtc_col = f'{self.metric}_webrtc'
        if self.metric == 'framesPerSecond' or self.metric == 'framesReceivedPerSecond' or self.metric == 'framesReceived':
            # df_merge = df_merge.rename(columns={'fps': metric_col, self.metric: webrtc_col, 'ts': 'timestamp'})
            df_grp = df_grp.rename(columns={'fps': metric_col, col: 'timestamp'})
        df_grp['file'] = pcap_file
        # df_merge = df_merge[[webrtc_col, metric_col, 'timestamp']]
        # df_merge = df_merge.dropna()
        # df_merge['file'] = pcap_file
        return df_grp
