from util.webrtc_reader import WebRTCReader
from util.helper_functions import read_net_file, filter_video_frames_rtp, get_net_stats
import sys
import pandas as pd
import numpy as np
from os.path import dirname, abspath

d = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(d)


class RTPWithoutInterleavesModel:
    def __init__(self, vca, metric):
        self.metric = metric

    def is_freeze(self, x, avg):
        if x > max(3*avg, (avg + 0.150)):
            return 1
        else:
            return 0

    def estimate(self, file_tuple):
        pcap_file = file_tuple[0]
        csv_file = file_tuple[1]
        webrtc_file = file_tuple[2]
        df = read_net_file(csv_file)
        if df is None:
            return None
        df = df.sort_values(by=['frame.time_relative'])
        df['frame.time_relative'] = df['frame.time_relative'].astype(float)
        df = filter_video_frames_rtp(df)
        df_grp = df.groupby("rtp.timestamp").agg({"frame.time_epoch": list, "udp.length": list}).reset_index()
        df_grp["frame_st"] = df_grp["frame.time_epoch"].apply(lambda x: min(x))
        df_grp["frame_et_arr"] = df_grp["frame.time_epoch"].apply(lambda x: max(x))
        df_grp['frame_et'] = 0
        # Reorder end timestamps
        prev = None
        for idx, row in df_grp.iterrows():
            if prev is None:
                df_grp.at[idx, 'frame_et'] = row['frame_et_arr']
                prev = row['frame_et_arr']
                continue
            df_grp.at[idx, 'frame_et'] = max(prev, row['frame_et_arr'])
            prev = df_grp['frame_et'].iloc[idx]
        df_grp["frame_size"] = df_grp["udp.length"].apply(lambda x: sum(x))
        df_grp = get_net_stats(df_video=df_grp)
        webrtc_reader = WebRTCReader(webrtc_file)
        df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
        col = "frame_et_int"
        df_merge = pd.merge(df_grp, df_webrtc, left_on=col, right_on="ts")
        metric_col = f'{self.metric}_rtp_no_interleaves'
        webrtc_col = f'{self.metric}_webrtc'
        if self.metric == 'framesPerSecond' or self.metric == 'framesReceivedPerSecond':
            df_merge = df_merge.rename(
                columns={'fps': metric_col, self.metric: webrtc_col, 'ts': 'timestamp'})
        df_merge = df_merge[[webrtc_col, metric_col, 'timestamp']]
        df_merge = df_merge.dropna()
        df_merge['file'] = pcap_file
        return df_merge