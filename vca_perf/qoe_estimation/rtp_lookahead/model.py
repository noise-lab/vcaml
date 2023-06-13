from util.webrtc_reader import WebRTCReader
from util.helper_functions import read_net_file, filter_video_frames_rtp, get_net_stats
import sys
import pandas as pd
from os.path import dirname, abspath

d = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(d)


class RTPLookaheadModel:
    def __init__(self, vca, metric):
        self.metric = metric
        self.vca=vca
        self.ptypes = {'meet': {'vid': '98', 'rtx': '99'}}

    def estimate(self, file_tuple):
        pcap_file = file_tuple[0]
        csv_file = file_tuple[1]
        webrtc_file = file_tuple[2]
        df = read_net_file(csv_file)
        if df is None:
            print('df is none')
            return None
        df = df.sort_values(by=['frame.time_relative'])
        df['frame.time_relative'] = df['frame.time_relative'].astype(float)
        df = filter_video_frames_rtp(df, self.vca)

        # df_vid = df[df['rtp.p_type'] == self.ptypes[self.vca]['vid']]
        # df_rtx = df[df['rtp.p_type'] == self.ptypes[self.vca]['rtx']]

        # df_grp_vid = df_vid.groupby("rtp.timestamp").agg({"frame.time_epoch": list, "udp.length": list, "rtp.marker": list, "rtp.seq": list}).reset_index()
        # df_grp_vid["frame_st"] = df_grp_vid["frame.time_epoch"].apply(lambda x: min(x))
        # df_grp_vid["frame_et"] = df_grp_vid["frame.time_epoch"].apply(lambda x: max(x))

        # df_grp_rtx = df_rtx.groupby("rtp.timestamp").agg({"frame.time_epoch": list, "udp.length": list, "rtp.marker": list, "rtp.seq": list}).reset_index()
        # df_grp_rtx["frame_st"] = df_grp_rtx["frame.time_epoch"].apply(lambda x: min(x))
        # df_grp_rtx["frame_et"] = df_grp_rtx["frame.time_epoch"].apply(lambda x: max(x))

        # df_grp_vid['frame_size'] = 0

        # for i in range(df_grp_vid.shape[0]):
        #     valid = True
        #     last_rtx_index = 0
        #     current_ts = df_grp_vid.iloc[i]['rtp.timestamp']
        #     rtx_df = df_grp_rtx[df_grp_rtx['rtp.timestamp'] == current_ts]
        #     seq_no_list = list(df_grp_vid.iloc[i]['rtp.seq'])
        #     frame_time_list = list(df_grp_vid.iloc[i]['frame.time_epoch'])
        #     for j in range(len(seq_no_list)):
        #         if j < len(df_grp_vid.iloc[i]['rtp.seq']) - 1:
        #             next_frame_time = frame_time_list[j+1]
        #         elif j == len(df_grp_vid.iloc[i]['rtp.seq']) - 1 and i < len(df_grp_vid) - 1:
        #             next_frame_time = list(df_grp_vid.iloc[i+1]['frame.time_epoch'])[0]
        #         elif j == len(df_grp_vid.iloc[i]['rtp.seq']) and i == len(df_grp_vid) - 1:
        #             next_frame_time = list(df_grp_vid.iloc[i]['frame.time_epoch'])[-1]
        #         if i > 0 and seq_no_list[j] != last_seq + 1:

        #             rtx_sizes = list(rtx_df['udp.length'])
        #             rtx_times = list(rtx_df['frame.time_epoch'])
                    
        #             if len(rtx_sizes) > 0:              
        #                 count = 0
        #                 prev_pkt_size = rtx_sizes[last_rtx_index]
        #                 while last_rtx_index < len(rtx_times) and rtx_sizes[0][last_rtx_index] == prev_pkt_size and rtx_times[0][last_rtx_index] < next_frame_time:
        #                     if count > 10:
        #                         valid = False
        #                         break
        #                     last_rtx_index += 1    
        #                     prev_pkt_size = rtx_sizes[last_rtx_index]                     
        #                     count += 1
        #             else:
        #                 valid = False
        #             last_seq = seq_no_list[j]
        #         last_seq = list(df_grp_vid.iloc[i]['rtp.seq'])[j]
        #         if not valid:
        #             break
        #     df_grp_vid.at[i, 'frame_size'] = (sum(df_grp_vid.iloc[i]['udp.length']) if valid else 0)

        df_video = df.groupby("rtp.timestamp").agg({"frame.time_epoch": list, "udp.length": list}).reset_index()
        df_video["frame_st"] = df_video["frame.time_epoch"].apply(lambda x: min(x))
        df_video["frame_et"] = df_video["frame.time_epoch"].apply(lambda x: max(x))
        df_video["frame_size"] = df_video["udp.length"].apply(lambda x: sum(x))

        rtp_timestamps = sorted(list(df_video['rtp.timestamp']))
        rtp_timestamp_diff = [0 for _ in range(len(rtp_timestamps))]
        for i in range(1, len(rtp_timestamps)):
            rtp_timestamp_diff[i] = rtp_timestamps[i] - rtp_timestamps[i-1]           
        
        df_video['rtp_timestamp_diff'] = rtp_timestamp_diff
        df_video['frame_rate'] = 90000 / df_video['rtp_timestamp_diff']
        
        df_grp = get_net_stats(df_video)

        webrtc_reader = WebRTCReader(webrtc_file)
        df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
        col = "frame_et_int"
        df_merge = pd.merge(df_grp, df_webrtc, left_on=col, right_on="ts")
        metric_col = f'{self.metric}_rtp-lookahead'
        webrtc_col = f'{self.metric}_webrtc'
        if self.metric == 'framesPerSecond' or self.metric == 'framesReceived' or self.metric == 'framesReceivedPerSecond':
            df_merge = df_merge.rename(
                columns={'fps': metric_col, self.metric: webrtc_col, 'ts': 'timestamp'})
        df_merge = df_merge[[webrtc_col, metric_col, 'timestamp']]
        df_merge = df_merge.dropna()
        df_merge['file'] = pcap_file
        print(df_merge.head(7))
        return df_merge
