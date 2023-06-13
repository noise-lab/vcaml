from util.webrtc_reader import WebRTCReader
from util.helper_functions import read_net_file, filter_video_frames, is_freeze, get_freeze_dur
import sys
import pandas as pd
import numpy as np
from os.path import dirname, abspath
from collections import defaultdict
from scipy.stats import uniform, gamma

d = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(d)


class BayesianModel:

    def __init__(self, vca, metric):
        self.lookback = 3
        self.metric = metric
        self.vca = vca

    def train(self, list_of_files):
        frames_df_list = []
        default_probs = pd.Series(data=np.ones(
            101)*1e-8, index=np.arange(1, -100, -1))
        retransmission_probs = [default_probs]
        print(
            f'\nTraining Bayesian model...\nVCA: {self.vca}\nMetric: {self.metric}\n')
        for file_tuple in list_of_files:
            csv_file = file_tuple[1]
            pcap = read_net_file(csv_file)
            if pcap is None:
                print('Invalid file encountered...skipping...')
                continue
            pcap = filter_video_frames(pcap)
            pcap = pcap.sort_values(by=['frame.time_relative'])
            pcap['frame.time_relative'] = pcap['frame.time_relative'].astype(
                float)

            pcap["rtp.ts_rel"], frame_ts = pd.factorize(pcap["rtp.timestamp"])
            pcap["rtp.latest_ts_rel"] = pcap["rtp.ts_rel"].cummax()
            pcap['rtp.ts_rel_diff'] = pcap['rtp.ts_rel'] - \
                pcap['rtp.latest_ts_rel'].shift(1)
            retransmission_probs.append(
                pcap["rtp.ts_rel_diff"].value_counts(normalize=True))

            frame_df = pcap.groupby("rtp.timestamp").agg(
                {"udp.length": [list, "mean"]}).reset_index()
            frame_df['intra'] = frame_df[[("udp.length", "list"), ("udp.length", "mean")]].apply(
                lambda r: np.abs(np.array(r[0]) - r[1])[1:], axis=1)
            frames_df_list.append(frame_df.iloc[1:])

        self.transition_probs = pd.concat(
            retransmission_probs, axis=1).fillna(0).mean(axis=1)
        frames_df = pd.concat(frames_df_list).reset_index()
        self.intra_frame_dist = gamma.fit(np.concatenate(frames_df["intra"]))
        self.packet_size_dist = uniform.fit(
            np.concatenate(frames_df[("udp.length", "list")]))

    def estimate(self, file_tuple):
        csv_file = file_tuple[1]
        webrtc_file = file_tuple[2]
        df = read_net_file(csv_file)
        if df is None:
            return None
        df = df.sort_values(by=['frame.time_relative'])
        df['frame.time_relative'] = df['frame.time_relative'].astype(float)
        df = filter_video_frames(df)

        # Only consider lookbacks that have positive probability
        lookback = int(
            min(-np.min(self.transition_probs.index), self.lookback))
        frame_id_assignment = [0]
        frame_id = 0
        pkt_sizes = defaultdict(list)
        num_pkts = defaultdict(int)

        pkt_sizes[0].append(df.iloc[0]['udp.length'])
        num_pkts[0] += 1
        for i in range(1, df.shape[0]):
            curr_size = df.iloc[i]['udp.length']
            probs = np.zeros(lookback+2)
            for j, prev_frame_id in enumerate(range(frame_id-lookback, frame_id+1)):
                if prev_frame_id < 0:
                    probs[j] = -np.Inf
                    continue

                rel_idx = prev_frame_id - frame_id
                diff = np.abs(curr_size - pkt_sizes[prev_frame_id][-1])
                probs[j] += gamma.logpdf(diff, *self.intra_frame_dist)
                probs[j] += np.log(self.transition_probs[rel_idx])

            probs[-1] = uniform.logpdf(curr_size, *self.packet_size_dist) + \
                np.log(self.transition_probs[rel_idx])

            best_frame = np.argmax(probs) + frame_id - lookback
            frame_id_assignment.append(best_frame)
            pkt_sizes[best_frame].append(curr_size)
            num_pkts[best_frame] += 1
            frame_id = max(best_frame, frame_id)

        df["frame.time_epoch"] = df["frame.time_epoch"].astype(float)
        df["frame_num"] = frame_id_assignment

        df['udp.length'] = df['udp.length'] - 12
        df_grp = df.groupby("frame_num").agg(
            {"frame.time_epoch": list, "udp.length": list}).reset_index()
        df_grp["frame_st"] = df_grp["frame.time_epoch"].apply(lambda x: min(x))
        df_grp["frame_et"] = df_grp["frame.time_epoch"].apply(lambda x: max(x))
        df_grp["frame_size"] = df_grp["udp.length"].apply(lambda x: sum(x))
        df_grp["ft_end"] = df_grp['frame_et'].apply(lambda x: int(x))

        # frame duration calculations
        df_grp["frame_dur"] = df_grp["frame_et"].diff()
        df_grp["avg_frame_dur"] = df_grp["frame_dur"].rolling(
            30).mean()
        df_grp = df_grp.fillna(0)

        # freeze calculation
        df_grp["is_freeze"] = df_grp.apply(is_freeze, axis=1)
        df_grp["freeze_dur"] = df_grp.apply(get_freeze_dur, axis=1)

        df_grp = df_grp.groupby("ft_end").agg({"frame_size": ["count", "sum"], "is_freeze": "sum",
                                               "freeze_dur": "sum",
                                               "frame_dur": "std"}).reset_index()

        # rename columns
        df_grp.columns = ['_'.join(col).strip('_')
                          for col in df_grp.columns.values]
        df_grp = df_grp.rename(columns={'frame_size_count': 'fps',
                                        'is_freeze_sum': 'freeze_count',
                                        'frame_size_sum': 'bitrate',
                                        'freeze_dur_sum': 'freeze_dur',
                                        'frame_dur_std': 'interframe_delay_std'
                                        })
        df_grp['bitrate'] = df_grp['bitrate']*8/1000

        webrtc_reader = WebRTCReader(webrtc_file)
        df_webrtc = webrtc_reader.get_webrtc()[[self.metric, 'ts']]
        col = "ft_end"
        df_merge = pd.merge(df_grp, df_webrtc, left_on=col, right_on="ts")
        metric_col = f'{self.metric}_bayesian'
        webrtc_col = f'{self.metric}_webrtc'
        if self.metric == 'framesPerSecond' or self.metric == 'framesReceivedPerSecond':
            df_merge = df_merge.rename(
                columns={'fps': metric_col, self.metric: webrtc_col, 'ts': 'timestamp'})
        df_merge = df_merge[[webrtc_col, metric_col, 'timestamp']]
        df_merge = df_merge.dropna()
        return df_merge[[webrtc_col, metric_col, 'timestamp']]
