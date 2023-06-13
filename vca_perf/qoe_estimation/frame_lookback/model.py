from util.webrtc_reader import WebRTCReader
from util.helper_functions import read_net_file, filter_video_frames, is_freeze, get_freeze_dur
import sys
import pandas as pd
from os.path import dirname, abspath
from collections import defaultdict
import numpy as np
d = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(d)


class FrameLookbackModel:
    def __init__(self, vca, metric, config, dataset):

        if vca == "meet":
            self.intra = 2
        elif vca == "webex":
            self.intra = 2
        elif vca == "teams":
            self.intra = 2
        else:
            raise ValueError('Unsupported VCA')
        self.config = config
        self.vca = vca
        self.metric = metric
        self.net_columns = ['frame.time_relative','frame.time_epoch','ip.src','ip.dst','ip.proto','ip.len','udp.srcport','udp.dstport', 'udp.length','rtp.ssrc','rtp.timestamp','rtp.seq','rtp.p_type', 'rtp.marker']
        self.max_lookback = {'meet': 3, 'teams': 2, 'webex': 1}
        self.dataset = dataset

    
    def train(self, file_tuples):
        idx = 1
        dfs = []
        for file_tuple in file_tuples:
            print(f'Training for {self.vca}: {idx} of {len(file_tuples)}')
            csv_file = file_tuple[1]
            webrtc_file = file_tuple[2]
            df = pd.read_csv(csv_file, header=None, sep='\t', names=self.net_columns, lineterminator='\n', encoding='ascii')
            df = df[~df['ip.proto'].isna()]
            if df['ip.proto'].dtype == object:
                df = df[df['ip.proto'].str.contains(',') == False]
            df['ip.proto'] = df['ip.proto'].astype(int)
            df = df[(df['ip.proto'] == 17) & (df['ip.dst'] == self.config['destination_ip'][self.dataset])]
            src = df.groupby('ip.src').agg({'udp.length': sum, 'rtp.p_type': 'count'}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.src'].iloc[0]
            df = df[df['ip.src'] == src]
            df = df[df['udp.length'] > 306]
            df = df.sort_values(by=['frame.time_relative'])
            df['packet_rank'] = list(range(len(df)))
            dfg = df.groupby('rtp.timestamp').agg(intra_diff = ('udp.length', lambda x: np.diff(np.array(x)).tolist()), min_udp_length = ('udp.length', min), max_udp_length = ('udp.length', max), udp_length_list = ('udp.length', list), ptype_list = ('rtp.p_type', list), seq_no_list = ('rtp.seq', list), marker_list = ('rtp.marker', list), time_list = ('frame.time_relative', list), rank_list = ('packet_rank', list))
            dfg = dfg.reset_index()
            dfg['min_intra_diff'] = dfg['intra_diff'].apply(lambda x: np.array(x).min() if len(x) > 0 else 0)
            dfg['max_intra_diff'] = dfg['intra_diff'].apply(lambda x: np.array(x).max() if len(x) > 0 else 0)
            dfg['avg_intra_diff'] = dfg['intra_diff'].apply(lambda x: np.array(x).mean() if len(x) > 0 else 0)
            dfg['median_intra_diff'] = dfg['intra_diff'].apply(lambda x: np.array(np.array(x)) if len(x) > 0 else 0)
            dfg['std_intra_diff'] = dfg['intra_diff'].apply(lambda x: np.array(x).std() if len(x) > 0 else 0)           
            dfg['min_abs_intra_diff'] = dfg['intra_diff'].apply(lambda x: np.abs(np.array(x)).min() if len(x) > 0 else 0)
            dfg['max_abs_intra_diff'] = dfg['intra_diff'].apply(lambda x: np.abs(np.array(x)).max() if len(x) > 0 else 0) 
            dfg['avg_abs_intra_diff'] = dfg['intra_diff'].apply(lambda x: np.abs(np.array(x)).mean() if len(x) > 0 else 0)
            dfg['median_abs_intra_diff'] = dfg['intra_diff'].apply(lambda x: np.median(np.abs(np.array(x))) if len(x) > 0 else 0)
            dfg['std_abs_intra_diff'] = dfg['intra_diff'].apply(lambda x: np.abs(np.array(x)).std() if len(x) > 0 else 0)
            dfg['inter_diff'] = np.nan
            dfg['frame_length'] = dfg['udp_length_list'].apply(lambda x: len(x))
            for i in range(1, dfg.shape[0]):
                dfg.at[i, 'inter_diff'] = dfg['udp_length_list'].iloc[i][0] - dfg['udp_length_list'].iloc[i-1][-1]
            dfg = dfg.dropna()
            print(dfg.shape)
            dfs.append(dfg)
            idx += 1
        df = pd.concat(dfs, axis=0)
        return df
    
    def assign(self, df, vca):
        max_lookback = {'meet': 16, 'teams': 2, 'webex': 1}
        l = max_lookback[vca]
        frame_id_assignment = [-1 for _ in range(df.shape[0])]
        frame_id = 1
        frame_size_sum = defaultdict(lambda: 0)
        frame_count = defaultdict(lambda: 0)
        frame_size_sum[1] = df.iloc[0]['udp.length']
        frame_count[1] = 1
        frame_id_assignment[0] = 1
        for i in range(1, df.shape[0]):
            found = False
            s = df.iloc[i]['udp.length']
            for j in range(frame_id, max(frame_id-l, 0), -1):
                avg = frame_size_sum[j] / frame_count[j]
                if abs(avg - s) <= self.intra:
                    if abs(avg - s) == 0 or (abs(avg - s) > 0 and frame_count[j] < 10):
                        frame_id_assignment[i] = j
                        found = True
                        break
            if not found:
                frame_id += 1
                frame_id_assignment[i] = frame_id
            frame_size_sum[frame_id_assignment[i]] += s
            frame_count[frame_id_assignment[i]] += 1
        return frame_id_assignment
    
    def assign_1(self, df, vca):
        max_lookback = {'meet': 3, 'teams': 2, 'webex': 1}
        l = max_lookback[vca]
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
        pcap_file = file_tuple[0]
        csv_file = file_tuple[1]
        webrtc_file = file_tuple[2]
        df = pd.read_csv(csv_file, header=None, sep='\t', names=self.net_columns, lineterminator='\n', encoding='ascii')
        df = df[~df['ip.proto'].isna()]
        df['ip.proto'] = df['ip.proto'].astype(str)
        df = df[df['ip.proto'].str.contains(',') == False]
        df['ip.proto'] = df['ip.proto'].apply(lambda x: int(float(x)))
        ip_addr = self.config['destination_ip'][self.dataset]
        if ip_addr == 'dynamic':
            ip_addr = df.groupby('ip.dst').agg({'udp.length': sum}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
        df = df[df["ip.dst"] == ip_addr]
        df = df[(df['ip.proto'] == 17) & (df['ip.dst'] == ip_addr)]
        src = df.groupby('ip.src').agg({'udp.length': sum, 'rtp.p_type': 'count'}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.src'].iloc[0]
        df = df[df['ip.src'] == src]
        df = df[(df['udp.length'] > 306)]
        df = df.sort_values(by=['frame.time_relative'])
        # df['iat'] = df['frame.time_relative'].diff().shift(-1)
        # cutoff_time = df[df['iat'] > 3].sort_values(by='frame.time_relative', ascending=False)
        # if cutoff_time.shape[0] > 0:
        #     cutoff_time = cutoff_time.iloc[0]['frame.time_relative']
        #     df = df[df['frame.time_relative'] > cutoff_time]
        frame_id_assignment = self.assign_1(df, self.vca)
        df["frame_num"] = frame_id_assignment
        df['udp.length'] = df['udp.length'] - 12
        df_grp_udp = df.groupby("frame_num").agg({"frame.time_epoch": list, "udp.length": list}).reset_index()
        df_grp_udp["frame_st"] = df_grp_udp["frame.time_epoch"].apply(lambda x: min(x))
        df_grp_udp["frame_et"] = df_grp_udp["frame.time_epoch"].apply(lambda x: max(x))
        # df_grp_udp = df_grp_udp[df_grp_udp['frame_et'] - df_grp_udp['frame_st'] <= self.max_buffer_time[self.vca]]
        df_grp_udp["frame_size"] = df_grp_udp["udp.length"].apply(lambda x: sum(x))
        df_grp_udp["ft_end"] = df_grp_udp['frame_et'].apply(lambda x: int(x)+1)

        df_grp_udp["frame_dur"] = df_grp_udp["frame_et"].diff()
        df_grp_udp["avg_frame_dur"] = df_grp_udp["frame_dur"].rolling(
            30).mean()  # Why 30?
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
        metric_col = f'{self.metric}_frame-lookback'
        webrtc_col = f'{self.metric}_gt'
        df_merge = df_merge.rename(columns={f'predicted_{self.metric}': metric_col, self.metric: webrtc_col, 'ts': 'timestamp'})
        df_merge['file'] = pcap_file
        df_merge['dataset'] = self.dataset
        df_merge = df_merge[[webrtc_col, metric_col, 'timestamp', 'file', 'dataset']]
        df_merge = df_merge.dropna()
        if df_merge.shape[0] == 0:
            return None
        if self.metric == 'framesReceivedPerSecond':
            df_merge[webrtc_col] = df_merge[webrtc_col].apply(lambda x: round(x))
        return df_merge