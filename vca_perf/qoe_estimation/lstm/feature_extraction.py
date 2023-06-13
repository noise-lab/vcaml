import pandas as pd
import numpy as np
from collections import defaultdict

pd.options.mode.chained_assignment = None


class FeatureExtractor:

    def __init__(self, feature_subset, config, vca):
        self.feature_subset = feature_subset
        self.config = config
        self.vca = vca

        self.max_size = -1
        self.max_iat = -1

    def extract_rtp_features(self, df_net):
        df_net['rtp.timestamp'] = df_net['rtp.timestamp'].astype(float)
        df_net['rtp.seq'] = df_net['rtp.seq'].astype(int)
        rtp_seq_nos = {}
        feature_data = {'rtp_ts_mean': [], 'rtp_ts_min': [], 'rtp_ts_max': [], 'rtp_ts_q1': [], 'rtp_ts_q2': [], 'rtp_ts_q3': [], 'rtp_ts_unique': [], 'et': []}
        for pt in df_net['rtp.p_type'].unique():
            rtp_seq_nos[pt] = df_net[df_net['rtp.p_type'] == pt]['rtp.seq'].unique()
            feature_data[f'rtp_loss_{pt}'] = []
        ts_list = (df_net['rtp.timestamp'].tolist())
        ts_dict = {x: idx for idx, x in enumerate(ts_list)}
        df_net['ts_rank'] = df_net['rtp.timestamp'].replace(ts_dict)
        df_net = df_net.sort_values(by='time_normed')
        prev = df_net['time_normed'].iloc[0]
        window_size = self.config['prediction_window']
        ts_rank_vals = []
        ts_vals = set()
        seq_min = {pt: ((1<<16)+1) for pt in df_net['rtp.p_type'].unique()}
        seq_max = {pt: -1*((1<<16)+1) for pt in df_net['rtp.p_type'].unique()}

        arr_times = {df_net['rtp.seq'].iloc[i]: df_net['time_normed'].iloc[i] for i in range(len(df_net))}

        # Discard sequence numbers that arrive late (mark them as -1)
        # seqno_mapping = {}
        # for pt in df_net['rtp.p_type'].unique():
        #     seq_list =  sorted(list(rtp_seq_nos[pt]))
        #     new_seq_numbers = [-1 for _ in range(len(seq_list))]
        #     for idx, seqno in enumerate(seq_list):
        #         if idx == 0:
        #             new_seq_numbers[idx] = seqno
        #             continue
        #         if arr_times[seqno] - arr_times[seq_list[idx-1]] < -1*self.config['jitter_buffer_size']:
        #             new_seq_numbers[idx-1] = -1
        #         new_seq_numbers[idx] = seqno

        #     seqno_mapping.update({seq_list[i]: new_seq_numbers[i] for i in range(len(seq_list))})
        # df_net['rtp.seq'] = df_net['rtp.seq'].replace(seqno_mapping)

        for i in range(df_net.shape[0]):
            if df_net['time_normed'].iloc[i] - prev > window_size:
                if i > 0:
                    arr = np.array(ts_rank_vals)
                    arr = np.diff(arr)
                    if len(arr) > 0:
                        feature_data['rtp_ts_mean'].append(arr.mean())
                        feature_data['rtp_ts_min'].append(arr.min())
                        feature_data['rtp_ts_max'].append(arr.max())
                        feature_data['rtp_ts_q1'].append(np.quantile(arr, 0.25))
                        feature_data['rtp_ts_q2'].append(np.quantile(arr, 0.50))
                        feature_data['rtp_ts_q3'].append(np.quantile(arr, 0.75))
                        feature_data['rtp_ts_unique'].append(len(ts_vals))
                        feature_data['et'].append(int(df_net['time'].iloc[i]))
                        for pt in df_net['rtp.p_type'].unique():
                            missing_seq_nos = [x for x in range(seq_min[pt], seq_max[pt]+1) if x not in rtp_seq_nos[pt]]
                            lost = len(missing_seq_nos)
                            loss_frac = lost / (seq_max[pt]-seq_min[pt]+1)
                            feature_data[f'rtp_loss_{pt}'].append(loss_frac)
                        ts_rank_vals = []
                        ts_vals = set()
                        seq_min = {pt: ((1<<16)+1) for pt in df_net['rtp.p_type'].unique()}
                        seq_max = {pt: -1*((1<<16)+1) for pt in df_net['rtp.p_type'].unique()}
                    prev = df_net['time_normed'].iloc[i]
            ts_rank_vals.append(df_net['ts_rank'].iloc[i])
            ts_vals.add(df_net['rtp.timestamp'].iloc[i])
            pt = df_net['rtp.p_type'].iloc[i]
            seq_min[pt] = min(seq_min[pt], df_net['rtp.seq'].iloc[i])
            seq_max[pt] = max(seq_max[pt], df_net['rtp.seq'].iloc[i])

        arr = np.array(ts_rank_vals)
        arr = np.diff(arr)
        if len(arr) > 0:
            feature_data['rtp_ts_mean'].append(arr.mean())
            feature_data['rtp_ts_min'].append(arr.min())
            feature_data['rtp_ts_max'].append(arr.max())
            feature_data['rtp_ts_q1'].append(np.quantile(arr, 0.25))
            feature_data['rtp_ts_q2'].append(np.quantile(arr, 0.50))
            feature_data['rtp_ts_q3'].append(np.quantile(arr, 0.75))
            feature_data['rtp_ts_unique'].append(len(ts_vals))
            feature_data['et'].append(int(df_net['time'].iloc[len(df_net)-1]))
            for pt in df_net['rtp.p_type'].unique():
                missing_seq_nos = [x for x in range(seq_min[pt], seq_max[pt]+1) if x not in rtp_seq_nos[pt]]
                lost = len(missing_seq_nos)
                loss_frac = lost / (seq_max[pt]-seq_min[pt]+1)
                feature_data[f'rtp_loss_{pt}'].append(loss_frac)

        df = pd.DataFrame(feature_data)
        return df

    def extract_features(self, df_net):
        features = []
        for feature_type in self.feature_subset:
            if feature_type == 'SIZE':
                df_size = self.extract_size_features(df_net)
                features.append(df_size)
            elif feature_type == 'IAT':
                df_iat = self.extract_iat_features(df_net)
                features.append(df_iat)
            elif feature_type == 'LSTATS':
                df_stats = self.extract_length_stat_features(df_net)
                features.append(df_stats)
            elif feature_type == 'TSTATS':
                df_stats = self.extract_iat_stat_features(df_net)
                features.append(df_stats)
            elif feature_type == 'BPS':
                df_stats = self.extract_bps_features(df_net)
                features.append(df_stats)
        # Remove duplicate 'et'
        if len(features) > 1:
            for j in range(len(features)):
                if j > 0:
                    df = features[j]
                    df = df[df.columns.difference(['et'])]
                    features[j] = df
                    
        return pd.concat(features, axis=1)

    def extract_size_features(self, df_net):
        prev = df_net['time_normed'].iloc[0]
        current_interval = 1
        end_times = []
        pkt_sizes = {'interval': [], 'sizes': []}
        n_features = self.config['n_features_size'][self.vca]
        window_size = self.config['prediction_window']
        for i in range(df_net.shape[0]):
            if df_net['time_normed'].iloc[i] - prev > window_size:
                current_interval += 1
                prev = df_net['time_normed'].iloc[i]
                end_times.append(df_net['time'].iloc[i])
            if i == df_net.shape[0]-1:
                end_times.append(df_net['time'].iloc[i])
            pkt_sizes['interval'].append(current_interval)
            pkt_sizes['sizes'].append(df_net['length'].iloc[i])
        df_agg = pd.DataFrame(pkt_sizes)
        df_agg = df_agg.groupby('interval').agg({'sizes': list})
        for idx, row in df_agg.iterrows():
            size_list = row['sizes']
            self.max_size = max(self.max_size, len(size_list))
            if len(size_list) < n_features:
                size_list += [0]*(n_features-len(size_list))
            df_agg.at[idx, 'sizes'] = size_list[:n_features]
        size_col_names = [f'size_{i}' for i in range(1, n_features+1)]
        df_size = pd.DataFrame(
            df_agg['sizes'].to_list(), columns=size_col_names)
        df_size['et'] = 0.0
        df_size.loc[:, 'et'] = end_times
        df_size['et'] = df_size['et'].apply(lambda x: int(x))
        return df_size

    def extract_iat_features(self, df_net):
        prev = df_net['time_normed'].iloc[0]
        current_interval = 1
        end_times = []
        pkt_iats = {'interval': [], 'iats': []}
        n_features = self.config['n_features_iat'][self.vca]
        window_size = self.config['prediction_window']
        for i in range(df_net.shape[0]):
            if df_net['time_normed'].iloc[i] - prev > window_size:
                current_interval += 1
                prev = df_net['time_normed'].iloc[i]
                end_times.append(df_net['time'].iloc[i])
            if i == df_net.shape[0]-1:
                end_times.append(df_net['time'].iloc[i])
            pkt_iats['interval'].append(current_interval)
            if i == 0:
                pkt_iats['iats'].append(df_net['time_normed'].iloc[i]*1000)
            else:
                pkt_iats['iats'].append(
                    (df_net['time_normed'].iloc[i]-df_net['time_normed'].iloc[i-1])*1000)

        df_agg = pd.DataFrame(pkt_iats)
        df_agg = df_agg.groupby('interval').agg({'iats': list})
        for idx, row in df_agg.iterrows():
            iat_list = row['iats']
            self.max_iat = max(self.max_size, len(iat_list))
            if len(iat_list) < n_features:
                iat_list += [0]*(n_features-len(iat_list))
            df_agg.at[idx, 'iats'] = iat_list[:n_features]
        iat_col_names = [f'iat_{i}' for i in range(1, n_features+1)]
        df_iat = pd.DataFrame(df_agg['iats'].to_list(), columns=iat_col_names)
        df_iat['et'] = 0.0
        df_iat.loc[:, 'et'] = end_times
        df_iat['et'] = df_iat['et'].apply(lambda x: int(x))
        return df_iat

    def extract_length_stat_features(self, df_net):
        prev = df_net['time_normed'].iloc[0]
        current_interval = 1
        end_times = []
        pkt_sizes = {'interval': [], 'sizes': []}
        window_size = self.config['prediction_window']
        for i in range(df_net.shape[0]):
            if df_net['time_normed'].iloc[i] - prev > window_size:
                current_interval += 1
                prev = df_net['time_normed'].iloc[i]
                end_times.append(df_net['time'].iloc[i])
            if i == df_net.shape[0]-1:
                end_times.append(df_net['time'].iloc[i])
            pkt_sizes['interval'].append(current_interval)
            pkt_sizes['sizes'].append(df_net['length'].iloc[i])
        df_agg = pd.DataFrame(pkt_sizes)
        df_agg = df_agg.groupby('interval').agg({'sizes': [list, sum, len]})
        df_agg.columns = ['size_list', 'size_sum', 'size_len']
        df_agg = df_agg.assign(**{'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                               'q1': 0, 'q2': 0, 'q3': 0, 'num_pkts': 0, 'num_bytes': 0, 'num_unique': 0, 'num_rtx': 0})
        for idx, row in df_agg.iterrows():
            d = defaultdict(int)
            sl = row['size_list']
            for x in sl:
                d[x] += 1
            sl_set = set(sl)
            count = 0
            for x in sl_set:
                if x+2 in d:
                    count += d[x+2]
            df_agg.at[idx, 'num_rtx'] = count
            df_agg.at[idx, 'num_unique'] = len(sl_set)
            df_agg.at[idx, 'size_list'] = sl
            df_agg.at[idx, 'mean'] = np.array(sl).mean()
            df_agg.at[idx, 'std'] = np.array(sl).std()
            df_agg.at[idx, 'min'] = min(sl)
            df_agg.at[idx, 'max'] = max(sl)
            df_agg.at[idx, 'q1'], df_agg.at[idx, 'q2'], df_agg.at[idx,'q3'] = np.quantile(np.array(sl), q=[0.25, 0.5, 0.75])
            df_agg.at[idx, 'num_pkts'] = len(sl)
            df_agg.at[idx, 'num_bytes'] = sum(sl)
        df_stats = df_agg[['mean', 'std', 'min', 'max', 'q1', 'q2', 'q3', 'num_pkts', 'num_bytes', 'num_unique']]
        df_stats['et'] = 0.0
        df_stats.loc[:, 'et'] = end_times
        df_stats['et'] = df_stats['et'].apply(lambda x: int(x))
        df_stats = df_stats.reset_index()
        df_stats = df_stats[df_stats.columns.difference(['interval'])]
        col_map = {}
        for col in df_stats.columns:
            if col == 'et':
                continue
            col_map[col] = 'l_'+col
        df_stats = df_stats.rename(columns=col_map)
        return df_stats


    def extract_iat_stat_features(self, df_net):
        prev = df_net['time_normed'].iloc[0]
        current_interval = 1
        end_times = []
        pkt_iats = {'interval': [], 'iats': []}
        window_size = self.config['prediction_window']
        for i in range(df_net.shape[0]):
            if df_net['time_normed'].iloc[i] - prev > window_size:
                current_interval += 1
                prev = df_net['time_normed'].iloc[i]
                end_times.append(df_net['time'].iloc[i])
            if i == df_net.shape[0]-1:
                end_times.append(df_net['time'].iloc[i])
            pkt_iats['interval'].append(current_interval)
            if i == 0:
                pkt_iats['iats'].append(df_net['time_normed'].iloc[i]*1000)
            else:
                pkt_iats['iats'].append((df_net['time_normed'].iloc[i]-df_net['time_normed'].iloc[i-1])*1000)

        df_agg = pd.DataFrame(pkt_iats)
        df_agg = df_agg.groupby('interval').agg({'iats': list})
        df_agg = df_agg.assign(**{'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'q1': 0, 'q2': 0, 'q3': 0})
        for idx, row in df_agg.iterrows():
            sl = row['iats']
            df_agg.at[idx, 'iats'] = sl
            df_agg.at[idx, 'mean'] = np.array(sl).mean()
            df_agg.at[idx, 'std'] = np.array(sl).std()
            df_agg.at[idx, 'min'] = min(sl)
            df_agg.at[idx, 'max'] = max(sl)
            df_agg.at[idx, 'q1'], df_agg.at[idx, 'q2'], df_agg.at[idx,
                                                                  'q3'] = np.quantile(np.array(sl), q=[0.25, 0.5, 0.75])
        df_stats = df_agg[['mean', 'std', 'min', 'max','q1', 'q2', 'q3']]
        df_stats['et'] = 0.0
        df_stats.loc[:, 'et'] = end_times
        df_stats['et'] = df_stats['et'].apply(lambda x: int(x))
        df_stats = df_stats.reset_index()
        df_stats = df_stats[df_stats.columns.difference(['interval'])]
        col_map = {}
        for col in df_stats.columns:
            if col == 'et':
                continue
            col_map[col] = 't_'+col
        df_stats = df_stats.rename(columns=col_map)
        df_stats['t_et'] = df_stats['et']
        return df_stats

    def extract_bps_features(self, df_net):
        prev = df_net['time_normed'].iloc[0]
        current_interval = 1
        end_times = []
        pkt_bps = {'interval': [], 'sizes': [], 'iats': []}
        n_features = self.config['n_features_bps'][self.vca]
        window_size = self.config['prediction_window']
        prev_pkt_time = 0
        for i in range(df_net.shape[0]):
            if i == 0:
                prev_pkt_time = df_net['time_normed'].iloc[i]
                continue
            if df_net['time_normed'].iloc[i] - prev > window_size:
                current_interval += 1
                prev = df_net['time_normed'].iloc[i]
                end_times.append(df_net['time'].iloc[i])
            if i == df_net.shape[0]-1:
                end_times.append(df_net['time'].iloc[i])
            pkt_bps['interval'].append(current_interval)
            pkt_bps['sizes'].append(df_net['length'].iloc[i])
            pkt_bps['iats'].append(df_net['time_normed'].iloc[i] - prev_pkt_time)
            prev_pkt_time = df_net['time_normed'].iloc[i]
        df_agg = pd.DataFrame(pkt_bps)
        df_agg = df_agg.groupby('interval').agg({'sizes': list, 'iats': list})
        size_iat = []
        for idx, row in df_agg.iterrows():
            size_list = row['sizes']
            iat_list = row['iats']
            f = []
            for i in range(len(size_list)):
                f.append(size_list[i])
                f.append(iat_list[i])
            self.max_size = max(self.max_size, len(f))
            if len(f) < n_features:
                f += [-1]*(n_features-len(f))
            size_iat.append(f[:n_features])
        col_names = [f'f_{i}' for i in range(1, n_features+1)]
        df_bps = pd.DataFrame(size_iat, columns=col_names)
        df_bps['et'] = 0.0
        df_bps.loc[:, 'et'] = end_times
        df_bps['et'] = df_bps['et'].apply(lambda x: int(x)+1)
        return df_bps




