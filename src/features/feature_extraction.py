import pandas as pd
import numpy as np
from collections import defaultdict
import bisect
import itertools
pd.options.mode.chained_assignment = None


class FeatureExtractor:

    def __init__(self, feature_subset, config, vca, dataset):
        self.feature_subset = feature_subset
        self.config = config
        self.vca = vca
        print('prediction window =', self.config['prediction_window'])
        # self.video_ptype = {'meet': ['98'], 'teams': ['102'], 'webex': ['102']}
        # self.rtx_ptype = {'meet': ['99'], 'teams': ['123'], 'webex': [None]}
        self.max_size = -1
        self.max_iat = -1
        self.dataset = dataset
    
    def calculate_stats(self, feature_data, col_name, arr, default_val):
        arr = np.array(arr)
        if len(arr) > 0: 
            feature_data[f'{col_name}_min'].append(np.min(arr))
            feature_data[f'{col_name}_max'].append(np.max(arr))
            feature_data[f'{col_name}_q1'].append(np.percentile(arr, 25))
            feature_data[f'{col_name}_q2'].append(np.percentile(arr, 50))
            feature_data[f'{col_name}_q3'].append(np.percentile(arr, 75))
            feature_data[f'{col_name}_mean'].append(np.mean(arr))
            feature_data[f'{col_name}_std'].append(np.std(arr))
        else:
            feature_data[f'{col_name}_min'].append(default_val)
            feature_data[f'{col_name}_max'].append(default_val)
            feature_data[f'{col_name}_q1'].append(default_val)
            feature_data[f'{col_name}_q2'].append(default_val)
            feature_data[f'{col_name}_q3'].append(default_val)
            feature_data[f'{col_name}_mean'].append(default_val)
            feature_data[f'{col_name}_std'].append(default_val)

        return feature_data

    def extract_rtp_features(self, df_net):
        df_net['rtp.timestamp'] = df_net['rtp.timestamp'].astype(float)
        df_net['rank'] = list(range(df_net.shape[0]))
        df_net['rtp.seq'] = df_net['rtp.seq'].astype(int)
        feature_data = {'vid_ts_unique': [], 'rtx_ts_unique': [], 'vid_marker_sum': [], 'rtx_marker_sum': [], 'common_vid_rtx_ts_unique': [], 'union_ts_unique': [], 'ooo_seqno_vid': [], 'buffer_time_mean': [], 'buffer_time_std': [], 'buffer_time_min': [], 'buffer_time_max': [], 'buffer_time_q1': [], 'buffer_time_q2': [], 'buffer_time_q3': [], 'n_pkt_diff_mean': [],  'n_pkt_diff_std': [],'n_pkt_diff_min': [],'n_pkt_diff_max': [],'n_pkt_diff_q1': [],'n_pkt_diff_q2': [],'n_pkt_diff_q3': [], 'rtp_lag_mean': [],  'rtp_lag_std': [],'rtp_lag_min': [],'rtp_lag_max': [],'rtp_lag_q1': [],'rtp_lag_q2': [],'rtp_lag_q3': [],'et': []}
        prev = df_net['time_normed'].iloc[0]
        uniq_vid_ts = set()
        uniq_rtx_ts = set()
        uniq_common_ts = set()
        uniq_union_ts = set()
        vid_marker_sum = 0 
        rtx_marker_sum = 0 
        ooo_seqno_vid = 0
        min_rank = defaultdict(lambda: 1000000)
        max_rank = defaultdict(lambda: -1)
        window_size = self.config['prediction_window']
        ft = df_net.groupby('rtp.timestamp').agg(time_normed_min = ('time_normed', 'min'), time_normed_max = ('time_normed', 'max')).reset_index()
        ft = ft.sort_values(by='rtp.timestamp')
        rtp2idx = {ft['rtp.timestamp'][i]: i for i in range(ft.shape[0])}
        rtp0 = ft['rtp.timestamp'][0]
        t0 = ft.iloc[0]['time_normed_max']
        last_pkt = defaultdict(lambda : 0)
        lags = defaultdict(lambda: 0)
        idx = 0
        for j, row in df_net.iterrows():
            last_pkt[row['rtp.timestamp']] = idx
            idx += 1
        for i in range(df_net.shape[0]):
            curr = df_net.iloc[i]['rtp.timestamp']
            ridx = rtp2idx[curr]
            if last_pkt[curr] == i:
                actual_dur = ft[ft['rtp.timestamp'] == curr]['time_normed_max'].iloc[0] - t0
                expected_dur = (curr - rtp0) / 90000
                lags[curr] = actual_dur - expected_dur
            if df_net.iloc[i]['rtp.p_type'] in self.config['video_ptype'][self.dataset][self.vca]:
                uniq_vid_ts.add(df_net.iloc[i]['rtp.timestamp'])
                vid_marker_sum += df_net.iloc[i]['rtp.marker']
                if i > 0 and (df_net.iloc[i]['rtp.seq'] - df_net.iloc[i-1]['rtp.seq'] != 1):
                    ooo_seqno_vid += 1
            if df_net.iloc[i]['rtp.p_type'] in self.config['rtx_ptype'][self.dataset][self.vca]:
                uniq_rtx_ts.add(df_net.iloc[i]['rtp.timestamp'])
                rtx_marker_sum += df_net.iloc[i]['rtp.marker']

            if df_net['time_normed'].iloc[i] - prev > window_size:
                if len(uniq_vid_ts) > 0 or len(uniq_rtx_ts) > 0:
                    btime = 90 / np.diff(np.array(sorted(list(uniq_vid_ts.union(uniq_rtx_ts)))))
                    rank_arr = []
                    for ts, rank in min_rank.items():
                        d = max_rank[ts] - rank
                        rank_arr.append(d)
                    feature_data = self.calculate_stats(feature_data, 'buffer_time', btime, 0)
                    lag_arr = []
                    for l in lags:
                        lag_arr.append(lags[l])
                    feature_data['vid_ts_unique'].append(len(uniq_vid_ts))
                    feature_data['rtx_ts_unique'].append(len(uniq_rtx_ts))
                    feature_data['common_vid_rtx_ts_unique'].append(len(uniq_rtx_ts.intersection(uniq_vid_ts)))
                    feature_data['union_ts_unique'].append(len(uniq_rtx_ts.union(uniq_vid_ts)))
                    feature_data['vid_marker_sum'].append(vid_marker_sum)
                    feature_data['rtx_marker_sum'].append(rtx_marker_sum)
                    feature_data['ooo_seqno_vid'].append(ooo_seqno_vid)
                    
                    feature_data = self.calculate_stats(feature_data, 'n_pkt_diff', rank_arr, -1)
                    feature_data = self.calculate_stats(feature_data, 'rtp_lag', lag_arr, 100000)
                    
                    feature_data['et'].append(df_net.iloc[i]['time'])
                    uniq_vid_ts = set()
                    uniq_rtx_ts = set()
                    uniq_common_ts = set()
                    vid_marker_sum = 0 
                    rtx_marker_sum = 0 
                    ooo_seqno_vid = 0
                    min_rank = defaultdict(lambda: 1000000)
                    max_rank = defaultdict(lambda: -1)
                    lags = defaultdict(lambda: 0)

                prev = df_net['time_normed'].iloc[i]
        if len(uniq_vid_ts) > 0 or len(uniq_rtx_ts) > 0:
            btime = 90 / np.diff(np.array(sorted(list(uniq_vid_ts.union(uniq_rtx_ts)))))
            rank_arr = []
            for ts, rank in min_rank.items():
                d = max_rank[ts] - rank
                rank_arr.append(d)
            feature_data['vid_ts_unique'].append(len(uniq_vid_ts))
            feature_data['rtx_ts_unique'].append(len(uniq_rtx_ts))
            feature_data['common_vid_rtx_ts_unique'].append(len(uniq_rtx_ts.intersection(uniq_vid_ts)))
            feature_data['union_ts_unique'].append(len(uniq_rtx_ts.union(uniq_vid_ts)))
            feature_data['vid_marker_sum'].append(vid_marker_sum)
            feature_data['rtx_marker_sum'].append(rtx_marker_sum)
            feature_data['ooo_seqno_vid'].append(ooo_seqno_vid)
            feature_data = self.calculate_stats(feature_data, 'buffer_time', btime, 0)
            feature_data = self.calculate_stats(feature_data, 'n_pkt_diff', rank_arr, -1)
            lag_arr = []
            for l in lags:
                lag_arr.append(lags[l])
            feature_data = self.calculate_stats(feature_data, 'rtp_lag', lag_arr, 100000)
            feature_data['et'].append(df_net.iloc[i]['time'])
        df = pd.DataFrame(feature_data)
        df['et'] = df['et'].apply(lambda x: int(x))
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
                
        def calculate_burst_count(x):
            x = np.array(x)
            if len(x) <= 1:
                return 0
            mask = x >= 30
            return mask.sum()
        
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
        df_agg['burst_count'] = df_agg['iats'].apply(lambda x: calculate_burst_count(x))
        df_stats = df_agg[['mean', 'std', 'min', 'max','q1', 'q2', 'q3', 'burst_count']]
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
