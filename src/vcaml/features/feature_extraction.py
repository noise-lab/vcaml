import logging
from collections import defaultdict

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


class FeatureExtractor:

    def __init__(self, featureSubset, config, vca, dataset):
        self.featureSubset = featureSubset
        self.config = config
        self.vca = vca
        self.maxSize = -1
        self.maxIat = -1
        self.dataset = dataset
        logger.debug('FeatureExtractor  vca=%s  window=%ss', vca, config['prediction_window'])

    # ------------------------------------------------------------------ helpers

    def _buildIntervals(self, timeNormed: pd.Series, timeEpoch: pd.Series):
        """Assign each packet to a 1-second window.

        Returns (intervalIds, endTimes) where endTimes[k] is the epoch timestamp
        of the first packet that crossed into window k+1 (representative end time
        for joining against WebRTC labels).
        """
        windowSize = self.config['prediction_window']
        intervalIds, endTimes = [], []
        prev = timeNormed.iloc[0]
        currentInterval = 1
        n = len(timeNormed)
        for i in range(n):
            if timeNormed.iloc[i] - prev > windowSize:
                currentInterval += 1
                prev = timeNormed.iloc[i]
                endTimes.append(timeEpoch.iloc[i])
            if i == n - 1:
                endTimes.append(timeEpoch.iloc[i])
            intervalIds.append(currentInterval)
        return intervalIds, endTimes

    def _appendStats(self, featureData: dict, colName: str, arr, defaultVal):
        """Append per-window summary statistics for one array into featureData."""
        arr = np.array(arr)
        if len(arr) > 0:
            q1, q2, q3 = np.percentile(arr, [25, 50, 75])
            featureData[f'{colName}_min'].append(float(np.min(arr)))
            featureData[f'{colName}_max'].append(float(np.max(arr)))
            featureData[f'{colName}_q1'].append(float(q1))
            featureData[f'{colName}_q2'].append(float(q2))
            featureData[f'{colName}_q3'].append(float(q3))
            featureData[f'{colName}_mean'].append(float(np.mean(arr)))
            featureData[f'{colName}_std'].append(float(np.std(arr)))
        else:
            for suffix in ('_min', '_max', '_q1', '_q2', '_q3', '_mean', '_std'):
                featureData[f'{colName}{suffix}'].append(defaultVal)
        return featureData

    @staticmethod
    def _newRtpWindowState() -> dict:
        return {
            'vidTimestamps': set(),
            'rtxTimestamps': set(),
            'vidMarkerCount': 0,
            'rtxMarkerCount': 0,
            'oooSeqCount': 0,
            'minPktIdx': defaultdict(lambda: 10 ** 9),
            'maxPktIdx': defaultdict(lambda: -1),
            'lags': {},
        }

    @staticmethod
    def _emptyRtpFeatures() -> dict:
        return {
            'vid_ts_unique': [], 'rtx_ts_unique': [], 'vid_marker_sum': [],
            'rtx_marker_sum': [], 'common_vid_rtx_ts_unique': [],
            'union_ts_unique': [], 'ooo_seqno_vid': [],
            'buffer_time_mean': [], 'buffer_time_std': [], 'buffer_time_min': [],
            'buffer_time_max': [], 'buffer_time_q1': [], 'buffer_time_q2': [],
            'buffer_time_q3': [], 'n_pkt_diff_mean': [], 'n_pkt_diff_std': [],
            'n_pkt_diff_min': [], 'n_pkt_diff_max': [], 'n_pkt_diff_q1': [],
            'n_pkt_diff_q2': [], 'n_pkt_diff_q3': [], 'rtp_lag_mean': [],
            'rtp_lag_std': [], 'rtp_lag_min': [], 'rtp_lag_max': [],
            'rtp_lag_q1': [], 'rtp_lag_q2': [], 'rtp_lag_q3': [], 'et': [],
        }

    def _buildRtpLookups(self, df_net: pd.DataFrame) -> dict:
        """Precompute per-RTP-timestamp metadata used during the main packet loop."""
        ft = (df_net.groupby('rtp.timestamp')
              .agg(timeNormedMax=('time_normed', 'max'))
              .reset_index()
              .sort_values('rtp.timestamp'))
        lastPktIdx = {ts: i for i, ts in enumerate(df_net['rtp.timestamp'])}
        return {
            'tsToMaxTime': dict(zip(ft['rtp.timestamp'], ft['timeNormedMax'])),
            'rtp0': ft['rtp.timestamp'].iloc[0],
            't0': ft['timeNormedMax'].iloc[0],
            'lastPktIdx': lastPktIdx,
        }

    def _updateRtpState(self, state: dict, packetIdx: int, row: pd.Series,
                        lookups: dict, videoPtypes: set, rtxPtypes: set,
                        prevSeq) -> None:
        """Update per-window accumulators for one incoming packet (in-place)."""
        ts = row['rtp.timestamp']
        state['minPktIdx'][ts] = min(state['minPktIdx'][ts], packetIdx)
        state['maxPktIdx'][ts] = max(state['maxPktIdx'][ts], packetIdx)

        if lookups['lastPktIdx'][ts] == packetIdx:
            actualDur = lookups['tsToMaxTime'][ts] - lookups['t0']
            expectedDur = (ts - lookups['rtp0']) / 90000
            state['lags'][ts] = actualDur - expectedDur

        ptype = row['rtp.p_type']
        if ptype in videoPtypes:
            state['vidTimestamps'].add(ts)
            state['vidMarkerCount'] += row['rtp.marker']
            if prevSeq is not None and row['rtp.seq'] - prevSeq != 1:
                state['oooSeqCount'] += 1
        if ptype in rtxPtypes:
            state['rtxTimestamps'].add(ts)
            state['rtxMarkerCount'] += row['rtp.marker']

    def _flushRtpWindow(self, featureData: dict, state: dict, et: float) -> dict:
        """Append one closed window's RTP features into featureData."""
        allTs = state['vidTimestamps'].union(state['rtxTimestamps'])
        bufferTimes = 90 / np.diff(np.array(sorted(allTs)))
        pktSpans = [state['maxPktIdx'][ts] - state['minPktIdx'][ts]
                    for ts in state['minPktIdx']]
        featureData = self._appendStats(featureData, 'buffer_time', bufferTimes, 0)
        featureData['vid_ts_unique'].append(len(state['vidTimestamps']))
        featureData['rtx_ts_unique'].append(len(state['rtxTimestamps']))
        featureData['common_vid_rtx_ts_unique'].append(
            len(state['rtxTimestamps'].intersection(state['vidTimestamps'])))
        featureData['union_ts_unique'].append(len(allTs))
        featureData['vid_marker_sum'].append(state['vidMarkerCount'])
        featureData['rtx_marker_sum'].append(state['rtxMarkerCount'])
        featureData['ooo_seqno_vid'].append(state['oooSeqCount'])
        featureData = self._appendStats(featureData, 'n_pkt_diff', pktSpans, -1)
        featureData = self._appendStats(
            featureData, 'rtp_lag', list(state['lags'].values()), 100000)
        featureData['et'].append(et)
        return featureData

    # ------------------------------------------------------------------ public

    def extract_features(self, df_net: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for featureType in self.featureSubset:
            if featureType == 'SIZE':
                parts.append(self._extractSizeFeatures(df_net))
            elif featureType == 'IAT':
                parts.append(self._extractIatFeatures(df_net))
            elif featureType == 'LSTATS':
                parts.append(self._extractLengthStatFeatures(df_net))
            elif featureType == 'TSTATS':
                parts.append(self._extractIatStatFeatures(df_net))

        for j in range(1, len(parts)):
            parts[j] = parts[j][parts[j].columns.difference(['et'])]
        return pd.concat(parts, axis=1)

    def extract_rtp_features(self, df_net: pd.DataFrame) -> pd.DataFrame:
        df_net = df_net.copy()
        df_net['rtp.timestamp'] = df_net['rtp.timestamp'].astype(float)
        df_net['rtp.seq'] = df_net['rtp.seq'].astype(int)

        lookups = self._buildRtpLookups(df_net)
        videoPtypes = set(self.config['video_ptype'][self.dataset][self.vca])
        rtxPtypes = set(self.config['rtx_ptype'][self.dataset][self.vca])
        windowSize = self.config['prediction_window']

        featureData = self._emptyRtpFeatures()
        state = self._newRtpWindowState()
        prevTime = df_net['time_normed'].iloc[0]
        n = df_net.shape[0]

        for i in range(n):
            row = df_net.iloc[i]
            prevSeq = df_net.iloc[i - 1]['rtp.seq'] if i > 0 else None
            self._updateRtpState(state, i, row, lookups, videoPtypes, rtxPtypes, prevSeq)

            if row['time_normed'] - prevTime > windowSize:
                if state['vidTimestamps'] or state['rtxTimestamps']:
                    featureData = self._flushRtpWindow(featureData, state, row['time'])
                    state = self._newRtpWindowState()
                prevTime = row['time_normed']

        if state['vidTimestamps'] or state['rtxTimestamps']:
            featureData = self._flushRtpWindow(
                featureData, state, df_net.iloc[n - 1]['time'])

        df = pd.DataFrame(featureData)
        df['et'] = df['et'].apply(int)
        return df

    # ------------------------------------------------------------------ private

    def _extractSizeFeatures(self, df_net: pd.DataFrame) -> pd.DataFrame:
        nFeatures = self.config['n_features_size'][self.vca]
        intervalIds, endTimes = self._buildIntervals(df_net['time_normed'], df_net['time'])
        df_agg = (pd.DataFrame({'interval': intervalIds, 'sizes': df_net['length'].tolist()})
                  .groupby('interval')
                  .agg({'sizes': list}))
        for idx, row in df_agg.iterrows():
            sl = row['sizes']
            self.maxSize = max(self.maxSize, len(sl))
            df_agg.at[idx, 'sizes'] = sl[:nFeatures] + [0] * max(0, nFeatures - len(sl))
        colNames = [f'size_{i}' for i in range(1, nFeatures + 1)]
        dfOut = pd.DataFrame(df_agg['sizes'].tolist(), columns=colNames)
        dfOut['et'] = [int(t) for t in endTimes]
        return dfOut

    def _extractIatFeatures(self, df_net: pd.DataFrame) -> pd.DataFrame:
        nFeatures = self.config['n_features_iat'][self.vca]
        tn = df_net['time_normed']
        iats = [tn.iloc[0] * 1000] + [
            (tn.iloc[i] - tn.iloc[i - 1]) * 1000 for i in range(1, len(tn))]
        intervalIds, endTimes = self._buildIntervals(tn, df_net['time'])
        df_agg = (pd.DataFrame({'interval': intervalIds, 'iats': iats})
                  .groupby('interval')
                  .agg({'iats': list}))
        for idx, row in df_agg.iterrows():
            il = row['iats']
            self.maxIat = max(self.maxIat, len(il))
            df_agg.at[idx, 'iats'] = il[:nFeatures] + [0] * max(0, nFeatures - len(il))
        colNames = [f'iat_{i}' for i in range(1, nFeatures + 1)]
        dfOut = pd.DataFrame(df_agg['iats'].tolist(), columns=colNames)
        dfOut['et'] = [int(t) for t in endTimes]
        return dfOut

    def _extractLengthStatFeatures(self, df_net: pd.DataFrame) -> pd.DataFrame:
        intervalIds, endTimes = self._buildIntervals(df_net['time_normed'], df_net['time'])
        df_agg = (pd.DataFrame({'interval': intervalIds, 'sizes': df_net['length'].tolist()})
                  .groupby('interval')
                  .agg({'sizes': list}))
        rows = []
        for _, row in df_agg.iterrows():
            sl = np.array(row['sizes'])
            q1, q2, q3 = np.quantile(sl, [0.25, 0.5, 0.75])
            rows.append({
                'l_mean': sl.mean(), 'l_std': sl.std(),
                'l_min': sl.min(), 'l_max': sl.max(),
                'l_q1': q1, 'l_q2': q2, 'l_q3': q3,
                'l_num_pkts': len(sl), 'l_num_bytes': int(sl.sum()),
                'l_num_unique': len(set(sl.tolist())),
            })
        dfOut = pd.DataFrame(rows)
        dfOut['et'] = [int(t) for t in endTimes]
        return dfOut

    def _extractIatStatFeatures(self, df_net: pd.DataFrame) -> pd.DataFrame:
        tn = df_net['time_normed']
        iats = [tn.iloc[0] * 1000] + [
            (tn.iloc[i] - tn.iloc[i - 1]) * 1000 for i in range(1, len(tn))]
        intervalIds, endTimes = self._buildIntervals(tn, df_net['time'])
        df_agg = (pd.DataFrame({'interval': intervalIds, 'iats': iats})
                  .groupby('interval')
                  .agg({'iats': list}))
        rows = []
        for _, row in df_agg.iterrows():
            sl = np.array(row['iats'])
            q1, q2, q3 = np.quantile(sl, [0.25, 0.5, 0.75])
            burstCount = int((sl >= 30).sum()) if len(sl) > 1 else 0
            rows.append({
                't_mean': sl.mean(), 't_std': sl.std(),
                't_min': sl.min(), 't_max': sl.max(),
                't_q1': q1, 't_q2': q2, 't_q3': q3,
                't_burst_count': burstCount,
            })
        dfOut = pd.DataFrame(rows)
        dfOut['et'] = [int(t) for t in endTimes]
        return dfOut
