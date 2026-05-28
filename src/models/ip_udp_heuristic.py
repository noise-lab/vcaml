import logging
import sys
import warnings
from os.path import abspath, dirname

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

from util.helper_functions import get_freeze_dur, is_freeze
from util.webrtc_reader import WebRTCReader

logger = logging.getLogger(__name__)


class IP_UDP_Heuristic:

    def __init__(self, vca, metric, config, dataset):
        self.intraTolerance = 2
        self.config = config
        self.vca = vca
        self.metric = metric
        self.maxLookback = {'meet': 3, 'teams': 2, 'webex': 1}
        self.dataset = dataset

    def _assignFrameIds(self, df, vca):
        lookback = self.maxLookback[vca]
        frameIds = [-1] * df.shape[0]
        frameId = 0
        for i in range(df.shape[0]):
            pktLen = df.iloc[i]['udp.length']
            matched = False
            for j in range(i - 1, max(0, i - lookback - 1), -1):
                if abs(df.iloc[j]['udp.length'] - pktLen) <= self.intraTolerance:
                    frameIds[i] = frameId
                    matched = True
                    break
            if not matched:
                frameId += 1
                frameIds[i] = frameId
        return frameIds

    def estimate(self, fileTuple):
        csvFile, webrtcFile = fileTuple[0], fileTuple[1]
        logger.debug('Estimating: %s', csvFile)
        df = pd.read_csv(csvFile)
        df = df[~df['ip.proto'].isna()]
        df['ip.proto'] = df['ip.proto'].astype(str)
        df = df[df['ip.proto'].str.contains(',') == False]
        df['ip.proto'] = df['ip.proto'].apply(lambda x: int(float(x)))
        try:
            dstIp = (df.groupby('ip.dst').agg({'udp.length': 'sum'})
                     .reset_index()
                     .sort_values(by='udp.length', ascending=False)
                     .head(1)['ip.dst'].iloc[0])
        except IndexError:
            logger.warning('Faulty trace, skipping: %s', csvFile)
            return None
        df = df[(df['ip.dst'] == dstIp) & (df['ip.proto'] == 17)]
        try:
            dstIp = (df.groupby('ip.dst')
                     .agg({'udp.length': 'sum', 'rtp.p_type': 'count'})
                     .reset_index()
                     .sort_values(by='udp.length', ascending=False)
                     .head(1)['ip.dst'].iloc[0])
            df = df[df['ip.dst'] == dstIp]
        except IndexError:
            logger.warning('Faulty trace, skipping: %s', csvFile)
            return None
        df = df[df['udp.length'] > 306].sort_values('frame.time_relative')
        df['frame_num'] = self._assignFrameIds(df, self.vca)
        df['udp.length'] = df['udp.length'] - 12

        df_frames = (df.groupby('frame_num')
                     .agg({'frame.time_epoch': list, 'udp.length': list})
                     .reset_index())
        df_frames['frame_st'] = df_frames['frame.time_epoch'].apply(min)
        df_frames['frame_et'] = df_frames['frame.time_epoch'].apply(max)
        df_frames['frame_size'] = df_frames['udp.length'].apply(sum)
        df_frames['ft_end'] = df_frames['frame_et'].apply(lambda x: int(x) + 1)
        df_frames['frame_dur'] = df_frames['frame_et'].diff()
        df_frames['avg_frame_dur'] = df_frames['frame_dur'].rolling(30).mean()
        df_frames = df_frames.fillna(0)

        # Drop leading frames with huge gaps (likely pre-call silence)
        longGapIdx = df_frames.index[df_frames['frame_dur'] >= 8].tolist()
        df_frames = df_frames.iloc[longGapIdx[0] + 1 if longGapIdx else 0:]

        df_frames['is_freeze'] = df_frames.apply(is_freeze, axis=1)
        df_frames['freeze_dur'] = df_frames.apply(get_freeze_dur, axis=1)
        df_frames = (df_frames.groupby('ft_end')
                     .agg({'frame_size': ['count', 'sum'], 'is_freeze': 'sum',
                           'freeze_dur': 'sum', 'frame_dur': 'std'})
                     .reset_index())
        df_frames.columns = ['_'.join(col).strip('_') for col in df_frames.columns.values]
        df_frames = df_frames.rename(columns={
            'frame_size_count': 'predicted_framesReceivedPerSecond',
            'is_freeze_sum': 'freeze_count',
            'frame_size_sum': 'predicted_bitrate',
            'freeze_dur_sum': 'freeze_dur',
            'frame_dur_std': 'predicted_frame_jitter',
        })
        df_frames['predicted_bitrate'] *= 8
        df_frames['predicted_frame_jitter'] *= 1000

        webrtcReader = WebRTCReader(webrtcFile, self.dataset)
        df_webrtc = webrtcReader.get_webrtc()[[self.metric, 'ts']]
        df_merged = pd.merge(df_frames, df_webrtc, left_on='ft_end', right_on='ts')
        metricCol = f'{self.metric}_ip-udp-heuristic'
        gtCol = f'{self.metric}_gt'
        df_merged = df_merged.rename(columns={
            f'predicted_{self.metric}': metricCol,
            self.metric: gtCol,
            'ts': 'timestamp',
        })
        df_merged['file'] = csvFile
        df_merged['dataset'] = self.dataset
        df_merged = df_merged[[gtCol, metricCol, 'timestamp', 'file', 'dataset']].dropna()
        if df_merged.shape[0] == 0:
            return None
        if self.metric == 'framesReceivedPerSecond':
            df_merged[gtCol] = df_merged[gtCol].apply(round)
        return df_merged
