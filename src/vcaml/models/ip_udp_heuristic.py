import logging
import warnings

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

from vcaml.util.helper_functions import aggregateFrameStats, get_freeze_dur, is_freeze, readIpUdpFile
from vcaml.util.webrtc_reader import WebRTCReader

logger = logging.getLogger(__name__)


class IP_UDP_Heuristic:

    def __init__(self, vca, metric, config, dataset):
        self.intraTolerance = config['intra_tolerance']
        self.config = config
        self.vca = vca
        self.metric = metric
        self.maxLookback = config['max_lookback']
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
        df = readIpUdpFile(csvFile)
        if df is None:
            logger.warning('Faulty trace, skipping: %s', csvFile)
            return None
        df = df.sort_values('frame.time_relative')
        df['frame_num'] = self._assignFrameIds(df, self.vca)
        df['udp.length'] = df['udp.length'] - self.config['rtp_header_size']

        df_frames = (df.groupby('frame_num')
                     .agg({'frame.time_epoch': list, 'udp.length': list})
                     .reset_index())
        df_frames['frame_st'] = df_frames['frame.time_epoch'].apply(min)
        df_frames['frame_et'] = df_frames['frame.time_epoch'].apply(max)
        df_frames['frame_size'] = df_frames['udp.length'].apply(sum)
        df_frames['ft_end'] = df_frames['frame_et'].apply(lambda x: int(x) + 1)
        df_frames['frame_dur'] = df_frames['frame_et'].diff()
        df_frames['avg_frame_dur'] = df_frames['frame_dur'].rolling(self.config['freeze_window']).mean()
        df_frames = df_frames.fillna(0)

        # Drop leading frames with huge gaps (likely pre-call silence)
        longGapIdx = df_frames.index[df_frames['frame_dur'] >= self.config['long_gap_threshold']].tolist()
        df_frames = df_frames.iloc[longGapIdx[0] + 1 if longGapIdx else 0:]

        df_frames['is_freeze'] = df_frames.apply(is_freeze, axis=1)
        df_frames['freeze_dur'] = df_frames.apply(get_freeze_dur, axis=1)
        df_frames = aggregateFrameStats(df_frames, groupCol='ft_end')

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
