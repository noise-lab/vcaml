import logging
import sys
import warnings
from os.path import abspath, dirname

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

from util.helper_functions import get_net_stats, read_net_file
from util.webrtc_reader import WebRTCReader

logger = logging.getLogger(__name__)


class RTP_Heuristic:

    def __init__(self, vca, metric, config, dataset):
        self.metric = metric
        self.vca = vca
        self.config = config
        self.dataset = dataset

    def estimate(self, fileTuple):
        csvFile, webrtcFile = fileTuple[0], fileTuple[1]
        logger.debug('Estimating: %s', csvFile)
        df = read_net_file(self.dataset, csvFile)
        if df is None:
            return None
        videoPtypes = self.config['video_ptype'][self.dataset][self.vca]
        rtxPtypes = self.config['rtx_ptype'][self.dataset][self.vca]
        df = df.sort_values('frame.time_relative')
        df['frame.time_relative'] = df['frame.time_relative'].astype(float)
        df = df[df['rtp.p_type'].isin([*videoPtypes, *rtxPtypes])]
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
        df = df[df['udp.length'] > 306]
        df['udp.length'] = df['udp.length'] - 12

        # Group packets by RTP timestamp → one row per frame
        df_frames = (df.groupby('rtp.timestamp')
                     .agg({'frame.time_epoch': list, 'udp.length': list,
                           'rtp.seq': list, 'rtp.marker': list})
                     .reset_index())
        # Keep only complete frames (exactly one marker bit set)
        df_frames['is_valid'] = df_frames['rtp.marker'].apply(lambda x: int(sum(x)))
        df_frames = df_frames[df_frames['is_valid'] == 1]
        df_frames['frame_st'] = df_frames['frame.time_epoch'].apply(min)
        df_frames['frame_et'] = df_frames['frame.time_epoch'].apply(max)
        df_frames['frame_size'] = df_frames['udp.length'].apply(sum)
        df_frames = get_net_stats(df_video=df_frames)

        webrtcReader = WebRTCReader(webrtcFile, self.dataset)
        df_webrtc = webrtcReader.get_webrtc()[[self.metric, 'ts']]
        df_merged = pd.merge(df_frames, df_webrtc, left_on='frame_et_int', right_on='ts')
        metricCol = f'{self.metric}_rtp-heuristic'
        gtCol = f'{self.metric}_gt'
        df_merged = df_merged.rename(columns={
            f'predicted_{self.metric}': metricCol,
            self.metric: gtCol,
            'frame_et_int': 'timestamp',
        })
        df_merged['file'] = csvFile
        df_merged['dataset'] = self.dataset
        df_merged = df_merged[[gtCol, metricCol, 'timestamp', 'file', 'dataset']].dropna()
        if self.metric == 'framesReceivedPerSecond':
            df_merged[gtCol] = df_merged[gtCol].apply(round)
        return df_merged
