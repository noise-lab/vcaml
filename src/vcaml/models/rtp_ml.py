import warnings

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

from vcaml.models.base_ml_estimator import BaseMLEstimator
from vcaml.pipeline.net_utils import read_net_file, renameNetColumns, selectDstIp


class RTP_ML(BaseMLEstimator):

    @property
    def modelTag(self):
        return 'rtp-ml'

    def _loadAndPrepareFile(self, csvFile):
        df = read_net_file(self.dataset, csvFile)
        if df is None:
            return None
        videoPtypes = self.config['video_ptype'][self.dataset][self.vca]
        rtxPtypes = self.config['rtx_ptype'][self.dataset][self.vca]
        df = df[
            (df['rtp.p_type'].isin(videoPtypes)) |
            ((df['rtp.p_type'].isin(rtxPtypes)) & (df['udp.length'] > self.config['video_thresh']))
        ]
        df = selectDstIp(df)
        if df is None:
            return None
        df = renameNetColumns(df)
        return df.sort_values('time_normed')[
            ['length', 'time', 'time_normed',
             'rtp.timestamp', 'rtp.seq', 'rtp.marker', 'rtp.p_type']]

    def _extractFeatures(self, df_net, featureExtractor):
        df_features = featureExtractor.extract_features(df_net=df_net)
        df_rtp = featureExtractor.extract_rtp_features(df_net=df_net)
        return pd.merge(df_features, df_rtp, on='et')
