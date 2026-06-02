import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from vcaml.models.base_ml_estimator import BaseMLEstimator
from vcaml.pipeline.net_utils import readIpUdpFile, renameNetColumns


class IP_UDP_ML(BaseMLEstimator):

    @property
    def modelTag(self):
        return 'ip-udp-ml'

    def _loadAndPrepareFile(self, csvFile):
        df = readIpUdpFile(csvFile)
        if df is None:
            return None
        df = renameNetColumns(df)
        return df.sort_values('time_normed')[['length', 'time', 'time_normed']]

    def _extractFeatures(self, df_net, featureExtractor):
        return featureExtractor.extract_features(df_net=df_net)
