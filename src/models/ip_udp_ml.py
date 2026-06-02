import sys
import warnings
from os.path import abspath, dirname

warnings.simplefilter(action='ignore', category=FutureWarning)

d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

from models.base_ml_estimator import BaseMLEstimator
from util.helper_functions import readIpUdpFile, renameNetColumns


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
