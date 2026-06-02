import pandas as pd

from .net_utils import filter_ptype
from vcaml.io.webrtc_reader import WebRTCReader


class FileValidator:

    def __init__(self, fileTuple, config, dataset):
        self.fileTuple = fileTuple
        self.config = config
        self.dataset = dataset

    def validate(self) -> bool:
        csvFile = self.fileTuple[0]
        webrtcFile = self.fileTuple[1]

        df_net = pd.read_csv(csvFile)
        try:
            dstIp = (df_net.groupby('ip.dst')
                     .agg({'udp.length': 'sum'})
                     .reset_index()
                     .sort_values(by='udp.length', ascending=False)
                     .head(1)['ip.dst'].iloc[0])
        except (IndexError, KeyError):
            return False

        df_net = df_net[(df_net['ip.dst'] == dstIp) & (~pd.isna(df_net['rtp.ssrc']))]
        df_net['rtp.p_type'] = df_net['rtp.p_type'].apply(filter_ptype)
        df_net['ip.proto'] = df_net['ip.proto'].astype(str)
        df_net = df_net[~df_net['ip.proto'].str.contains(',')]
        df_net = df_net[~df_net['ip.proto'].isna()]
        try:
            df_net['ip.proto'] = df_net['ip.proto'].apply(lambda x: int(float(x)))
        except ValueError:
            return False

        df_videoPackets = df_net[(df_net['ip.proto'] == 17) & (df_net['udp.length'] > 306)]
        if len(df_videoPackets) == 0:
            return False

        df_rtpPackets = df_net[~pd.isna(df_net['rtp.p_type'])]
        if len(df_net) == 0 or len(df_rtpPackets) == 0:
            return False

        webrtcReader = WebRTCReader(webrtcFile=webrtcFile, dataset=self.dataset)
        df_webrtc = webrtcReader.get_webrtc()

        if 'framesPerSecond' not in df_webrtc.columns:
            return False

        webrtcTsMin = df_webrtc['ts'].min()
        webrtcTsMax = df_webrtc['ts'].max()
        pcapTsMin = df_rtpPackets['frame.time_epoch'].min()
        pcapTsMax = df_rtpPackets['frame.time_epoch'].max()
        if webrtcTsMax < pcapTsMin or pcapTsMax < webrtcTsMin:
            return False

        if int(df_webrtc['duration'].max()) > df_webrtc['num_vals'].max():
            return False

        return True
