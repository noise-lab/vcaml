import pandas as pd
from .webrtc_reader import WebRTCReader
from .helper_functions import filter_ptype
import ast


class FileValidator:
    def __init__(self, file_tuple, config, dataset):
        self.file_tuple = file_tuple
        # Wireshark fields
        self.net_columns = ['frame.time_relative', 'frame.time_epoch', 'ip.src', 'ip.dst', 'ip.proto', 'ip.len',
                            'udp.srcport', 'udp.dstport', 'udp.length', 'rtp.ssrc', 'rtp.timestamp', 'rtp.seq', 'rtp.p_type', 'rtp.marker']
        self.config = config
        self.dataset = dataset

    def validate(self):
        csv_file = self.file_tuple[0]
        webrtc_file = self.file_tuple[1]
        print(f'Validating {csv_file} and {webrtc_file}')
        df_net = pd.read_csv(csv_file)
        ip_addr = df_net.groupby('ip.dst').agg({'udp.length': sum}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
        df_net = df_net[(df_net["ip.dst"] == ip_addr) &
                        (~pd.isna(df_net["rtp.ssrc"]))]
        df_net['rtp.p_type'] = df_net['rtp.p_type'].apply(filter_ptype)
        df_net['rtp.p_type'] = df_net['rtp.p_type'].apply(filter_ptype)
        df_net['ip.proto'] = df_net['ip.proto'].astype(str)
        df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
        df_net = df_net[~df_net['ip.proto'].isna()]
        try:
            df_net['ip.proto'] = df_net['ip.proto'].apply(lambda x: int(float(x)))
        except ValueError:
            print('Malformed PCAP..')
            return False
        df_video = df_net[(df_net['ip.proto'] == 17) & (
            df_net['ip.dst'] == ip_addr) & (df_net['udp.length'] > 306)]
        if len(df_video) == 0:
            print('No video packets found...')
            return False
        df_rtp = df_net[~pd.isna(df_net["rtp.p_type"])]
        webrtc_reader = WebRTCReader(
            webrtc_file=webrtc_file, dataset=self.dataset)
        df_webrtc = webrtc_reader.get_webrtc()

        # Check if CSV file is empty
        if len(df_net) == 0 or len(df_rtp) == 0:
            print('Empty CSV File...')
            return False

        # Check if no streams are detected
        if 'framesPerSecond' not in df_webrtc.columns:
            return False

        # Check if timestamps align
        (webrtc_min_time, webrtc_max_time) = (
            df_webrtc["ts"].min(), df_webrtc["ts"].max())
        (pcap_min_time, pcap_max_time) = (
            df_rtp["frame.time_epoch"].min(), df_rtp["frame.time_epoch"].max())

        if webrtc_max_time < pcap_min_time or pcap_max_time < webrtc_min_time:
            print(f'Timestamps do not align : {csv_file} {webrtc_file}')
            return False

        # If duration for webrtc logs exceeds the number of FPS samples, invalidate

        if int(df_webrtc['duration'].max()) > df_webrtc['num_vals'].max():
            print('Less number of frames than call duration.')
            return False

        return True
