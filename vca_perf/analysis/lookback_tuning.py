import sys
import pandas as pd
from os.path import dirname, abspath
import matplotlib.pyplot as plt
import json, re, ast, numpy as np
import dateutil.parser
from datetime import datetime
from sklearn.metrics import mean_absolute_error

def is_freeze(x):
    if x["frame_dur"] > max(3*x["avg_frame_dur"], (x["avg_frame_dur"] + 0.150)):
        return 1
    else:
        return 0


def get_freeze_dur(x):
    if x["is_freeze"] == 1:
        return x["frame_dur"]
    else:
        return 0
    
project_config = {'webrtc_format': {'conext_data': 1, 'IMC_Zero_Loss': 2, 'IMC_Lab_data': 2}}
    
class WebRTCReader:
    def __init__(self, webrtc_file, dataset):
        self.webrtc_file = webrtc_file
        self.dataset = dataset
        if project_config['webrtc_format'][self.dataset] == 1:
            self.wanted_stats = {"RTCInboundRTPVideoStream" : ["ssrc", "lastPacketReceivedTimestamp", \
                    "framesPerSecond", "[bytesReceived_in_bits/s]", "[codec]", "packetsLost",\
                    "framesDropped", "framesReceived", "[framesReceived/s]", "[interFrameDelayStDev_in_ms]", "nackCount", "packetsReceived", "jitterBufferDelay", "[framesDecoded/s]", "jitterBufferEmittedCount", "frameHeight"], \
                    "RTCMediaStreamTrack_receiver" : ["trackIdentifier", "freezeCount*","totalFreezesDuration*", \
                    "totalFramesDuration*", "pauseCount*", "totalPausesDuration*"]}
        else:
            self.wanted_stats = {"IT01V" : ["ssrc", "lastPacketReceivedTimestamp", \
            "framesPerSecond", "[bytesReceived_in_bits/s]", "[codec]", "packetsLost",\
            "framesDropped", "framesReceived", "[framesReceived/s]", "[interFrameDelayStDev_in_ms]", "nackCount", "packetsReceived", "trackIdentifier", "freezeCount","totalFreezesDuration", "pauseCount", "totalPausesDuration", "jitterBufferDelay", "[framesDecoded/s]", "jitterBufferEmittedCount", "frameHeight", "qpSum"]}

        self.cum_stat_list = ["freezeCount*", "totalFreezesDuration*", "totalFramesDuration*", "framesReceived", "pauseCount*", "totalPausesDuration*", "jitterBufferDelay", "jitterBufferEmittedCount", "qpSum"]


    def get_most_active(self, webrtc_stats, id_list):
        stat_temp = "RTCInboundRTPVideoStream_%s-framesPerSecond" if project_config['webrtc_format'][self.dataset] == 1 else "IT01V%s-framesPerSecond"
        valid_id_list = [id_list[i] for i in range(len(id_list)) if stat_temp % id_list[i] in webrtc_stats]
        sum_list = [sum(ast.literal_eval(webrtc_stats[stat_temp % ssrc_id]["values"])) for ssrc_id in valid_id_list]
        if len(sum_list) == 0:
            return None
        index_max = np.argmax(sum_list)
        return valid_id_list[index_max]
        
    def is_cum_stat(self, x):
        for cum_stat in self.cum_stat_list:
            if '-'+cum_stat in x:
                return True
        return False

    def get_active_stream(self, webrtc_stats, pref):
        id_map = {}
        for k in webrtc_stats:
            m = re.search(f"{pref}_(\d+)-", k) if project_config['webrtc_format'][self.dataset] == 1 else re.search(f"{pref}(\d+)-", k)
            if not m:
                continue
            id1 = m.group(1)
            id_map[id1] = 1
        return list(id_map.keys())

    def get_stat(self, stat_name, st_time, et_time, val_list):
        st_dt = datetime.timestamp(dateutil.parser.parse(st_time))
        et_dt = datetime.timestamp(dateutil.parser.parse(et_time))
        (t, i) = (int(st_dt), 0)
        l = []
        while t < et_dt and i < len(val_list):
            l.append([t, val_list[i]])
            i += 1
            t += 1
        stat_suff = stat_name.split("-")[1]
        df = pd.DataFrame(l, columns=["ts", stat_suff])
        return df 

    def get_webrtc(self):
        webrtc = json.load(open(self.webrtc_file))
        active_ids = []
        try:
            unknown_key = None
            for k in webrtc["PeerConnections"].keys():
                if len(webrtc["PeerConnections"][k]["stats"]) == 0:
                    continue
                webrtc_stats = webrtc["PeerConnections"][k]["stats"]
                pref = "RTCInboundRTPVideoStream" if project_config['webrtc_format'][self.dataset] == 1 else "IT01V"
                active_ids = self.get_active_stream(webrtc_stats, pref) # Gets a list of SSRC IDs
                id1 = self.get_most_active(webrtc_stats, active_ids)
                if id1 is not None and len(id1) > 0:
                    break
        except Exception as e:
            print(e)
            return pd.DataFrame()
        if len(active_ids) == 0:
            print("no inbound stream")
            return pd.DataFrame()

        if id1 is None:
            print('No frames seen in this trace')
            return pd.DataFrame()

        ##

        stat_names = [f"{pref}_{id1}-{stat}" for stat in self.wanted_stats[pref]] if project_config['webrtc_format'][self.dataset] == 1 else [f"{pref}{id1}-{stat}" for stat in self.wanted_stats[pref]]

#         media_field = f"{pref}_{id1}-trackId" if project_config['webrtc_format'][self.dataset] == 1 else f"{pref}{id1}-trackId"
#         media_track = list(set(ast.literal_eval(webrtc_stats[media_field]["values"])))
        
#         if len(media_track) == 0:
#             print(f"No media track in file {self.webrtc_file}")
#             return pd.DataFrame()
#         elif len(media_track) > 1:
#             print(f"More than 1 media track in {self.webrtc_file}")
        
        df_all = pd.DataFrame()
        # if project_config['webrtc_format'][self.dataset] == 1:
        #     pref = "RTCMediaStreamTrack_receiver"
        #     stat_names += [f"{media_track[0]}-{stat}" for stat in self.wanted_stats[pref]]
        duration = None
        num_val = None
        try:
            for stat in stat_names:
                if stat.startswith('DEPRECATED'):
                    continue

                (st_time, et_time) = (webrtc_stats[stat]["startTime"], webrtc_stats[stat]["endTime"])
                if "framesReceived" in stat:
                    st = datetime.timestamp(dateutil.parser.parse(st_time))
                    et = datetime.timestamp(dateutil.parser.parse(et_time))
                    duration = et-st
                val_str = webrtc_stats[stat]["values"]
                val_list = ast.literal_eval(val_str)
                if self.is_cum_stat(stat):
                    val_list = [val_list[0]] + [val_list[i] - val_list[i-1] for i in range(1, len(val_list))] # [start_val, diff_between_consecutive_vals]

                df_stat = self.get_stat(stat, st_time, et_time, val_list)
                
                if "framesReceived" in stat:
                    num_val = len(df_stat)
                # print(stat)
                # print(df_stat.isna())
                if df_all.empty:
                    df_all = df_stat
                else:
                    df_all = pd.merge(df_all, df_stat, on="ts", how="outer")
        except Exception as e:
            print(f'Something went wrong for stat {stat} in file {self.webrtc_file}')
            print(e)
            return pd.DataFrame()
        df_all = df_all.rename(columns={'[framesReceived/s]': 'framesReceivedPerSecond', '[framesDecoded/s]': 'framesDecodedPerSecond'})
        df_all['duration'] = duration
        df_all['num_vals'] = num_val
        df_all = df_all.rename(columns={"[bytesReceived_in_bits/s]": "bitrate", "[interFrameDelayStDev_in_ms]": "frame_jitter"})
        return df_all

def assign_1(df, vca, l):
    frame_id_assignment = [-1 for _ in range(df.shape[0])]
    frame_id = 0
    for i in range(df.shape[0]):
        found = False
        s = df.iloc[i]['udp.length']
        for j in range(i-1, max(0, i-l-1), -1):
            if abs(df.iloc[j]['udp.length'] - s) <= 2:
                frame_id_assignment[i] = frame_id
                found = True
                break
        if not found:
            frame_id += 1
            frame_id_assignment[i] = frame_id
    return frame_id_assignment
for vca in ['meet', 'webex', 'teams']:
    df = pd.read_csv('imc_lab_results.csv', index_col=None)
    df = df[df['VCA'] == vca]
    ftuples = list(df.groupby(['pcap_file','csv_file', 'webrtc_file']).groups.keys())[:50]
    res = {'L': [], 'VCA': [], 'MAE': []}
    for l in range(1, 11):
        print(f'L = {l} VCA = {vca}')
        for trace in ftuples:
            pcap_file = trace[0]
            csv_file = trace[1]
            webrtc_file = trace[2]
            thresh = 2
            net_columns = ['frame.time_relative','frame.time_epoch','ip.src','ip.dst','ip.proto','ip.len','udp.srcport','udp.dstport', 'udp.length','rtp.ssrc','rtp.timestamp','rtp.seq','rtp.p_type', 'rtp.marker']
            df = pd.read_csv(csv_file, header=None, sep='\t', names=net_columns, lineterminator='\n', encoding='ascii')
            if df['ip.proto'].dtype == object:
                df = df[df['ip.proto'].str.contains(',') == False]
            df = df[~df['ip.proto'].isna()]
            df['ip.proto'] = df['ip.proto'].astype(int)
            ip_addr = df.groupby('ip.dst').agg({'udp.length': sum}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
            src = df.groupby('ip.src').agg({'udp.length': sum, 'rtp.p_type': 'count'}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.src'].iloc[0]
            df = df[(df['ip.proto'] == 17) & (df['ip.src'] == src) & (df['ip.dst'] == ip_addr)]
            df = df[(~df['rtp.ssrc'].isna()) & (df['udp.length'] > 400)]
            df = df.sort_values(by=['frame.time_relative'])
            df['frame_num'] = assign_1(df, vca, l)
            df['udp.length'] = df['udp.length'] - 12
            df_grp_udp = df.groupby("frame_num").agg({"frame.time_epoch": list, "udp.length": list}).reset_index()
            df_grp_udp["frame_st"] = df_grp_udp["frame.time_epoch"].apply(lambda x: min(x))
            df_grp_udp["frame_et"] = df_grp_udp["frame.time_epoch"].apply(lambda x: max(x))
            # df_grp_udp = df_grp_udp[df_grp_udp['frame_et'] - df_grp_udp['frame_st'] <= self.max_buffer_time[self.vca]]
            df_grp_udp["frame_size"] = df_grp_udp["udp.length"].apply(lambda x: sum(x))
            df_grp_udp["ft_end"] = df_grp_udp['frame_et'].apply(lambda x: int(x)+1)

            df_grp_udp["frame_dur"] = df_grp_udp["frame_et"].diff()
            df_grp_udp["avg_frame_dur"] = df_grp_udp["frame_dur"].rolling(
                30).mean()  # Why 30?
            df_grp_udp = df_grp_udp.fillna(0)
            idx = df_grp_udp.index[df_grp_udp['frame_dur'] >= 8].tolist()
            if len(idx) > 0:
                idx = idx[0]+1
            else:
                idx = 0
            df_grp_udp = df_grp_udp.iloc[idx:]
            # freeze calculation
            df_grp_udp["is_freeze"] = df_grp_udp.apply(is_freeze, axis=1)
            df_grp_udp["freeze_dur"] = df_grp_udp.apply(get_freeze_dur, axis=1)

            df_grp_udp = df_grp_udp.groupby("ft_end").agg({"frame_size": ["count", "sum"], "is_freeze": "sum",
                                                "freeze_dur": "sum",
                                                "frame_dur": "std"}).reset_index()

            # rename columns
            df_grp_udp.columns = ['_'.join(col).strip('_')
                            for col in df_grp_udp.columns.values]
            df_grp_udp = df_grp_udp.rename(columns={'frame_size_count': 'predicted_framesReceivedPerSecond',
                                            'is_freeze_sum': 'freeze_count',
                                            'frame_size_sum': 'predicted_bitrate',
                                            'freeze_dur_sum': 'freeze_dur',
                                            'frame_dur_std': 'predicted_frame_jitter'
                                            })
            df_grp_udp['predicted_bitrate'] = df_grp_udp['predicted_bitrate']*8
            df_grp_udp['predicted_frame_jitter'] *= 1000
            webrtc_reader = WebRTCReader(webrtc_file, 'IMC_Lab_data')
            df_webrtc = webrtc_reader.get_webrtc()[['framesReceivedPerSecond', 'ts']]
            col = "ft_end"
            df_merge = pd.merge(df_grp_udp, df_webrtc, left_on=col, right_on="ts")
            metric_col = f'framesReceivedPerSecond_frame-lookback'
            webrtc_col = f'framesReceivedPerSecond_gt'
            df_merge = df_merge.rename(columns={f'predicted_framesReceivedPerSecond': metric_col, 'framesReceivedPerSecond': webrtc_col, 'ts': 'timestamp'})
            df_merge = df_merge[[webrtc_col, metric_col, 'timestamp']]
            df_merge = df_merge.dropna()
            mae = mean_absolute_error(df_merge[webrtc_col], df_merge[metric_col])
            res['L'].append(l)
            res['VCA'].append(vca)
            res['MAE'].append(mae)
            print(f'VCA = {vca} | L = {l} | MAE = {mae}')
            
    dfl = pd.DataFrame(res)
    dfl.to_csv(f'{vca}_lookback.csv', index=False)