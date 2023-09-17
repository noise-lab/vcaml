import pandas as pd
import json
import re
import ast
import numpy as np
import dateutil.parser
from datetime import datetime
import sys
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)
from config import project_config

class WebRTCReader:
    def __init__(self, webrtc_file, dataset):
        self.webrtc_file = webrtc_file
        self.dataset = dataset
        self.wanted_stats = {"IT01V": ["ssrc", "lastPacketReceivedTimestamp",
                                       "framesPerSecond", "[bytesReceived_in_bits/s]", "[codec]", "packetsLost",
                                       "framesDropped", "framesReceived", "[framesReceived/s]", "[interFrameDelayStDev_in_ms]", "nackCount", "packetsReceived", "trackIdentifier", "freezeCount", "totalFreezesDuration", "pauseCount", "totalPausesDuration", "jitterBufferDelay", "[framesDecoded/s]", "jitterBufferEmittedCount", "frameHeight", "qpSum"]}

        self.cum_stat_list = ["freezeCount*", "totalFreezesDuration*", "totalFramesDuration*", "framesReceived",
                              "pauseCount*", "totalPausesDuration*", "jitterBufferDelay", "jitterBufferEmittedCount", "qpSum"]

    def get_most_active(self, webrtc_stats, id_list):
        stat_temp = "IT01V%s-framesPerSecond"
        valid_id_list = [id_list[i] for i in range(
            len(id_list)) if stat_temp % id_list[i] in webrtc_stats]
        sum_list = [sum(ast.literal_eval(webrtc_stats[stat_temp %
                        ssrc_id]["values"])) for ssrc_id in valid_id_list]
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
            m = re.search(f"{pref}(\d+)-", k)
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
        try:
            webrtc = json.load(open(self.webrtc_file))
            active_ids = []
            unknown_key = None
            for k in webrtc["PeerConnections"].keys():
                if len(webrtc["PeerConnections"][k]["stats"]) == 0:
                    continue
                webrtc_stats = webrtc["PeerConnections"][k]["stats"]
                pref = "IT01V"
                active_ids = self.get_active_stream(
                    webrtc_stats, pref)  # Gets a list of SSRC IDs
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

        stat_names = [f"{pref}{id1}-{stat}" for stat in self.wanted_stats[pref]]

        df_all = pd.DataFrame()
        duration = None
        num_val = None
        try:
            for stat in stat_names:
                if stat.startswith('DEPRECATED'):
                    continue

                (st_time, et_time) = (
                    webrtc_stats[stat]["startTime"], webrtc_stats[stat]["endTime"])
                if "framesReceived" in stat:
                    st = datetime.timestamp(dateutil.parser.parse(st_time))
                    et = datetime.timestamp(dateutil.parser.parse(et_time))
                    duration = et-st
                val_str = webrtc_stats[stat]["values"]
                val_list = ast.literal_eval(val_str)
                if self.is_cum_stat(stat):
                    # [start_val, diff_between_consecutive_vals]
                    val_list = [val_list[0]] + [val_list[i] - val_list[i-1]
                                                for i in range(1, len(val_list))]

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
            print(
                f'Something went wrong for stat {stat} in file {self.webrtc_file}')
            print(e)
            return pd.DataFrame()
        df_all = df_all.rename(columns={
                               '[framesReceived/s]': 'framesReceivedPerSecond', '[framesDecoded/s]': 'framesDecodedPerSecond'})
        df_all['duration'] = duration
        df_all['num_vals'] = num_val
        df_all = df_all.rename(columns={
                               "[bytesReceived_in_bits/s]": "bitrate", "[interFrameDelayStDev_in_ms]": "frame_jitter"})
        return df_all
