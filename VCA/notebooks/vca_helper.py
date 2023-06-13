import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm

import re, json, operator
from dateutil.parser import parse

ip_addr = {
    "turris" : "128.135.123.204",
    "A"      : "192.168.1.193",
    "B"      : "192.168.1.107"
}


def get_iperf(glob_str = "zoom-iperf-*.json"):

    df_iperfs = []
    for fname in glob(glob_str):

        try:
            with open(fname) as file:
                j = json.load(file)
        except:
            print(fname, "bad")
            return None

        if "intervals" not in j:
            print(fname, "did not have intervals")
            continue

        if not j["intervals"]:
            print(fname, "had no intervals")
            continue

        iperf_mbps = [i["sum"]["bits_per_second"] / 1e6 for i in j["intervals"]]

        df_iperf = pd.DataFrame({"iperf_mbps" : iperf_mbps})


        try:
            df_iperf["ts"] = j["start"]["timestamp"]["timesecs"] + 1 + df_iperf.index
            df_iperf["fname"] = fname.replace(".json", "")
        except:
            print(fname, j["start"], iperf_mbps)

        df_iperfs.append(df_iperf)

    iperf = pd.concat(df_iperfs)

    return iperf



## find candidate keys w
def find_keys(key_list, regex):
   valid_keys = [key for key in key_list if regex in key]
   return valid_keys


## util function
def cast_float(l):
	l_new = [0 if elem == "null" else float(elem) for elem in l]    
	return l_new


def webrtc_get_active_stream(stats, key_list, pref, stat_suff = "[packetsSent/s]"):
    id_map = {}
    for key in key_list:
        key_id = key.split('-')[0].split('_')[-1]
        id_map[key_id] = f'{pref}_{key_id}-{stat_suff}'

    if len(id_map) == 1:
        return list(id_map.keys())[0]
    
    for key_id in list(id_map.keys()):
        if id_map[key_id] not in stats:
            id_map.pop(key_id)
  
    
    total_count_list = [np.sum(cast_float(stats[id_map[x]]['values'][1:-1].split(','))) for x in id_map]
    if len(total_count_list) == 0:
        return ""
    max_index, max_value = max(enumerate(total_count_list), key=operator.itemgetter(1))
    return list(id_map.keys())[max_index]

def webrtc_stream_stats(stats, stream_pref, stat_pref, active_stat_name, stat_name_list):
    stat_key_list = stats.keys()
    qos_data = {}
    candidate_keys = find_keys(stat_key_list, stream_pref)
    active_stream_id = webrtc_get_active_stream(stats, candidate_keys, stream_pref, stat_suff=active_stat_name)

    if active_stream_id == "":
        return -1
    
    for stat_name in stat_name_list:
        stat_full_name = f'{stream_pref}_{active_stream_id}-{stat_name}'
        st_time = int(parse(stats[stat_full_name]["startTime"]).timestamp())
        val_list = cast_float(stats[stat_full_name]['values'][1:-1].split(',')) 
        qos_data[f'{stat_pref}_{stat_name}'] = [(st_time+i, val_list[i]) for i in range(0, len(val_list))]
        
    return qos_data


def get_webrtc(vca, fname):
    qos_data = {}
    
    all_stats = []
    ## load the json object
    #content = ''.join(list(map(str.strip, open(fname).readlines())))
    f = open(fname)
    json_data = json.load(f)
    #print(json_data.keys())
    json_stats = json_data["PeerConnections"]
    pc_list = list(json_stats.keys())
    if len(pc_list) == 0:
        print("len pc_list is zero")
        return pd.DataFrame()
    stats = json_stats[pc_list[0]]["stats"]

    '''
    if len(pc_list) > 1:
        stats1 = json_stats[pc_list[1]]["stats"]
        print_list(list(stats.keys())[:100])
        print_list(list(stats1.keys())[:100])
    '''
    
    # inbound stats
    stat_name_list = ['frameHeight', 'frameWidth', 'framesPerSecond', "[bytesReceived_in_bits/s]", "[qpSum/framesDecoded]", "pliCount", "firCount"]
    stream_pref, active_stat_name, stat_pref = ("RTCInboundRTPVideoStream", "[packetsReceived/s]", "received")
    stream_qos_data = webrtc_stream_stats(stats, stream_pref, stat_pref, active_stat_name, stat_name_list)
    all_stats += [f'{stat_pref}_{stat_name}' for stat_name in stat_name_list]

    if stream_qos_data == -1:
        print("error parsing inbound stats")
        return pd.DataFrame()
    qos_data.update(stream_qos_data)

    
    ## outbound stats 
    stat_name_list = ['frameHeight', 'frameWidth', 'framesPerSecond', "[bytesSent_in_bits/s]", "[qpSum/framesEncoded]"]
    stream_pref, active_stat_name, stat_pref = ("RTCOutboundRTPVideoStream", "[packetsSent/s]", "sent")
    stream_qos_data = webrtc_stream_stats(stats, stream_pref, stat_pref, active_stat_name, stat_name_list)
    all_stats += [f'{stat_pref}_{stat_name}' for stat_name in stat_name_list]

    if stream_qos_data == -1:
        print("error parsing outbound stats")
        return pd.DataFrame()
    qos_data.update(stream_qos_data)

    ## received media stream stats
    stat_name_list = ["jitterBufferDelay", "[jitterBufferDelay/jitterBufferEmittedCount_in_ms]", "freezeCount*", "pauseCount*", "totalFreezesDuration*", "totalPausesDuration*"]    
    stream_pref, active_stat_name, stat_pref = ("RTCMediaStreamTrack_receiver", "freezeCount*", "received")
    stream_qos_data = webrtc_stream_stats(stats, stream_pref, stat_pref, active_stat_name, stat_name_list)
    all_stats += [f'{stat_pref}_{stat_name}' for stat_name in stat_name_list]
    if stream_qos_data == -1:
        print("error parsing media stream stats")
        return pd.DataFrame()
    qos_data.update(stream_qos_data)
    
    ## re-shape the data indexed on timestamps
    qos_data_time = {}
    for i in range(0, len(all_stats)):
        stat_name = all_stats[i]
        stat_val_list = qos_data[stat_name]
        stat_val_map = {ts: stat_val for (ts, stat_val) in stat_val_list}
        for ts in qos_data_time:
            if ts not in stat_val_map:
                qos_data_time[ts] += [-1]
            else:
                qos_data_time[ts] += [stat_val_map[ts]]
                stat_val_map.pop(ts)
                
        for ts in stat_val_map:
            qos_data_time[ts] = [-1 for k in range(i)] + [stat_val_map[ts]]

    fname_last = fname.split('/')[-1]
    for ts in qos_data_time:
        qos_data_time[ts] += [fname_last]
    
    qos_data_time = [[k]+v for (k,v) in qos_data_time.items()]
    col_names = ["ts"] + all_stats + ["fname"]
    
    df = pd.DataFrame(qos_data_time, columns=col_names)
    df["vca"] = vca

    df.set_index(df.ts, inplace = True)
    df.sort_index(inplace = True)
    return df  


webrtc_stats = {"ts" : "ts", 
                'received_frameHeight' : "recv_resy", 
                'received_frameWidth' : "recv_resx",
                'received_framesPerSecond' : "recv_fps", 
                'received_[bytesReceived_in_bits/s]' : "recv_bps",
#                 'received_[qpSum/framesDecoded]' : "recv_frames_enc", 
#                 'received_pliCount',
#                 'received_firCount', 
                'sent_frameHeight' : "sent_resy", 
                'sent_frameWidth' : "sent_resx",
                'sent_framesPerSecond' : "sent_fps", 
                'sent_[bytesSent_in_bits/s]' : "sent_bps",
                'sent_[qpSum/framesEncoded]' : "sent_frames_enc", 
                'received_jitterBufferDelay' : "recv_jitter_buff_delay",
#                 'received_[jitterBufferDelay/jitterBufferEmittedCount_in_ms]',
                'received_freezeCount*' : "recv_freeze_count", 
                'received_pauseCount*' : "recv_pause_count",
#                 'received_totalFreezesDuration*', 
#                 'received_totalPausesDuration*'
               }

def get_all_webrtc(vca, glob_str):

    dfs = []
    for f in glob(glob_str):
        dfs.append(get_webrtc(vca, f))

    df = pd.concat(dfs)
    df = df[webrtc_stats].copy()
    df.rename(columns = webrtc_stats, inplace = True)

    df.replace({-1 : np.nan}, inplace = True)
    df.recv_bps = df.recv_bps.replace({-1 : np.nan, 0: np.nan})
    df.sent_bps = df.sent_bps.replace({-1 : np.nan, 0: np.nan})

    df["sent_mbps"] = df["sent_bps"] / 1e6
    df["recv_mbps"] = df["recv_bps"] / 1e6
    df.drop(["sent_bps", "recv_bps"], axis = 1, inplace = True)

    return df



def get_zoom_api(ip_addr, glob_str = "congestion_zoom_?.json"):
    
    qos_data = {}

    for fname in glob(glob_str):
        with open(fname) as file:
            j = json.load(file)

        for p in j["participants"]:
            if p["ip_address"] != ip_addr: continue

            for qos in p["user_qos"]:
                date_time = qos["date_time"]

                out_qos = {}
                
                for io in ["input", "output"]:
                
                    raw_qos = qos["video_" + io]

                    if raw_qos["bitrate"]:
                        out_qos[io + "_mbps"]         = float(raw_qos["bitrate"].replace(" kbps", "")) / 1000

                    if raw_qos["latency"]:
                        out_qos[io + "_latency_ms"]   = float(raw_qos["latency"].replace(" ms", ""))

                    if raw_qos["jitter"]:
                        out_qos[io + "_jitter_ms"]    = float(raw_qos["jitter"].replace(" ms", ""))

                    if raw_qos["avg_loss"]:
                        out_qos[io + "_avg_loss_pct"] = float(raw_qos["avg_loss"].replace("%", ""))

                    if raw_qos["max_loss"]:
                        out_qos[io + "_max_loss_pct"] = float(raw_qos["max_loss"].replace("%", ""))

                    if raw_qos["resolution"]:
                        out_qos[io + "_xres"]   = int(re.sub("\*.*", "", raw_qos["resolution"]))
                        out_qos[io + "_yres"]   = int(re.sub(".*\*", "", raw_qos["resolution"]))

                qos_data[date_time] = out_qos

    data = pd.DataFrame.from_dict(qos_data, orient = "index")
    data.set_index(pd.to_datetime(data.index), inplace = True)
    data.sort_index(inplace = True)
    data["ts"] = (data.index.astype(int) // 1e9).astype(int)

    return data



# %%bash
# for f in *pcap; do 
#     c=$(echo $f | sed "s/pcap/csv/")
#     tshark -r $f -d udp.port==1024-49152,rtp -t e -T fields -e frame.time -e ip.src -e ip.dst -e frame.len > $c
# done

def parse_meta(f):

    f = f.split("/")[-1]
    f = f.replace(".csv", "")
    f = f.replace("comp_up_", "")
    f = f.replace("comp_", "")
    f = re.sub(r"^a-", "", f)
    f = re.sub(r"^b-", "", f)
    f = re.sub(r"^up-", "", f)
    meta = f.replace("_", "-").split("-")

    output = {"A" : meta[0], "B" : meta[1], "shape_dl" : float(meta[3]), "shape_ul" : float(meta[4])}

    return output


def get_pcap_csv(ip_addr, glob_str = "*.csv", B = False):

    dfs = []
    for f in tqdm(glob(glob_str)):

        df = pd.read_csv(f, delimiter = "\t",
                         names = ["datetime", "src", "dst", "len"],
                         parse_dates = ["datetime"])

        df["s"]  = df.datetime.dt.floor("s")
        df["ts"] = (df.s.astype(np.int64) // 1e9).astype(int)

        ts_min, ts_max = df.ts.min(), df.ts.max()

        dl = df.query(f"dst == '{ip_addr}'")
        dl.rename(columns = {"len" : "pcap_dl_mbps"}, inplace = True)
        dl = dl.groupby("ts").pcap_dl_mbps.sum() * 8 / 1e6

        ul = df.query(f"src == '{ip_addr}'")
        ul.rename(columns = {"len" : "pcap_ul_mbps"}, inplace = True)
        ul = ul.groupby("ts").pcap_ul_mbps.sum() * 8 / 1e6

        null = pd.Series(name = "null",
                         index = range(ts_min, ts_max+1), 
                         data = np.zeros(ts_max - ts_min + 1))

        df = pd.concat([dl, ul, null], axis = 1)
        df.drop("null", axis = 1, inplace = True)
        df.index.name = "ts"
        df.reset_index(inplace = True)
        df = df.fillna(0)

        if B:
            starting_ts = df.set_index("ts").pcap_dl_mbps.rolling(120).sum().idxmax() - 120
            df["active"] = (starting_ts <= df.ts) & (df.ts < starting_ts + 120)

        meta = parse_meta(f)
        for k, v in meta.items():
            if k in df.columns:
                continue

            df[k] = v

        dfs.append(df)

    pcap = pd.concat(dfs)

    return pcap



def reduce_pcap_csv(f, comp):


    IP = ip_addr[comp]

    df = pd.read_csv(f, delimiter = "\t",
                     names = ["datetime", "src", "dst", "len"],
                     parse_dates = ["datetime"])

    df["s"]  = df.datetime.dt.floor("s")
    df["ts"] = (df.s.astype(np.int64) // 1e9).astype(int)

    ab_addresses = {'192.168.1.107', '192.168.1.193'}
    df.query("~((src in @ab_addresses) & (dst in @ab_addresses))", inplace = True)

    ts_min, ts_max = df.ts.min(), df.ts.max()

    dl = df.query(f"dst == '{IP}'").copy()
    dl.rename(columns = {"len" : "pcap_dl_mbps"}, inplace = True)
    dl = dl.groupby("ts").pcap_dl_mbps.sum() * 8 / 1e6

    ul = df.query(f"src == '{IP}'").copy()
    ul.rename(columns = {"len" : "pcap_ul_mbps"}, inplace = True)
    ul = ul.groupby("ts").pcap_ul_mbps.sum() * 8 / 1e6

    null = pd.Series(name = "null",
                     index = range(ts_min, ts_max+1),
                     data = np.zeros(ts_max - ts_min + 1))

    df = pd.concat([dl, ul, null], axis = 1)
    df.drop("null", axis = 1, inplace = True)
    df.index.name = "ts"
    df.reset_index(inplace = True)
    df.fillna(0, inplace = True)

    if comp == "B":
        starting_ts = df.set_index("ts").pcap_dl_mbps.rolling(120).sum().idxmax() - 120
        df["active"] = (starting_ts <= df.ts) & (df.ts < starting_ts + 120)

    meta = parse_meta(f)
    for k, v in meta.items():
        if k in df.columns:
            continue

        df[k] = v

    # print(f, comp)
    # print(" >", meta)

    df.to_csv(f.replace(".csv", f"_{comp}_red.csv"), index = False)  



