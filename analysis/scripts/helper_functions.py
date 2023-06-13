import pandas as pd
import json, re, ast, numpy as np
import functools
import operator, os
from glob import glob


def read_data(data_dir, vca, ip):
    def get_dirname(x):
        for dirname in dirlist:
            if dirname in x:
                return dirname
    
    ## Setting the variables
    cols= ['frame.time_relative','frame.time_epoch','ip.src','ip.dst','ip.proto','ip.len','udp.srcport','udp.dstport', 'udp.length','rtp.ssrc','rtp.timestamp','rtp.seq','rtp.p_type', 'rtp.marker','rtp.padding','rtp.ext','rtp.ext.len','rtcp.pt','rtcp.senderssrc','rtcp.ssrc.high_seq','rtcp.ssrc.dlsr','rtcp.psfb.fmt', 'rtcp.rtpfb.fmt','rtcp.sender.octetcount']
    webrtc_files = []
    no_file, no_webrtc_stat, no_net_data, no_overlap = (0, 0, 0, 0)
    
    ## Reading the data 
    dirlist = glob(data_dir)
    for dirname in dirlist:
        webrtc_files += glob(f"{dirname}/{vca}/webrtc/*.json")
    
    ## Reading the webrtc 
    file_map = {}
    for webrtc_filename in webrtc_files:
        pref = os.path.basename(webrtc_filename)
        dirname = get_dirname(webrtc_filename)

        net_filename = glob(f"{dirname}/{vca}/captures/*.csv")[0]

        if not os.path.exists(net_filename):
            no_file += 1
            continue

        df_webrtc = get_webrtc(webrtc_filename)
        if df_webrtc.empty:
            no_webrtc_stat += 1
            continue

        df_net = pd.read_csv(net_filename, header=None, sep='\t', names=cols)

        df_net = df_net[(df_net["ip.dst"] == ip) & (~pd.isna(df_net["rtp.ssrc"]))]
        if df_net.empty:
            no_net_data += 1
            continue

        df_net["time_normed"] = df_net["frame.time_relative"]
        df_net["length"] = df_net["udp.length"]
        df_net["time"] = df_net["frame.time_epoch"]
        time_col = "frame.time_epoch"

        df_rtp = df_net[~pd.isna(df_net["rtp.p_type"])]
        
        (webrtc_min_time, webrtc_max_time) = (df_webrtc["ts"].min(), df_webrtc["ts"].max())
        (pcap_min_time, pcap_max_time) = (df_rtp[time_col].min(), df_rtp[time_col].max())

        if webrtc_max_time < pcap_min_time or pcap_max_time < webrtc_min_time:
            no_overlap += 1
            continue

        file_map[webrtc_filename] = {"webrtc": df_webrtc, "pcap": df_net, "rtp": df_rtp, "file": webrtc_filename} 
    return file_map, (no_file, no_webrtc_stat, no_net_data, no_overlap)



def get_bitrate(x):
    if x["dur"] == 0:
        return 0
    return x["size"]*8 / (1000*x["dur"])

def get_pktrate(x):
    if x["dur"] == 0:
        return 0
    return x["num"] / x["dur"]

def get_stats(x):
    try:
        q1, q2, q3 = np.quantile(x, q=[0.25, 0.5, 0.75])
        min_x, max_x, mean_x, std_x = (min(x), max(x), np.mean(x), np.std(x))
        feat = [min_x, max_x, mean_x, std_x, q1, q2, q3]
        return feat
    except:
        print(x)
        
        
def pcap_to_csv(fname, outfile):
    tshark_cmd = "tshark -r %s -d udp.port==1024-49152,rtp -t e -T fields -e frame.time_relative -e frame.time_epoch -e ip.src -e ip.dst -e ip.proto -e ip.len -e udp.srcport -e udp.dstport -e udp.length -e rtp.ssrc -e rtp.timestamp -e rtp.seq -e rtp.p_type -e rtp.marker -e rtp.padding -e rtp.ext -e rtp.ext.len -e rtcp.pt -e rtcp.senderssrc -e rtcp.ssrc.high_seq -e rtcp.ssrc.dlsr -e rtcp.psfb.fmt -e rtcp.rtpfb.fmt -e rtcp.sender.octetcount > %s"
    cmd = tshark_cmd % (fname, outfile)
    os.system(cmd)
    

def get_net(filename):
    cols= ['frame.time_relative','frame.time_epoch','ip.src','ip.dst','ip.proto','ip.len','udp.srcport','udp.dstport', 'udp.length','rtp.ssrc','rtp.timestamp','rtp.seq','rtp.p_type', 'rtp.marker','rtp.padding','rtp.ext','rtp.ext.len','rtcp.pt','rtcp.senderssrc','rtcp.ssrc.high_seq','rtcp.ssrc.dlsr','rtcp.psfb.fmt', 'rtcp.rtpfb.fmt','rtcp.sender.octetcount']
    df = pd.read_csv(filename, delimiter='\t', header=None, names=cols)
    return get_net_stats(df)


def get_net_stats(df, interval=1):
    df["is_rtp"] = df["rtp.ssrc"].apply(lambda x: 1 if not pd.isna(x) else 0)
    df["time_win"] = df["frame.time_epoch"].apply(lambda x: int(x/interval)*interval)
    grp_cols = ["time_win", "ip.src", "ip.dst", "udp.srcport", "udp.dstport", "is_rtp"]
    df_grp = df.groupby(grp_cols)["frame.time_epoch", "udp.length"].agg(list).reset_index()


    stat_name = ["min", "max", "mean", "std", "q1", "q2", "q3"]
    df_grp["dur"] = interval#df_grp["frame.time_relative"].apply(lambda x: max(x) - min(x))
    df_grp["size"] = df_grp["udp.length"].apply(lambda x: sum(x))
    df_grp["num"] = df_grp["udp.length"].apply(lambda x: len(x))
    df_grp["dur1"] = df_grp["frame.time_epoch"].apply(lambda x: max(x) - min(x))
    df_grp["bitrate_kbps"] = df_grp.apply(get_bitrate, axis=1)
    df_grp["pktrate_pps"] = df_grp.apply(get_pktrate, axis=1)

    df_grp["size_stats"] = df_grp["udp.length"].apply(get_stats)
    size_stat_name = [f"size_{x}" for x in stat_name]
    df_size = pd.DataFrame(df_grp["size_stats"].tolist(), columns=size_stat_name)

    df_grp["iat"] = df_grp["frame.time_epoch"].apply(lambda x: [0] if len(x) == 1 else [x[i] - x[i-1] for i in range(1, len(x))])
    df_grp["iat_stats"] = df_grp["iat"].apply(get_stats)
    iat_stat_name = [f"iat_{x}" for x in stat_name]
    df_iat = pd.DataFrame(df_grp["iat_stats"].tolist(), columns=iat_stat_name)

    df_grp = pd.concat([df_grp, df_size, df_iat], axis=1)

    cols = ['time_win', 'ip.src', 'ip.dst', 'udp.srcport', 'udp.dstport', 'is_rtp',
            'dur', 'dur1', 'bitrate_kbps', 'pktrate_pps', 'size_min', 'size_max', 'size_mean', 'size_std', 'size_q1',
           'size_q2', 'size_q3', 'iat_min', 'iat_max', 'iat_mean', 'iat_std',
           'iat_q1', 'iat_q2', 'iat_q3']
    df_stats = df_grp[cols]
    return df_stats


def concat_list(a):
    return functools.reduce(operator.iconcat, a, [])


def fill_dummy_missing_timewin(df_grp, df_grp_small):
    l = []
    for i in range(0, df_grp_small.shape[0]):
        blank_row = df_grp_small.iloc[i].tolist()
        diff = int(blank_row[-1])
        for j in range(1, diff):
            blank_row[1] = blank_row[1] - diff + j
            blank_row[7] = [0]
            blank_row[8] = [0]
            l.append(blank_row[1:])
    df_append = pd.DataFrame(l, columns=df_grp.columns)
    df_grp = pd.concat([df_grp, df_append]).reset_index().sort_values(by=["time_win"])
    df_grp["time_win_diff"] = df_grp["time_win"].diff()
    return df_grp



def get_net_stats_rolling(df, interval=1):
    df["is_rtp"] = df["rtp.ssrc"].apply(lambda x: 1 if not pd.isna(x) else 0)
    df["time_win"] = df["frame.time_epoch"].apply(lambda x: int(x))
    df = df.fillna(0)
    grp_cols = ["time_win", "ip.src", "ip.dst", "udp.srcport", "udp.dstport", "is_rtp"]
    cols = ["frame.time_epoch", "udp.length"]
    df_grp = df.groupby(grp_cols)[cols].agg(list).reset_index()
    df_grp = df_grp.sort_values(by="time_win")
    df_grp["time_win_diff"] = df_grp["time_win"].diff()
    df_grp_small = df_grp[df_grp["time_win_diff"] > 1].reset_index()
    
    
    if interval > 1:
        if not df_grp_small.empty:
            df_grp = fill_dummy_missing_timewin(df_grp, df_grp_small)

        df_grp1 = df_grp.copy()

        for i in range(interval-1, df_grp.shape[0]):
            for col in cols:
                df_grp.at[i, col] = concat_list(df_grp1.iloc[i-interval+1:i+1][col].tolist())

    stat_name = ["min", "max", "mean", "std", "q1", "q2", "q3"]
    df_grp["dur"] = interval#df_grp["frame.time_relative"].apply(lambda x: max(x) - min(x))
    df_grp["size"] = df_grp["udp.length"].apply(lambda x: sum(x))
    df_grp["num"] = df_grp["udp.length"].apply(lambda x: len(x))
    df_grp["dur1"] = df_grp["frame.time_epoch"].apply(lambda x: max(x) - min(x))
    df_grp["bitrate_kbps"] = df_grp.apply(get_bitrate, axis=1)
    df_grp["pktrate_pps"] = df_grp.apply(get_pktrate, axis=1)

    df_grp["size_stats"] = df_grp["udp.length"].apply(get_stats)
    size_stat_name = [f"size_{x}" for x in stat_name]
    df_size = pd.DataFrame(df_grp["size_stats"].tolist(), columns=size_stat_name)

    df_grp["iat"] = df_grp["frame.time_epoch"].apply(lambda x: [0] if len(x) == 1 else [x[i] - x[i-1] for i in range(1, len(x))])
    df_grp["iat_stats"] = df_grp["iat"].apply(get_stats)
    iat_stat_name = [f"iat_{x}" for x in stat_name]
    df_iat = pd.DataFrame(df_grp["iat_stats"].tolist(), columns=iat_stat_name)

    df_grp = pd.concat([df_grp, df_size, df_iat], axis=1)

    cols = ['time_win', 'ip.src', 'ip.dst', 'udp.srcport', 'udp.dstport', 'is_rtp',
            'dur', 'dur1', 'bitrate_kbps', 'pktrate_pps', 'size_min', 'size_max', 'size_mean', 'size_std', 'size_q1',
           'size_q2', 'size_q3', 'iat_min', 'iat_max', 'iat_mean', 'iat_std',
           'iat_q1', 'iat_q2', 'iat_q3']
    df_stats = df_grp[cols]
    return df_stats.iloc[interval-1:]



import dateutil.parser
from datetime import datetime

wanted_stats = {"RTCInboundRTPVideoStream" : ["ssrc", "lastPacketReceivedTimestamp", \
                "framesPerSecond", "[bytesReceived_in_bits/s]", "[codec]", "packetsLost",\
                "framesDropped", "framesReceived", "framesDecoded", "[interFrameDelayStDev_in_ms]",
                "nackCount", "packetsReceived"], 
                "RTCMediaStreamTrack_receiver" : ["trackIdentifier", "freezeCount*","totalFreezesDuration*", \
                "totalFramesDuration*", "pauseCount*", "totalPausesDuration*"]}

def get_stat(stat_name, st_time, et_time, val_list):
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

def get_active_stream(webrtc_stats, pref):
    id_map = {}
    for k in webrtc_stats:
        m = re.search(f"{pref}_(\d+)-", k)
        if not m:
            continue
        id1 = m.group(1)
        id_map[id1] = 1
    return list(id_map.keys())


# Are these cumulative? stats
cum_stat_list = ["freezeCount*", "totalFreezesDuration*", "totalFramesDuration*", "framesReceived", "pauseCount*", "totalPausesDuration*", "framesDecoded"]
def is_cum_stat(x):
    for cum_stat in cum_stat_list:
        if cum_stat in x:
            return True
    return False


def get_sampling_rate(filename, ptype="98"):
    webrtc = json.load(open(filename))
    try:
        unknown_key = list(webrtc["PeerConnections"].keys())[0]
    except:
        return None
    
    webrtc_stats = webrtc["PeerConnections"][unknown_key]["stats"]
    k = f"RTCCodec_1_Inbound_{ptype}-clockRate"
    if k not in webrtc_stats:
        return []
    else:
        return list(set(ast.literal_eval(webrtc_stats[k]["values"])))
    
def get_most_active(webrtc_stats, id_list):
    stat_temp = "RTCInboundRTPVideoStream_%s-framesPerSecond"
    valid_id_list = [ssrc_id for ssrc_id in id_list if stat_temp % ssrc_id in webrtc_stats]
    sum_list = [sum(ast.literal_eval(webrtc_stats[stat_temp % ssrc_id]["values"])) for ssrc_id in valid_id_list]
    index_max = np.argmax(sum_list)
    return valid_id_list[index_max]
    
    
def get_webrtc(filename):
    webrtc = json.load(open(filename))
    try:
        unknown_key = list(webrtc["PeerConnections"].keys())[0]
    except:
        return pd.DataFrame()
    
    webrtc_stats = webrtc["PeerConnections"][unknown_key]["stats"]
    
    pref = "RTCInboundRTPVideoStream"
    active_ids = get_active_stream(webrtc_stats, pref) # Gets a list of SSRC IDs
    if len(active_ids) == 0:
        #print("no inbound stream")
        return pd.DataFrame()
    
    id1 = get_most_active(webrtc_stats, active_ids)#active_ids[0]
    ##

    stat_names = [f"{pref}_{id1}-{stat}" for stat in wanted_stats[pref]]
    
    media_field = f"{pref}_{id1}-trackId"
    media_track = list(set(ast.literal_eval(webrtc_stats[media_field]["values"])))
    
    if len(media_track) == 0:
        print("no media track")
        return pd.DataFrame()
    elif len(media_track) > 1:
        print("more than 1 media track %s" % filename)
    
    df_all = pd.DataFrame()
    pref = "RTCMediaStreamTrack_receiver"
    stat_names += [f"{media_track[0]}-{stat}" for stat in wanted_stats[pref]]
    try:
        for stat in stat_names:
            (st_time, et_time) = (webrtc_stats[stat]["startTime"], webrtc_stats[stat]["endTime"])
            val_str = webrtc_stats[stat]["values"]
            val_list = ast.literal_eval(val_str)
            if is_cum_stat(stat):
                val_list = [val_list[0]] + [val_list[i] - val_list[i-1] for i in range(1, len(val_list))] # [start_val, diff_between_consecutive_vals]

            df_stat = get_stat(stat, st_time, et_time, val_list)
            if df_all.empty:
                df_all = df_stat
            else:
                df_all = pd.merge(df_all, df_stat, on="ts", how="outer")
    except:
        print(stat, filename)
        return pd.DataFrame()
    return df_all