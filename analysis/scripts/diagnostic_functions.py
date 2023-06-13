import pandas as pd
import numpy as np

def filter_ptype(p_type, vca):
    if isinstance(p_type, str) and ',' in p_type:
        p_type = p_type.split(',')[0]
    p_type = int(p_type)
    return p_type

        
    return str(int(float(x)))

video_size_thresh = {'meet': 300, 'teams': 400}
video_ptypes = {'meet': [98, 99], 'teams': [102, 123]}
net_cols = ['frame.time_relative','frame.time_epoch','ip.src','ip.dst','ip.proto','ip.len','udp.srcport','udp.dstport', 'udp.length','rtp.ssrc','rtp.timestamp','rtp.seq','rtp.p_type', 'rtp.marker','rtp.padding','rtp.ext','rtp.ext.len','rtcp.pt','rtcp.senderssrc','rtcp.ssrc.high_seq','rtcp.ssrc.dlsr','rtcp.psfb.fmt', 'rtcp.rtpfb.fmt','rtcp.sender.octetcount']



def is_video_rtp(p_type, vca):
    if p_type in video_ptypes[vca]:
        return 1
    else:
        return 0

def is_video_udp(x, vca):
    if int(x) > video_size_thresh[vca]:
        return 1
    else:
        return 0


def preprocess_trace(df_net = pd.DataFrame(), filename = None, vca = "teams"):
    if df_net.empty:
        df_net = pd.read_csv(filename, header=None, sep='\t', names=net_cols, lineterminator='\n', encoding='ascii')
        vca = filename.split('/')[-1].split('-')[0]
        df_net['vca'] = vca
        df_net = df_net[(df_net["ip.dst"] == ip) & (~pd.isna(df_net["rtp.ssrc"]))]
    df_net['rtp.p_type'] = df_net['rtp.p_type'].apply(lambda x: filter_ptype(x, vca))
    df_net["is_video_actual"] = df_net["rtp.p_type"].apply(lambda x: is_video_rtp(x, vca))
    df_net["is_video_pred"] = df_net["udp.length"].apply(lambda x: is_video_udp(x, vca))
    #df_rtp = label_video_non_rtp(df_rtp)
    return df_net

def get_violations_per_frame(l, vca):
    count = 0
    for i in range(1, len(l)):
        cur_diff = abs(l[i] - l[i-1])
        if cur_diff > size_diff_threshold[vca]:
            count += 1
    return count

def get_violations_with_lookback(df_grp, vca):
    v = [0]
    for idx, frame in df_grp.iterrows():
        if idx == 0:
            continue
        for pkt in frame['udp.length']:
            prev_avgs = [np.array(df_grp.iloc[idx-j]['udp.length']).mean() for j in range(1, preferred_lookback[vca]+1) if idx-j > 0]
            count = 0
            vfound = False
            for a in prev_avgs:
                if abs(a-pkt) <= size_diff_threshold[vca]:
                    vfound = True
                    break
            if vfound:
                count += 1
        v.append(count)
    return v

def get_frames(df_rtp):
    df_video = df_rtp[df_rtp['is_video_actual'] == 1]
    df_grp = df_video.groupby('rtp.timestamp').agg({"udp.length": list, "rtp.p_type": list, "frame.time_relative": list}).reset_index()
    return df_grp
        
def get_thresh_violations(df_rtp = pd.DataFrame(), filename = None, with_lookback=False):
    if df_rtp.empty:
        df_rtp = preprocess_trace(filename)
    vca = df_rtp['vca'].unique()[0]
    df_grp = get_frames(df_rtp)
    if with_lookback:
        df_grp['lookback_violations'] = get_violations_with_lookback(df_grp, vca)
        return df_grp['lookback_violations'], len(df_rtp)
    else:
        df_grp['num_violations'] = df_grp['udp.length'].apply(lambda x: get_violations_per_frame(x, vca))
        print(df_grp[df_grp['num_violations'] >= 1][['udp.length', 'num_violations']])
        return df_grp['num_violations'], len(df_rtp)
    
def get_packet_reorders(df_rtp = pd.DataFrame(), filename=None, max_lookback=2):
    vca = df_rtp['vca'].unique()[0]
    if df_rtp.empty:
        df_rtp = preprocess_trace(filename)
    else:
        df_rtp = df_rtp[(df_rtp["is_video_actual"] == 1) & (df_rtp["udp.length"] > video_size_thresh[vca])]

    df_rtp['rtp.ts_rel'] = 0
    seen_rtp_ts = {}
    within_lookback_rtp_timestamp_list = [-1 for i in range(max_lookback)]
    count = 0
    for idx, row in df_rtp.iterrows():
        if row['rtp.timestamp'] not in seen_rtp_ts:
            seen_rtp_ts[row['rtp.timestamp']] = 1
            within_lookback_rtp_timestamp_list  = [row["rtp.timestamp"]] + within_lookback_rtp_timestamp_list[:-1]
        else:
            if row['rtp.timestamp'] in within_lookback_rtp_timestamp_list:
                continue
            else:
                within_lookback_rtp_timestamp_list  = [row["rtp.timestamp"]] + within_lookback_rtp_timestamp_list[:-1]
                count +=1 
    return count, len(seen_rtp_ts)


    
size_diff_threshold = {'meet': 4, 'teams': 2}
def exceed_intraframe_threshold(df_rtp = pd.DataFrame(), filename=None):
    vca = df_rtp['vca'].unique()[0]
    if df_rtp.empty:
        df_rtp = preprocess_trace(filename)
    else:
        df_rtp = df_rtp[(df_rtp["is_video_actual"] == 1) & (df_rtp["udp.length"] > video_size_thresh[vca])]
    count = 0
    df_grp = df_rtp.groupby("rtp.timestamp").agg({"udp.length": list}).reset_index()
    for idx, row in df_grp.iterrows():
        pkt_sizes = row["udp.length"]
        for i in range(1, len(pkt_sizes)):
            if abs(pkt_sizes[i] - pkt_sizes[i-1]) > size_diff_threshold[vca]:
                count += 1
    return count
        
                
def interframe_size_diff_within_intraframe(df_rtp = pd.DataFrame(), max_lookback=2, filename=None):
    vca = df_rtp['vca'].unique()[0]
    if df_rtp.empty:
        df_rtp = preprocess_trace(filename)
    else:
        df_rtp = df_rtp[(df_rtp["is_video_actual"] == 1) & (df_rtp["udp.length"] > video_size_thresh[vca])]
    count = 0

    df_grp = df_rtp.groupby("rtp.timestamp").agg({"udp.length": list}).reset_index()
    df_grp["avg_frame_size"] = df_grp["udp.length"].apply(np.mean)
    size_list = df_grp["avg_frame_size"].tolist()
    for idx1 in range(1, len(size_list)):
        for j in range(1, max_lookback+1):
            idx2 = idx1 - j
            if idx2 < 0:
                continue
            if abs(size_list[idx1] - size_list[idx2]) <= size_diff_threshold[vca]:
                count += 1
                break
    return count
    
def plot_media_classification(filename):
    df_rtp = preprocess_trace(filename)
    cm = confusion_matrix(df_rtp["is_video_actual"], df_rtp["is_video_pred"])
    np.set_printoptions(3)
    cm_frac = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_frac, annot=True)