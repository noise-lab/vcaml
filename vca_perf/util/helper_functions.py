from config import project_config
import math
import pandas as pd
import sys
from os.path import dirname, abspath
import numpy as np
from collections import defaultdict
pd.set_option('display.float_format', lambda x: '%.2f' % x)
d = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(d)


def filter_ptype(x):
    if type(x) != str and math.isnan(x):
        return x
    x = str(x)
    if ',' in x:
        return str(int(float(x.split(',')[0])))
    return str(int(float(x)))


def mark_video_frames(pcap):
    pcap["is_video_pred"] = (
        pcap["udp.length"] > project_config['video_thresh']).astype(np.int32)
    return pcap


def filter_video_frames(pcap):
    pcap = pcap[pcap["udp.length"] > project_config['video_thresh']]
    return pcap

def filter_video_frames_rtp(pcap, vca):
    # if vca == 'webex':
    #     top_num = 1
    # else: 
    #     top_num = 2
    top_num = 1
    pcap['rtp.p_type'] = pcap['rtp.p_type'].apply(filter_ptype)
    top_x = pcap.groupby(['rtp.p_type'])['udp.length'].mean().nlargest(top_num).index.tolist()
    condition = ((pcap['rtp.p_type'].isin(top_x)))
    return pcap[condition]

def read_net_file(dataset, filename):
    csv_columns = ['frame.time_relative', 'frame.time_epoch', 'ip.src', 'ip.dst', 'ip.proto', 'ip.len', 'udp.srcport', 'udp.dstport', 'udp.length', 'rtp.ssrc', 'rtp.timestamp', 'rtp.seq', 'rtp.p_type', 'rtp.marker']
    df_net = pd.read_csv(filename, header=None, sep='\t',
                         names=csv_columns, lineterminator='\n', encoding='ascii')
    ip_addr = project_config['destination_ip'][dataset]
    if ip_addr == 'dynamic':
        ip_addr = df_net.groupby('ip.dst').agg({'udp.length': sum}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
    df_net = df_net[(df_net["ip.dst"] == ip_addr) & (~pd.isna(df_net["rtp.ssrc"]))]
    df_net = df_net[~df_net['ip.proto'].isna()]
    df_net['rtp.p_type'] = df_net['rtp.p_type'].apply(filter_ptype)
    df_net['rtp.p_type'] = df_net['rtp.p_type'].dropna()
    df_net['ip.proto'] = df_net['ip.proto'].astype(str)
    df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
    df_net['ip.proto'] = df_net['ip.proto'].apply(lambda x: int(float(x)))

    if df_net.empty:
        return
    return df_net


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

def get_net_stats(df_video, ft_end_col="frame_et"):
    ## frame duration calculations
    df_video = df_video.sort_values(by=ft_end_col)
    df_video["frame_size"] = df_video["frame_size"].apply(lambda x: float(x))
    df_video["frame_dur"] = df_video[ft_end_col].diff()

    df_video["avg_frame_dur"] = df_video["frame_dur"].rolling(30).mean()
    df_video = df_video.fillna(0)
    
    df_video["frame_dur"] = df_video["frame_dur"].apply(lambda x: 0 if x < 0 else x)

    ## freeze calculation
    df_video["is_freeze"] = df_video.apply(is_freeze, axis=1)
    df_video["freeze_dur"] = df_video.apply(get_freeze_dur, axis=1)
    
    ## obtain per second stats
    df_video["frame_et_int"] = df_video[ft_end_col].apply(lambda x: int(x)+1)
    df_grp = df_video.groupby("frame_et_int").agg({"frame_size" : ["sum", "count"], "is_freeze": "sum", 
                                             "freeze_dur": "sum", "frame_dur": "std"}).reset_index()
    
    ## rename columns
    df_grp.columns = ['_'.join(col).strip('_') for col in df_grp.columns.values]    
    df_grp = df_grp.rename(columns={'frame_size_count': 'predicted_framesReceivedPerSecond',
                                    'is_freeze': 'freeze_count',
                                    'frame_size_sum': 'predicted_bitrate',
                                    'freeze_dur': 'freeze_dur',
                                    'lost_frame': 'frames_lost',
                                    'frame_dur_std': 'predicted_frame_jitter'
                                   })
    df_grp['predicted_bitrate'] = df_grp['predicted_bitrate']*8
    df_grp['predicted_frame_jitter'] = df_grp['predicted_frame_jitter']*1000
    return df_grp
