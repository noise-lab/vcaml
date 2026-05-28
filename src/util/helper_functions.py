import logging
import math
import sys
from collections import defaultdict
from os.path import abspath, dirname

import numpy as np
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.2f' % x)

d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

from config import project_config

logger = logging.getLogger(__name__)


def filter_ptype(x):
    if type(x) != str and math.isnan(x):
        return x
    x = str(x)
    if ',' in x:
        return str(int(float(x.split(',')[0])))
    return str(int(float(x)))


def mark_video_frames(pcap):
    pcap['is_video_pred'] = (
        pcap['udp.length'] > project_config['video_thresh']).astype(np.int32)
    return pcap


def filter_video_frames(pcap):
    return pcap[pcap['udp.length'] > project_config['video_thresh']]


def filter_video_frames_rtp(pcap, vca):
    pcap['rtp.p_type'] = pcap['rtp.p_type'].apply(filter_ptype)
    topPtype = (pcap.groupby('rtp.p_type')['udp.length']
                .mean().nlargest(1).index.tolist())
    return pcap[pcap['rtp.p_type'].isin(topPtype)]


def read_net_file(dataset, filename):
    df_net = pd.read_csv(filename)
    try:
        dstIp = (df_net.groupby('ip.dst')
                 .agg({'udp.length': 'sum'})
                 .reset_index()
                 .sort_values(by='udp.length', ascending=False)
                 .head(1)['ip.dst'].iloc[0])
    except (IndexError, KeyError):
        return None
    df_net = df_net[(df_net['ip.dst'] == dstIp) & (~pd.isna(df_net['rtp.ssrc']))]
    df_net = df_net[~df_net['ip.proto'].isna()]
    df_net['rtp.p_type'] = df_net['rtp.p_type'].apply(filter_ptype).dropna()
    df_net['ip.proto'] = df_net['ip.proto'].astype(str)
    df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
    df_net['ip.proto'] = df_net['ip.proto'].apply(lambda x: int(float(x)))
    return df_net if not df_net.empty else None


def is_freeze(x):
    return 1 if x['frame_dur'] > max(3 * x['avg_frame_dur'],
                                      x['avg_frame_dur'] + 0.150) else 0


def get_freeze_dur(x):
    return x['frame_dur'] if x['is_freeze'] == 1 else 0


def get_net_stats(df_video, ftEndCol='frame_et'):
    df_video = df_video.sort_values(by=ftEndCol).copy()
    df_video['frame_size'] = df_video['frame_size'].astype(float)
    df_video['frame_dur'] = df_video[ftEndCol].diff()
    df_video['avg_frame_dur'] = df_video['frame_dur'].rolling(30).mean()
    df_video = df_video.fillna(0)
    df_video['frame_dur'] = df_video['frame_dur'].clip(lower=0)
    df_video['is_freeze'] = df_video.apply(is_freeze, axis=1)
    df_video['freeze_dur'] = df_video.apply(get_freeze_dur, axis=1)
    df_video['frame_et_int'] = df_video[ftEndCol].apply(lambda x: int(x) + 1)
    df_grp = (df_video.groupby('frame_et_int')
              .agg({'frame_size': ['sum', 'count'], 'is_freeze': 'sum',
                    'freeze_dur': 'sum', 'frame_dur': 'std'})
              .reset_index())
    df_grp.columns = ['_'.join(col).strip('_') for col in df_grp.columns.values]
    df_grp = df_grp.rename(columns={
        'frame_size_count': 'predicted_framesReceivedPerSecond',
        'is_freeze': 'freeze_count',
        'frame_size_sum': 'predicted_bitrate',
        'freeze_dur': 'freeze_dur',
        'frame_dur_std': 'predicted_frame_jitter',
    })
    df_grp['predicted_bitrate'] *= 8
    df_grp['predicted_frame_jitter'] *= 1000
    return df_grp
