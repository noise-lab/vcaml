import os
import shutil


def convert(x):
    for t in os.listdir(x):
        if t.endswith('.pcap'):
            # if f'{t[:-5]}' in os.listdir(x):
            #     continue
            print(f'{t}->{t[:-5]}.csv')
            tshark_cmd = f"tshark -r {x}/{t} -d udp.port==1024-49152,rtp -t e -T fields -e frame.time_relative -e frame.time_epoch -e ip.src -e ip.dst -e ip.proto -e ip.len -e udp.srcport -e udp.dstport -e udp.length -e rtp.ssrc -e rtp.timestamp -e rtp.seq -e rtp.p_type -e rtp.marker > {x}/{t[:-5]}.csv"
            os.system(tshark_cmd)
