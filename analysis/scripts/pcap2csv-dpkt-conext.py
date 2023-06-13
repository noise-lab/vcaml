import os, sys
import binascii
import pandas as pd
import dpkt
import socket

cols= ['frame.time_relative','frame.time_epoch','ip.src','ip.dst','ip.proto','ip.len','udp.srcport','udp.dstport', 'udp.length','rtp.ssrc','rtp.timestamp','rtp.seq','rtp.p_type', 'rtp.marker','rtp.padding']
       # 'rtp.ext', 'rtp.ext_len' ,'rtcp.pt' ,'rtcp.version','rtcp.padding','rtcp.count','rtcp.packet_type', 'rtcp.length','rtcp.ssrc', 'rtcp.sender_info', 'rtcp.report_blocks', 'rtcp.sdes_chunks']

print(len(cols))
dname = sys.argv[1]

def inet_to_str(inet):
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)

for exp_dir in os.listdir(dname):
    vca = exp_dir.split('_')[1].lower()
    capture_dir = os.path.join(dname, exp_dir, vca, 'capture')
    for filename in os.listdir(capture_dir):
        data = []
        filename = os.path.join(capture_dir, filename)
        if ".pcap" not in filename:
            continue
        outfile = filename[:filename.rfind(".")] + ".csv"
        with open(filename, 'rb') as f:
            first = True
            ts0 = None
            pcap = dpkt.pcapng.Reader(f)
            for ts, buf in pcap:
                if first:
                    ts0 = ts
                eth=dpkt.ethernet.Ethernet(buf)
                ip_pkt = eth.data
                if type(ip_pkt.data) == dpkt.udp.UDP:
                    udp_pkt = ip_pkt.data
                    dst_ip = inet_to_str(ip_pkt.dst)
                    if dst_ip == '192.168.1.187':
                        if (vca == 'meet' and udp_pkt.sport == 3478) or vca == 'teams':
                            rtp_packet = dpkt.rtp.RTP(udp_pkt.data)
                            if rtp_packet.version == 2 and not (rtp_packet.pt >= 64 and rtp_packet.pt <= 95):
                                time_relative = ts - ts0
                                time_epoch = ts
                                src = inet_to_str(ip_pkt.src)
                                dst = inet_to_str(ip_pkt.dst)
                                proto = ip_pkt.get_proto(ip_pkt.p).__name__
                                ip_len = len(ip_pkt)
                                srcport = udp_pkt.sport
                                dstport = udp_pkt.dport
                                length = len(udp_pkt)
                                ssrc = rtp_packet.ssrc
                                timestamp = rtp_packet.ts
                                sequence = rtp_packet.seq
                                p_type = rtp_packet.pt
                                marker = rtp_packet.m
                                padding = rtp_packet.p
                                # ext = rtp_packet.extension
                                row = [
                                    time_relative,
                                    time_epoch,
                                    src,
                                    dst,
                                    proto,
                                    ip_len,
                                    srcport,
                                    dstport,
                                    length,
                                    ssrc,
                                    timestamp,
                                    sequence,
                                    p_type,
                                    marker,
                                    padding,
                                    # ext
                                ]
                                data.append(row)
                first = False
        df = pd.DataFrame(data, columns = cols)
        df.to_csv(outfile, index = False)
        print(outfile)