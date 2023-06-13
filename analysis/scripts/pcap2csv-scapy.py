import os, sys
from scapy.all import IP, UDP, RTP, RTPExtension, rdpcap, hexdump, import_hexcap, Ether
from scapy.utils import PcapNgReader
import binascii
import pandas as pd

cols= ['frame.time_relative','frame.time_epoch','ip.src','ip.dst','ip.proto','ip.len','udp.srcport','udp.dstport', 'udp.length','rtp.ssrc','rtp.timestamp','rtp.seq','rtp.p_type', 'rtp.marker','rtp.padding','rtp.ext']
       # 'rtp.ext_len' ,'rtcp.pt' ,'rtcp.version','rtcp.padding','rtcp.count','rtcp.packet_type', 'rtcp.length','rtcp.ssrc', 'rtcp.sender_info', 'rtcp.report_blocks', 'rtcp.sdes_chunks']

print(len(cols))
dname = sys.argv[1]
vca = sys.argv[2]

for exp_dir in os.listdir(dname):
    capture_dir = os.path.join(dname, exp_dir, vca, 'captures')
    for filename in os.listdir(capture_dir):
        data = []
        filename = os.path.join(capture_dir, filename)
        if ".pcap" not in filename:
            continue
        print(filename)
        pkts = rdpcap(filename)
        outfile = filename[:filename.rfind(".")] + ".csv"
        for pkt in pkts:
                if pkt.haslayer(IP):
                    ip_pkt = pkt[IP]
                    if pkt.haslayer(UDP):
                        udp_pkt = pkt[UDP]
                    if ip_pkt.dst == '192.168.1.187':
                        if (vca == 'meet' and udp_pkt.sport == 3478) or vca == 'teams':
                            try:
                                rtp_packet = RTP(bytes(udp_pkt.payload))
                                if rtp_packet.version == 2 and not (rtp_packet.payload_type >= 64 and rtp_packet.payload_type <= 95):
                                    time_relative = pkt.time - pkts[0].time
                                    time_epoch = pkt.time
                                    src = ip_pkt.src
                                    dst = ip_pkt.dst
                                    proto = ip_pkt.proto
                                    ip_len = len(ip_pkt)
                                    srcport = udp_pkt.sport
                                    dstport = udp_pkt.dport
                                    length = len(udp_pkt)
                                    ssrc = rtp_packet.sourcesync
                                    timestamp = rtp_packet.timestamp
                                    sequence = rtp_packet.sequence
                                    p_type = rtp_packet.payload_type
                                    marker = rtp_packet.marker
                                    padding = rtp_packet.padding
                                    ext = rtp_packet.extension
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
                                        ext
                                    ]
                                    data.append(row)
                            except:
                                continue
        df = pd.DataFrame(data, columns = cols)
        df.to_csv(outfile, index = False)
        print(outfile)