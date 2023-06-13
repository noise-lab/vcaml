import os, sys, glob

def get_pcap(fname)
    tshark_cmd = "tshark -r %s -d udp.port==1024-49152,rtp -t e -T fields -e frame.time_relative -e frame.time_epoch -e ip.src -e ip.dst -e ip.proto -e ip.len -e udp.srcport -e udp.dstport -e udp.length -e rtp.ssrc -e rtp.timestamp -e rtp.seq -e rtp.p_type -e rtp.marker -e rtp.padding -e rtp.ext -e rtp.ext.len -e rtcp.pt -e rtcp.senderssrc -e rtcp.ssrc.high_seq -e rtcp.ssrc.dlsr -e rtcp.psfb.fmt -e rtcp.rtpfb.fmt -e rtcp.sender.octetcount > %s"
    outfile = fname[:fname.rfind(".")] + ".csv"
    cmd = tshark_cmd % (fname, outfile)
    os.system(cmd)

'''
dname = sys.argv[1]
vca = sys.argv[2]

for exp_dir in os.listdir(dname):
    capture_dir = os.path.join(dname, exp_dir, vca, 'captures')
    for filename in os.listdir(capture_dir):
        filename = os.path.join(capture_dir, filename)
        if ".pcap" not in filename:
            continue
        print(filename)
        outfile = filename[:filename.rfind(".")] + ".csv"
        print(outfile)
        cmd = tshark_cmd % (filename, outfile)
        os.system(cmd)
'''