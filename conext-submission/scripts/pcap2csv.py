import os, sys, glob

tshark_cmd = "tshark -r %s -d udp.port==1024-49152,rtp -t e -T fields -e frame.time_relative -e ip.src -e ip.dst -e ip.proto -e ip.len -e udp.srcport -e udp.dstport -e rtp.ssrc -e rtp.timestamp -e rtp.seq > %s"

dirname = "../data/*"#sys.argv[1]
filelist = glob.glob(dirname)
print(filelist)
for filename in filelist:
    if ".pcap" not in filename:
        continue
    print(filename)
    outfile = filename[:filename.rfind(".")] + ".csv"
    print(outfile)
    cmd = tshark_cmd % (filename, outfile)
    os.system(cmd)

