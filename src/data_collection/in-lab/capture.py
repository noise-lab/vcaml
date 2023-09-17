"""

    Functions for capture network traffic
"""
import time
import subprocess
import os


def capture_traffic(trace, args, new_dir):
    """ Capture network traffic using input filter """
    bname = os.path.basename(trace).split('.')[0]

    filename = f'{new_dir}/{args.website}-{args.browser}-{bname}-{int(time.time())}.pcap'
    
    cmd = f'tshark {args.filter} -w {filename} -a duration:{str(args.time)}' 
    res = subprocess.Popen(cmd, shell=True)
    res.wait()
    print("done")
