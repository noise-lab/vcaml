from inspect import trace
import os
import time
from datetime import datetime
from traceback import print_tb
import numpy as np
from threading import Thread
import subprocess
import requests

net_profiles = [
    # tput, lat, loss
    ([5], [100], range(10, 21, 5)),
    # (range(8, 11), [0], [0]),
    # ([5], range(0, 151, 30), [0]),
    # ([5], [10], range(5, 21, 5)),
]

iface = 'lan1'
vca = 'webex'
duration = 50
num_reps = 5

total_runs = 0

for net_profile in net_profiles:
    tvar = net_profile[0]
    lat_var = net_profile[1]
    loss_var = net_profile[2]
    total_runs += len(tvar)*len(lat_var)*len(loss_var)*num_reps
    
print(total_runs)

curr_date = datetime.today().strftime('%Y-%m-%d')

def vcqoe(vca, duration, exp_id, link, throughput, latency, loss, exp_time):
    os.system("python3 vcqoe {} -p /home/noise/.config/google-chrome/Default -i -t {} -e {} -d -b chrome -id {} -c -tr {} 0 {} 0 {} 0 {}".format(vca, duration, exp_id, link, throughput, latency, loss, exp_time))

def get_total_meeting_duration():
    return int(read_remote_file('/home/noise/vca-webex-server/duration.txt').decode("utf-8").strip())

def get_meeting_link():
    info = read_remote_file('/home/noise/vca-webex-server/link.txt').decode("utf-8")
    print(f'Meeting Link Info = {info}')
    if info == '':
        return None
    ls = info.split('\n')
    return ls[0]

def get_meeting_start_time():
    info = read_remote_file('/home/noise/vca-webex-server/link.txt').decode("utf-8")
    print(f'Start time Info = {info}')
    if info == '':
        return 0
    ls = info.split('\n')
    return float(ls[1])

def read_remote_file(file_path):
    p = subprocess.Popen(["ssh", "noise@192.168.1.166", f"cat {file_path}"], stdout=subprocess.PIPE)
    out, err = p.communicate()
    return out

def go():
    if vca == 'teams':
        link = 'https://teams.microsoft.com/l/meetup-join/19%3A0AyOi6eZzNQDEDij0ItNwtihyE3UEufAd3XyHhBY_2w1%40thread.tacv2/1665672520951?context=%7B%22Tid%22%3A%2283b02c92-5f26-48ed-9e5b-6c2fca46a8e6%22%2C%22Oid%22%3A%22bbb88ca9-fbac-49ed-8d69-ee518382813e%22%7D'
    elif vca == 'meet':
        link = 'https://meet.google.com/mot-oknp-yum'
    elif vca == 'webex':
        link = None
    
    runs_so_far = 4
    
    for net_profile in net_profiles:
        tvar = net_profile[0]
        lat_var = net_profile[1]
        loss_var = net_profile[2]
        for throughput in tvar:
            for latency in lat_var:
                for loss in loss_var:
                    start = 1
                    if runs_so_far == 4:
                        start = 4
                    for rep_no in range(start, num_reps+1):
                        if vca == 'webex':
                            meeting_start_time = get_meeting_start_time()
                            current_time = time.time()
                            if (current_time-meeting_start_time) >= 36*60:
                                print('Previous call has expired! Initiating a new one...')
                                res = requests.get('http://192.168.1.166:5000/initiate-call')
                                print(res)
                                time.sleep(60)

                            link = get_meeting_link()
                            dur = get_total_meeting_duration()

                            if dur >= 36:
                                print('Pausing experiments until a new call link is received....')
                                time.sleep(245)
                                res = requests.get('http://192.168.1.166:5000/initiate-call')
                                print(res)
                                time.sleep(60)
                                link = get_meeting_link()
                            else:
                                print('Continuing with the old link...')

                        exp_time = int(time.time())
                        trace_label = '_'.join(map(str, [throughput, latency, loss, exp_time]))
                        exp_id = '{}_{}_{}_rep_{}'.format(curr_date, vca, trace_label, rep_no)
                        s = time.time()
                        print(f'\n######################## Running config {runs_so_far} of {total_runs} ##############################\n')
                        print(exp_id)
                        vcqoe(vca, duration, exp_id, link, throughput, latency, loss, exp_time)
                        time.sleep(duration+5)
                        runs_so_far += 1
                        dur = round(time.time() - s, 2)
                        print(f'\nThis config took {dur} seconds...\n')

if __name__ == '__main__':
    go()