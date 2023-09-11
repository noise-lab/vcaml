import os
import time
from datetime import datetime
import numpy as np
from threading import Thread
import subprocess
import requests

iface = 'lan1'
vca = 'meet'
trace_folder = 'zero_loss_traces'
total_runs = 192

max_duration = {'meet': 60, 'webex': 40}


curr_date = datetime.today().strftime('%Y-%m-%d')

def vcqoe(vca, duration, exp_id, link, throughput, tj, latency, lj, loss, loj, runs_so_far):
    os.system("python3 vcqoe {} -p /home/noise/.config/google-chrome/Default -i -t {} -e {} -d -b chrome -id {} -c -tf {} -tr {} {} {} {} {} {} {} {}".format(vca, duration, exp_id, link, trace_folder, throughput, tj, latency, lj, loss, loj, duration, runs_so_far))

def get_total_meeting_duration():
    d = read_remote_file('/home/noise/vca-webex-server/duration.txt').decode("utf-8").strip()
    if d == '':
        return 1000000
    return int(d)

def get_meeting_link():
    info = read_remote_file('/home/noise/vca-webex-server/link.txt').decode("utf-8")
    if info == '':
        return None
    ls = info.split('\n')
    return ls[0]

def get_meeting_start_time():
    info = read_remote_file('/home/noise/vca-webex-server/link.txt').decode("utf-8")
    if info == '':
        return 0
    ls = info.split('\n')
    return float(ls[1])

def read_remote_file(file_path):
    p = subprocess.Popen(["ssh", "noise@192.168.1.166", f"cat {file_path}"], stdout=subprocess.PIPE)
    out, err = p.communicate()
    return out

def get_trace_filename(trace_num):
    return [x for x in os.listdir(trace_folder) if x.endswith('_'+str(trace_num)+'.csv')][0]

def go():
    if vca == 'teams':
        link = 'https://teams.microsoft.com/l/meetup-join/19%3A0AyOi6eZzNQDEDij0ItNwtihyE3UEufAd3XyHhBY_2w1%40thread.tacv2/1665672520951?context=%7B%22Tid%22%3A%2283b02c92-5f26-48ed-9e5b-6c2fca46a8e6%22%2C%22Oid%22%3A%22bbb88ca9-fbac-49ed-8d69-ee518382813e%22%7D'
    elif vca == 'webex' or vca == 'meet':
        link = None
    
    runs_so_far = 1
    # to_run = list(range(1, 89)) + [93, 319,320,321,322,323, 324,325, 326, 327,328, 329, 330,487,490,492,493]
    # link = get_meeting_link()
    # if link is None:
    print('Initiating a call...')
    res = requests.get(f'http://192.168.1.166:5000/initiate-call?vca={vca}')
    print(res)
    
    time.sleep(70)

    # for runs_so_far in to_run:
    while runs_so_far <= total_runs:
        if vca == 'webex' or vca == 'meet':

            link = get_meeting_link()
            dur = get_total_meeting_duration()

            if dur >= max_duration[vca]-3:
                print('Pausing experiments until a new call link is received....')
                while True:
                    new_link = get_meeting_link()
                    if new_link != link:
                        print('Got the link...')
                        link = new_link
                        break
            else:
                print('Continuing with the old link...')

        exp_time = int(time.time())
        trace_label = get_trace_filename(runs_so_far)[:-4]
        tsp = trace_label.split('_')
        throughput = int(tsp[0])
        tj = int(tsp[1]) 
        latency = int(tsp[2])
        lj = int(tsp[3])
        loss = int(tsp[4])
        loj = int(tsp[5])
        duration = int(tsp[6])
        exp_id = '{}_{}_{}'.format(curr_date, vca, trace_label)
        s = time.time()
        print(f'\n######################## Running config {runs_so_far} of {total_runs} ##############################\n')
        print(exp_id)
        vcqoe(vca, duration, exp_id, link, throughput, tj, latency, lj, loss, loj, runs_so_far)
        res = subprocess.Popen(['pkill chrome'], shell=True)
        res = subprocess.Popen(['pkill tshark'], shell=True)
        runs_so_far += 1
        dur = round(time.time() - s, 2)
        print(f'\nThis config took {dur} seconds...\n')
        time.sleep(5)

if __name__ == '__main__':
    go()