import time
from subprocess import Popen, PIPE
from shlex import split
import requests
import json
from trace_handler import TraceHandler
import os
from pathlib import Path

class Experiment:
    
    def __init__(self, vca, exp_label, duration=30, repeats=5, continue_from=0):
        self.vca = vca
        self.max_meeting_time = {'meet': 60, 'webex': 40}
        self.exp_label = exp_label
        self.duration = duration
        self.repeats = repeats
        self.continue_from = continue_from

    def vcqoe(self, link, trace):
        os.system("python3 vcqoe {} -p /home/noise/.config/google-chrome/Default -i -t {} -e {} -d -b chrome -id {} -c -tr {}".format(self.vca, self.duration, self.exp_label, link, trace))


    def create_remote_path(self, path):
        Popen(split(f'ssh root@192.168.1.1 "mkdir {path}"'), stdout=PIPE, stderr=PIPE)

    def execute(self):
        
        self.setup_call_server()
        first = True

        for rep in range(max(0, self.continue_from), self.repeats):

            if first:
                time.sleep(5)
                self.initiate_call()
                first = False
            
            print(f'############# {self.exp_label}: rep {rep} of {self.repeats} #############')
            start_time = time.time()
            config = self.read_config()

            call_id = f'rep_{rep}'
            rep_path = f'Data/{self.exp_label}/{call_id}'

            Path(rep_path).mkdir(parents=True, exist_ok=True)
            self.create_remote_path(f'/root/vcqoe/traces/{self.exp_label}')

            trace_handler = TraceHandler(
                host_path=rep_path, 
                router_path=f'root@192.168.1.1:~/vcqoe/traces/{self.exp_label}', 
                config=config
            )
            trace_handler.generate_profile(call_id)
            print('Polling for link...')
            link = self.poll_for_link()

            self.vcqoe(link, f'{self.exp_label}/{call_id}.csv')
            
            print('Cleaning up...')
            self.reset_receiver()

            end_time = time.time()
            dur = round(end_time-start_time, 2)
            print(f'\nThis rep took {dur} seconds...\n')

        self.reset_call_server()

    def initiate_call(self):
        print('Initiating a call...')
        requests.get(f'http://192.168.1.166:5000/initiate-call?vca={self.vca}')


    def read_config(self):
        with open(f'experiment_configs/{self.exp_label}.json') as fd:
            config = json.load(fd)
        return config
    
    def poll_for_link(self):
        if self.vca == 'webex' or self.vca == 'meet':
            link = self.get_meeting_link()
            dur = self.get_total_meeting_duration()

            if dur >= self.max_meeting_time[self.vca]-3:
                print('Pausing experiments until a new call link is received....')
                while True:
                    new_link = self.get_meeting_link()
                    if new_link != link:
                        print('Got the link...')
                        link = new_link
                        break
            else:
                print('Continuing with the old link...')
        elif self.vca == 'teams':
            link = 'https://teams.microsoft.com/l/meetup-join/19%3A0AyOi6eZzNQDEDij0ItNwtihyE3UEufAd3XyHhBY_2w1%40thread.tacv2/1665672520951?context=%7B%22Tid%22%3A%2283b02c92-5f26-48ed-9e5b-6c2fca46a8e6%22%2C%22Oid%22%3A%22bbb88ca9-fbac-49ed-8d69-ee518382813e%22%7D'
        return link

    def read_remote_file(self, file_path):
        p = Popen(["ssh", "noise@192.168.1.166", f"cat {file_path}"], stdout=PIPE)
        out, err = p.communicate()
        return out

    def get_total_meeting_duration(self):
        d = self.read_remote_file('/home/noise/vca-server/duration.txt').decode("utf-8").strip()
        if d == '':
            return 1000000
        return int(d)

    def get_meeting_link(self):
        info = self.read_remote_file('/home/noise/vca-server/link.txt').decode("utf-8")
        if info == '':
            return None
        ls = info.split('\n')
        return ls[0]

    def setup_call_server(self):
        self.reset_call_server()
        p = Popen(['ssh', 'noise@192.168.1.166', 'ps aux | pgrep -fl "call_server.py"'], stdout=PIPE)
        out, err = p.communicate()
        if 'python3' in out.decode('utf-8'):
            print('Server already running..')
        else:
            p = Popen(split('ssh noise@192.168.1.166 "/home/noise/vca-server/start_server.sh"'), stdout=PIPE, stderr=PIPE)
        time.sleep(2)
    
    def reset_call_server(self):
        Popen(split('ssh noise@192.168.1.166 "pkill python3"'), stdout=PIPE, stderr=PIPE)
        Popen(split('ssh noise@192.168.1.166 "pkill chrome"'), stdout=PIPE, stderr=PIPE)
        Popen(split('ssh noise@192.168.1.166 "> /home/noise/vca-server/duration.txt"'), stdout=PIPE, stderr=PIPE)
        Popen(split('ssh noise@192.168.1.166 "> /home/noise/vca-server/link.txt"'), stdout=PIPE, stderr=PIPE)

    def reset_receiver(self):
        Popen(['pkill','chrome'], stdout=PIPE, stderr=PIPE)
        Popen(['pkill', 'tshark'], stdout=PIPE, stderr=PIPE)

if __name__ == '__main__':
    vals = [x for x in range(800, 1001, 100)] + [2000, 5000, 10000]
    for i in vals:
        e = Experiment('meet', f'fixed_profile_{i}_0_0', duration=120, repeats=5)
        e.execute()

    # e = Experiment('meet', f'fixed_profile', duration=120, repeats=5)
    # e.reset_call_server()