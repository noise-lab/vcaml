import pandas as pd 
import numpy as np 
import os
import time
import json

class TraceHandler:

    def __init__(self, host_path, router_path, config):
        self.host_path = host_path
        self.router_path = router_path
        self.config = config

    def generate_profile(self, identifier):
        config = self.config
        profile = []
        for idx, f in enumerate(config['functions']):
            if 'default' in f:
                for _ in range(config['slots'][idx]):
                    profile.append((-1, -1, -1))
            elif 'fixed' in f:
                th = f['fixed'][0]
                lat = f['fixed'][1]
                loss = f['fixed'][2]
                for _ in range(config['slots'][idx]):
                    profile.append((th, lat, loss))
            elif 'linear' in f:
                th, lat, loss = [], [], []
                th_profile = f['linear'][0]
                lat_profile = f['linear'][1]
                loss_profile = f['linear'][2]
                
                for i in range(config['slots'][idx]):
                    rate, start = th_profile['rate'], th_profile['start']
                    th.append(start + i*rate)
                
                for i in range(config['slots'][idx]):
                    rate, start = lat_profile['rate'], lat_profile['start']
                    lat.append(start + i*rate)

                for i in range(config['slots'][idx]):
                    rate, start = loss_profile['rate'], loss_profile['start']
                    loss.append(start + i*rate)

                for i in range(config['slots'][idx]):
                    profile.append((th[i], lat[i], loss[i]))

            elif 'random' in f:
                th, lat, loss = [], [], []
                th_profile = f['random'][0]
                lat_profile = f['random'][1]
                loss_profile = f['random'][2]

                mean, std = th_profile['mean'], th_profile['std']
                th += map(int, list(np.random.normal(mean, std, config['slots'][idx])))

                mean, std = lat_profile['mean'], lat_profile['std']
                lat += map(int, list(np.random.normal(mean, std, config['slots'][idx])))

                mean, std = loss_profile['mean'], loss_profile['std']
                loss += map(int, list(np.random.normal(mean, std, config['slots'][idx])))

                for i in range(config['slots'][idx]):
                    profile.append((th[i], lat[i], loss[i]))

        self.place_spec_file(profile, identifier)

    def place_spec_file(self, profile, identifier):
        df = pd.DataFrame(profile)
        df.to_csv(f'{self.host_path}/{identifier}.csv', header=False, index=False)

        print(f'Copying {identifier}.csv to {self.router_path}...')
        command = f'scp -r {self.host_path}/{identifier}.csv {self.router_path}'
        os.system(command)
        time.sleep(1)

def generate_fixed_config():
    temp = {"slots": [30],"functions": [{"fixed": [1000, 0, 0]}]}
    vals = [x for x in range(200, 1001, 100)] + [2000, 5000, 10000]
    for v in vals:
        d = dict(temp)
        d['functions'][0]['fixed'][0] = v
        with open(f'experiment_configs/fixed_profile_{v}_0_0.json', 'w') as fd:
            json.dump(d, fd)

if __name__ == '__main__':
    generate_fixed_config()
    # handler = TraceHandler(host_path=os.getcwd(), router_path='root@192.168.1.1:~/vcqoe/traces/test', config={
    #     'slots': [10, 10, 10, 10],
    #     'functions': [
    #         {'fixed': [300, 10, 2]},
    #         {'linear': [{'rate': 1, 'start': 100}, {'rate': 2, 'start': 5},{'rate': 3, 'start': 1}]},
    #         {'random':[{'mean': 300, 'std': 5}, {'mean': 10, 'std': 5}, {'mean': 2, 'std': 0.5}]}
    #     ]
    # })
    # handler.generate_profile('trace')