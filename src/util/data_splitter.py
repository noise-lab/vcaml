from .validator import FileValidator
import random
import os
import numpy as np
from sklearn.model_selection import KFold

"""
Implements different types of splits on data files:
    1. Random split
    2. Split over files of similar network configuration
    3. Split over files with either side of the throughput jitter threshold.
    
Input is a dictionary indexed by VCA containing linked files
"""


class DataSplitter:
    def __init__(self, file_dict, criterion, split_ratio, config):
        self.file_dict = file_dict
        self.split_ratio = split_ratio
        self.criterion = criterion
        self.config = config
        self.dataset = dataset

    def filter_files(self):
        print('\nFiltering out anomalous files...\n')
        filtered_files = {}
        for vca in self.file_dict:
            if vca not in filtered_files:
                filtered_files[vca] = []
            for pcap_file, csv_file, webrtc_file in self.file_dict[vca]:
                print(pcap_file)
                validator = FileValidator(file_tuple=(
                    pcap_file, csv_file, webrtc_file), config=self.config, dataset=self.dataset)
                if validator.validate():
                    filtered_files[vca].append(
                        (pcap_file, csv_file, webrtc_file))
        return filtered_files

    def split(self):
        self.file_dict = self.filter_files()
        if self.criterion == 'random':
            criterion = RandomSplitCriterion(self.file_dict, self.split_ratio)
        elif self.criterion == 'network-condition':
            criterion = NetworkConditionSplitCriterion(
                self.file_dict, self.split_ratio)
        elif self.criterion == 'throughput-std':
            criterion = ThroughputStdSplitCriterion(self.file_dict)
        return criterion.split()


class RandomSplitCriterion:

    def __init__(self, file_dict, split_ratio):
        self.file_dict = file_dict
        self.split_ratio = split_ratio

    def split(self):
        splits = {}
        for vca in self.file_dict:
            n = len(self.file_dict[vca])
            num_samples = int(self.split_ratio*n)
            file_set = set(self.file_dict[vca])
            split_1 = set(random.sample(self.file_dict[vca], num_samples))
            split_2 = file_set.difference(split_1)
            splits[vca] = {'train': list(split_1), 'test': list(split_2)}
        return splits


class NetworkConditionSplitCriterion:

    def __init__(self, file_dict, split_ratio):
        self.file_dict = file_dict
        self.split_ratio = split_ratio

    def split(self):
        splits = [{}]
        for vca in self.file_dict:
            splits[0][vca] = {'train': [], 'test': []}
            net_cond_files = {}
            net_cond_strs = set()
            for file_tuple in self.file_dict[vca]:
                csv_file = file_tuple[1]
                bname = os.path.basename(csv_file)
                bname = bname.split('.')[0]
                bname = bname.split('-')[2]
                bname = '_'.join(bname.split('_')[:5])
                net_cond_strs.add(bname)

            for net_cond in net_cond_strs:
                tups = [x for x in self.file_dict[vca]
                        if 'chrome-'+net_cond in x[1]]
                n = len(tups)
                num_samples = int(self.split_ratio*n)
                splits[0][vca]['train'] += tups[:num_samples]
                splits[0][vca]['test'] += tups[num_samples:]
        return splits


class ThroughputStdSplitCriterion:
    def __init__(self, file_dict):
        self.file_dict = file_dict

    def split(self):
        splits = {}
        for vca in self.file_dict:
            splits[vca] = {'train': [], 'test': []}
            devs = []
            for file_tuple in self.file_dict[vca]:
                # Get a list of standard deviations in throughput
                pcap_file = file_tuple[0]
                fsp = os.path.basename(pcap_file).split('_')
                devs.append(int(fsp[1]))
            devs = np.array(devs)
            median_dev = np.median(devs)
            for file_tuple in self.file_dict[vca]:
                pcap_file = file_tuple[0]
                fsp = os.path.basename(pcap_file).split('_')
                if int(fsp[1]) <= median_dev:
                    splits[vca][0]['train'].append(file_tuple)
                else:
                    splits[vca][0]['test'].append(file_tuple)
        return splits


class KfoldCVOverFiles:

    def __init__(self, k, file_dict, config, dataset):
        self.k = k
        self.file_dict = file_dict
        self.config = config
        self.dataset = dataset

    def filter_files(self):
        print('\nFiltering out anomalous files...\n')
        filtered_files = {}
        for vca in self.file_dict:
            if vca not in filtered_files:
                filtered_files[vca] = []
            for file_tuple in self.file_dict[vca]:
                csv_file = file_tuple[0]
                webrtc_file = file_tuple[1]
                validator = FileValidator(
                    file_tuple=file_tuple, config=self.config, dataset=self.dataset)
                if validator.validate():
                    filtered_files[vca].append(file_tuple)
        return filtered_files

    def split(self):
        self.file_dict = self.filter_files()
        cross_validation_splits = [{} for _ in range(self.k)]
        for vca in self.file_dict:
            kf = KFold(n_splits=self.k, random_state=None)
            X = np.array(self.file_dict[vca])
            idx = 1
            for train_index, test_index in kf.split(X):
                cross_validation_splits[idx-1][vca] = {}
                X_train, X_test = list(X[train_index]), list(X[test_index])
                cross_validation_splits[idx-1][vca]['train'] = X_train
                cross_validation_splits[idx-1][vca]['test'] = X_test
                print(
                    f'\nSplit # {idx} | VCA = {vca} | n_files_train = {len(X_train)} | n_files_test = {len(X_test)}\n')
                idx += 1
        return cross_validation_splits
