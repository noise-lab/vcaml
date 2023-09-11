import time
import sys
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

import os
import pickle
from config import project_config
from models.ip_udp_heuristic import IP_UDP_Heuristic
from models.ip_udp_ml import IP_UDP_ML
from models.rtp_heuristic import RTP_Heuristic
from models.rtp_ml import RTP_ML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from pathlib import Path
import pandas as pd
from datetime import datetime as dt
from itertools import product

class ModelRunner:

    def __init__(self, metric, estimation_method, feature_subset, data_dir, cv_index):

        self.metric = metric
        self.estimation_method = estimation_method
        self.feature_subset = 'none' if feature_subset is None else feature_subset
        self.data_dir = data_dir

        if feature_subset:
            feature_subset_tag = '-'.join(feature_subset)
        else:
            feature_subset_tag = 'none'

        data_bname = os.path.basename(data_dir)
        self.trial_id = '_'.join([metric, estimation_method, feature_subset_tag, data_bname])

        self.intermediates_dir = f'{self.data_dir}_intermediates/{self.trial_id}'

        self.cv_index = cv_index
        
        self.model = None
        path = Path(self.intermediates_dir)
        path.mkdir(parents=True, exist_ok=True)

    def save_intermediate(self, data_object, pickle_filename):
        pickle_filename = f'{self.trial_id}_{pickle_filename}'
        with open(f'{self.intermediates_dir}/{pickle_filename}.pkl', 'wb') as fd:
            pickle.dump(data_object, fd)

    def load_intermediate(self, pickle_filename):
        with open(f'{self.intermediates_dir}/{pickle_filename}.pkl', 'rb') as fd:
            data_object = pickle.load(fd)
        return data_object

    def fps_prediction_accuracy(self, pred, truth):
        n = len(pred)
        df = pd.DataFrame({'pred': pred.to_numpy(), 'truth': truth.to_numpy()})
        df['deviation'] = df['pred']-df['truth']
        df['deviation'] = df['deviation'].abs()
        return len(df[df['deviation'] <= 2])/n

    def train_model(self, split_files):
        bname = os.path.basename(self.data_dir)
        vca_model = {}
        importances = {}
        for vca in split_files:
            print(f'\nVCA = {vca}')
            if self.estimation_method == 'ip-udp-ml':
                estimator = RandomForestClassifier() if self.metric == 'frameHeight' else RandomForestRegressor()
                model = IP_UDP_ML(
                    vca=vca,
                    feature_subset=self.feature_subset,
                    estimator=estimator,
                    config=project_config,
                    metric=self.metric,
                    dataset=bname
                )
                model.train(split_files[vca]['train'])

            elif self.estimation_method == 'rtp-ml':
                estimator = RandomForestClassifier() if self.metric == 'frameHeight' else RandomForestRegressor()
                model = RTP_ML(
                    vca=vca,
                    feature_subset=self.feature_subset,
                    estimator=estimator,
                    config=project_config,
                    metric=self.metric,
                    dataset=bname
                )
                model.train(split_files[vca]['train'])

            elif self.estimation_method == 'ip-udp-heuristic':
                model = IP_UDP_Heuristic(vca=vca, metric=self.metric, config=project_config, dataset=bname)

            elif self.estimation_method == 'rtp-heuristic':
                model = RTP_Heuristic(vca=vca, metric=self.metric, config=project_config, dataset=bname)
            vca_model[vca] = model
        self.save_intermediate(vca_model, 'vca_model')
        return vca_model

    def get_test_set_predictions(self, split_files, vca_model):
        predictions = {}
        maes = {}
        accs = {}
        for vca in split_files:
            predictions[vca] = []
            maes[vca] = []
            accs[vca] = []
            idx = 1
            total = len(split_files[vca]['test'])
            for file_tuple in split_files[vca]['test']:
                print(file_tuple[0])
                model = vca_model[vca]
                print(
                    f'Testing {self.estimation_method} on file {idx} out of {total}...')
                output = model.estimate(file_tuple)
                if output is None:
                    idx += 1
                    predictions[vca].append(output)
                    continue
                if self.metric != 'frameHeight':
                    mae = mean_absolute_error(output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
                if self.metric == 'framesPerSecond' or self.metric == 'framesReceived' or self.metric == 'framesReceivedPerSecond' or self.metric == 'framesDecodedPerSecond' or self.metric == 'framesRendered':
                    acc = self.fps_prediction_accuracy(output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
                    accs[vca].append(acc)
                    print(f'Accuracy = {round(acc, 2)}')
                if self.metric != 'frameHeight':
                    print(f'MAE = {round(mae, 2)}')
                    maes[vca].append(mae)
                else:
                    a = accuracy_score(output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
                    print(f'Accuracy = {round(a, 2)}')
                    accs[vca].append(a)
                idx += 1
                predictions[vca].append(output)
        for vca in split_files:
            if self.metric == 'frameHeight':
                mae_avg = "None"
            else:
                mae_avg = round(sum(maes[vca])/len(maes[vca]), 2)
            accuracy_str = ''
            if self.metric == 'framesPerSecond' or self.metric == 'framesReceivedPerSecond' or self.metric == 'framesDecodedPerSecond' or self.metric == 'framesRendered':
                acc_avg = round(100*sum(accs[vca])/len(accs[vca]), 2)
                accuracy_str = f'|| Accuracy_avg = {acc_avg}'
            line = f'{dt.now()}\tVCA: {vca} || Experiment : {self.trial_id} || MAE_avg = {mae_avg} {accuracy_str}\n'
            with open('log.txt', 'a') as fd:
                fd.write(line)
        self.save_intermediate(predictions, 'predictions')
        return predictions

if __name__ == '__main__':

    # Example usage

    metrics = ['framesReceivedPerSecond', 'bitrate', 'frame_jitter', 'frameHeight']  # what to predict
    estimation_methods = ['ip-udp-heuristic', 'rtp-heuristic', 'ip-udp-ml', 'rtp-ml']  # how to predict
    feature_subsets = [['LSTATS', 'TSTATS']] # groups of features as per `features.feature_extraction.py`
    data_dir = '../data/raw/IMC_Lab_Data'

    param_list = [metrics, estimation_methods, feature_subsets, data_dir]
    for metric, estimation_method, feature_subset, data_dir in product(*param_list):
        if metric == 'frameHeight' and 'heuristic' in estimation_method:
            continue
        cv_idx = 1