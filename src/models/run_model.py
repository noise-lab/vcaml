from collections import defaultdict
from itertools import product
from datetime import datetime as dt
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from config import project_config
import pickle
import os
import time
import sys
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)
from util.file_processor import FileProcessor
from util.file_processor import FileValidator
from util.data_splitter import KfoldCVOverFiles
from models.rtp_ml import RTP_ML
from models.rtp_heuristic import RTP_Heuristic
from models.ip_udp_ml import IP_UDP_ML
from models.ip_udp_heuristic import IP_UDP_Heuristic

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
        self.trial_id = '_'.join(
            [metric, estimation_method, feature_subset_tag, data_bname, f'cv_{cv_index}'])

        self.intermediates_dir = f'{self.data_dir}_intermediates/{self.trial_id}'

        self.cv_index = cv_index

        self.model = None

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
                estimator = RandomForestClassifier(
                ) if self.metric == 'frameHeight' else RandomForestRegressor()
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
                estimator = RandomForestClassifier(
                ) if self.metric == 'frameHeight' else RandomForestRegressor()
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
                model = IP_UDP_Heuristic(
                    vca=vca, metric=self.metric, config=project_config, dataset=bname)

            elif self.estimation_method == 'rtp-heuristic':
                model = RTP_Heuristic(
                    vca=vca, metric=self.metric, config=project_config, dataset=bname)
            vca_model[vca] = model
        # self.save_intermediate(vca_model, 'vca_model')
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
                    mae = mean_absolute_error(
                        output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
                if self.metric == 'framesPerSecond' or self.metric == 'framesReceived' or self.metric == 'framesReceivedPerSecond' or self.metric == 'framesDecodedPerSecond' or self.metric == 'framesRendered':
                    acc = self.fps_prediction_accuracy(
                        output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
                    accs[vca].append(acc)
                    print(f'Accuracy = {round(acc, 2)}')
                if self.metric != 'frameHeight':
                    print(f'MAE = {round(mae, 2)}')
                    maes[vca].append(mae)
                else:
                    a = accuracy_score(
                        output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
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
        # self.save_intermediate(predictions, 'predictions')
        return predictions


if __name__ == '__main__':

    # Example usage

    metrics = ['framesReceivedPerSecond', 'bitrate',
               'frame_jitter', 'frameHeight']  # what to predict
    estimation_methods = ['ip-udp-ml']  # how to predict
    # groups of features as per `features.feature_extraction.py`
    feature_subsets = [['LSTATS', 'TSTATS']]
    data_dir = ['/home/taveesh/Documents/vcaml/data/hashed_real_world']

    bname = os.path.basename(data_dir[0])

    # Get a list of pairs (trace_csv_file, ground_truth)

    fp = FileProcessor(data_directory=data_dir[0])
    file_dict = fp.get_linked_files()

    # Create 5-fold cross validation splits and validate files. Refer `util/validator.py` for more details

    kcv = KfoldCVOverFiles(5, file_dict, project_config, bname)
    file_splits = kcv.split()

    vca_preds = defaultdict(list)

    param_list = [metrics, estimation_methods, feature_subsets, data_dir]

    # Run models over 5 cross validations

    for metric, estimation_method, feature_subset, data_dir in product(*param_list):
        if metric == 'frameHeight' and 'heuristic' in estimation_method:
            continue
        models = []
        cv_idx = 1
        for fsp in file_splits:
            model_runner = ModelRunner(
                metric, estimation_method, feature_subset, data_dir, cv_idx)
            vca_model = model_runner.train_model(fsp)
            predictions = model_runner.get_test_set_predictions(fsp, vca_model)
            models.append(vca_model)

            for vca in predictions:
                vca_preds[vca].append(pd.concat(predictions[vca], axis=0))

            cv_idx += 1
