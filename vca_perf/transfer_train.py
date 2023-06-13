import os, glob
from config import project_config
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
import time 

train_dir = '/home/taveesh/Documents/Projects/vca-qoe-inference/data/IMC_Lab_data'
test_dir = '/home/taveesh/Documents/Projects/vca-qoe-inference/data/Sensitivity_Data'

metrics = ['framesReceivedPerSecond', 'bitrate', 'frame_jitter', 'frameHeight']
models = ['rtp_ml', 'ml']


def get_lab_feature_csvs(dir_name, model):
    bname = os.path.basename(dir_name)
    vca_dict = {'meet': [], 'teams': [], 'webex': []}
    for vca in vca_dict.keys():
        for feature_file in glob.glob(f"{dir_name}/*/{vca}/captures/*_{model}_{vca}_{bname}.csv"):
            if model == 'ml' and 'rtp_ml' in feature_file:
                continue
            vca_dict[vca].append(feature_file)
    return vca_dict

train_traces = get_lab_feature_csvs(train_dir, 'rtp_ml')
test_traces = get_lab_feature_csvs(test_dir, 'rtp_ml')

for model in models:
    for metric in metrics:
        data = []
        for vca in train_traces.keys():
            for feature_file in train_traces[vca]:
                feature_matrix = pd.read_csv(feature_file, index_col=False)
                feature_matrix = feature_matrix[project_config['features_list'][model] + ['et', 'ts', metric]]
                data.append(feature_matrix)
            print('\nFitting the model...\n')
            X = pd.concat(data, axis=0)
            X = X.dropna()
            y = X[metric]
            X = X[X.columns.difference([metric, 'et', 'ts', 'file'])]
            if metric == 'framesPerSecond' or metric == 'framesRendered' or metric == 'framesReceived' or metric == 'framesReceivedPerSecond':
                y = y.apply(lambda x: round(x))
            t1 = time.time()
            if metric == 'frameHeight':
                estimator = RandomForestClassifier()
            else:
                estimator = RandomForestRegressor()
            estimator.fit(X, y)
            dur = round(time.time() - t1, 2)
            print(f'\nModel training took {dur} seconds.\n')

            print('\nTesting begins...\n')
            
            for feature_file in test_traces[vca]:
                feature_matrix = pd.read_csv(feature_file, index_col=False)
                X = feature_matrix[project_config['features_list'][model] + ['et', 'ts', metric]]
                X = X.dropna()
                timestamps = X['ts']
                y_test = X[metric]
                X = X[X.columns.difference([metric, 'et', 'ts', 'file'])]
                if X.shape[0] == 0:
                    continue
                y_pred = estimator.predict(X)
                if metric == 'framesPerSecond' or metric == 'framesRendered' or metric == 'framesReceived' or metric == 'framesReceivedPerSecond':
                    y_pred = list(map(lambda x: round(x), y_pred))
                    y_test = y_test.apply(lambda x: round(x))
                X[metric+f'_{model}'] = y_pred
                X[metric+'_gt'] = y_test
                X['timestamp'] = timestamps
                X['file'] = feature_file
                X = X[[metric+f'_{model}', metric+'_gt', 'timestamp', 'file']]
                print(feature_file)
                print(f'Metric = {metric} || Model = {model} || VCA = {vca}')
                if metric == 'frameHeight':
                    acc = accuracy_score(y_pred, y_test)
                    acc = round(acc, 2)
                    line = f'Accuracy = {acc}\n'
                else:
                    mae = mean_absolute_error(y_pred, y_test)
                    mae = round(mae, 2)
                    line = f'MAE = {mae}\n'
                print(line)
