import pickle
import pandas as pd
import itertools
import os
from model_trial import Experiment
from util.data_splitter import KfoldCVOverFiles, NetworkConditionSplitCriterion
from util.file_processor import FileProcessor
from util.preprocessor import Preprocessor
from config import project_config
from pathlib import Path
from sklearn.metrics import mean_absolute_error, accuracy_score


def fps_prediction_accuracy(pred, truth):
    if len(pred) != len(truth):
        raise ValueError('Length mismatch for predictions and ground truth!')
    n = len(pred)
    df = pd.DataFrame({'pred': pred.to_numpy(), 'truth': truth.to_numpy()})
    df['deviation'] = df['pred']-df['truth']
    df['deviation'] = df['deviation'].abs()
    return 100*len(df[df['deviation'] <= 2])/n

def load_previous_split(model_trial, data_dir):
    intermediates_dir = f'{data_dir}_intermediates/{model_trial.trial_id}'
    with open(f'{intermediates_dir}/{model_trial.trial_id}_split_files.pkl', 'rb') as fd:
        split_files = pickle.load(fd)
    return split_files

def save_cv_splits(cv_splits, data_dir):
    cv_dir = f'{data_dir}_cv_splits'
    path = Path(cv_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(f'{cv_dir}/cv_file_splits.pkl', 'wb') as fd:
        pickle.dump(cv_splits, fd)

def load_cv_splits(data_dir):
    cv_dir = f'{data_dir}_cv_splits'
    with open(f'{cv_dir}/cv_file_splits.pkl', 'rb') as fd:
        cv_splits = pickle.load(fd)
    return cv_splits

# data_dirs =['/home/taveesh/Documents/Projects/vca-qoe-inference/data/Data_combined', '/home/taveesh/Documents/Projects/vca-qoe-inference/data/conext_data']
# preprocess_datasets = [False, False]
data_dirs =['/home/taveesh/Documents/Projects/vca-qoe-inference/data/IMC_Lab_data']
preprocess_datasets = [False]

for i, data_dir in enumerate(data_dirs):
    if preprocess_datasets[i]:
        bname = os.path.basename(data_dir)
        fp = FileProcessor(data_directory=data_dir, data_format=project_config['data_format'][bname])
        file_dict = fp.get_linked_files()
        sp = NetworkConditionSplitCriterion(file_dict, 0.5)
        fsplits = sp.split()
        # kcv = KfoldCVOverFiles(k = 5, config=project_config, file_dict=file_dict, dataset=bname)
        # cv_splits = kcv.split()
        save_cv_splits(fsplits, data_dir)
        # fp = FileProcessor(data_dir, project_config['data_format'][bname])
        # files = fp.get_linked_files()
        # for vca in files:
        #     p = Preprocessor(vca, ['LSTATS', 'TSTATS'], bname)
        #     p.process_input(files[vca])    

metrics = ['framesReceivedPerSecond', 'bitrate', 'frame_jitter', 'frameHeight']  # what to predict
estimation_methods = ['ml', 'rtp_ml', 'rtp']  # how to predict
split_types = ['random']  # how to split data
split_ratios = [0.5]  # in what proportion should data be split
feature_subsets = [['LSTATS', 'TSTATS']]  # netml features
ml_model_names = ['rf']   # sklearn model name
data_splits = ['full_video']    # train on full video or partial
special_identifiers = ['original']

combinations = [metrics, estimation_methods, split_types, split_ratios, feature_subsets, ml_model_names, data_splits, data_dirs, special_identifiers]

idx = 0
for metric, estimation_method, split_type, split_ratio, feature_subset, ml_model_name, data_split, data_dir, special_identifier in itertools.product(*combinations):
    if metric == 'frameHeight' and (estimation_method == 'rtp' or estimation_method == 'frame-lookback'):
        continue
    bname = os.path.basename(data_dir)
    cv_splits = load_cv_splits(data_dir)
    n_cv = len(cv_splits)
    model_trials = []
    cv_idx = 1
    vca_preds = {}
    feature_importances = []
    models = []
    for split_files in cv_splits:
        sid = f'cv_{cv_idx}'
        if special_identifier is not None:
            sid = f'{sid}_{special_identifier}'
        mt = Experiment(metric, estimation_method, split_type, split_ratio,
            feature_subset, ml_model_name, data_split, data_dir, 2, sid, cv_index=cv_idx)
        vca_model = mt.train_model(split_files=split_files)
        p = mt.get_test_set_predictions(split_files=split_files, vca_model=vca_model)
        models.append(vca_model)
        for vca in p:
            if vca not in vca_preds:
                vca_preds[vca] = []
            df_files = pd.concat(p[vca], axis=0)
            vca_preds[vca].append(df_files)
        intermediates_dir = f'{data_dir}_intermediates/{mt.trial_id}'
        with open(f'{intermediates_dir}/{mt.trial_id}.pkl', 'wb') as fd:
            pickle.dump(mt, fd)
        cv_idx += 1
    idx += 1
    with open('accuracy.txt', 'a') as fd:
        feature_tag = 'none' if feature_subset is None else '-'.join(feature_subset)
        if special_identifier is None:
            special_identifier = 'original'
        fd.write('*'*120)
        fd.write('\n')
        fd.write(f'Model = {metric}_{estimation_method}_{feature_tag}_{ml_model_name}_{bname}_{special_identifier}\n')
        for vca in vca_preds.keys():
            df = pd.concat(vca_preds[vca], axis=0)
            if metric != 'frameHeight':
                mae = round(mean_absolute_error(df[f'{metric}_{estimation_method}'], df[f'{metric}_gt']), 2)
            if metric == 'framesReceived':
                acc = round(fps_prediction_accuracy(df[f'{metric}_{estimation_method}'], df[f'{metric}_gt']), 2)
                fd.write(f'VCA = {vca}\n | MAE = {mae} | Acc = {acc}%\n')
            elif metric == 'frameHeight':
                acc = round(100*accuracy_score(df[f'{metric}_{estimation_method}'], df[f'{metric}_gt']), 2)
                fd.write(f'VCA = {vca}\n | Acc = {acc}\n')
            else:
                fd.write(f'VCA = {vca}\n | MAE = {mae}\n')
    if 'ml' in estimation_method:
        with open('features.txt', 'a') as fd:
            fd.write('*'*120)
            fd.write('\n')
            fd.write(f'Model = {metric}_{estimation_method}_{feature_tag}_{ml_model_name}_{bname}_{special_identifier}\n')

        for vca in models[0].keys():
            scores = {}
            for vca_model in models:
                f = vca_model[vca].feature_importances
                for feature in f:
                    if feature not in scores:
                        scores[feature] = []
                    scores[feature].append(f[feature])
            scr = str(pd.DataFrame(scores).mean().sort_values(ascending=False).head(10).to_dict())
            with open('features.txt', 'a') as fd:
                fd.write(f'VCA: {vca} || {scr}\n')