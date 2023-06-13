import os, sys
import pandas as pd
import pickle
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)
from model_trial import Experiment
from util.file_processor import FileProcessor
from sklearn.metrics import mean_absolute_error
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import graphviz
import pathlib

def load_model_trial(path):
    with open(path, 'rb') as fd:
        model_trial = pickle.load(fd)
    return model_trial

def get_predictions():
    intermediates_dir='data/intermediates'
    preds = {}
    for pkl_file in os.listdir(intermediates_dir):
        if pkl_file.endswith('_predictions.pkl'):
            with open(f'{intermediates_dir}/{pkl_file}', 'rb') as fd:
                data_object = pickle.load(fd)
                preds[f'{intermediates_dir}/{pkl_file}'] = data_object
    return preds

def get_vca_models():
    intermediates_dir='data/intermediates'
    models = {}
    for pkl_file in os.listdir(intermediates_dir):
        if pkl_file.endswith('_vca_model.pkl'):
            with open(f'{intermediates_dir}/{pkl_file}', 'rb') as fd:
                data_object = pickle.load(fd)
                models[f'{intermediates_dir}/{pkl_file}'] = data_object
    return models

def load_previous_split(file_prefix):
    intermediates_dir = f'data/intermediates'
    with open(f'{intermediates_dir}/{file_prefix}_split.pkl', 'rb') as fd:
        split_files = pickle.load(fd)
    return split_files

class FeatureImportanceVisualizer:
    def acquire_data(self, model_trial, data_dir):
        merged_data = {'Feature Importance': [], 'VCA': [], 'Feature': []}
        with open(f'{data_dir}_intermediates/{model_trial.trial_id}/{model_trial.trial_id}_vca_model.pkl', 'rb') as fd:
            vca_model = pickle.load(fd)
            for vca in vca_model:
                f_imp = vca_model[vca].feature_importances
                for feature in f_imp:
                    if f_imp[feature] == 0:
                        continue
                    merged_data['Feature Importance'].append(f_imp[feature])
                    merged_data['Feature'].append(feature)
                    merged_data['VCA'].append(vca)
        return pd.DataFrame(merged_data)

    def visualize_overall_importance(self, model_trial, df_merged):
        n_cols = len(df_merged['VCA'].unique())
        fig, ax = plt.subplots(1, n_cols, figsize=(20, 5))
        idx = 0
        for vca in df_merged['VCA'].unique():
            if n_cols <= 1:
                axis = ax
            else:
                axis = ax[idx]
            axis.set_title(f'VCA = {vca}')
            axis.set_ylim([0, 100])
            axis.grid(visible=1)
            g = sns.barplot(data=df_merged[df_merged['VCA'] == vca], x = 'Feature', y = 'Feature Importance', ax=axis, errorbar=None)
            for i in g.containers:
                g.bar_label(i,fmt='%.2f')
            idx += 1
        pathlib.Path(f'analysis/plots/{model_trial.trial_id}').mkdir(parents=True, exist_ok=True) 
        plt.savefig(f'analysis/plots/{model_trial.trial_id}/feature_importance.png')
        # plt.show()

    def visualize_correlation(self, model_trial, data_dir):
        with open(f'{data_dir}_intermediates/{model_trial.trial_id}/{model_trial.trial_id}_vca_model.pkl', 'rb') as fd:
            vca_model = pickle.load(fd)

        fig, ax = plt.subplots(nrows=1, ncols=len(vca_model), figsize=(20, 5))
        i = 0
        if model_trial.estimation_method == 'ml':
            for vca in vca_model:
                X = vca_model[vca].feature_matrix
                y = vca_model[vca].target_vals
                if model_trial.ml_model_name == 'decision_tree':
                    dot_data = tree.export_graphviz(vca_model[vca].estimator, out_file=None, feature_names=X.columns) 
                    graph = graphviz.Source(dot_data) 
                    graph.render(filename=f'analysis/plots/{vca}_{model_trial.trial_id}_decision_tree.png')
                df = pd.concat([X, y], axis=1)
                corr = df.corr()
                ax[i].set_title(f'VCA = {vca}')
                sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax[i], annot=corr)
                i += 1
            plt.savefig(f'analysis/plots/{model_trial.trial_id}_correlations.png')
            # plt.show()

class EstimationMethodVisualizer:

    def acquire_data(self, preds, metric):
        merged_data = []
        for pkl_file in preds:
            base_name = os.path.basename(pkl_file)
            bsp = base_name.split('_')
            if bsp[0] == 'ml':
                if not 'IAT-SIZE-STATS' in bsp:
                    continue
            metric_col = f'{metric}_{bsp[0]}'
            webrtc_col = f'{metric}_webrtc'
            for vca in preds[pkl_file]:
                df_files = pd.concat(preds[pkl_file][vca], axis=0)
                fps_cols = [col for col in df_files.columns if metric in col]
                fps_cols += ['timestamp']
                df_files = df_files[fps_cols]
                em = pd.DataFrame([bsp[0]]*len(df_files), index=df_files.index, columns=['estimation_method'])
                st = pd.DataFrame([bsp[1]]*len(df_files), index=df_files.index, columns=['split_type'])
                vc = pd.DataFrame([vca]*len(df_files), index=df_files.index, columns = ['vca'])
                mae = mean_absolute_error(df_files[metric_col], df_files[webrtc_col])
                err = pd.DataFrame([mae]*len(df_files), index=df_files.index, columns=['mean_absolute_error'])
                df_files = pd.concat([em, st, vc, err], axis=1)
                merged_data.append(df_files[['mean_absolute_error', 'split_type', 'estimation_method', 'vca']])
        return pd.concat(merged_data, axis=0)

    def visualize(self, df_merged):
        n_cols = len(df_merged['vca'].unique())
        fig, ax = plt.subplots(1, n_cols)
        idx = 0
        for vca in df_merged['vca'].unique():
            ax[idx].set_title(f'VCA = {vca}')
            ax[idx].set_ylim([0, 10])
            ax[idx].grid(visible=1)
            g = sns.barplot(data=df_merged[df_merged['vca'] == vca], x = 'estimation_method', y = 'mean_absolute_error', hue='split_type', ax=ax[idx], errorbar=None)
            for i in g.containers:
                g.bar_label(i,fmt='%.2f')
            idx += 1
        plt.show()

class TrainTestSplitCriteriaVisualizer:
    pass

class FeatureSubsetVisualizer:
    pass

class NetworkConditionVisualizer:

    def get_net_condition(self, file_tuple):

        webrtc_file = os.path.basename(file_tuple[2])
        flsp = webrtc_file.split('-')

        th_pattern = re.compile('^chrome\-\d*\-0\-0\-0\-0\-0\-[0-9]+\-[0-9]+\.json')
        sd_th_pattern = re.compile('^chrome\-5\-\d*\.\d*\-0\-0\-0\-0\-[0-9]+\-[0-9]+\.json')
        lat_pattern = re.compile('^chrome\-5\-0\-\d*\-0\-0\-0\-[0-9]+\-[0-9]+\.json')
        sd_lat_pattern = re.compile('^chrome\-5\-0\-50\-\d*\-0\-0\-[0-9]+\-[0-9]+\.json')
        loss_pattern = re.compile('^chrome\-5\-0\-50\-0\-[0-9]\-0\-[0-9]+\-[0-9]+\.json')
        sd_loss_pattern = re.compile('^chrome\-[0-9]\-0\-50\-0\-2\-\d*\.\d*\-[0-9]+\-[0-9]+\.json')
        patterns = {'throughput': th_pattern, 'std_throughput' : sd_th_pattern, 'latency': lat_pattern, 'std_latency': sd_lat_pattern , 'loss' : loss_pattern , 'std_loss': sd_loss_pattern}

        for pattern_name, pattern_regex in patterns.items():
            match_obj = pattern_regex.match(webrtc_file)
            if match_obj:
                if pattern_name == 'throughput':
                    val = float(flsp[1])
                elif pattern_name == 'std_throughput':
                    val = float(flsp[2])
                elif pattern_name == 'latency':
                    val = float(flsp[3])
                elif pattern_name == 'std_latency':
                    val = float(flsp[4])
                elif pattern_name == 'loss':
                    val = float(flsp[5])
                elif pattern_name == 'std_loss':
                    val = float(flsp[6])
                return (pattern_name, val)

    def acquire_data(self, preds, metric):
        merged_data = []
        for pkl_file in preds:
            base_name = os.path.basename(pkl_file)
            bsp = base_name.split('_')
            if bsp[0] == 'ml':
                if not 'IAT-SIZE-STATS' in bsp:
                    continue
            metric_col = f'{metric}_{bsp[0]}'
            webrtc_col = f'{metric}_webrtc'
            
            file_split = load_previous_split(bsp[1])

            for vca in preds[pkl_file]:
                filtered_frames = []
                for idx, file_tuple in enumerate(file_split[vca]['test']):
                    net_cond, val = self.get_net_condition(file_tuple)
                    if preds[pkl_file][vca][idx] is None:
                        continue
                    df_file = preds[pkl_file][vca][idx]
                    df_file['network_variation_type'] = net_cond
                    df_file['network_metric_value'] = val
                    df_file['vca'] = vca
                    filtered_frames.append(df_file)

                df_files = pd.concat(filtered_frames, axis=0)
                fps_cols = [col for col in df_files.columns if metric in col]
                fps_cols += ['timestamp', 'network_variation_type', 'network_metric_value', 'vca']
                df_files = df_files[fps_cols]
                em = pd.DataFrame([bsp[0]]*len(df_files), index=df_files.index, columns=['estimation_method'])
                df_files = pd.concat([df_files, em], axis=1)
                df_files['mean_absolute_error'] = 0.0
                for nv in df_files['network_variation_type'].unique():
                    df = df_files[df_files['network_variation_type'] == nv]
                    for nm in df['network_metric_value'].unique():
                        df_val = df[df['network_metric_value'] == nm]
                        mae = mean_absolute_error(df_val[metric_col], df_val[webrtc_col])
                        df_files.loc[df_val.index, 'mean_absolute_error'] = mae
                merged_data.append(df_files)
        return pd.concat(merged_data, axis=0)

    def visualize(self, df_merged):
        for net_cond in df_merged['network_variation_type'].unique():
            if net_cond == 'throughput':
                x_label = 'Throughput (Mbps)'
            elif net_cond == 'std_throughput':
                x_label = 'Throughput standard deviation (Mbps)'
            elif net_cond == 'latency':
                x_label = 'Latency (ms)'
            elif net_cond == 'std_latency':
                x_label = 'Latency standard deviation (ms)'
            elif net_cond == 'loss':
                x_label = 'Packet loss %'
            elif net_cond == 'std_loss':
                x_label = 'Packet loss standard deviation (%)'
            y_label = 'Mean Absolute Error'
            df_cond = df_merged[df_merged['network_variation_type'] == net_cond]
            n_cols = len(df_cond['vca'].unique())
            fig, ax = plt.subplots(1, n_cols)
            idx = 0
            plt.title(f'Network variation: {net_cond}')
            for vca in df_cond['vca'].unique():
                ax[idx].set_title(f'VCA = {vca}', fontsize=16)
                ax[idx].set_ylim([0, 10])
                ax[idx].grid(visible=1)
                data = df_cond[df_cond['vca'] == vca]
                g = sns.lineplot(data=data, x = 'network_metric_value', y = 'mean_absolute_error', hue='estimation_method', ax=ax[idx], marker='o', errorbar=None)
                legend_labels, _= ax[idx].get_legend_handles_labels()
                sns.move_legend(ax[idx], 'upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=len(legend_labels), title = 'Estimation model', fontsize=16)
                g.set_xlabel(x_label, fontsize=16)
                g.set_ylabel(y_label, fontsize=16)
                g.tick_params(labelsize=12)
                idx += 1
            plt.show()


class MetricVarianceVisualizer:
    pass

def fps_prediction_accuracy(pred, truth):
    if len(pred) != len(truth):
        raise ValueError('Length mismatch for predictions and ground truth!')
    n = len(pred)
    df = pd.DataFrame({'pred': pred.to_numpy(), 'truth': truth.to_numpy()})
    df['deviation'] = df['pred']-df['truth']
    df['deviation'] = df['deviation'].abs()
    return df['deviation'].apply(lambda x: int(x <= 2))

class AccuracyVSQoSVisualizer:
    def __init__(self, mt_list, data_dir) -> None:
        self.mt_list = mt_list
        self.data_dir = data_dir

    def acquire_data(self):
        res = []
        for mt in self.mt_list:
            with open(f'{data_dir}_intermediates/{mt.trial_id}/{mt.trial_id}_predictions.pkl', 'rb') as fd:
                preds = pickle.load(fd)
            pred_merged = pd.concat(preds['webex'], axis=0)
            pred_merged['Deviation'] = pred_merged[f'{mt.metric}_{mt.estimation_method}'] - pred_merged[f'{mt.metric}_webrtc']
            pred_merged['MAE'] = pred_merged['Deviation'].apply(abs)
            pred_merged['Accuracy'] = 100*fps_prediction_accuracy(pred_merged[f'{mt.metric}_{mt.estimation_method}'], pred_merged[f'{mt.metric}_webrtc'])
            trace_files = []
            for f in pred_merged['file'].unique():
                trace_num = f[:-5].split('-')[-3]
                trace_files.append([x for x in os.listdir(f'{data_dir}_traces') if x.endswith('_'+trace_num+'.csv')][0])
            qos_time = []
            for f in trace_files:
                trace = pd.read_csv(f'{data_dir}_traces/{f}', names = ['throughput', 'latency', 'loss'])
                trace = trace.reset_index()
                timestamps = pd.read_csv(f'{data_dir}_traces/{f}.txt', names = ['timestamp'])
                timestamps = timestamps.reset_index()
                qt = pd.concat([trace, timestamps], axis=1)
                qos_time.append(qt)
            qos_merged = pd.concat(qos_time, axis=0)
            qos_profile = pd.merge(pred_merged, qos_merged, on='timestamp')
            wdata = {'Window Size': [], 'Ground truth FPS': [], 'Predicted FPS': [], 'MAE': [], 'Accuracy': [], 'Deviation in prediction' : [], 'th_avg': [], 'lat_avg': [], 'loss': [], 'th_std': [], 'lat_std': [], 'file': []}
            for w in range(1, 2):
                for f in qos_profile['file'].unique():
                    df_file = qos_profile[qos_profile['file'] == f]
                    n = len(df_file)
                    wdata['file'] += [f]*n
                    wdata['Window Size'] += [w]*n
                    for idx in range(len(df_file)):
                        wdata['th_avg'] += [df_file['throughput'].iloc[idx-w+1:idx+1].mean()]
                        wdata['th_std'] += [df_file['throughput'].iloc[idx-w+1:idx+1].std()]
                        wdata['lat_avg'] += [df_file['latency'].iloc[idx-w+1:idx+1].mean()]
                        wdata['lat_std'] += [df_file['latency'].iloc[idx-w+1:idx+1].std()]
                        wdata['loss'] += [df_file['loss'].iloc[idx-w+1:idx+1].mean()]

                        wdata['Ground truth FPS'] += [df_file[f'{mt.metric}_webrtc'].iloc[idx]]
                        wdata['Predicted FPS'] += [df_file[f'{mt.metric}_{mt.estimation_method}'].iloc[idx]]
                        wdata['MAE'] += [df_file['MAE'].iloc[idx]]
                        wdata['Accuracy'] += [df_file['Accuracy'].iloc[idx]]
                        wdata['Deviation in prediction'] += [df_file['Deviation'].iloc[idx]]
            wdf = pd.DataFrame(wdata)
            res.append(wdf)
        df = pd.concat(res, axis=0)
        df = df.reset_index()
        return df
    
    def visualize(self, data, x, y, z):
        cmap = sns.color_palette("rocket", as_cmap=True)
        f, ax = plt.subplots()
        points = ax.scatter(data[x], data[y], c=data[z], cmap=cmap)
        f.colorbar(points, label = z)
        labels = {'latency': 'Latency (ms)', 'throughput': 'Throughput (kbps)'}
        ax.set_xlabel(labels[x])
        ax.set_ylabel(labels[y])
        ax.grid(visible=1)
        plt.show()

    def visualize_counts(self, data, x):
        acc_mean = data['Accuracy'].mean()
        data['Accuracy Bin'] = data['Accuracy'].apply(lambda x: 'Accuracy > average' if x > acc_mean else 'Accuracy <= average')
        print(len(data.index) == len(set(data.index))) 
        sns.ecdfplot(data=data, x = x, hue = 'Accuracy Bin')
        labels = {'latency': 'Latency (ms)', 'throughput': 'Throughput (kbps)', 'loss': 'Loss %'}
        acc_mean = round(acc_mean, 2)
        plt.title(f'Accuracy = {acc_mean}%')
        plt.xlabel(labels[x])
        plt.grid(visible=1)
        plt.show()


    def visualize_boxplots(self, data, x, xlabel, y, y_bounds=None, n_bins=5, custom_bins = None):
        if x != 'loss':
            bins = [data[x].quantile(i/n_bins) for i in range(n_bins)] + [data[x].max()] if not custom_bins else custom_bins
            labels = [f'{round(bins[i])}-{round(bins[i+1])}' for i in range(len(bins)-1)]
            data[xlabel] = pd.cut(x = data[x], bins = bins, labels = labels, include_lowest = True)
        else:
            data[xlabel] = data[x]
        sns.boxplot(x=xlabel, y = y, data=data, showfliers=False, color='lightblue')
        plt.xlabel(xlabel)
        plt.ylabel(y)
        if y_bounds:
            plt.ylim(y_bounds)
        plt.grid(visible=1)
        plt.show()

if __name__ == '__main__':
    fv = FeatureImportanceVisualizer()
    data_dir = '/home/taveesh/Documents/PhD-Research/Code/vca-qoe-inference/data/WebEx_Data'
    model = 'framesPerSecond_ml_random_LSTATS-TSTATS_rf_regressor_full_video_WebEx_Data_cv_%s_original'
    # feature_data = []
    mt_list = []
    for cv in range(1, 6):
        mt_dir = model % cv
        mt_path = f'{data_dir}_intermediates/{mt_dir}/{mt_dir}.pkl'
        mt = load_model_trial(mt_path)
        mt_list.append(mt)
    
    # with open(f'{data_dir}_intermediates/{mt.trial_id}/{mt.trial_id}_predictions.pkl', 'rb') as fd:
    #     preds = pickle.load(fd)
    
    # dfs = []
    # for vca in ['meet', 'teams']:
    #     pred_merged = pd.concat(preds[vca], axis=0)
    #     pred_merged['Deviation'] = pred_merged[f'{mt.metric}_{mt.estimation_method}'] - pred_merged[f'{mt.metric}_webrtc']
    #     pred_merged['MAE'] = pred_merged['Deviation'].apply(abs)
    #     pred_merged['Accuracy'] = 100*fps_prediction_accuracy(pred_merged[f'{mt.metric}_{mt.estimation_method}'], pred_merged[f'{mt.metric}_webrtc'])
    #     pred_merged['VCA'] = vca
    #     dfs.append(pred_merged)
    # predictions = pd.concat(dfs, axis=0)
        # if mt.ml_model_name == 'rf_regressor':
        #     df_merged = fv.acquire_data(mt, data_dir)
        #     feature_data.append(df_merged)
            # fv.visualize_overall_importance(mt, df_merged)
    av = AccuracyVSQoSVisualizer(mt_list, data_dir)
    # av.visualize_boxplots(predictions, 'framesPerSecond_webrtc', 'Ground truth FPS', 'MAE', y_bounds=[-1, 15], custom_bins=[0, 10, 20, 30, 40])
    df = av.acquire_data()
    av.visualize_boxplots(df, 'Ground truth FPS', 'Ground truth FPS', 'Deviation in prediction', custom_bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000])
    # av.visualize_boxplots(df, 'th_avg', 'Average windowed throughput (Kbps)', 'Deviation in prediction', y_bounds = [-8, 8], custom_bins=list(range(0, 20001, 2000)))
    # av.visualize_boxplots(df, 'lat_avg', 'Average windowed latency (ms)', 'Deviation in prediction', y_bounds = [-8, 8], custom_bins=list(range(0, 201, 20)))
    # av.visualize_boxplots(df, 'th_std', 'Windowed standard deviation in throughput (Kbps)', 'Deviation in prediction', y_bounds = [-8, 8], custom_bins=list(range(0, 2001, 500)))
    # av.visualize_boxplots(df, 'lat_std', 'Windowed standard deviation in latency (ms)', 'Deviation in prediction', y_bounds = [-8, 8], custom_bins=list(range(0, 51, 10)))
    # av.visualize_boxplots(df, 'loss', 'Packet loss %', 'Deviation in prediction', y_bounds = [-8, 8])

    # av.visualize_boxplots(df, 'Average ground truth FPS', 'Average ground truth FPS', 'Average MAE', y_bounds=[0, 170], custom_bins=[0, 10, 20, 30, 40, 900])
    
    # sns.ecdfplot(data=df[df['Window Size'] == 1], x = 'Average ground truth FPS')
    # plt.show()



    # av.visualize_counts(df, 'throughput')
    # av.visualize_counts(df, 'latency')
    # av.visualize_counts(df, 'loss')
    # av.visualize(df, 'throughput', 'latency', 'Accuracy')
    # df = pd.concat(feature_data, axis=0)
    # dfg = df.groupby(['Feature', 'VCA'])['Feature Importance'].mean()
    # model = model % 'X'
    # pathlib.Path(f'analysis/tables/{model}').mkdir(parents=True, exist_ok=True) 
    # dfg.to_csv(f'analysis/tables/{model}/feature_importance.csv')