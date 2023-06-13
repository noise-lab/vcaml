import os
import pickle
from util.file_processor import FileProcessor
from util.data_splitter import DataSplitter
from config import project_config
from qoe_estimation.ml.model import MLBasedModel
from qoe_estimation.frame_lookback.model import FrameLookbackModel
from qoe_estimation.rtp_lookahead.model import RTPLookaheadModel
from qoe_estimation.bayesian.model import BayesianModel
from qoe_estimation.rtp.model import RTPModel
from qoe_estimation.rtp_no_interleaves.model import RTPWithoutInterleavesModel
from qoe_estimation.rtp_ml.model import RTPMLModel
from qoe_estimation.lstm.model import LSTM_Model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.svm import SVR
from pathlib import Path
import pandas as pd
from datetime import datetime as dt
import operator
from os.path import dirname, abspath
project_root = dirname(dirname(abspath(__file__)))

class Experiment:
    """
    Represents an experiment. Trains or tests an inference mdoel. Also stores every useful intermediate in the form of pickle files.

    The pickle files for the intermediates are stored under the parent directory of the dataset source. Following could be one of the intermediates:

    1. File split: Division of the data into train and test sets based on a criterion.
    2. VCA model: Contains a separate model that is unique to an experiment.
    3. Predictions: Contains the output of the model.
    
    Most of the parameters of this class are chosen from a discrete set of possible values. Each experiment has a unique ID that follows the below format:

    `{metric}_{estimation_method}_{split_type}_{feature_subset_tag}_{ml_model_name}_{data_split}_{data_dir}`

    Here feature_subset_tag is the same as a string denoting members of feature_subset joined using a '-'. If any of the above strings is empty, a 'none' is used in place of them.

    :param metric: The metric to be predicted and/or trained for. Could be `"framesReceived"`, `"[bytesReceived_in_bits/s]"`, or `"[interFrameDelayStDev_in_ms]"`
    :type metric: str
    :param estimation_method: The inference method to be used. Currently, `"bayesian"`, `"frame-lookback"`, `"rtp"`, and `"ml"` are supported.
    :type estimation_method: str
    :param split_type: The kind of split that is required on the data. Currently, `"random"`, `"network-condition"`, and `"throughput-std"` are supported.
    :type split_type: str
    :param split_ratio: The ratio of train to test set in the split. Takes in values between 0 and 1.
    :type split_ratio: float
    :param feature_subset: A list of features to be used in the ML model. Currently, any subset of [`"IAT"`, `"SIZE"`, `"STATS"`] is supported.
    :type feature_subset: list
    :param ml_model_name: Name of the ML model to be trained. Currently, `"rf_regressor"` and `"svm"` are supported.
    :type ml_model_name: str
    :param data_split: A filter on data that could be used to divide the complete dataset. An example could be the video duration.
    :type data_split: str
    :param data_dir: The full path of the dataset's location within the same host.
    :type data_dir: str
    :param data_format: Only two formats are supported. Takes discrete values 1 or 2.
    :type data_format: int
    :param special_identifier: Anything else that might distinguish this experiment from others
    :type special_identifier: str
    """

    def __init__(self, metric, estimation_method, split_type, split_ratio, feature_subset, ml_model_name, data_split, data_dir, data_format, special_identifier, cv_index):

        self.metric = metric
        self.estimation_method = estimation_method
        self.split_type = split_type
        self.split_ratio = split_ratio
        self.feature_subset = 'none' if feature_subset is None else feature_subset
        ml_model_name = 'none' if ml_model_name is None else ml_model_name
        self.ml_model_name = ml_model_name
        self.data_split = data_split
        self.data_dir = data_dir
        self.special_identifier = special_identifier

        if feature_subset:
            feature_subset_tag = '-'.join(feature_subset)
        else:
            feature_subset_tag = 'none'

        data_bname = os.path.basename(data_dir)
        self.trial_id = '_'.join([metric, estimation_method, split_type, feature_subset_tag, ml_model_name, data_split, data_bname])
        
        if self.special_identifier is not None:
            self.trial_id += '_'+self.special_identifier

        self.intermediates_dir = f'{self.data_dir}_intermediates/{self.trial_id}'
        self.data_format = data_format

        self.cv_index = cv_index
        
        self.model = None
        path = Path(self.intermediates_dir)
        path.mkdir(parents=True, exist_ok=True)

    def save_intermediate(self, data_object, pickle_filename):
        """A wrapper function to store an intermediate under the parent directory of the directory containing the dataset.

        :param data_object: Any object that holds information related to this class
        :type data_object: object
        :param pickle_filename: Name of the pickle file (excluding the extension)
        :type pickle_filename: str
        """
        pickle_filename = f'{self.trial_id}_{pickle_filename}'
        with open(f'{self.intermediates_dir}/{pickle_filename}.pkl', 'wb') as fd:
            pickle.dump(data_object, fd)

    def load_intermediate(self, pickle_filename):
        """A wrapper function to load an intermediate from the filesystem. The location is relative to the dataset's location by default.

        :param pickle_filename: Name of the pickle file to be loaded (excluding the extension)
        :type pickle_filename: str
        :return: An object that holds any information related to this class
        :rtype: object
        """
        with open(f'{self.intermediates_dir}/{pickle_filename}.pkl', 'rb') as fd:
            data_object = pickle.load(fd)
        return data_object

    def acquire_data(self):
        """
        Performs 3 functions:
            1. Reads PCAP files from the data location and converts them to CSVs if not done already.
            2. Performs validations over the dataset and filters out anomalous files.
            3. Links the PCAP, CSV and WebRTC files for the same experiment together.

        :return: A dictionary indexed using the VCA's name in lowercase and either 'train' or 'test' at the next level. Values hold tuples of the form (<PCAP file path>, <CSV file path>, <WebRTC file path>).

        :rtype: dict
        """
        file_processor = FileProcessor(data_directory=self.data_dir, data_format=self.data_format)
        file_dict = file_processor.get_linked_files()
        data_splitter = DataSplitter(file_dict=file_dict, criterion=self.split_type, split_ratio=self.split_ratio, config=project_config)
        split_files = data_splitter.split()
        self.save_intermediate(split_files, 'split_files')
        return split_files

    def fps_prediction_accuracy(self, pred, truth):
        """A utility function to be used only while predicting the FPS. Reports the number of instances for which the predictions are within 2 frames of the ground truth.

        :param pred: A pandas Series containing the predictions indexed over the prediction time window.
        :type pred: pandas.Series
        :param truth: A pandas Series containing the ground truth value of FPS indexed over the prediction time window.
        :type truth: pandas.Series
        :raises LengthMismatchException: This exception is raised if the length of the predictions and the ground truth do not match.
        :return: A number between 0 and 1.
        :rtype: float
        """
        if len(pred) != len(truth):
            raise self.LengthMismatchException('Length mismatch for predictions and ground truth!')

        n = len(pred)
        df = pd.DataFrame({'pred': pred.to_numpy(), 'truth': truth.to_numpy()})
        df['deviation'] = df['pred']-df['truth']
        df['deviation'] = df['deviation'].abs()
        return len(df[df['deviation'] <= 2])/n

    def train_model(self, split_files):
        """Trains only those models that require training, example: ML, Bayesian

        :param split_files: The output of :func:`~vca_perf.Experiment.acquire_data`
        :type split_files: dict
        :return: A dictionary containing a trained model for each VCA.
        :rtype: dict
        """
        bname = os.path.basename(self.data_dir)
        vca_model = {}
        importances = {}
        for vca in split_files:
            print(f'\nVCA = {vca}')
            if self.estimation_method == 'ml':
                
                if self.ml_model_name == 'rf':
                    if self.metric == 'frameHeight':
                        estimator = RandomForestClassifier()
                    else:
                        estimator = RandomForestRegressor()
 
                elif self.ml_model_name == 'decision_tree':
                    estimator = DecisionTreeRegressor(max_depth=3)
                elif self.ml_model_name == 'svm':
                    estimator = SVR()

                model = MLBasedModel(
                    vca=vca,
                    feature_subset=self.feature_subset,
                    estimator=estimator,
                    data_partition_criteria=None,
                    config=project_config,
                    metric=self.metric,
                    dataset=bname
                )

                model.train(split_files[vca]['train'])

            elif self.estimation_method == 'rtp_ml':
                
                if self.ml_model_name == 'rf':
                    if self.metric == 'frameHeight':
                        estimator = RandomForestClassifier()
                    else:
                        estimator = RandomForestRegressor()
                elif self.ml_model_name == 'decision_tree':
                    estimator = DecisionTreeRegressor(max_depth=3)
                elif self.ml_model_name == 'svm':
                    estimator = SVR()

                model = RTPMLModel(
                    vca=vca,
                    feature_subset=self.feature_subset,
                    estimator=estimator,
                    data_partition_criteria=None,
                    config=project_config,
                    metric=self.metric,
                    dataset=bname
                )
                model.train(split_files[vca]['train'])
            elif self.estimation_method == 'lstm':
                model = LSTM_Model(
                    vca=vca,
                    feature_subset=self.feature_subset,
                    estimator = None,
                    data_partition_criteria=None,
                    config=project_config,
                    metric=self.metric
                )
                model.train(split_files[vca]['train'])
            elif self.estimation_method == 'frame-lookback':
                model = FrameLookbackModel(vca=vca, metric=self.metric, config=project_config, dataset=bname)
                # df = model.train(split_files[vca]['train']+split_files[vca]['test'])
                # df.to_csv(f'../data/{vca}_frame_stats.csv')
            elif self.estimation_method == 'frame-lookahead':
                model = FrameLookAheadModel(vca=vca, metric=self.metric, cv_index=self.cv_index, data_dir=self.data_dir)
            elif self.estimation_method == 'bayesian':
                model = BayesianModel(vca=vca, metric=self.metric)
                model.train(split_files[vca]['train'])
            elif self.estimation_method == 'rtp':
                model = RTPModel(vca=vca, metric=self.metric, config=project_config, dataset=bname)
            elif self.estimation_method == 'rtp_no_interleaves':
                model = RTPWithoutInterleavesModel(vca=vca, metric=self.metric)
            elif self.estimation_method == 'rtp-lookahead':
                model = RTPLookaheadModel(vca=vca, metric=self.metric)
            vca_model[vca] = model
        self.save_intermediate(vca_model, 'vca_model')
        return vca_model

    def get_test_set_predictions(self, split_files, vca_model):
        """Calculates and returns predictions of a model over each VCA

        :param split_files: The output of :func:`acquire_data`
        :type split_files: dict
        :param vca_model:  The output of :func:`train_model`
        :type vca_model: dict
        :return: A dictionary containing prediction dataframes indexed against the VCA
        :rtype: dict
        """
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

    class LengthMismatchException(Exception):
        """
        This exception is thrown if the lengths of the input FPS dataframe and ground truth do not match.
        """
        pass
