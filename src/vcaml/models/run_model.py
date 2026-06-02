import logging
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from vcaml.config import project_config
from vcaml.models.ip_udp_heuristic import IP_UDP_Heuristic
from vcaml.models.ip_udp_ml import IP_UDP_ML
from vcaml.models.rtp_heuristic import RTP_Heuristic
from vcaml.models.rtp_ml import RTP_ML

logger = logging.getLogger(__name__)

_FPS_METRICS = frozenset({
    'framesPerSecond', 'framesReceived', 'framesReceivedPerSecond',
    'framesDecodedPerSecond', 'framesRendered',
})


class ModelRunner:

    def __init__(self, metric, estimationMethod, featureSubset, dataDir, cvIndex):
        self.metric = metric
        self.estimationMethod = estimationMethod
        self.featureSubset = 'none' if featureSubset is None else featureSubset
        self.dataDir = dataDir

        featureTag = '-'.join(featureSubset) if featureSubset else 'none'
        datasetName = os.path.basename(dataDir)
        self.trialId = '_'.join(
            [metric, estimationMethod, featureTag, datasetName, f'cv_{cvIndex}'])
        self.cvIndex = cvIndex

    def _fpsPredictionAccuracy(self, predicted, groundTruth) -> float:
        n = len(predicted)
        deviation = (predicted.to_numpy() - groundTruth.to_numpy())
        return (abs(deviation) <= 2).sum() / n

    def trainModel(self, splitFiles: dict) -> dict:
        datasetName = os.path.basename(self.dataDir)
        vcaModels = {}
        for vca in splitFiles:
            logger.info('Training  vca=%s  trial=%s', vca, self.trialId)
            trainFiles = splitFiles[vca]['train']
            if self.estimationMethod == 'ip-udp-ml':
                estimator = (RandomForestClassifier()
                             if self.metric == 'frameHeight'
                             else RandomForestRegressor())
                model = IP_UDP_ML(vca=vca, featureSubset=self.featureSubset,
                                  estimator=estimator, config=project_config,
                                  metric=self.metric, dataset=datasetName)
                model.train(trainFiles)
            elif self.estimationMethod == 'rtp-ml':
                estimator = (RandomForestClassifier()
                             if self.metric == 'frameHeight'
                             else RandomForestRegressor())
                model = RTP_ML(vca=vca, featureSubset=self.featureSubset,
                               estimator=estimator, config=project_config,
                               metric=self.metric, dataset=datasetName)
                model.train(trainFiles)
            elif self.estimationMethod == 'ip-udp-heuristic':
                model = IP_UDP_Heuristic(vca=vca, metric=self.metric,
                                         config=project_config, dataset=datasetName)
            elif self.estimationMethod == 'rtp-heuristic':
                model = RTP_Heuristic(vca=vca, metric=self.metric,
                                      config=project_config, dataset=datasetName)
            vcaModels[vca] = model
        return vcaModels

    def getTestSetPredictions(self, splitFiles: dict, vcaModels: dict) -> dict:
        predictions = {}
        maes = {}
        accuracies = {}
        for vca in splitFiles:
            predictions[vca] = []
            maes[vca] = []
            accuracies[vca] = []
            testFiles = splitFiles[vca]['test']
            totalFiles = len(testFiles)
            for fileIdx, fileTuple in enumerate(testFiles, 1):
                logger.debug('Testing %s on file %d/%d: %s',
                             self.estimationMethod, fileIdx, totalFiles, fileTuple[0])
                output = vcaModels[vca].estimate(fileTuple)
                if output is None:
                    predictions[vca].append(None)
                    continue
                predCol = f'{self.metric}_{self.estimationMethod}'
                gtCol = f'{self.metric}_gt'
                if self.metric != 'frameHeight':
                    mae = mean_absolute_error(output[gtCol], output[predCol])
                    maes[vca].append(mae)
                    logger.debug('MAE = %.2f', mae)
                if self.metric in _FPS_METRICS:
                    acc = self._fpsPredictionAccuracy(output[gtCol], output[predCol])
                    accuracies[vca].append(acc)
                    logger.debug('Accuracy = %.2f', acc)
                if self.metric == 'frameHeight':
                    acc = accuracy_score(output[gtCol], output[predCol])
                    accuracies[vca].append(acc)
                    logger.debug('Accuracy = %.2f', acc)
                predictions[vca].append(output)

        for vca in splitFiles:
            maeAvg = ('None' if self.metric == 'frameHeight'
                      else round(sum(maes[vca]) / len(maes[vca]), 2) if maes[vca] else 'N/A')
            accStr = ''
            if self.metric in _FPS_METRICS and accuracies[vca]:
                accAvg = round(100 * sum(accuracies[vca]) / len(accuracies[vca]), 2)
                accStr = f'  accuracy_avg={accAvg}'
            logger.info('RESULT  vca=%s  trial=%s  mae_avg=%s%s',
                        vca, self.trialId, maeAvg, accStr)
        return predictions
