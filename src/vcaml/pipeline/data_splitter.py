import logging

import numpy as np
from sklearn.model_selection import KFold

from .validator import FileValidator

logger = logging.getLogger(__name__)


class KfoldCVOverFiles:

    def __init__(self, k, fileDict, config, dataset):
        self.k = k
        self.fileDict = fileDict
        self.config = config
        self.dataset = dataset

    def _filterFiles(self) -> dict:
        logger.info('Filtering anomalous files...')
        filteredFiles = {}
        for vca, filePairs in self.fileDict.items():
            filteredFiles[vca] = []
            for fileTuple in filePairs:
                validator = FileValidator(
                    fileTuple=fileTuple, config=self.config, dataset=self.dataset)
                if validator.validate():
                    filteredFiles[vca].append(fileTuple)
        return filteredFiles

    def split(self) -> list:
        self.fileDict = self._filterFiles()
        splits = [{} for _ in range(self.k)]
        for vca, filePairs in self.fileDict.items():
            nFiles = len(filePairs)
            if nFiles < self.k:
                logger.warning(
                    'vca=%s has only %d valid files (need %d for %d-fold CV) — skipping',
                    vca, nFiles, self.k, self.k)
                continue
            kf = KFold(n_splits=self.k, random_state=None)
            X = np.array(filePairs)
            for foldIdx, (trainIdx, testIdx) in enumerate(kf.split(X)):
                trainFiles = list(X[trainIdx])
                testFiles = list(X[testIdx])
                splits[foldIdx][vca] = {'train': trainFiles, 'test': testFiles}
                logger.info('Fold %d  vca=%s  train=%d  test=%d',
                            foldIdx + 1, vca, len(trainFiles), len(testFiles))
        return splits
