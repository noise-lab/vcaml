import logging

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from .validator import FileValidator

logger = logging.getLogger(__name__)


class KfoldCVOverFiles:

    def __init__(self, k, fileDict, config, dataset):
        self.k = k
        self.fileDict = fileDict
        self.config = config
        self.dataset = dataset

    def _filterFiles(self) -> dict:
        allPairs = [(vca, ft) for vca, pairs in self.fileDict.items() for ft in pairs]
        filteredFiles = {vca: [] for vca in self.fileDict}
        valid = 0
        with tqdm(allPairs, desc='Validating data files', unit='file') as bar:
            for vca, fileTuple in bar:
                if FileValidator(fileTuple=fileTuple, config=self.config, dataset=self.dataset).validate():
                    valid += 1
                    filteredFiles[vca].append(fileTuple)
        total = len(allPairs)
        logger.info('File validation: %d total, %d valid, %d invalid', total, valid, total - valid)
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
