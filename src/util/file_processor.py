import logging
import os
from glob import glob

from .validator import FileValidator

logger = logging.getLogger(__name__)


class FileProcessor:

    def __init__(self, dataDirectory):
        self.dataDirectory = dataDirectory

    def get_linked_files(self) -> dict:
        datasetName = os.path.basename(self.dataDirectory)
        if datasetName in ('in_lab_data', 'hashed_in_lab'):
            return self._getInLabFiles()
        elif datasetName in ('real_world_data', 'hashed_real_world'):
            return self._getRealWorldFiles()
        return {}

    def _getInLabFiles(self) -> dict:
        linkedFiles = {}
        for experimentDir in os.listdir(self.dataDirectory):
            vca = experimentDir.split('_')[1]
            experimentPath = os.path.join(self.dataDirectory, experimentDir)
            if not os.path.isdir(experimentPath):
                continue
            linkedFiles.setdefault(vca, [])
            csvMatches = glob(f'{experimentPath}/*.csv')
            jsonMatches = glob(f'{experimentPath}/*.json')
            if not csvMatches or not jsonMatches:
                logger.warning('Missing CSV or JSON in %s — skipping', experimentPath)
                continue
            linkedFiles[vca].append((csvMatches[0], jsonMatches[0]))
        return linkedFiles

    def _getRealWorldFiles(self) -> dict:
        linkedFiles = {}
        for device in os.listdir(self.dataDirectory):
            devicePath = os.path.join(self.dataDirectory, device)
            if not os.path.isdir(devicePath):
                continue
            for csvFile in glob(f'{devicePath}/*.csv'):
                vca = os.path.basename(csvFile).split('-')[1]
                linkedFiles.setdefault(vca, [])
                webrtcFilename = os.path.basename(csvFile)[:-4] + '.json'
                if webrtcFilename not in os.listdir(devicePath):
                    logger.warning('WebRTC file not found for %s — skipping', csvFile)
                    continue
                linkedFiles[vca].append((csvFile, os.path.join(devicePath, webrtcFilename)))
        return linkedFiles
