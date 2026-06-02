import ast
import json
import logging
import re
from datetime import datetime

import dateutil.parser
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WebRTCReader:

    # Cumulative stats that must be differenced to get per-second values
    _CUM_STATS = frozenset({
        'freezeCount*', 'totalFreezesDuration*', 'totalFramesDuration*',
        'framesReceived', 'pauseCount*', 'totalPausesDuration*',
        'jitterBufferDelay', 'jitterBufferEmittedCount', 'qpSum',
    })

    _WANTED_STATS = {
        'IT01V': [
            'ssrc', 'lastPacketReceivedTimestamp', 'framesPerSecond',
            '[bytesReceived_in_bits/s]', '[codec]', 'packetsLost', 'framesDropped',
            'framesReceived', '[framesReceived/s]', '[interFrameDelayStDev_in_ms]',
            'nackCount', 'packetsReceived', 'trackIdentifier', 'freezeCount',
            'totalFreezesDuration', 'pauseCount', 'totalPausesDuration',
            'jitterBufferDelay', '[framesDecoded/s]', 'jitterBufferEmittedCount',
            'frameHeight', 'qpSum',
        ]
    }

    _RENAME_COLS = {
        '[framesReceived/s]': 'framesReceivedPerSecond',
        '[framesDecoded/s]': 'framesDecodedPerSecond',
        '[bytesReceived_in_bits/s]': 'bitrate',
        '[interFrameDelayStDev_in_ms]': 'frame_jitter',
    }

    def __init__(self, webrtcFile, dataset):
        self.webrtcFile = webrtcFile
        self.dataset = dataset

    # ------------------------------------------------------------------ helpers

    def _isCumStat(self, statName: str) -> bool:
        return any('-' + cs in statName for cs in self._CUM_STATS)

    def _getActiveStreamIds(self, webrtcStats: dict, prefix: str) -> list:
        seen = {}
        for key in webrtcStats:
            m = re.search(f'{prefix}(\\d+)-', key)
            if m:
                seen[m.group(1)] = True
        return list(seen.keys())

    def _getMostActiveStreamId(self, webrtcStats: dict, streamIds: list):
        statTemplate = 'IT01V%s-framesPerSecond'
        validIds = [sid for sid in streamIds if statTemplate % sid in webrtcStats]
        fpsSums = [
            sum(ast.literal_eval(webrtcStats[statTemplate % sid]['values']))
            for sid in validIds
        ]
        if not fpsSums:
            return None
        return validIds[int(np.argmax(fpsSums))]

    def _parseStatSeries(self, statName: str, startTime: str,
                         endTime: str, values: list) -> pd.DataFrame:
        tStart = datetime.timestamp(dateutil.parser.parse(startTime))
        tEnd = datetime.timestamp(dateutil.parser.parse(endTime))
        colName = statName.split('-')[1]
        rows = []
        t, i = int(tStart), 0
        while t < tEnd and i < len(values):
            rows.append([t, values[i]])
            i += 1
            t += 1
        return pd.DataFrame(rows, columns=['ts', colName])

    # ------------------------------------------------------------------ public

    def get_webrtc(self) -> pd.DataFrame:
        try:
            rawData = json.load(open(self.webrtcFile))
            activeIds = []
            streamId = None
            for connKey in rawData['PeerConnections']:
                connStats = rawData['PeerConnections'][connKey]['stats']
                if not connStats:
                    continue
                activeIds = self._getActiveStreamIds(connStats, 'IT01V')
                streamId = self._getMostActiveStreamId(connStats, activeIds)
                if streamId:
                    webrtcStats = connStats
                    break
        except Exception as exc:
            logger.error('Failed to parse WebRTC file %s: %s', self.webrtcFile, exc)
            return pd.DataFrame()

        if not activeIds:
            logger.debug('No inbound stream found: %s', self.webrtcFile)
            return pd.DataFrame()
        if streamId is None:
            logger.debug('No frames seen: %s', self.webrtcFile)
            return pd.DataFrame()

        prefix = 'IT01V'
        statNames = [f'{prefix}{streamId}-{s}' for s in self._WANTED_STATS[prefix]]
        df_all = pd.DataFrame()
        callDuration = None
        numSamples = None

        try:
            for statName in statNames:
                if statName.startswith('DEPRECATED') or statName not in webrtcStats:
                    continue
                statEntry = webrtcStats[statName]
                startTime = statEntry['startTime']
                endTime = statEntry['endTime']
                valList = ast.literal_eval(statEntry['values'])

                if 'framesReceived' in statName:
                    tStart = datetime.timestamp(dateutil.parser.parse(startTime))
                    tEnd = datetime.timestamp(dateutil.parser.parse(endTime))
                    callDuration = tEnd - tStart

                if self._isCumStat(statName):
                    valList = [valList[0]] + [
                        valList[i] - valList[i - 1] for i in range(1, len(valList))
                    ]

                df_stat = self._parseStatSeries(statName, startTime, endTime, valList)
                if 'framesReceived' in statName:
                    numSamples = len(df_stat)
                df_all = df_stat if df_all.empty else pd.merge(
                    df_all, df_stat, on='ts', how='outer')
        except Exception as exc:
            logger.error('Error parsing stat %s in %s: %s', statName, self.webrtcFile, exc)
            return pd.DataFrame()

        df_all = df_all.rename(columns=self._RENAME_COLS)
        df_all['duration'] = callDuration
        df_all['num_vals'] = numSamples
        return df_all
