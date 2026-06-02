import json
from datetime import datetime, timezone

import pytest

from vcaml.io.webrtc_reader import WebRTCReader

START = '2023-01-01T00:00:00+00:00'
END = '2023-01-01T00:00:03+00:00'
EPOCH_START = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())


def _bare_reader():
    """Return a WebRTCReader without calling __init__ (for testing pure helper methods)."""
    return WebRTCReader.__new__(WebRTCReader)


def _webrtc_json(stream_id='1', fps=None, frames_received=None, bitrate=None):
    sid = stream_id
    return {
        'PeerConnections': {
            'conn1': {
                'stats': {
                    f'IT01V{sid}-framesPerSecond': {
                        'startTime': START, 'endTime': END,
                        'values': str(fps or [30, 29, 30]),
                    },
                    f'IT01V{sid}-framesReceived': {
                        'startTime': START, 'endTime': END,
                        'values': str(frames_received or [0, 30, 60]),
                    },
                    f'IT01V{sid}-[bytesReceived_in_bits/s]': {
                        'startTime': START, 'endTime': END,
                        'values': str(bitrate or [100_000, 200_000, 150_000]),
                    },
                }
            }
        }
    }


@pytest.fixture
def make_reader(tmp_path):
    def _make(content):
        f = tmp_path / 'webrtc.json'
        f.write_text(json.dumps(content))
        return WebRTCReader(str(f), 'in_lab_data')
    return _make


class TestIsCumStat:
    def test_frames_received_is_cumulative(self):
        assert _bare_reader()._isCumStat('IT01V1-framesReceived') is True

    def test_jitter_buffer_delay_is_cumulative(self):
        assert _bare_reader()._isCumStat('IT01V1-jitterBufferDelay') is True

    def test_qp_sum_is_cumulative(self):
        assert _bare_reader()._isCumStat('IT01V1-qpSum') is True

    def test_frames_per_second_is_not_cumulative(self):
        assert _bare_reader()._isCumStat('IT01V1-framesPerSecond') is False

    def test_ssrc_is_not_cumulative(self):
        assert _bare_reader()._isCumStat('IT01V1-ssrc') is False


class TestGetActiveStreamIds:
    def test_finds_all_unique_stream_ids(self):
        stats = {
            'IT01V1-framesPerSecond': {},
            'IT01V2-framesPerSecond': {},
            'IT01V1-ssrc': {},
            'OTHER-key': {},
        }
        ids = _bare_reader()._getActiveStreamIds(stats, 'IT01V')
        assert set(ids) == {'1', '2'}

    def test_empty_stats_returns_empty(self):
        assert _bare_reader()._getActiveStreamIds({}, 'IT01V') == []

    def test_no_matching_prefix_returns_empty(self):
        stats = {'OTHER-1-foo': {}, 'OTHER-2-bar': {}}
        assert _bare_reader()._getActiveStreamIds(stats, 'IT01V') == []


class TestGetMostActiveStreamId:
    def test_picks_stream_with_higher_cumulative_fps(self):
        stats = {
            'IT01V1-framesPerSecond': {'values': '[10, 10, 10]'},
            'IT01V2-framesPerSecond': {'values': '[30, 30, 30]'},
        }
        assert _bare_reader()._getMostActiveStreamId(stats, ['1', '2']) == '2'

    def test_single_stream_returned(self):
        stats = {'IT01V5-framesPerSecond': {'values': '[25, 25]'}}
        assert _bare_reader()._getMostActiveStreamId(stats, ['5']) == '5'

    def test_no_valid_streams_returns_none(self):
        assert _bare_reader()._getMostActiveStreamId({}, []) is None

    def test_stream_ids_without_fps_stat_excluded(self):
        # stream '2' has no framesPerSecond entry → only '1' is valid
        stats = {'IT01V1-framesPerSecond': {'values': '[20]'}}
        assert _bare_reader()._getMostActiveStreamId(stats, ['1', '2']) == '1'


class TestParseStatSeries:
    def test_column_names(self):
        df = _bare_reader()._parseStatSeries(
            'IT01V1-framesPerSecond', START, END, [30, 29, 31])
        assert list(df.columns) == ['ts', 'framesPerSecond']

    def test_row_count_equals_seconds_in_span(self):
        # END - START = 3 seconds → 3 rows
        df = _bare_reader()._parseStatSeries(
            'IT01V1-framesPerSecond', START, END, [30, 29, 31])
        assert len(df) == 3

    def test_values_preserved_in_order(self):
        df = _bare_reader()._parseStatSeries(
            'IT01V1-framesPerSecond', START, END, [30, 29, 31])
        assert df['framesPerSecond'].tolist() == [30, 29, 31]

    def test_ts_starts_at_epoch_of_start_time(self):
        df = _bare_reader()._parseStatSeries(
            'IT01V1-framesPerSecond', START, END, [30, 29, 31])
        assert df['ts'].iloc[0] == EPOCH_START

    def test_ts_increments_by_one_second(self):
        df = _bare_reader()._parseStatSeries(
            'IT01V1-framesPerSecond', START, END, [30, 29, 31])
        assert df['ts'].tolist() == [EPOCH_START, EPOCH_START + 1, EPOCH_START + 2]


class TestGetWebrtc:
    def test_returns_nonempty_dataframe_on_valid_input(self, make_reader):
        df = make_reader(_webrtc_json()).get_webrtc()
        assert not df.empty

    def test_cumulative_stat_is_differenced(self, make_reader):
        # framesReceived [0, 30, 60] → differenced to [0, 30, 30]
        df = make_reader(_webrtc_json(frames_received=[0, 30, 60])).get_webrtc()
        assert 'framesReceived' in df.columns
        assert df['framesReceived'].tolist() == [0, 30, 30]

    def test_bytes_received_renamed_to_bitrate(self, make_reader):
        df = make_reader(_webrtc_json()).get_webrtc()
        assert 'bitrate' in df.columns
        assert '[bytesReceived_in_bits/s]' not in df.columns

    def test_duration_and_num_vals_columns_added(self, make_reader):
        df = make_reader(_webrtc_json()).get_webrtc()
        assert 'duration' in df.columns
        assert 'num_vals' in df.columns

    def test_missing_file_returns_empty_dataframe(self, tmp_path):
        r = WebRTCReader(str(tmp_path / 'nonexistent.json'), 'in_lab_data')
        assert r.get_webrtc().empty

    def test_empty_peer_connections_returns_empty_dataframe(self, make_reader):
        df = make_reader({'PeerConnections': {}}).get_webrtc()
        assert df.empty

    def test_most_active_stream_selected(self, make_reader):
        # Two streams; stream '2' has triple the FPS of stream '1'
        content = {
            'PeerConnections': {
                'conn1': {
                    'stats': {
                        'IT01V1-framesPerSecond': {
                            'startTime': START, 'endTime': END,
                            'values': '[10, 10, 10]',
                        },
                        'IT01V1-framesReceived': {
                            'startTime': START, 'endTime': END,
                            'values': '[0, 10, 20]',
                        },
                        'IT01V2-framesPerSecond': {
                            'startTime': START, 'endTime': END,
                            'values': '[30, 30, 30]',
                        },
                        'IT01V2-framesReceived': {
                            'startTime': START, 'endTime': END,
                            'values': '[0, 30, 60]',
                        },
                    }
                }
            }
        }
        df = make_reader(content).get_webrtc()
        # Stream '2' framesReceived after differencing: [0, 30, 30]
        assert df['framesReceived'].tolist() == [0, 30, 30]
