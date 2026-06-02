import pandas as pd
import pytest

from vcaml.feature_extraction import FeatureExtractor

DATASET = 'in_lab_data'
VCA = 'meet'


@pytest.fixture
def config():
    return {
        'prediction_window': 1,
        'n_features_size': {VCA: 5},
        'n_features_iat': {VCA: 5},
        'video_ptype': {DATASET: {VCA: ['96']}},
        'rtx_ptype': {DATASET: {VCA: ['99']}},
    }


@pytest.fixture
def extractor(config):
    return FeatureExtractor(
        featureSubset=['LSTATS', 'TSTATS'],
        config=config,
        vca=VCA,
        dataset=DATASET,
    )


def make_net_df(times, lengths):
    return pd.DataFrame({
        'time_normed': times,
        'time': [1_700_000_000.0 + t for t in times],
        'length': lengths,
    })


class TestBuildIntervals:
    def test_single_packet(self, extractor):
        t = pd.Series([0.0])
        e = pd.Series([1e9])
        ids, ets = extractor._buildIntervals(t, e)
        assert ids == [1]
        assert len(ets) == 1

    def test_all_in_one_window(self, extractor):
        t = pd.Series([0.0, 0.3, 0.9])
        e = pd.Series([1e9, 1e9 + 0.3, 1e9 + 0.9])
        ids, ets = extractor._buildIntervals(t, e)
        assert ids == [1, 1, 1]
        assert len(ets) == 1

    def test_two_windows(self, extractor):
        t = pd.Series([0.0, 0.5, 1.5, 1.8])
        e = pd.Series([0.0, 0.5, 1.5, 1.8])
        ids, ets = extractor._buildIntervals(t, e)
        assert ids == [1, 1, 2, 2]
        assert len(ets) == 2

    def test_three_windows(self, extractor):
        t = pd.Series([0.0, 1.5, 3.0])
        e = pd.Series([0.0, 1.5, 3.0])
        ids, ets = extractor._buildIntervals(t, e)
        assert ids == [1, 2, 3]
        assert len(ets) == 3

    def test_boundary_uses_strict_greater_than(self, extractor):
        # Gap of exactly windowSize (1.0) does NOT trigger a new window (uses >)
        t = pd.Series([0.0, 1.0])
        e = pd.Series([0.0, 1.0])
        ids, ets = extractor._buildIntervals(t, e)
        assert ids == [1, 1]
        assert len(ets) == 1


class TestAppendStats:
    def _empty_data(self):
        return {f'x{s}': [] for s in ('_min', '_max', '_q1', '_q2', '_q3', '_mean', '_std')}

    def test_all_seven_stats_populated(self, extractor):
        result = extractor._appendStats(self._empty_data(), 'x', [2.0, 4.0, 6.0, 8.0], 0)
        assert result['x_min'] == [2.0]
        assert result['x_max'] == [8.0]
        assert result['x_mean'] == [pytest.approx(5.0)]
        assert result['x_q2'] == [pytest.approx(5.0)]
        assert len(result['x_std']) == 1

    def test_empty_array_fills_default(self, extractor):
        result = extractor._appendStats(self._empty_data(), 'x', [], -1)
        for suffix in ('_min', '_max', '_q1', '_q2', '_q3', '_mean', '_std'):
            assert result[f'x{suffix}'] == [-1]


class TestLengthStatFeatures:
    def test_single_window_all_stats(self, extractor):
        df = make_net_df([0.0, 0.3, 0.6], [100, 200, 300])
        out = extractor._extractLengthStatFeatures(df)
        assert len(out) == 1
        row = out.iloc[0]
        assert row['l_mean'] == pytest.approx(200.0)
        assert row['l_min'] == 100.0
        assert row['l_max'] == 300.0
        assert row['l_num_pkts'] == 3
        assert row['l_num_bytes'] == 600
        assert row['l_num_unique'] == 3
        assert 'et' in out.columns

    def test_splits_into_two_windows(self, extractor):
        df = make_net_df([0.0, 0.5, 1.5, 1.8], [100, 200, 300, 400])
        out = extractor._extractLengthStatFeatures(df)
        assert len(out) == 2

    def test_duplicate_lengths_counted_once_for_unique(self, extractor):
        df = make_net_df([0.0, 0.3, 0.6], [100, 100, 200])
        out = extractor._extractLengthStatFeatures(df)
        assert out.iloc[0]['l_num_unique'] == 2


class TestIatStatFeatures:
    def test_burst_count_above_threshold(self, extractor):
        # IATs: [0ms, 50ms, 50ms] — two values >= 30ms
        df = make_net_df([0.0, 0.05, 0.10], [100, 100, 100])
        out = extractor._extractIatStatFeatures(df)
        assert out.iloc[0]['t_burst_count'] == 2

    def test_burst_count_below_threshold(self, extractor):
        # IATs: [0ms, 1ms, 1ms] — none >= 30ms
        df = make_net_df([0.0, 0.001, 0.002], [100, 100, 100])
        out = extractor._extractIatStatFeatures(df)
        assert out.iloc[0]['t_burst_count'] == 0

    def test_et_column_present(self, extractor):
        df = make_net_df([0.0, 0.3, 0.6], [100, 100, 100])
        out = extractor._extractIatStatFeatures(df)
        assert 'et' in out.columns

    def test_splits_into_two_windows(self, extractor):
        df = make_net_df([0.0, 0.5, 1.5, 1.8], [100, 100, 100, 100])
        out = extractor._extractIatStatFeatures(df)
        assert len(out) == 2


class TestSizeFeatures:
    def test_pads_short_window_with_zeros(self, extractor):
        # nFeatures=5, only 3 packets → pad size_4 and size_5 with 0
        df = make_net_df([0.0, 0.3, 0.6], [10, 20, 30])
        out = extractor._extractSizeFeatures(df)
        assert out.iloc[0]['size_1'] == 10
        assert out.iloc[0]['size_3'] == 30
        assert out.iloc[0]['size_4'] == 0
        assert out.iloc[0]['size_5'] == 0

    def test_truncates_long_window(self, extractor):
        # nFeatures=5, 7 packets → keep only first 5
        df = make_net_df([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                         [10, 20, 30, 40, 50, 60, 70])
        out = extractor._extractSizeFeatures(df)
        assert out.iloc[0]['size_5'] == 50

    def test_et_column_present(self, extractor):
        df = make_net_df([0.0, 0.3, 0.6], [100, 100, 100])
        out = extractor._extractSizeFeatures(df)
        assert 'et' in out.columns


class TestIatFeatures:
    def test_pads_short_window_with_zeros(self, extractor):
        # nFeatures=5, only 3 packets → pad iat_4 and iat_5 with 0
        df = make_net_df([0.0, 0.3, 0.6], [100, 100, 100])
        out = extractor._extractIatFeatures(df)
        assert out.iloc[0]['iat_4'] == 0
        assert out.iloc[0]['iat_5'] == 0

    def test_truncates_long_window(self, extractor):
        # nFeatures=5, 7 packets → 5 IAT columns + et = 6 total
        df = make_net_df([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [100] * 7)
        out = extractor._extractIatFeatures(df)
        assert len(out.columns) == 6  # iat_1..iat_5 + et

    def test_et_column_present(self, extractor):
        df = make_net_df([0.0, 0.3, 0.6], [100, 100, 100])
        out = extractor._extractIatFeatures(df)
        assert 'et' in out.columns


class TestExtractFeatures:
    def test_lstats_columns_present(self, extractor):
        df = make_net_df([0.0, 0.3, 0.6], [100, 200, 300])
        out = extractor.extract_features(df)
        for col in ('l_mean', 'l_std', 'l_min', 'l_max', 'l_num_pkts', 'l_num_bytes'):
            assert col in out.columns

    def test_tstats_columns_present(self, extractor):
        df = make_net_df([0.0, 0.3, 0.6], [100, 200, 300])
        out = extractor.extract_features(df)
        for col in ('t_mean', 't_std', 't_min', 't_max', 't_burst_count'):
            assert col in out.columns

    def test_size_subset(self, config):
        fe = FeatureExtractor(['SIZE'], config, VCA, DATASET)
        df = make_net_df([0.0, 0.3, 0.6], [100, 200, 300])
        out = fe.extract_features(df)
        assert 'size_1' in out.columns
        assert 'et' in out.columns

    def test_iat_subset(self, config):
        fe = FeatureExtractor(['IAT'], config, VCA, DATASET)
        df = make_net_df([0.0, 0.3, 0.6], [100, 200, 300])
        out = fe.extract_features(df)
        assert 'iat_1' in out.columns
        assert 'et' in out.columns

    def test_et_appears_exactly_once_with_multiple_subsets(self, config):
        fe = FeatureExtractor(['LSTATS', 'TSTATS'], config, VCA, DATASET)
        df = make_net_df([0.0, 0.3, 0.6], [100, 200, 300])
        out = fe.extract_features(df)
        assert list(out.columns).count('et') == 1

    def test_row_count_matches_windows(self, extractor):
        # 4 packets spanning two 1-second windows → 2 output rows
        df = make_net_df([0.0, 0.5, 1.5, 1.8], [100, 200, 300, 400])
        out = extractor.extract_features(df)
        assert len(out) == 2
