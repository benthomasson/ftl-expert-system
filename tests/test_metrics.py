"""Tests for fast-path metrics."""

from ftl_expert_system.metrics import FastPathMetrics


class TestFastPathMetrics:
    def test_initial_state(self):
        m = FastPathMetrics()
        assert m.total_queries == 0
        assert m.hit_rate == 0.0

    def test_record_fast_path(self):
        m = FastPathMetrics()
        m.record_fast_path()
        m.record_fast_path()
        m.record_slow_path()
        assert m.fast_path_hits == 2
        assert m.slow_path_falls == 1
        assert m.hit_rate == 2 / 3

    def test_record_belief_extraction(self):
        m = FastPathMetrics()
        m.record_slow_path(belief_extracted=True)
        m.record_slow_path(belief_extracted=False)
        assert m.beliefs_extracted == 1

    def test_save_and_load(self, tmp_path):
        m = FastPathMetrics()
        m.record_fast_path()
        m.record_slow_path(belief_extracted=True)
        path = tmp_path / "metrics.json"
        m.save(path)

        loaded = FastPathMetrics.load(path)
        assert loaded.fast_path_hits == 1
        assert loaded.slow_path_falls == 1
        assert loaded.beliefs_extracted == 1

    def test_load_missing_file(self, tmp_path):
        m = FastPathMetrics.load(tmp_path / "nonexistent.json")
        assert m.total_queries == 0

    def test_summary(self):
        m = FastPathMetrics()
        m.record_fast_path()
        m.record_fast_path()
        m.record_slow_path()
        s = m.summary()
        assert "66.7%" in s
        assert "fast: 2" in s
