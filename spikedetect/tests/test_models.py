"""Tests for spikedetect.models."""

import numpy as np
import pytest

from spikedetect.models import (
    Recording,
    SpikeDetectionParams,
    SpikeDetectionResult,
)


class TestSpikeDetectionParams:
    def test_creation_with_defaults(self):
        params = SpikeDetectionParams(fs=10000)
        assert params.fs == 10000
        assert params.hp_cutoff == 200.0
        assert params.lp_cutoff == 800.0
        assert params.diff_order == 1
        assert params.peak_threshold == 5.0
        assert params.distance_threshold == 15.0
        assert params.amplitude_threshold == 0.2
        assert params.polarity == 1
        assert params.spike_template is None

    def test_spike_template_width_auto(self):
        params = SpikeDetectionParams(fs=10000)
        assert params.spike_template_width == round(0.005 * 10000) + 1

    def test_spike_template_width_custom(self):
        params = SpikeDetectionParams(fs=10000, spike_template_width=100)
        assert params.spike_template_width == 100

    def test_spike_template_coerced_to_float64(self):
        template = np.array([1, 2, 3], dtype=np.int32)
        params = SpikeDetectionParams(fs=10000, spike_template=template)
        assert params.spike_template.dtype == np.float64

    def test_default_factory(self):
        params = SpikeDetectionParams.default(fs=20000)
        assert params.fs == 20000
        assert params.spike_template_width == round(0.005 * 20000) + 1

    def test_validate_valid_params(self):
        params = SpikeDetectionParams(fs=10000)
        result = params.validate()
        assert result is params  # returns self

    def test_validate_negative_fs(self):
        params = SpikeDetectionParams(fs=-1)
        with pytest.raises(ValueError, match="Sample rate.*must be positive"):
            params.validate()

    def test_validate_hp_above_nyquist(self):
        params = SpikeDetectionParams(fs=10000, hp_cutoff=6000)
        with pytest.raises(ValueError, match="High-pass cutoff.*must be below"):
            params.validate()

    def test_validate_lp_above_nyquist(self):
        params = SpikeDetectionParams(fs=10000, lp_cutoff=6000)
        with pytest.raises(ValueError, match="Low-pass cutoff.*must be below"):
            params.validate()

    def test_validate_bad_diff_order(self):
        params = SpikeDetectionParams(fs=10000, diff_order=3)
        with pytest.raises(ValueError, match="diff_order must be 0, 1, or 2"):
            params.validate()

    def test_validate_bad_polarity(self):
        params = SpikeDetectionParams(fs=10000, polarity=0)
        with pytest.raises(ValueError, match="polarity must be -1 or 1"):
            params.validate()

    def test_validate_updates_template_width(self):
        template = np.ones(30)
        params = SpikeDetectionParams(fs=10000, spike_template=template)
        params.validate()
        assert params.spike_template_width == 30

    def test_to_dict(self):
        params = SpikeDetectionParams(
            fs=10000,
            spike_template=np.array([1.0, 2.0, 3.0]),
            likely_inflection_point_peak=25,
        )
        d = params.to_dict()
        assert d["fs"] == 10000
        assert d["spike_template"] == [1.0, 2.0, 3.0]
        assert d["likely_inflection_point_peak"] == 25
        assert "last_filename" in d

    def test_to_dict_no_template(self):
        params = SpikeDetectionParams(fs=10000)
        d = params.to_dict()
        assert "spike_template" not in d
        assert "likely_inflection_point_peak" not in d

    def test_from_dict_roundtrip(self):
        original = SpikeDetectionParams(
            fs=20000,
            hp_cutoff=300.0,
            lp_cutoff=1000.0,
            diff_order=2,
            peak_threshold=3.0,
            distance_threshold=10.0,
            amplitude_threshold=0.5,
            spike_template=np.array([1.0, 2.0, 3.0]),
            polarity=-1,
            likely_inflection_point_peak=15,
            last_filename="test.mat",
        )
        d = original.to_dict()
        restored = SpikeDetectionParams.from_dict(d)
        assert restored.fs == original.fs
        assert restored.hp_cutoff == original.hp_cutoff
        assert restored.lp_cutoff == original.lp_cutoff
        assert restored.diff_order == original.diff_order
        assert restored.polarity == original.polarity
        assert (restored.likely_inflection_point_peak
                == original.likely_inflection_point_peak)
        assert restored.last_filename == original.last_filename
        np.testing.assert_array_equal(
            restored.spike_template,
            original.spike_template,
        )

    def test_from_dict_minimal(self):
        d = {"fs": 10000}
        params = SpikeDetectionParams.from_dict(d)
        assert params.fs == 10000
        assert params.hp_cutoff == 200.0

    def test_template_is_fresh_no_template(self):
        params = SpikeDetectionParams(fs=10000)
        assert not params.template_is_fresh()

    def test_template_is_fresh_no_timestamp(self):
        params = SpikeDetectionParams(
            fs=10000, spike_template=np.ones(5),
        )
        assert not params.template_is_fresh()

    def test_template_is_fresh_within_ttl(self):
        from datetime import datetime, timedelta
        params = SpikeDetectionParams(
            fs=10000,
            spike_template=np.ones(5),
            template_updated_at=datetime.now() - timedelta(hours=1),
        )
        assert params.template_is_fresh(ttl_hours=24)

    def test_template_is_fresh_beyond_ttl(self):
        from datetime import datetime, timedelta
        params = SpikeDetectionParams(
            fs=10000,
            spike_template=np.ones(5),
            template_updated_at=datetime.now() - timedelta(hours=48),
        )
        assert not params.template_is_fresh(ttl_hours=24)

    def test_timestamp_roundtrip(self):
        from datetime import datetime
        # Microseconds preserved by isoformat round-trip.
        ts = datetime(2026, 4, 28, 14, 30, 15, 123456)
        original = SpikeDetectionParams(
            fs=10000,
            spike_template=np.ones(5),
            template_updated_at=ts,
        )
        d = original.to_dict()
        assert d["template_updated_at"] == ts.isoformat()
        restored = SpikeDetectionParams.from_dict(d)
        assert restored.template_updated_at == ts


class TestRecording:
    def test_creation(self):
        voltage = np.array([1, 2, 3, 4, 5])
        rec = Recording(name="test", voltage=voltage, sample_rate=10000)
        assert rec.name == "test"
        assert rec.voltage.dtype == np.float64
        assert rec.sample_rate == 10000
        assert rec.current is None
        assert rec.result is None

    def test_voltage_coerced_to_float64(self):
        voltage = np.array([1, 2, 3], dtype=np.int32)
        rec = Recording(name="test", voltage=voltage, sample_rate=10000)
        assert rec.voltage.dtype == np.float64

    def test_voltage_raveled(self):
        voltage = np.array([[1, 2], [3, 4]])
        rec = Recording(name="test", voltage=voltage, sample_rate=10000)
        assert rec.voltage.ndim == 1
        assert len(rec.voltage) == 4

    def test_current_coerced(self):
        voltage = np.array([1.0, 2.0])
        current = np.array([0.1, 0.2], dtype=np.float32)
        rec = Recording(
            name="test", voltage=voltage,
            sample_rate=10000, current=current,
        )
        assert rec.current.dtype == np.float64


class TestSpikeDetectionResult:
    def test_creation(self):
        params = SpikeDetectionParams(fs=10000)
        result = SpikeDetectionResult(
            spike_times=np.array([100, 200, 300]),
            spike_times_uncorrected=np.array([101, 201, 301]),
            params=params,
        )
        assert result.spike_times.dtype == np.int64
        assert result.spike_times_uncorrected.dtype == np.int64
        assert result.spot_checked is False

    def test_n_spikes(self):
        params = SpikeDetectionParams(fs=10000)
        result = SpikeDetectionResult(
            spike_times=np.array([100, 200, 300]),
            spike_times_uncorrected=np.array([101, 201, 301]),
            params=params,
        )
        assert result.n_spikes == 3

    def test_spike_times_seconds(self):
        params = SpikeDetectionParams(fs=10000)
        result = SpikeDetectionResult(
            spike_times=np.array([10000, 20000]),
            spike_times_uncorrected=np.array([10000, 20000]),
            params=params,
        )
        np.testing.assert_allclose(result.spike_times_seconds, [1.0, 2.0])

    def test_empty_result(self):
        params = SpikeDetectionParams(fs=10000)
        result = SpikeDetectionResult(
            spike_times=np.array([], dtype=np.int64),
            spike_times_uncorrected=np.array([], dtype=np.int64),
            params=params,
        )
        assert result.n_spikes == 0
        assert len(result.spike_times_seconds) == 0
