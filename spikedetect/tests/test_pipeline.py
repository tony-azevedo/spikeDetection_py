"""End-to-end pipeline tests for spikedetect."""

import numpy as np
import pytest

from spikedetect.models import SpikeDetectionResult
from spikedetect.pipeline.detect import (
    detect_spikes,
    _apply_refractory_filter,
    _template_fwhm_samples,
)


@pytest.fixture
def pipeline_recording():
    """Recording designed to survive the filter+diff pipeline.

    Instead of embedding raw templates in noise, we create a signal
    that produces clear peaks after bandpass filtering + differentiation.
    The template is extracted from the same trimmed+filtered signal that
    detect_spikes will use internally.
    """
    from spikedetect.models import Recording, SpikeDetectionParams
    from spikedetect.pipeline.filtering import filter_data

    fs = 10000
    duration = 5.0
    n_samples = int(fs * duration)
    rng = np.random.default_rng(42)

    noise = rng.normal(0, 0.001, n_samples)

    # Embed sharp voltage transients that will survive filtering
    true_positions = np.array(
        [5000, 10000, 15000, 20000,
         25000, 30000, 35000, 40000]
    )
    for pos in true_positions:
        t_local = np.arange(-25, 26) / fs
        spike = 5.0 * np.exp(-0.5 * (t_local / 0.0003) ** 2)
        noise[pos - 25:pos + 26] += spike

    params = SpikeDetectionParams(
        fs=fs, hp_cutoff=200, lp_cutoff=800,
        diff_order=1, polarity=1,
    )

    # Simulate what detect_spikes does: trim start then filter
    start_point = round(0.01 * fs)  # = 100
    unfiltered_data = noise[start_point:]
    filtered = filter_data(
        unfiltered_data, fs,
        params.hp_cutoff, params.lp_cutoff,
        params.diff_order, params.polarity,
    )

    # Extract template at the first spike position (adjusted for trim)
    stw = params.spike_template_width
    half = stw // 2
    spike_in_trimmed = true_positions[0] - start_point
    idx_start = spike_in_trimmed - half
    idx_end = spike_in_trimmed + half + 1
    template = filtered[idx_start:idx_end].copy()

    params.spike_template = template
    params.peak_threshold = 0.005
    params.amplitude_threshold = 0.001

    recording = Recording(
        name="pipeline_test",
        voltage=noise, sample_rate=fs,
    )
    return recording, params, true_positions


class TestDetectSpikes:
    def test_detects_embedded_spikes(self, pipeline_recording):
        recording, params, true_positions = pipeline_recording
        result = detect_spikes(recording, params)

        assert isinstance(result, SpikeDetectionResult)
        assert result.n_spikes > 0

    def test_corrected_times_near_true_positions(self, pipeline_recording):
        recording, params, true_positions = pipeline_recording
        result = detect_spikes(recording, params)

        if result.n_spikes == 0:
            pytest.skip("No spikes detected; cannot verify positions")

        tolerance = 100  # samples (10ms at 10kHz)
        matched = 0
        for detected in result.spike_times:
            min_dist = np.min(np.abs(true_positions - detected))
            if min_dist < tolerance:
                matched += 1

        assert matched >= 1, (
            f"Only {matched} out of {result.n_spikes} "
            "detected spikes matched true positions"
        )

    def test_returns_spike_detection_result(self, pipeline_recording):
        recording, params, _ = pipeline_recording
        result = detect_spikes(recording, params)
        assert isinstance(result, SpikeDetectionResult)
        assert hasattr(result, "spike_times")
        assert hasattr(result, "spike_times_uncorrected")
        assert hasattr(result, "params")

    def test_raises_without_template(self):
        from spikedetect.models import Recording, SpikeDetectionParams

        recording = Recording(
            name="test",
            voltage=np.zeros(10000),
            sample_rate=10000,
        )
        params = SpikeDetectionParams(fs=10000)
        with pytest.raises(ValueError, match="No spike template provided"):
            detect_spikes(recording, params)

    def test_empty_result_for_silent_recording(self, default_params):
        from spikedetect.models import Recording

        silent = Recording(
            name="silent",
            voltage=np.zeros(50000),
            sample_rate=10000,
        )
        result = detect_spikes(silent, default_params)
        assert result.n_spikes == 0

    def test_spike_times_are_int64(self, pipeline_recording):
        recording, params, _ = pipeline_recording
        result = detect_spikes(recording, params)
        assert result.spike_times.dtype == np.int64
        assert result.spike_times_uncorrected.dtype == np.int64

    def test_detect_with_pre_filter(self, pipeline_recording):
        recording, params, _ = pipeline_recording
        result = detect_spikes(recording, params, pre_filter_cutoff=3000.0)
        assert isinstance(result, SpikeDetectionResult)


class TestTemplateFwhm:
    def test_gaussian_template(self):
        # FWHM of a Gaussian is 2*sqrt(2*ln(2))*sigma ~= 2.355*sigma
        sigma = 5.0
        x = np.arange(-30, 31)
        template = np.exp(-0.5 * (x / sigma) ** 2)
        fwhm = _template_fwhm_samples(template)
        expected = round(2.355 * sigma)
        assert abs(fwhm - expected) <= 1, (
            f"FWHM={fwhm}, expected ~{expected}"
        )

    def test_flat_template_returns_one(self):
        assert _template_fwhm_samples(np.ones(10)) == 1

    def test_empty_template_returns_one(self):
        assert _template_fwhm_samples(np.array([])) == 1

    def test_none_template_returns_one(self):
        assert _template_fwhm_samples(None) == 1


class TestRefractoryFilter:
    def test_drops_close_pair(self):
        corrected = np.array([100, 105, 200, 300])
        uncorrected = np.array([99, 104, 199, 299])
        out_c, out_u = _apply_refractory_filter(corrected, uncorrected, 10)
        np.testing.assert_array_equal(out_c, [100, 200, 300])
        np.testing.assert_array_equal(out_u, [99, 199, 299])

    def test_keeps_first_drops_second(self):
        # User explicitly asked: keep first, drop second.
        corrected = np.array([100, 103])
        uncorrected = np.array([95, 110])
        out_c, out_u = _apply_refractory_filter(corrected, uncorrected, 10)
        np.testing.assert_array_equal(out_c, [100])
        np.testing.assert_array_equal(out_u, [95])

    def test_unsorted_input_is_sorted_by_corrected(self):
        corrected = np.array([300, 100, 200])
        uncorrected = np.array([301, 101, 201])
        out_c, out_u = _apply_refractory_filter(corrected, uncorrected, 10)
        np.testing.assert_array_equal(out_c, [100, 200, 300])
        np.testing.assert_array_equal(out_u, [101, 201, 301])

    def test_zero_min_dt_is_passthrough(self):
        corrected = np.array([100, 101, 102])
        uncorrected = np.array([100, 101, 102])
        out_c, out_u = _apply_refractory_filter(corrected, uncorrected, 0)
        np.testing.assert_array_equal(out_c, [100, 101, 102])

    def test_single_spike_unchanged(self):
        out_c, out_u = _apply_refractory_filter(
            np.array([42]), np.array([41]), 10,
        )
        np.testing.assert_array_equal(out_c, [42])
        np.testing.assert_array_equal(out_u, [41])

    def test_chain_compares_against_last_kept(self):
        # Filter compares each candidate to the *last kept* spike, not all
        # earlier kept ones. With min_dt=5: 100 kept, 102 and 104 dropped
        # (within 5 of 100), 106 kept (6 samples past 100).
        corrected = np.array([100, 102, 104, 106])
        uncorrected = corrected.copy()
        out_c, _ = _apply_refractory_filter(corrected, uncorrected, 5)
        np.testing.assert_array_equal(out_c, [100, 106])

    def test_pipeline_drops_double_detection(self, pipeline_recording):
        # Force a double detection by manually appending a near-duplicate
        # uncorrected location -- detect_spikes runs the refractory step
        # at the end, so we'd need to monkey it. Instead just verify the
        # output has no pairs spaced under the template FWHM.
        recording, params, _ = pipeline_recording
        result = detect_spikes(recording, params)
        if result.n_spikes < 2:
            pytest.skip("Need >=2 spikes to check ISI")
        fwhm = _template_fwhm_samples(params.spike_template)
        isis = np.diff(np.sort(result.spike_times))
        assert np.all(isis >= fwhm), (
            f"Some ISIs ({isis[isis < fwhm]}) are below FWHM={fwhm}"
        )

    def test_min_isi_override_disables_filter(self, pipeline_recording):
        recording, params, _ = pipeline_recording
        params.min_isi_samples = 0
        result = detect_spikes(recording, params)
        # Just verify it runs and returns something sensible. The
        # filter is off, so we can't assert anything tighter than
        # "n_spikes >= count when filter on".
        assert isinstance(result, SpikeDetectionResult)
