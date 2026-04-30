"""Smoke tests for interactive GUI components.

These tests use the non-interactive 'Agg' backend so no windows appear.
They verify that each GUI can be instantiated, that internal methods
(_build_figure, _apply_filter, _update_plots, _on_pick, etc.) run
without errors, and that callbacks produce expected state changes.

The full .run() methods are NOT tested here because they block on user
input via matplotlib's event loop.
"""

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must be set before importing pyplot

import numpy as np
import pytest
import matplotlib.pyplot as plt

from spikedetect.models import (
    Recording,
    SpikeDetectionParams,
    SpikeDetectionResult,
)
from spikedetect.gui.filter_gui import FilterGUI
from spikedetect.gui.template_gui import TemplateSelectionGUI
from spikedetect.gui.threshold_gui import ThresholdGUI
from spikedetect.gui.spotcheck_gui import SpotCheckGUI
from spikedetect.pipeline.filtering import SignalFilter
from spikedetect.pipeline.peaks import PeakFinder
from spikedetect.pipeline.template import TemplateMatcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Create a synthetic recording with embedded spikes."""
    fs = 10000
    duration = 2.0
    n_samples = int(fs * duration)
    rng = np.random.default_rng(42)
    voltage = rng.normal(0, 0.001, n_samples)

    positions = np.array([3000, 6000, 9000, 12000, 15000])
    for pos in positions:
        t_local = np.arange(-25, 26) / fs
        spike = 5.0 * np.exp(-0.5 * (t_local / 0.0003) ** 2)
        voltage[pos - 25:pos + 26] += spike

    return voltage, fs, positions


@pytest.fixture
def params(synthetic_data):
    """Default params for synthetic data."""
    _, fs, _ = synthetic_data
    return SpikeDetectionParams(
        fs=fs, hp_cutoff=200, lp_cutoff=800,
        diff_order=1, polarity=1, peak_threshold=0.005,
    )


@pytest.fixture
def filtered(synthetic_data, params):
    """Filtered version of synthetic data."""
    voltage, fs, _ = synthetic_data
    start_point = round(0.01 * fs)
    unfiltered = voltage[start_point:]
    filt = SignalFilter.filter_data(
        unfiltered, fs=fs,
        hp_cutoff=params.hp_cutoff, lp_cutoff=params.lp_cutoff,
        diff_order=params.diff_order, polarity=params.polarity,
    )
    return filt, unfiltered


@pytest.fixture
def params_with_template(synthetic_data, params, filtered):
    """Params with a spike template extracted from the first spike."""
    _, fs, positions = synthetic_data
    filt, _ = filtered
    start_point = round(0.01 * fs)
    stw = params.spike_template_width
    half = stw // 2
    spike_in_trimmed = positions[0] - start_point
    template = filt[spike_in_trimmed - half:spike_in_trimmed + half + 1].copy()
    params.spike_template = template
    params.amplitude_threshold = 0.001
    return params


@pytest.fixture
def match_result(filtered, params_with_template):
    """TemplateMatcher result for ThresholdGUI tests."""
    filt, unfiltered = filtered
    p = params_with_template
    locs = PeakFinder.find_spike_locations(
        filt, peak_threshold=p.peak_threshold,
        fs=p.fs, spike_template_width=p.spike_template_width,
    )
    return TemplateMatcher.match(
        spike_locs=locs,
        spike_template=p.spike_template,
        filtered_data=filt,
        unfiltered_data=unfiltered,
        spike_template_width=p.spike_template_width,
        fs=p.fs,
    )


@pytest.fixture
def recording_and_result(synthetic_data, params_with_template):
    """A Recording and SpikeDetectionResult for SpotCheckGUI tests."""
    voltage, fs, _ = synthetic_data
    rec = Recording(name="test", voltage=voltage, sample_rate=fs)

    import spikedetect as sd
    result = sd.detect_spikes(rec, params_with_template)
    return rec, result


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# FilterGUI tests
# ---------------------------------------------------------------------------

class TestFilterGUI:
    def test_instantiation(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        assert gui.params is not None
        assert gui.fig is None

    def test_apply_filter(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        gui._apply_filter()
        assert gui._filtered is not None
        assert len(gui._filtered) == len(voltage)
        assert gui._locs is not None

    def test_build_figure(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        gui._apply_filter()
        gui._build_figure()
        assert gui.fig is not None
        assert gui._ax_unfilt is not None
        assert gui._ax_filt is not None

    def test_update_plots(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        gui._apply_filter()
        gui._build_figure()
        gui._update_plots()  # should not raise

    def test_slider_change_updates_params(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        gui._apply_filter()
        gui._build_figure()

        original_hp = gui.params.hp_cutoff
        gui._sl_hp.set_val(300.0)
        assert gui.params.hp_cutoff == 300.0
        assert gui.params.hp_cutoff != original_hp

    def test_polarity_toggle(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        gui._apply_filter()
        gui._build_figure()

        original_pol = gui.params.polarity
        gui._on_polarity_toggle(None)
        assert gui.params.polarity == -original_pol

    def test_diff_order_change(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        gui._apply_filter()
        gui._build_figure()

        gui._on_diff_change("2")
        assert gui.params.diff_order == 2


# ---------------------------------------------------------------------------
# TemplateSelectionGUI tests
# ---------------------------------------------------------------------------

class TestTemplateSelectionGUI:
    def test_instantiation(self, filtered, params):
        filt, _ = filtered
        gui = TemplateSelectionGUI(filt, params)
        assert gui.fig is None
        assert gui._selected_indices == []

    def test_build_figure(self, filtered, params):
        filt, _ = filtered
        gui = TemplateSelectionGUI(filt, params)
        gui._build_figure()
        assert gui.fig is not None
        assert gui._ax_main is not None
        assert gui._ax_squig is not None

    def test_build_template_no_selection(self, filtered, params):
        filt, _ = filtered
        gui = TemplateSelectionGUI(filt, params)
        gui._build_figure()
        template = gui._build_template()
        assert template is None

    def test_build_template_with_selection(
        self, filtered, params, synthetic_data,
    ):
        filt, _ = filtered
        _, fs, positions = synthetic_data
        start_point = round(0.01 * fs)
        gui = TemplateSelectionGUI(filt, params)
        gui._build_figure()

        # Simulate selecting two spikes
        for pos in positions[:2]:
            loc = pos - start_point
            if loc >= 0 and loc < len(filt):
                gui._selected_indices.append(loc)

        template = gui._build_template()
        assert template is not None
        assert template.ndim == 1
        assert len(template) > 0

    def test_on_pick_with_mock_event(self, filtered, params):
        filt, _ = filtered
        gui = TemplateSelectionGUI(filt, params)
        gui._build_figure()

        # Create a mock pick event
        class MockMouseEvent:
            xdata = (
                float(gui._peak_locs[0])
                if len(gui._peak_locs) > 0
                else 100.0
            )

        class MockEvent:
            artist = True
            ind = np.array([0])
            mouseevent = MockMouseEvent()

        if len(gui._peak_locs) > 0:
            gui._on_pick(MockEvent())
            assert len(gui._selected_indices) == 1

    def test_fresh_template_kept_with_clock_reset(
        self, filtered, params_with_template,
    ):
        from datetime import datetime, timedelta
        filt, _ = filtered
        old_ts = datetime.now() - timedelta(hours=1)
        params_with_template.template_updated_at = old_ts
        gui = TemplateSelectionGUI(filt, params_with_template)
        assert gui._reuse_existing_template
        assert gui._existing_template_is_fresh
        gui._build_figure()
        # No clicks on a fresh template -> keep template, restart clock.
        before = datetime.now()
        template = gui._build_template()
        after = datetime.now()
        np.testing.assert_array_equal(
            template, params_with_template.spike_template,
        )
        assert gui.template_updated_at is not None
        assert before <= gui.template_updated_at <= after
        assert gui.template_updated_at != old_ts

    def test_stale_template_kept_with_clock_reset(
        self, filtered, params_with_template,
    ):
        from datetime import datetime, timedelta
        filt, _ = filtered
        params_with_template.template_updated_at = (
            datetime.now() - timedelta(hours=48)
        )
        gui = TemplateSelectionGUI(
            filt, params_with_template, template_ttl_hours=24,
        )
        assert gui._reuse_existing_template
        assert not gui._existing_template_is_fresh
        gui._build_figure()
        # No clicks on a stale template -> still keep it; clock restarts.
        before = datetime.now()
        template = gui._build_template()
        after = datetime.now()
        np.testing.assert_array_equal(
            template, params_with_template.spike_template,
        )
        assert before <= gui.template_updated_at <= after

    def test_template_without_timestamp_treated_as_stale(
        self, filtered, params_with_template,
    ):
        from datetime import datetime
        filt, _ = filtered
        params_with_template.template_updated_at = None
        gui = TemplateSelectionGUI(filt, params_with_template)
        assert gui._reuse_existing_template
        assert not gui._existing_template_is_fresh
        gui._build_figure()
        before = datetime.now()
        template = gui._build_template()
        after = datetime.now()
        np.testing.assert_array_equal(
            template, params_with_template.spike_template,
        )
        # Untimestamped templates also get the clock-reset on Enter-keep.
        assert before <= gui.template_updated_at <= after

    def test_no_template_returns_none_without_clicks(
        self, filtered, params,
    ):
        filt, _ = filtered
        # params here has no template
        gui = TemplateSelectionGUI(filt, params)
        assert not gui._reuse_existing_template
        gui._build_figure()
        assert gui._build_template() is None
        assert gui.template_updated_at is None

    def test_clicks_override_existing_template_and_stamp(
        self, filtered, params_with_template, synthetic_data,
    ):
        from datetime import datetime, timedelta
        filt, _ = filtered
        _, fs, positions = synthetic_data
        start_point = round(0.01 * fs)
        old_ts = datetime.now() - timedelta(hours=1)
        params_with_template.template_updated_at = old_ts
        gui = TemplateSelectionGUI(filt, params_with_template)
        gui._build_figure()
        for pos in positions[:2]:
            loc = pos - start_point
            if 0 <= loc < len(filt):
                gui._selected_indices.append(loc)

        before = datetime.now()
        template = gui._build_template()
        after = datetime.now()
        assert template is not None
        assert gui.template_updated_at is not None
        assert before <= gui.template_updated_at <= after
        assert gui.template_updated_at != old_ts


# ---------------------------------------------------------------------------
# ThresholdGUI tests
# ---------------------------------------------------------------------------

class TestThresholdGUI:
    def test_instantiation(self, match_result, params_with_template):
        gui = ThresholdGUI(match_result, params_with_template)
        assert gui.fig is None
        assert gui._active_threshold == "distance"

    def test_build_figure(self, match_result, params_with_template):
        gui = ThresholdGUI(match_result, params_with_template)
        gui._build_figure()
        assert gui.fig is not None
        assert gui._ax_scatter is not None

    def test_update_panels(self, match_result, params_with_template):
        gui = ThresholdGUI(match_result, params_with_template)
        gui._build_figure()
        gui._update_panels()  # should not raise

    def test_toggle_active_threshold(self, match_result, params_with_template):
        gui = ThresholdGUI(match_result, params_with_template)
        gui._build_figure()
        gui._update_panels()

        assert gui._active_threshold == "distance"
        gui._toggle_active()
        assert gui._active_threshold == "amplitude"
        gui._toggle_active()
        assert gui._active_threshold == "distance"

    def test_on_click_updates_distance_threshold(
        self, match_result, params_with_template,
    ):
        gui = ThresholdGUI(match_result, params_with_template)
        gui._build_figure()
        gui._update_panels()

        class MockEvent:
            inaxes = gui._ax_scatter
            xdata = 5.0
            ydata = 0.5

        gui._on_click(MockEvent())
        assert gui.params.distance_threshold == 5.0

    def test_on_click_updates_amplitude_threshold(
        self, match_result, params_with_template,
    ):
        gui = ThresholdGUI(match_result, params_with_template)
        gui._build_figure()
        gui._update_panels()

        gui._toggle_active()  # switch to amplitude

        class MockEvent:
            inaxes = gui._ax_scatter
            xdata = 5.0
            ydata = 0.3

        gui._on_click(MockEvent())
        assert gui.params.amplitude_threshold == 0.3


# ---------------------------------------------------------------------------
# SpotCheckGUI tests
# ---------------------------------------------------------------------------

class TestSpotCheckGUI:
    def test_instantiation(self, recording_and_result):
        rec, result = recording_and_result
        gui = SpotCheckGUI(rec, result)
        assert gui.fig is None
        assert gui._spike_idx == 0

    def test_setup_and_build_figure(self, recording_and_result):
        rec, result = recording_and_result
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        gui._build_figure()
        assert gui.fig is not None
        assert gui._ax_trace is not None
        assert gui._ax_spike is not None

    def test_show_current_spike(self, recording_and_result):
        rec, result = recording_and_result
        if result.n_spikes == 0:
            pytest.skip("No spikes detected")
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        gui._build_figure()
        gui._show_current_spike()  # should not raise

    def test_handle_key_accept(self, recording_and_result):
        rec, result = recording_and_result
        if result.n_spikes == 0:
            pytest.skip("No spikes detected")
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        gui._build_figure()
        gui._show_current_spike()

        gui._handle_key("y")
        assert gui._accepted[0] is True or gui._accepted[0] == True

    def test_handle_key_reject(self, recording_and_result):
        rec, result = recording_and_result
        if result.n_spikes == 0:
            pytest.skip("No spikes detected")
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        gui._build_figure()
        gui._show_current_spike()

        gui._handle_key("n")
        assert gui._accepted[0] == False

    def test_handle_key_enter_returns_done(self, recording_and_result):
        rec, result = recording_and_result
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        gui._build_figure()

        action = gui._handle_key("enter")
        assert action == "done"

    def test_advance_increments_index(self, recording_and_result):
        rec, result = recording_and_result
        if result.n_spikes < 2:
            pytest.skip("Need at least 2 spikes")
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        gui._build_figure()
        gui._show_current_spike()

        assert gui._spike_idx == 0
        gui._advance()
        assert gui._spike_idx == 1

    def test_setup_populates_dtw_data(self, recording_and_result):
        rec, result = recording_and_result
        if result.n_spikes == 0:
            pytest.skip("No spikes detected")
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        assert gui._dtw_distances is not None
        assert gui._amplitudes is not None
        assert len(gui._dtw_distances) > 0
        assert len(gui._amplitudes) == len(gui._dtw_distances)

    def test_scatter_plot_has_dots(self, recording_and_result):
        rec, result = recording_and_result
        if result.n_spikes == 0:
            pytest.skip("No spikes detected")
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        gui._build_figure()
        # Scatter should have accepted dots (blue)
        assert gui._scat_in is not None or gui._scat_out is not None

    def test_raster_ticks_pickable(self, recording_and_result):
        rec, result = recording_and_result
        if result.n_spikes == 0:
            pytest.skip("No spikes detected")
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        gui._build_figure()
        assert gui._raster_lines is not None
        assert gui._raster_lines.get_picker() is not None

    def test_scatter_current_dot_updates(self, recording_and_result):
        rec, result = recording_and_result
        if result.n_spikes == 0:
            pytest.skip("No spikes detected")
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        gui._build_figure()
        gui._show_current_spike()
        # Current dot should have data
        assert gui._current_dot is not None
        xdata = gui._current_dot.get_xdata()
        assert len(xdata) > 0 or len(gui._spike_to_candidate) == 0

    def test_scatter_colors_update_on_reject(self, recording_and_result):
        rec, result = recording_and_result
        if result.n_spikes == 0:
            pytest.skip("No spikes detected")
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        gui._build_figure()
        gui._show_current_spike()

        # Reject first spike
        gui._handle_key("n")
        assert gui._accepted[0] == False
        # _update_scatter_colors was called internally -- should not crash

    def test_spike_to_candidate_mapping(self, recording_and_result):
        rec, result = recording_and_result
        if result.n_spikes == 0:
            pytest.skip("No spikes detected")
        gui = SpotCheckGUI(rec, result)
        gui._setup()
        # Each spike should map to a candidate
        assert len(gui._spike_to_candidate) > 0
        for spike_i, cand_i in gui._spike_to_candidate.items():
            assert 0 <= spike_i < len(gui._spikes)
            assert 0 <= cand_i < len(gui._candidate_locs)


# ---------------------------------------------------------------------------
# InteractiveWorkflow tests
# ---------------------------------------------------------------------------

class TestInteractiveWorkflow:
    def test_import(self):
        from spikedetect.gui import InteractiveWorkflow
        assert hasattr(InteractiveWorkflow, "run")

    def test_run_method_signature(self):
        from spikedetect.gui.workflow import InteractiveWorkflow
        import inspect
        sig = inspect.signature(InteractiveWorkflow.run)
        param_names = list(sig.parameters.keys())
        assert "recording" in param_names
        assert "params" in param_names

    def test_ask_done_function_exists(self):
        from spikedetect.gui.workflow import _ask_done
        assert callable(_ask_done)


# ---------------------------------------------------------------------------
# Non-blocking entry points (for embedding in Qt host apps)
# ---------------------------------------------------------------------------

class TestNonBlockingAPI:
    """Verify each GUI's setup() / on_finished / close() contract.

    Hosts that embed these GUIs (e.g. a PyQt browser app) drive their
    own event loop, so the GUI must not call start_event_loop. Instead,
    setup() builds the figure and installs handlers, the host adds the
    canvas to a layout, and on_finished fires when the user accepts.
    """

    def test_filter_gui_setup_returns_canvas(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        canvas = gui.setup()
        assert canvas is not None
        assert canvas is gui.fig.canvas

    def test_filter_gui_on_finished_fires(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        received = []
        gui.on_finished = received.append
        gui.setup()
        gui.finish()
        assert len(received) == 1
        assert received[0] is gui.params

    def test_filter_gui_finish_is_idempotent(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        received = []
        gui.on_finished = received.append
        gui.setup()
        gui.finish()
        gui.finish()
        gui.close()
        gui.close()
        assert len(received) == 1

    def test_filter_gui_on_params_changed_fires(self, synthetic_data, params):
        voltage, _, _ = synthetic_data
        gui = FilterGUI(voltage, params)
        received = []
        gui.on_params_changed = received.append
        gui.setup()
        gui._sl_hp.set_val(300.0)
        assert len(received) >= 1
        assert received[-1].hp_cutoff == 300.0

    def test_template_gui_on_finished_fires(self, filtered, params_with_template):
        filt, _ = filtered
        gui = TemplateSelectionGUI(filt, params_with_template)
        received = []
        gui.on_finished = received.append
        gui.setup()
        gui.finish()
        assert len(received) == 1
        # No selection made + existing template in params -> existing kept
        np.testing.assert_array_equal(
            received[0], params_with_template.spike_template,
        )

    def test_threshold_gui_on_finished_fires(self, match_result, params_with_template):
        gui = ThresholdGUI(match_result, params_with_template)
        received = []
        gui.on_finished = received.append
        gui.setup()
        gui.finish()
        assert len(received) == 1
        assert received[0] is gui.params

    def test_threshold_gui_b_key_toggles_active(self, match_result, params_with_template):
        gui = ThresholdGUI(match_result, params_with_template)
        gui.setup()
        assert gui._active_threshold == "distance"
        gui._on_key("b")
        assert gui._active_threshold == "amplitude"
        gui._on_key("b")
        assert gui._active_threshold == "distance"

    def test_spotcheck_gui_on_finished_fires(self, recording_and_result):
        rec, result = recording_and_result
        gui = SpotCheckGUI(rec, result)
        received = []
        gui.on_finished = received.append
        gui.setup()
        gui.finish()
        assert len(received) == 1
        assert received[0].spot_checked is True
