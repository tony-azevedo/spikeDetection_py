"""Interactive spike detection workflow.

Ports the MATLAB ``spikeDetection.m`` interactive loop (lines 89-126)
where a "Done?" dialog lets the user cycle through Filter -> Template ->
Threshold repeatedly until satisfied, then optionally spot-check.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from spikedetect.gui._widgets import blocking_wait
from spikedetect.gui.filter_gui import FilterGUI
from spikedetect.gui.template_gui import TemplateSelectionGUI
from spikedetect.gui.threshold_gui import ThresholdGUI
from spikedetect.gui.spotcheck_gui import SpotCheckGUI
from spikedetect.models import (
    Recording, SpikeDetectionParams, SpikeDetectionResult,
)
from spikedetect.pipeline.detect import SpikeDetector
from spikedetect.pipeline.filtering import SignalFilter
from spikedetect.pipeline.peaks import PeakFinder
from spikedetect.pipeline.template import TemplateMatcher

logger = logging.getLogger(__name__)


def _ask_done() -> bool:
    """Show a matplotlib dialog asking 'Done with spike detection?'.

    Mirrors MATLAB ``questdlg('Done with spike detection?', ...)``.

    Returns:
        True if the user clicks Yes or closes the figure,
        False if No.
    """
    result = {"done": True}

    fig, ax = plt.subplots(figsize=(4, 1.5))
    fig.set_facecolor("white")
    ax.set_visible(False)
    fig.suptitle("Done with spike detection?", fontsize=12, y=0.85)

    ax_yes = fig.add_axes([0.2, 0.15, 0.25, 0.3])
    ax_no = fig.add_axes([0.55, 0.15, 0.25, 0.3])
    btn_yes = Button(ax_yes, "Yes")
    btn_no = Button(ax_no, "No")

    def _on_yes(_event):
        result["done"] = True
        fig.canvas.stop_event_loop()

    def _on_no(_event):
        result["done"] = False
        fig.canvas.stop_event_loop()

    def _on_close(_event):
        result["done"] = True
        fig.canvas.stop_event_loop()

    btn_yes.on_clicked(_on_yes)
    btn_no.on_clicked(_on_no)
    fig.canvas.mpl_connect("close_event", _on_close)

    try:
        fig.canvas.start_event_loop(timeout=0)
    except Exception:
        pass

    if plt.fignum_exists(fig.number):
        plt.close(fig)

    return result["done"]


class InteractiveWorkflow:
    """Run the full interactive spike detection workflow.

    Chains FilterGUI -> TemplateSelectionGUI -> detect_spikes ->
    ThresholdGUI -> detect_spikes -> SpotCheckGUI with an optional
    repeat loop (like MATLAB's "Done?" dialog).

    This mirrors the MATLAB ``spikeDetection.m`` interactive path
    (lines 89-126) where ``questdlg('Done?')`` lets the user cycle
    through Filter -> Template -> Threshold repeatedly until satisfied.

    Examples
    --------
    >>> from spikedetect.gui import InteractiveWorkflow
    >>> result, params = InteractiveWorkflow.run(recording)
    >>> print(result.summary())
    """

    @staticmethod
    def run(
        recording: Recording,
        params: SpikeDetectionParams | None = None,
        spot_check: bool = True,
    ) -> tuple[SpikeDetectionResult, SpikeDetectionParams]:
        """Run the full interactive workflow.

        Args:
            recording: The electrophysiology recording to
                analyze.
            params: Initial detection parameters. If None,
                creates defaults from ``recording.sample_rate``.
            spot_check: If True (default), run SpotCheckGUI
                after the detection loop.

        Returns:
            A tuple of (result, params) where result is the
            final detection result (spot-checked if
            ``spot_check=True``) and params is the final
            parameters after all user adjustments.
        """
        if params is None:
            params = SpikeDetectionParams.default(fs=recording.sample_rate)
        else:
            params = deepcopy(params)

        result = None

        while True:
            # Step 1: FilterGUI -- tune filter parameters
            logger.info("Opening FilterGUI for parameter tuning")
            filter_gui = FilterGUI(recording.voltage, params)
            params = filter_gui.run()

            # Step 2: Filter data for template selection
            filtered_data = SignalFilter.filter_data(
                recording.voltage,
                fs=params.fs,
                hp_cutoff=params.hp_cutoff,
                lp_cutoff=params.lp_cutoff,
                diff_order=params.diff_order,
                polarity=params.polarity,
            )

            # Step 3: TemplateSelectionGUI -- select seed spikes
            logger.info("Opening TemplateSelectionGUI for template selection")
            template_gui = TemplateSelectionGUI(filtered_data, params)
            template = template_gui.run()

            if template is None:
                logger.warning(
                    "No template selected -- cannot run detection. "
                    "Returning to filter step."
                )
                continue

            params.spike_template = template
            params.template_updated_at = template_gui.template_updated_at

            # Step 4: Run detection with current params
            logger.info("Running spike detection pipeline")
            result = SpikeDetector.detect(recording, params)
            logger.info("Detection found %d spikes", result.n_spikes)

            # Step 5: Run ThresholdGUI for threshold adjustment
            # Need to compute match_result for ThresholdGUI
            start_point = round(0.01 * params.fs)
            unfiltered_data = recording.voltage[start_point:]
            filtered_trimmed = SignalFilter.filter_data(
                unfiltered_data,
                fs=params.fs,
                hp_cutoff=params.hp_cutoff,
                lp_cutoff=params.lp_cutoff,
                diff_order=params.diff_order,
                polarity=params.polarity,
            )

            spike_locs = PeakFinder.find_spike_locations(
                filtered_trimmed,
                peak_threshold=params.peak_threshold,
                fs=params.fs,
                spike_template_width=params.spike_template_width,
            )

            if len(spike_locs) > 0:
                match_result = TemplateMatcher.match(
                    spike_locs,
                    params.spike_template,
                    filtered_trimmed,
                    unfiltered_data,
                    params.spike_template_width,
                    params.fs,
                )

                logger.info("Opening ThresholdGUI for threshold adjustment")
                threshold_gui = ThresholdGUI(match_result, params)
                params = threshold_gui.run()

                # Step 6: Re-run detection with updated thresholds
                logger.info("Re-running detection with updated thresholds")
                result = SpikeDetector.detect(recording, params)
                logger.info(
                    "Detection with updated thresholds found %d spikes",
                    result.n_spikes,
                )
            else:
                logger.warning("No candidate peaks found for threshold GUI")

            # Step 7: Ask "Done?" (MATLAB questdlg pattern)
            if _ask_done():
                break

        # Step 8: Optional spot-check
        if spot_check and result is not None and result.n_spikes > 0:
            logger.info("Opening SpotCheckGUI for spike review")
            spotcheck_gui = SpotCheckGUI(recording, result)
            result = spotcheck_gui.run()

        if result is None:
            # No detection was ever run
            # (edge case: user never selected template)
            result = SpikeDetectionResult(
                spike_times=np.array([], dtype=np.int64),
                spike_times_uncorrected=np.array([], dtype=np.int64),
                params=params,
            )

        return result, params
