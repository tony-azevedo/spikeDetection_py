"""Full spike detection pipeline orchestrator.

Ports MATLAB ``spikeDetectionNonInteractive.m`` -- runs
the complete detection pipeline: filter -> find peaks ->
template match -> threshold -> correct spike times via
inflection point.
"""

from __future__ import annotations

import logging

import numpy as np

from spikedetect.models import (
    Recording,
    SpikeDetectionParams,
    SpikeDetectionResult,
)
from spikedetect.pipeline.filtering import SignalFilter
from spikedetect.pipeline.inflection import (
    InflectionPointDetector,
)
from spikedetect.pipeline.peaks import PeakFinder
from spikedetect.pipeline.template import TemplateMatcher

logger = logging.getLogger(__name__)


def _template_fwhm_samples(template: np.ndarray) -> int:
    """Number of samples where the template exceeds (min+max)/2.

    A robust FWHM proxy for the min-max-normalized templates the GUI saves.
    Falls back to 1 for degenerate (flat / empty) templates.
    """
    if template is None or len(template) == 0:
        return 1
    t = np.asarray(template, dtype=np.float64)
    half = (float(t.min()) + float(t.max())) / 2.0
    return max(int(np.sum(t > half)), 1)


def _apply_refractory_filter(
    corrected: np.ndarray,
    uncorrected: np.ndarray,
    min_dt_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy forward sweep: keep each spike whose time is at least
    ``min_dt_samples`` after the previously kept spike. Both arrays are
    sorted by ``corrected`` time and returned as a paired (corrected,
    uncorrected) tuple."""
    if len(corrected) <= 1 or min_dt_samples <= 0:
        return corrected, uncorrected
    order = np.argsort(corrected)
    corr_s = corrected[order]
    unc_s = uncorrected[order]
    keep = np.zeros(len(corr_s), dtype=bool)
    last_kept = -np.inf
    for i, t in enumerate(corr_s):
        if t - last_kept >= min_dt_samples:
            keep[i] = True
            last_kept = t
    return corr_s[keep], unc_s[keep]


class SpikeDetector:
    """Full non-interactive spike detection pipeline.

    Ports MATLAB ``spikeDetectionNonInteractive.m``. Runs
    the complete pipeline: filter -> find peaks -> template
    match -> threshold -> correct spike times via inflection
    point.

    Example::

        result = SpikeDetector.detect(recording, params)
        print(result.n_spikes)
    """

    @staticmethod
    def detect(
        recording: Recording,
        params: SpikeDetectionParams,
        start_offset: float = 0.01,
        pre_filter_cutoff: float | None = None,
    ) -> SpikeDetectionResult:
        """Run the full spike detection pipeline.

        Args:
            recording: The electrophysiology recording to
                analyze.
            params: Detection parameters. Must have
                ``spike_template`` set (non-None) for
                detection to proceed.
            start_offset: Fraction of recording to skip at
                the start (default 0.01 = 1%). Matches
                MATLAB ``start_point = round(.01*fs)``.
            pre_filter_cutoff: If not None, apply a
                low-pass pre-filter at this cutoff (Hz)
                to the raw voltage before trimming and
                running the pipeline. Ports MATLAB
                ``lowPassFilterMembraneVoltage.m``.
                Default None (no pre-filtering).

        Returns:
            A ``SpikeDetectionResult`` containing
            corrected and uncorrected spike times (as
            0-based sample indices into the *original*
            recording voltage).

        Raises:
            ValueError: If ``params.spike_template`` is
                None.

        Note:
            Original MATLAB function:
            ``spikeDetectionNonInteractive.m``

            The returned spike times are offset by
            ``start_point`` so they index into the
            original ``recording.voltage`` array,
            matching the MATLAB convention
            ``trial.spikes = spikes_detected +
            start_point``.
        """
        if not isinstance(recording, Recording):
            raise TypeError(
                "Expected a Recording object, got "
                f"{type(recording).__name__}. "
                "Load your data first with "
                "load_recording('file.mat') or create "
                "a Recording(name='...', "
                "voltage=array, sample_rate=10000)."
            )
        if not isinstance(params, SpikeDetectionParams):
            raise TypeError(
                "Expected SpikeDetectionParams, got "
                f"{type(params).__name__}. "
                "Create params with "
                "SpikeDetectionParams(fs=10000) or "
                "SpikeDetectionParams.default(fs=10000)."
            )

        import copy
        params = copy.copy(params)
        params = params.validate()

        if params.spike_template is None:
            raise ValueError(
                "No spike template provided. Use the "
                "interactive GUI (FilterGUI / "
                "TemplateSelectionGUI) to select one, "
                "or pass a 1-D numpy array as "
                "params.spike_template."
            )

        voltage = recording.voltage.copy()
        fs = params.fs

        # Optional low-pass pre-filter
        # (MATLAB lowPassFilterMembraneVoltage)
        if pre_filter_cutoff is not None:
            logger.info(
                "Applying pre-filter: cutoff=%.0f Hz",
                pre_filter_cutoff,
            )
            voltage = SignalFilter.pre_filter(
                voltage, fs, cutoff=pre_filter_cutoff,
            )
        duration_sec = len(voltage) / fs
        logger.info(
            "Starting spike detection on '%s' "
            "(%.1f s, %.0f Hz)",
            recording.name,
            duration_sec,
            fs,
        )

        # Trim start (MATLAB: start_point = round(.01*fs))
        start_point = round(start_offset * fs)
        stop_point = len(voltage)
        unfiltered_data = voltage[start_point:stop_point]

        # Step 1: Filter
        logger.info(
            "Filtering: hp=%.0f Hz, lp=%.0f Hz, "
            "diff_order=%d, polarity=%+d",
            params.hp_cutoff,
            params.lp_cutoff,
            params.diff_order,
            params.polarity,
        )
        filtered_data = SignalFilter.filter_data(
            unfiltered_data,
            fs=fs,
            hp_cutoff=params.hp_cutoff,
            lp_cutoff=params.lp_cutoff,
            diff_order=params.diff_order,
            polarity=params.polarity,
        )

        # Step 2: Find peaks
        # Guard against absurd peak_threshold
        # (use local var, don't mutate params)
        peak_thresh = params.peak_threshold
        if peak_thresh > 1e4 * np.std(filtered_data):
            peak_thresh = 3 * np.std(filtered_data)

        spike_locs = PeakFinder.find_spike_locations(
            filtered_data,
            peak_threshold=peak_thresh,
            fs=fs,
            spike_template_width=(
                params.spike_template_width
            ),
        )

        logger.info(
            "Found %d candidate peaks", len(spike_locs),
        )

        if len(spike_locs) == 0:
            logger.info(
                "No candidate peaks found -- "
                "returning empty result",
            )
            return SpikeDetectionResult(
                spike_times=np.array(
                    [], dtype=np.int64,
                ),
                spike_times_uncorrected=np.array(
                    [], dtype=np.int64,
                ),
                params=params,
            )

        # Verify no duplicates
        assert len(spike_locs) == len(
            np.unique(spike_locs)
        ), "Duplicate peak locations detected"

        # Step 3: Template matching (DTW + amplitude)
        logger.info(
            "Computing DTW template match for "
            "%d candidates...",
            len(spike_locs),
        )
        match_result = TemplateMatcher.match(
            spike_locs,
            params.spike_template,
            filtered_data,
            unfiltered_data,
            params.spike_template_width,
            fs,
        )

        if len(match_result.dtw_distances) == 0:
            logger.info(
                "No valid candidates after "
                "template matching",
            )
            return SpikeDetectionResult(
                spike_times=np.array(
                    [], dtype=np.int64,
                ),
                spike_times_uncorrected=np.array(
                    [], dtype=np.int64,
                ),
                params=params,
            )

        # Step 4: Threshold -- keep spikes with
        # DTW < threshold AND amplitude > threshold.
        # Use match_result.spike_locs (not original
        # spike_locs) since NaN candidates may have been
        # filtered out during template matching.
        suspect = (
            (match_result.dtw_distances
             < params.distance_threshold)
            & (match_result.amplitudes
               > params.amplitude_threshold)
        )
        accepted_locs = match_result.spike_locs[suspect]

        logger.info(
            "Accepted %d / %d candidates "
            "(distance < %.1f, amplitude > %.3f)",
            len(accepted_locs),
            len(match_result.spike_locs),
            params.distance_threshold,
            params.amplitude_threshold,
        )

        if len(accepted_locs) == 0:
            logger.info(
                "No spikes passed thresholds -- "
                "returning empty result",
            )
            return SpikeDetectionResult(
                spike_times=np.array(
                    [], dtype=np.int64,
                ),
                spike_times_uncorrected=np.array(
                    [], dtype=np.int64,
                ),
                params=params,
            )

        # Step 5: Correct spike times via inflection point
        uf_cands = (
            match_result.unfiltered_candidates[
                :, suspect
            ]
        )
        waveforms = uf_cands - uf_cands[0:1, :]

        (corrected, uncorrected, infl_peak) = (
            InflectionPointDetector.estimate_spike_times(
                accepted_locs,
                waveforms,
                match_result.dtw_distances[suspect],
                params.spike_template_width,
                fs,
                params.distance_threshold,
                params.likely_inflection_point_peak,
            )
        )

        # Update params with computed inflection point
        params.likely_inflection_point_peak = infl_peak

        # Step 6: Refractory filter -- drop double detections where two
        # local peaks both pass threshold for a single underlying spike.
        # Use the template's FWHM as the natural minimum spacing unless
        # the user explicitly set ``params.min_isi_samples``.
        if params.min_isi_samples is None:
            min_isi = _template_fwhm_samples(params.spike_template)
        else:
            min_isi = int(params.min_isi_samples)
        n_before = len(corrected)
        min_isi = 3*min_isi # Just double the FWHM. This is a bit arbitrary, but so is the FWHM anyways.
        corrected, uncorrected = _apply_refractory_filter(
            corrected, uncorrected, min_isi,
        )       
        n_dropped = n_before - len(corrected)
        if n_dropped > 0:
            logger.info(
                "Refractory filter dropped %d double detections "
                "(min_isi=%d samples = %.2f ms)",
                n_dropped, min_isi, 1000 * min_isi / fs,
            )

        # Offset back to original recording indices
        corrected_global = corrected + start_point
        uncorrected_global = uncorrected + start_point

        logger.info(
            "Detection complete: %d spikes in '%s'",
            len(corrected_global),
            recording.name,
        )

        return SpikeDetectionResult(
            spike_times=corrected_global.astype(
                np.int64,
            ),
            spike_times_uncorrected=(
                uncorrected_global.astype(np.int64)
            ),
            params=params,
        )


# Backwards-compatible alias
detect_spikes = SpikeDetector.detect
