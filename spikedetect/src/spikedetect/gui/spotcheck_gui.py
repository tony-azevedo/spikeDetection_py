"""Interactive spike-by-spike review GUI.

Ports the MATLAB ``spikeSpotCheck.m`` function. Allows the user to step
through each detected spike, accept or reject it, adjust its position,
and add or remove spikes from the detection result.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from spikedetect.gui._widgets import (
    raster_ticks, blocking_wait, install_finish_handlers, UserCancelled,
)
from spikedetect.models import (
    Recording, SpikeDetectionParams, SpikeDetectionResult,
)
from spikedetect.pipeline.filtering import filter_data
from spikedetect.pipeline.peaks import PeakFinder
from spikedetect.pipeline.template import TemplateMatcher, TemplateMatchResult
from spikedetect.utils import smooth, smooth_and_differentiate


# Colors matching MATLAB spikeSpotCheck.m
_COLOR_ACCEPTED = (0.0, 0.45, 0.74)       # blue
_COLOR_REJECTED = (0.929, 0.694, 0.125)    # yellow/orange
_COLOR_THRESHOLD = (1.0, 0.0, 0.0)        # red
_COLOR_CURRENT = (1.0, 0.0, 0.0)          # red


class SpotCheckGUI:
    """Interactive GUI for reviewing spikes one by one.

    Args:
        recording: The electrophysiology recording.
        result: Detection result to review and potentially
            modify.

    Attributes:
        result: The (possibly modified) detection result.
        fig: The GUI figure.
    """

    def __init__(
        self,
        recording: Recording,
        result: SpikeDetectionResult,
    ) -> None:
        self._recording = recording
        self.result = deepcopy(result)
        self.fig = None
        self._spike_idx = 0
        self._accepted: np.ndarray | None = None
        self._direction = 1

        # Scatter plot data (populated in _setup)
        self._dtw_distances: np.ndarray | None = None
        self._amplitudes: np.ndarray | None = None
        self._candidate_locs: np.ndarray | None = None
        self._candidate_is_accepted: np.ndarray | None = None
        self._spike_to_candidate: dict[int, int] = {}
        self._current_dot = None
        self._scat_in = None
        self._scat_out = None
        self._raster_lines = None

        self._finished = False
        self._cancelled = False
        self._disconnect_handlers: Callable[[], None] | None = None
        # Callbacks for non-blocking (e.g. Qt) embedding.
        self.on_finished: Callable[[SpikeDetectionResult], None] | None = None
        self.on_spike_index_changed: Callable[[int], None] | None = None

    def setup(self):
        """Build the figure without starting an event loop. See
        :meth:`FilterGUI.setup` for details.

        Returns:
            The figure's canvas.
        """
        if self.fig is not None:
            return self.fig.canvas
        self._setup()
        self._build_figure()
        if self.result.n_spikes > 0:
            self._show_current_spike()
        self._center_window()
        self._disconnect_handlers = install_finish_handlers(
            self.fig, self._on_key, self._finish,
        )
        return self.fig.canvas

    def run(self) -> SpikeDetectionResult:
        """Display the GUI and block until the user finishes.

        Keyboard controls:
            y       : Accept spike and move to next
            n       : Reject spike (remove) and move to next
            right   : Shift spike position right
            left    : Shift spike position left
            tab     : Skip to next spike without decision
            enter   : Finish review

        Returns:
            Updated result with spot_checked set to True.

        Raises:
            UserCancelled: If the user presses Esc.
        """
        self.setup()

        if self.result.n_spikes == 0:
            self._finish()
            return self.result

        while not self._finished:
            blocking_wait(self.fig)
            # All key dispatch happens in _on_key, which calls _finish
            # for Enter; the wait returns when _finish stops the loop.

        self.close()
        if self._cancelled:
            raise UserCancelled("SpotCheckGUI cancelled by user (Esc)")
        return self.result

    def finish(self) -> None:
        """Programmatically signal that the user is done. Idempotent."""
        self._finish()

    def close(self) -> None:
        """Disconnect handlers and close the figure. Idempotent."""
        if self._disconnect_handlers is not None:
            self._disconnect_handlers()
            self._disconnect_handlers = None
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

    def _on_key(self, key: str | None) -> None:
        if key in ("enter", "return"):
            self._finish()
            return
        if key == "escape":
            self._cancelled = True
            self._finish()
            return
        if self._spikes is None or len(self._spikes) == 0:
            return
        self._handle_key(key)

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        # Build final result from accepted spikes.
        if self._spikes is not None and self._accepted is not None and len(self._spikes) > 0:
            self.result.spike_times = np.sort(
                self._spikes[self._accepted]
            ).astype(np.int64)
        self.result.spot_checked = True
        try:
            self.fig.canvas.stop_event_loop()
        except Exception:
            pass
        if self.on_finished is not None:
            self.on_finished(self.result)

    def _center_window(self) -> None:
        """Center the figure window on the screen."""
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager, 'window'):
                window = manager.window
                window.update_idletasks()
                screen_width = window.winfo_screenwidth()
                screen_height = window.winfo_screenheight()
                win_width = window.winfo_width()
                win_height = window.winfo_height()
                x = (screen_width - win_width) // 2
                y = (screen_height - win_height) // 2
                window.geometry(f"+{x}+{y}")
            elif hasattr(manager, 'canvas'):
                canvas = manager.canvas
                if hasattr(canvas, 'parent'):
                    parent = canvas.parent()
                    if parent is not None:
                        parent.move(
                            parent.x() + (parent.width() - self.fig.get_figwidth() * self.fig.dpi) // 2,
                            parent.y() + (parent.height() - self.fig.get_figheight() * self.fig.dpi) // 2
                        )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Prepare internal data structures."""
        params = self.result.params
        voltage = self._recording.voltage
        fs = params.fs

        # Filter the data
        self._filtered = filter_data(
            voltage, fs=fs,
            hp_cutoff=params.hp_cutoff,
            lp_cutoff=params.lp_cutoff,
            diff_order=params.diff_order,
            polarity=params.polarity,
        )

        # Working copy of spike positions
        self._spikes = self.result.spike_times.copy()
        self._accepted = np.ones(len(self._spikes), dtype=bool)
        self._spike_idx = 0

        # Compute context window size from template width
        stw = params.spike_template_width
        half = stw // 2
        self._window = np.arange(-half, half + 1)
        self._context_window_half = half * 4  # 4x the spike width

        # Compute mean spike waveform and 2nd derivative
        if params.spike_template is not None and len(self._spikes) > 0:
            self._compute_mean_waveform()

        # Run template matching to get DTW distances and amplitudes
        self._run_template_matching()

    def _run_template_matching(self) -> None:
        """Run peak finding and template matching to populate scatter data."""
        params = self.result.params

        if params.spike_template is None or len(self._spikes) == 0:
            self._dtw_distances = np.empty(0)
            self._amplitudes = np.empty(0)
            self._candidate_locs = np.array([], dtype=np.int64)
            self._candidate_is_accepted = np.array([], dtype=bool)
            self._spike_to_candidate = {}
            return

        # Find all candidate peaks (same as detection pipeline)
        candidate_locs = PeakFinder.find_spike_locations(
            self._filtered,
            peak_threshold=params.peak_threshold,
            fs=params.fs,
            spike_template_width=params.spike_template_width,
        )

        if len(candidate_locs) == 0:
            self._dtw_distances = np.empty(0)
            self._amplitudes = np.empty(0)
            self._candidate_locs = np.array([], dtype=np.int64)
            self._candidate_is_accepted = np.array([], dtype=bool)
            self._spike_to_candidate = {}
            return

        # Run template matching
        match_result = TemplateMatcher.match(
            candidate_locs,
            params.spike_template,
            self._filtered,
            self._recording.voltage,
            params.spike_template_width,
            params.fs,
        )

        self._candidate_locs = match_result.spike_locs
        self._dtw_distances = match_result.dtw_distances
        self._amplitudes = match_result.amplitudes

        # Build mapping: for each accepted spike, find the closest candidate
        # (mirrors MATLAB spikes_map logic)
        self._candidate_is_accepted = np.zeros(
            len(self._candidate_locs), dtype=bool,
        )
        self._spike_to_candidate = {}

        for i, spike_pos in enumerate(self._spikes):
            if len(self._candidate_locs) > 0:
                diffs = np.abs(
                    self._candidate_locs.astype(np.int64)
                    - int(spike_pos)
                )
                best = int(np.argmin(diffs))
                self._candidate_is_accepted[best] = True
                self._spike_to_candidate[i] = best

    def _compute_mean_waveform(self) -> None:
        """Compute the mean spike waveform and its 2nd derivative."""
        params = self.result.params
        stw = params.spike_template_width
        half = stw // 2
        window = self._window
        smth_w = max(round(params.fs / 2000), 1)

        waveforms = []
        for s in self._spikes:
            if (s + window[0] >= 0
                    and s + window[-1] < len(
                        self._recording.voltage)):
                waveforms.append(
                    self._recording.voltage[s + window]
                )

        if len(waveforms) > 0:
            mean_wf = np.mean(waveforms, axis=0)
            mean_wf = mean_wf - np.min(mean_wf)
            mx = np.max(mean_wf)
            if mx > 0:
                mean_wf = mean_wf / mx
            self._mean_waveform = smooth(
                mean_wf - mean_wf[0], smth_w,
            )
            self._mean_2d = smooth_and_differentiate(
                self._mean_waveform, smth_w,
            )
        else:
            self._mean_waveform = np.zeros(len(window))
            self._mean_2d = np.zeros(len(window))

    def _build_figure(self) -> None:
        """Create the figure with subplots for review."""
        self.fig = plt.figure(figsize=(16, 7))
        self.fig.set_facecolor("white")
        gs = GridSpec(4, 3, figure=self.fig, hspace=0.4, wspace=0.3,
                      height_ratios=[2, 1, 1, 4])

        # Top: unfiltered trace with raster ticks
        self._ax_trace = self.fig.add_subplot(gs[0, :])
        self._ax_trace.set_title("Unfiltered trace with spike ticks")

        # Second row: filtered trace
        self._ax_filt = self.fig.add_subplot(gs[1, :])

        # Third row: current channel (optional)
        self._ax_current = self.fig.add_subplot(gs[2, :])

        # Bottom left: DTW scatter
        self._ax_hist = self.fig.add_subplot(gs[3, 0])
        self._ax_hist.set_xlabel("DTW Distance")
        self._ax_hist.set_ylabel("Amplitude")

        # Bottom center: spike context (unfiltered)
        self._ax_spike = self.fig.add_subplot(gs[3, 1])
        self._ax_spike.set_title("Is this a spike? (y/n) Arrows to adjust")

        # Bottom right: filtered context
        self._ax_squig = self.fig.add_subplot(gs[3, 2])
        self._ax_squig.set_title("Filtered context")

        # Draw the full traces
        voltage = self._recording.voltage
        n = len(voltage)
        t = np.arange(n) / self.result.params.fs

        self._ax_trace.plot(
            t, voltage,
            color=(0.85, 0.325, 0.098), linewidth=0.5,
        )
        if len(self._spikes) > 0:
            y_top = np.max(voltage) + 0.02 * np.ptp(voltage)
            spike_t = self._spikes / self.result.params.fs
            tick_lines = raster_ticks(
                self._ax_trace, spike_t, y_top,
                picker=5,
            )
            if tick_lines:
                self._raster_lines = tick_lines[0]
        self._ax_trace.set_xlim(t[0], t[-1])

        filt_mean = self._filtered - np.mean(self._filtered)
        self._ax_filt.plot(
            t, filt_mean,
            color=(0.0, 0.45, 0.74), linewidth=0.5,
        )
        self._ax_filt.set_xlim(t[0], t[-1])

        if self._recording.current is not None:
            self._ax_current.plot(
                t, self._recording.current, color=(0.74, 0, 0), linewidth=0.5,
            )
            self._ax_current.set_xlim(t[0], t[-1])
        else:
            self._ax_current.set_visible(False)

        # Populate the DTW scatter plot
        self._build_scatter()

    def _build_scatter(self) -> None:
        """Populate the DTW distance vs amplitude scatter plot."""
        ax = self._ax_hist

        if self._dtw_distances is None or len(self._dtw_distances) == 0:
            ax.set_title("No DTW data available")
            return

        params = self.result.params
        dists = self._dtw_distances
        amps = self._amplitudes
        accepted_mask = self._candidate_is_accepted
        rejected_mask = ~accepted_mask

        # Plot rejected dots (yellow) first, then accepted (blue) on top
        if np.any(rejected_mask):
            self._scat_out = ax.plot(
                dists[rejected_mask], amps[rejected_mask], ".",
                color=_COLOR_REJECTED, markersize=10, picker=5,
            )[0]
        if np.any(accepted_mask):
            self._scat_in = ax.plot(
                dists[accepted_mask], amps[accepted_mask], ".",
                color=_COLOR_ACCEPTED, markersize=10, picker=5,
            )[0]

        # Threshold lines (red)
        amp_min = min(np.min(amps), 0) if len(amps) > 0 else 0
        amp_max = np.max(amps) if len(amps) > 0 else 1
        dist_max = np.max(dists) if len(dists) > 0 else 1

        ax.plot(
            [params.distance_threshold, params.distance_threshold],
            [amp_min, amp_max],
            color=_COLOR_THRESHOLD, linewidth=1,
        )
        ax.plot(
            [0, dist_max],
            [params.amplitude_threshold, params.amplitude_threshold],
            color=_COLOR_THRESHOLD, linewidth=1,
        )

        # Current spike marker (red filled circle) - initially invisible
        self._current_dot, = ax.plot(
            [], [], "o", markerfacecolor=_COLOR_CURRENT,
            markeredgecolor="none", markersize=7,
        )

        ax.set_xlim(0, dist_max * 1.05 if dist_max > 0 else 1)
        ax.set_ylim(amp_min - 0.05 * abs(amp_max - amp_min),
                     amp_max + 0.05 * abs(amp_max - amp_min))
        ax.set_title("Click to select spikes, tab to move")

        # Connect pick events for clicking on scatter dots and raster ticks
        self.fig.canvas.mpl_connect("pick_event", self._on_scatter_pick)
        self.fig.canvas.mpl_connect("pick_event", self._on_raster_pick)

    def _on_raster_pick(self, event) -> None:
        """Handle clicking on a raster tick mark to navigate to that spike."""
        if event.mouseevent.inaxes != self._ax_trace:
            return
        if self._raster_lines is None or event.artist is not self._raster_lines:
            return

        xdata = event.mouseevent.xdata
        if xdata is None:
            return

        # Convert x-position (seconds) to sample index
        fs = self.result.params.fs
        sample = round(xdata * fs)

        # Find the closest spike
        if len(self._spikes) == 0:
            return
        diffs = np.abs(self._spikes.astype(np.int64) - int(sample))
        self._spike_idx = int(np.argmin(diffs))
        self._show_current_spike()
        self.fig.canvas.draw_idle()

    def _on_scatter_pick(self, event) -> None:
        """Handle clicking on a scatter dot to navigate to that spike."""
        if event.mouseevent.inaxes != self._ax_hist:
            return

        artist = event.artist
        ind = event.ind
        if len(ind) == 0:
            return
        pick_idx = ind[0]

        # Determine which candidate index was clicked
        accepted_mask = self._candidate_is_accepted
        rejected_mask = ~accepted_mask

        if artist is self._scat_in and self._scat_in is not None:
            # Clicked an accepted dot -- map back to candidate index
            accepted_indices = np.where(accepted_mask)[0]
            if pick_idx < len(accepted_indices):
                cand_idx = accepted_indices[pick_idx]
            else:
                return
        elif artist is self._scat_out and self._scat_out is not None:
            # Clicked a rejected dot
            rejected_indices = np.where(rejected_mask)[0]
            if pick_idx < len(rejected_indices):
                cand_idx = rejected_indices[pick_idx]
            else:
                return
        else:
            return

        # Find the spike index that maps to this candidate, or the closest spike
        target_loc = self._candidate_locs[cand_idx]
        # Check if any spike maps to this candidate
        for spike_i, cand_i in self._spike_to_candidate.items():
            if cand_i == cand_idx:
                self._spike_idx = spike_i
                self._show_current_spike()
                self.fig.canvas.draw_idle()
                return

        # If no spike maps directly, find the closest spike
        if len(self._spikes) > 0:
            diffs = np.abs(
                self._spikes.astype(np.int64) - int(target_loc)
            )
            self._spike_idx = int(np.argmin(diffs))
            self._show_current_spike()
            self.fig.canvas.draw_idle()

    def _update_scatter_dot(self) -> None:
        """Move the current dot marker to highlight the current spike."""
        if self._current_dot is None or self._dtw_distances is None:
            return
        if len(self._dtw_distances) == 0:
            return

        cand_idx = self._spike_to_candidate.get(self._spike_idx)
        if cand_idx is not None and cand_idx < len(self._dtw_distances):
            self._current_dot.set_data(
                [self._dtw_distances[cand_idx]],
                [self._amplitudes[cand_idx]],
            )
        else:
            self._current_dot.set_data([], [])

    def _update_scatter_colors(self) -> None:
        """Rebuild the scatter dot colors after accept/reject changes."""
        if self._dtw_distances is None or len(self._dtw_distances) == 0:
            return

        # Rebuild candidate_is_accepted from current _accepted state
        self._candidate_is_accepted[:] = False
        for i, acc in enumerate(self._accepted):
            if acc:
                cand_idx = self._spike_to_candidate.get(i)
                if cand_idx is not None:
                    self._candidate_is_accepted[cand_idx] = True

        accepted_mask = self._candidate_is_accepted
        rejected_mask = ~accepted_mask
        dists = self._dtw_distances
        amps = self._amplitudes

        if self._scat_in is not None:
            if np.any(accepted_mask):
                self._scat_in.set_data(
                    dists[accepted_mask],
                    amps[accepted_mask],
                )
            else:
                self._scat_in.set_data([], [])
        if self._scat_out is not None:
            if np.any(rejected_mask):
                self._scat_out.set_data(
                    dists[rejected_mask],
                    amps[rejected_mask],
                )
            else:
                self._scat_out.set_data([], [])

    def _show_current_spike(self) -> None:
        """Update the bottom panels to show the current spike."""
        if self._spike_idx < 0 or self._spike_idx >= len(self._spikes):
            return

        params = self.result.params
        fs = params.fs
        spike = self._spikes[self._spike_idx]
        voltage = self._recording.voltage
        n = len(voltage)
        ctx_half = self._context_window_half

        # Context boundaries (clamp to valid range)
        ctx_start = max(0, spike - ctx_half)
        ctx_end = min(n, spike + ctx_half)

        t_ctx = np.arange(ctx_start, ctx_end) / fs

        # Spike context view
        self._ax_spike.cla()
        self._ax_spike.plot(t_ctx, voltage[ctx_start:ctx_end],
                           color=(0.49, 0.18, 0.56), linewidth=0.8)

        # Highlight the spike window
        win_start = max(0, spike + self._window[0])
        win_end = min(n, spike + self._window[-1] + 1)
        t_win = np.arange(win_start, win_end) / fs
        self._ax_spike.plot(t_win, voltage[win_start:win_end],
                           color=(0, 0, 0), linewidth=1.5)

        # Mean waveform overlay
        if hasattr(self, "_mean_waveform"):
            amp = np.mean(np.abs(
                voltage[win_start:win_end] - voltage[spike]
            ))
            scale = max(amp, 0.01)
            mean_scaled = (
                self._mean_waveform * scale + voltage[spike]
            )
            if len(mean_scaled) == len(t_win):
                self._ax_spike.plot(
                    t_win, mean_scaled,
                    color=(0.4, 0.3, 1.0), linewidth=2, alpha=0.7,
                )
            # 2nd derivative overlay
            smth_start = round(fs / 2000)
            smth_end = len(self._mean_2d) - smth_start
            if smth_end > smth_start + 2:
                region_2d = self._mean_2d[smth_start + 1 : smth_end - 1]
                if np.max(np.abs(region_2d)) > 0:
                    region_scaled = (
                        region_2d
                        / np.max(np.abs(region_2d))
                        * scale + voltage[spike]
                    )
                    t_2d = np.arange(
                        win_start + smth_start + 1,
                        win_start + smth_end - 1,
                    ) / fs
                    if len(t_2d) == len(region_scaled):
                        self._ax_spike.plot(
                            t_2d, region_scaled,
                            color=(0, 0.8, 0.4), linewidth=2, alpha=0.7,
                        )

        # Vertical spike marker
        self._ax_spike.axvline(
            spike / fs, color=(1, 0, 0), linewidth=1, linestyle="--",
        )

        is_accepted = self._accepted[self._spike_idx]
        status = "accepted" if is_accepted else "REJECTED"
        self._ax_spike.set_title(
            f"Spike {self._spike_idx + 1}/{len(self._spikes)} [{status}] "
            f"(y=accept, n=reject, arrows=adjust)"
        )

        self._ax_spike.set_xlim(t_ctx[0], t_ctx[-1])

        # Filtered context view
        self._ax_squig.cla()
        self._ax_squig.plot(
            t_ctx, self._filtered[ctx_start:ctx_end],
            color=(0.49, 0.18, 0.56), linewidth=0.8,
        )
        if params.spike_template is not None:
            tmpl = params.spike_template
            tmpl_scaled = tmpl / np.max(np.abs(tmpl)) * np.max(
                np.abs(self._filtered[ctx_start:ctx_end])
            )
            half = len(tmpl) // 2
            # Use uncorrected spike location for template overlay
            uc_spike = spike
            if self._spike_idx < len(self.result.spike_times_uncorrected):
                uc_spike = self.result.spike_times_uncorrected[self._spike_idx]
            tmpl_start = max(0, uc_spike - half)
            tmpl_end = min(n, uc_spike + half + 1)
            t_tmpl = np.arange(tmpl_start, tmpl_end) / fs
            tmpl_len = tmpl_end - tmpl_start
            if tmpl_len == len(tmpl):
                self._ax_squig.plot(
                    t_tmpl, tmpl_scaled,
                    color=(0.85, 0.325, 0.098), linewidth=1,
                )
        self._ax_squig.set_xlim(t_ctx[0], t_ctx[-1])
        self._ax_squig.set_title("Filtered context with template")

        # Update the current dot on the scatter plot
        self._update_scatter_dot()

        self.fig.canvas.draw_idle()

    def _handle_key(self, key: str) -> str | None:
        """Process a keypress and return 'done' to finish, or None."""
        if key == "y":
            self._accepted[self._spike_idx] = True
            self._update_scatter_colors()
            self._advance()
        elif key == "n":
            self._accepted[self._spike_idx] = False
            self._update_scatter_colors()
            self._advance()
        elif key == "right":
            self._spikes[self._spike_idx] += 10
            self._show_current_spike()
        elif key == "shift+right":
            self._spikes[self._spike_idx] += 1
            self._show_current_spike()
        elif key == "left":
            self._spikes[self._spike_idx] -= 10
            self._show_current_spike()
        elif key == "shift+left":
            self._spikes[self._spike_idx] -= 1
            self._show_current_spike()
        elif key == "tab":
            self._advance()
        elif key == "shift+tab":
            self._direction = -1
            self._advance()
            self._direction = 1
        elif key in ("enter", "return"):
            return "done"
        return None

    def _advance(self) -> None:
        """Move to the next (or previous) spike."""
        self._spike_idx += self._direction
        if self._spike_idx < 0:
            self._spike_idx = 0
        elif self._spike_idx >= len(self._spikes):
            self._spike_idx = len(self._spikes) - 1
        self._show_current_spike()
        if self.on_spike_index_changed is not None:
            self.on_spike_index_changed(self._spike_idx)
