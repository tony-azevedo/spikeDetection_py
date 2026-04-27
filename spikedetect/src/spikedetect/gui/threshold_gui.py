"""Interactive threshold tuning GUI for spike classification.

Ports the MATLAB ``spikeThresholdUpdateGUI.m`` function. Shows a scatter
plot of DTW distance vs. amplitude and allows the user to click to move
threshold lines. Waveform panels show good/weird/bad categories.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from spikedetect.gui._widgets import (
    raster_ticks, blocking_wait, install_finish_handlers,
)
from spikedetect.models import SpikeDetectionParams
from spikedetect.pipeline.classify import classify_spikes
from spikedetect.pipeline.template import TemplateMatchResult
from spikedetect.utils import smooth, smooth_and_differentiate


class ThresholdGUI:
    """Interactive GUI for adjusting DTW distance and amplitude thresholds.

    Args:
        match_result: Results from template matching (distances,
            amplitudes, waveforms).
        params: Current detection parameters.

    Attributes:
        params: Parameters updated by user interaction.
        fig: The GUI figure.
    """

    def __init__(
        self, match_result: TemplateMatchResult, params: SpikeDetectionParams
    ) -> None:
        self._match = match_result
        self.params = deepcopy(params)
        self.fig = None
        self._active_threshold = "distance"  # or "amplitude"
        self._finished = False
        self._disconnect_handlers: Callable[[], None] | None = None
        # Callbacks for non-blocking (e.g. Qt) embedding.
        self.on_finished: Callable[[SpikeDetectionParams], None] | None = None
        self.on_thresholds_changed: Callable[[float, float], None] | None = None

    def setup(self):
        """Build the figure without starting an event loop. See
        :meth:`FilterGUI.setup` for details.

        Returns:
            The figure's canvas.
        """
        if self.fig is not None:
            return self.fig.canvas
        self._build_figure()
        self._update_panels()
        self._center_window()
        self._disconnect_handlers = install_finish_handlers(
            self.fig, self._on_key, self._finish,
        )
        return self.fig.canvas

    def run(self) -> SpikeDetectionParams:
        """Display the GUI and block until the user finishes.

        The user clicks on the scatter plot to reposition the
        active threshold line. Press 'b' to toggle between
        distance and amplitude thresholds. Press Enter or close
        the figure to accept.

        Returns:
            Updated parameters with new threshold values.
        """
        self.setup()

        while True:
            key = blocking_wait(self.fig)
            if key is None or key in ("enter", "return"):
                break
            # 'b' toggle is dispatched in _on_key so it works in both
            # blocking and non-blocking modes.

        self.close()
        return self.params

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
        elif key == "b":
            self._toggle_active()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self.fig.canvas.stop_event_loop()
        except Exception:
            pass
        if self.on_finished is not None:
            self.on_finished(self.params)

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

    def _build_figure(self) -> None:
        """Create the multi-panel figure layout."""
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.set_facecolor("white")
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3)

        # Top-left: scatter plot of distance vs amplitude
        self._ax_scatter = self.fig.add_subplot(gs[0:2, 0])
        self._ax_scatter.set_xlabel("DTW Distance")
        self._ax_scatter.set_ylabel("Amplitude")
        self._ax_scatter.set_title(
            "Click to move distance threshold (press 'b' to toggle)"
        )

        # Top-center: good/weird filtered waveforms
        self._ax_good_filt = self.fig.add_subplot(gs[0, 1])
        self._ax_good_filt.set_title("Good + Weird (filtered)")

        # Top-right: bad/weirdbad filtered waveforms
        self._ax_bad_filt = self.fig.add_subplot(gs[0, 2])
        self._ax_bad_filt.set_title("Bad + Weirdbad (filtered)")

        # Middle-center: good/weird unfiltered waveforms (click to set template)
        self._ax_good_uf = self.fig.add_subplot(gs[1, 1])
        self._ax_good_uf.set_title("Good + Weird (unfiltered)")

        # Middle-right: bad/weirdbad unfiltered
        self._ax_bad_uf = self.fig.add_subplot(gs[1, 2])
        self._ax_bad_uf.set_title("Bad + Weirdbad (unfiltered)")

        # Bottom: mean spike waveform with 2nd derivative
        self._ax_mean = self.fig.add_subplot(gs[2, :])
        self._ax_mean.set_title("Click to use mean waveform as new template")

        # Connect click handlers
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _toggle_active(self) -> None:
        """Toggle between distance and amplitude threshold editing."""
        if self._active_threshold == "distance":
            self._active_threshold = "amplitude"
            self._ax_scatter.set_title(
                "Click to move amplitude threshold (press 'b' to toggle)"
            )
        else:
            self._active_threshold = "distance"
            self._ax_scatter.set_title(
                "Click to move distance threshold (press 'b' to toggle)"
            )
        self._update_panels()

    def _on_click(self, event) -> None:
        """Handle mouse click on scatter plot or mean waveform panel."""
        if event.inaxes == self._ax_scatter and event.xdata is not None:
            if self._active_threshold == "distance":
                self.params.distance_threshold = float(event.xdata)
            else:
                self.params.amplitude_threshold = float(event.ydata)
            self._update_panels()
            if self.on_thresholds_changed is not None:
                self.on_thresholds_changed(
                    self.params.distance_threshold,
                    self.params.amplitude_threshold,
                )
        elif event.inaxes == self._ax_mean:
            self._update_template_from_mean()

    def _update_template_from_mean(self) -> None:
        """Set the spike template to the mean of accepted waveforms."""
        m = self._match
        suspect = (m.dtw_distances < self.params.distance_threshold) & (
            m.amplitudes > self.params.amplitude_threshold
        )
        if np.sum(suspect) == 0:
            return

        good_nf = m.norm_filtered_candidates[:, suspect]
        mean_template = np.mean(good_nf, axis=1)
        self.params.spike_template = mean_template
        self._ax_mean.set_title("Template updated from mean waveform")
        self.fig.canvas.draw_idle()

    def _update_panels(self) -> None:
        """Redraw all panels with current thresholds."""
        m = self._match
        dists = m.dtw_distances
        amps = m.amplitudes
        dt = self.params.distance_threshold
        at = self.params.amplitude_threshold

        good, weird, weirdbad, bad = classify_spikes(dists, amps, dt, at)
        suspect = (dists < dt) & (amps > at)

        # --- Scatter plot ---
        self._ax_scatter.cla()
        if np.any(~suspect):
            self._ax_scatter.plot(
                dists[~suspect], amps[~suspect], ".",
                color=(0.929, 0.694, 0.125), markersize=8,
            )
        if np.any(suspect):
            self._ax_scatter.plot(
                dists[suspect], amps[suspect], ".",
                color=(0.0, 0.45, 0.74), markersize=8,
            )

        # Threshold lines
        ylim_lo = min(np.min(amps), at) - 0.1 * max(abs(np.ptp(amps)), 0.1)
        ylim_hi = max(np.max(amps), at) + 0.1 * max(abs(np.ptp(amps)), 0.1)
        xlim_hi = max(np.max(dists), dt) * 1.1

        dist_color = (
            (0, 1, 1) if self._active_threshold == "distance"
            else (1, 0, 0)
        )
        amp_color = (
            (0, 1, 1) if self._active_threshold == "amplitude"
            else (1, 0, 0)
        )

        self._ax_scatter.axvline(dt, color=dist_color, linewidth=1.5)
        self._ax_scatter.axhline(at, color=amp_color, linewidth=1.5)
        self._ax_scatter.set_xlim(0, xlim_hi)
        self._ax_scatter.set_ylim(ylim_lo, ylim_hi)
        self._ax_scatter.set_xlabel("DTW Distance")
        self._ax_scatter.set_ylabel("Amplitude")

        if self._active_threshold == "distance":
            self._ax_scatter.set_title(
                "Click to move distance threshold (press 'b' to toggle)"
            )
        else:
            self._ax_scatter.set_title(
                "Click to move amplitude threshold (press 'b' to toggle)"
            )

        # --- Waveform panels ---
        window = m.window
        spike_window = m.spike_window

        # Good/weird filtered
        self._ax_good_filt.cla()
        self._ax_good_filt.set_title(
            f"Good + Weird filtered ({np.sum(suspect)} spikes)"
        )
        if np.any(good) and m.norm_filtered_candidates.shape[1] > 0:
            self._ax_good_filt.plot(
                window, m.norm_filtered_candidates[:, good],
                color=(0.8, 0.8, 0.8), linewidth=0.5,
            )
        if np.any(weird) and m.norm_filtered_candidates.shape[1] > 0:
            self._ax_good_filt.plot(
                window, m.norm_filtered_candidates[:, weird],
                color=(0, 0, 0), linewidth=0.5,
            )
        if np.any(good) and m.norm_filtered_candidates.shape[1] > 0:
            mean_nf = np.mean(m.norm_filtered_candidates[:, good], axis=1)
            self._ax_good_filt.plot(
                window, mean_nf, color=(0.3, 0.6, 1.0), linewidth=1.5,
            )

        # Bad/weirdbad filtered
        self._ax_bad_filt.cla()
        self._ax_bad_filt.set_title("Bad + Weirdbad (filtered)")
        if np.any(bad) and m.norm_filtered_candidates.shape[1] > 0:
            self._ax_bad_filt.plot(
                window, m.norm_filtered_candidates[:, bad],
                color=(1.0, 0.7, 0.7), linewidth=0.5,
            )
        if np.any(weirdbad) and m.norm_filtered_candidates.shape[1] > 0:
            self._ax_bad_filt.plot(
                window, m.norm_filtered_candidates[:, weirdbad],
                color=(0.7, 0, 0), linewidth=0.5,
            )

        # Good/weird unfiltered
        self._ax_good_uf.cla()
        self._ax_good_uf.set_title("Good + Weird (unfiltered)")
        if np.any(good) and m.unfiltered_candidates.shape[1] > 0:
            self._ax_good_uf.plot(
                spike_window, m.unfiltered_candidates[:, good],
                color=(0.8, 0.8, 0.8), linewidth=0.5,
            )
        if np.any(weird) and m.unfiltered_candidates.shape[1] > 0:
            self._ax_good_uf.plot(
                spike_window, m.unfiltered_candidates[:, weird],
                color=(0, 0, 0), linewidth=0.5,
            )
        # Mean spike line (blue)
        if np.any(suspect) and m.unfiltered_candidates.shape[1] > 0:
            mean_uf = np.mean(m.unfiltered_candidates[:, suspect], axis=1)
            self._ax_good_uf.plot(
                spike_window, mean_uf,
                color=(0, 0.3, 1.0), linewidth=2, label="mean",
            )
            # 2nd derivative overlay
            smth_w = max(round(self.params.fs / 2000), 1)
            sw = smooth(mean_uf - mean_uf[0], smth_w)
            sw_2d = smooth_and_differentiate(sw, smth_w)
            smth_start = round(self.params.fs / 2000)
            smth_end = len(sw_2d) - smth_start
            if smth_end > smth_start + 2:
                region = sw_2d[smth_start + 1 : smth_end - 1]
                if np.max(region) > 0:
                    scaled = region / np.max(region) * np.max(mean_uf)
                    x_region = spike_window[smth_start + 1 : smth_end - 1]
                    self._ax_good_uf.plot(
                        x_region, scaled,
                        color=(0, 0.8, 0.4), linewidth=2,
                    )

        # Bad/weirdbad unfiltered
        self._ax_bad_uf.cla()
        self._ax_bad_uf.set_title("Bad + Weirdbad (unfiltered)")
        if np.any(bad) and m.unfiltered_candidates.shape[1] > 0:
            self._ax_bad_uf.plot(
                spike_window, m.unfiltered_candidates[:, bad],
                color=(1.0, 0.7, 0.7), linewidth=0.5,
            )
        if np.any(weirdbad) and m.unfiltered_candidates.shape[1] > 0:
            self._ax_bad_uf.plot(
                spike_window, m.unfiltered_candidates[:, weirdbad],
                color=(0.7, 0, 0), linewidth=0.5,
            )

        # --- Mean spike panel (bottom) ---
        self._ax_mean.cla()
        self._ax_mean.set_title(
            "Mean accepted spike (click to use as template)"
        )
        if np.any(suspect) and m.unfiltered_candidates.shape[1] > 0:
            mean_uf = np.mean(m.unfiltered_candidates[:, suspect], axis=1)
            self._ax_mean.plot(
                spike_window, mean_uf,
                color=(0, 0.3, 1.0), linewidth=2,
            )

        self.fig.canvas.draw_idle()
