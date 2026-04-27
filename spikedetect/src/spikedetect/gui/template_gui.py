"""Interactive spike template selection GUI.

Ports the ``getSeedTemplate`` nested function from MATLAB
``spikeDetection.m``. The user clicks on peaks in the filtered data to
select seed spikes. When Enter is pressed, the selected waveforms are
cross-correlation aligned and averaged to produce the spike template.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from spikedetect.gui._widgets import blocking_wait, install_finish_handlers
from spikedetect.models import SpikeDetectionParams
from spikedetect.pipeline.peaks import find_spike_locations


class TemplateSelectionGUI:
    """Interactive GUI for selecting seed spikes and building a template.

    Args:
        filtered_data: 1-D bandpass-filtered voltage trace.
        params: Current detection parameters (must have ``fs``
            and ``spike_template_width`` set).

    Attributes:
        params: Detection parameters (read-only reference).
        fig: The GUI figure.
    """

    def __init__(
        self, filtered_data: np.ndarray, params: SpikeDetectionParams
    ) -> None:
        self._filtered = np.asarray(filtered_data, dtype=np.float64).ravel()
        self.params = deepcopy(params)
        self.fig = None
        self._selected_indices: list[int] = []
        self._finished = False
        self._template: np.ndarray | None = None
        self._disconnect_handlers: Callable[[], None] | None = None
        # Callbacks for non-blocking (e.g. Qt) embedding.
        self.on_finished: Callable[[np.ndarray | None], None] | None = None
        self.on_selection_changed: Callable[[int], None] | None = None

    def setup(self):
        """Build the figure without starting an event loop. See
        :meth:`FilterGUI.setup` for details.

        Returns:
            The figure's canvas.
        """
        if self.fig is not None:
            return self.fig.canvas
        self._build_figure()
        self._center_window()
        self._disconnect_handlers = install_finish_handlers(
            self.fig, self._on_key, self._finish,
        )
        return self.fig.canvas

    def run(self) -> np.ndarray | None:
        """Display the GUI and block until the user presses Enter.

        Returns:
            The averaged spike template waveform, or None if no
            spikes were selected.
        """
        self.setup()

        while True:
            key = blocking_wait(self.fig)
            if key is None or key in ("enter", "return"):
                break

        # Build template synchronously for blocking callers; close()
        # will skip the rebuild because we set self._template here.
        self._template = self._build_template()
        self.close()
        return self._template

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

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._template is None:
            self._template = self._build_template()
        try:
            self.fig.canvas.stop_event_loop()
        except Exception:
            pass
        if self.on_finished is not None:
            self.on_finished(self._template)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        """Create the figure with filtered data, peaks, and click handler."""
        stw = self.params.spike_template_width
        locs = find_spike_locations(
            self._filtered,
            peak_threshold=self.params.peak_threshold,
            fs=self.params.fs,
            spike_template_width=stw,
        )
        self._peak_locs = locs

        self.fig, (self._ax_main, self._ax_squig) = plt.subplots(
            2, 1, figsize=(12, 7),
            gridspec_kw={"height_ratios": [3, 2], "hspace": 0.25},
        )
        self.fig.set_facecolor("white")

        n = len(self._filtered)
        self._ax_main.plot(np.arange(n), self._filtered, "k", linewidth=0.5)
        if len(locs) > 0:
            self._ax_main.plot(
                locs, self._filtered[locs], "ro", markersize=4, picker=5,
            )
        self._ax_main.set_xlim(0, n)
        self._ax_main.set_title(
            "Click peaks to select seed spikes (shift+click for multiple), "
            "then press Enter"
        )

        self._ax_squig.set_title("Selected waveforms")

        # Connect click handler
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

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

    def _on_pick(self, event) -> None:
        """Handle click on a peak marker."""
        if event.artist is None:
            return

        ind = event.ind
        if ind is None or len(ind) == 0:
            return

        # Find nearest peak location
        x_click = event.mouseevent.xdata
        if x_click is None:
            return

        locs = self._peak_locs
        distances = np.abs(locs[ind] - x_click)
        best = ind[np.argmin(distances)]
        loc = int(locs[best])

        if loc in self._selected_indices:
            return
        self._selected_indices.append(loc)

        # Mark selected peak in green
        self._ax_main.plot(loc, self._filtered[loc], "go", markersize=8)

        # Draw the waveform snippet
        stw = self.params.spike_template_width
        half = stw // 2
        window = np.arange(-half, half + 1)
        if loc - half >= 0 and loc + half < len(self._filtered):
            snippet = self._filtered[loc + window]
            self._ax_squig.plot(window, snippet, alpha=0.6)
            self._ax_squig.set_title(
                f"Selected waveforms ({len(self._selected_indices)})"
            )

        self.fig.canvas.draw_idle()
        if self.on_selection_changed is not None:
            self.on_selection_changed(len(self._selected_indices))

    def _build_template(self) -> np.ndarray | None:
        """Build averaged template from selected spike waveforms.

        Uses cross-correlation alignment when multiple spikes are
        selected, matching the MATLAB ``getSeedTemplate`` logic.
        """
        if len(self._selected_indices) == 0:
            return None

        stw = self.params.spike_template_width
        half = stw // 2
        wide_window = np.arange(-stw, stw + 1)
        narrow_half = half

        # Extract wide waveforms for alignment
        seeds = []
        for loc in self._selected_indices:
            if (loc + wide_window[0] >= 0
                    and loc + wide_window[-1] < len(self._filtered)):
                seeds.append(self._filtered[loc + wide_window])

        if len(seeds) == 0:
            return None

        seeds = np.array(seeds)  # shape (n_seeds, 2*stw+1)

        if seeds.shape[0] > 1:
            # Align via cross-correlation (matching MATLAB)
            aligned = seeds.copy()
            ref = seeds[0]
            for r in range(1, seeds.shape[0]):
                corr = np.correlate(ref, seeds[r], mode="full")
                lags = np.arange(-(len(ref) - 1), len(ref))
                best_lag = lags[np.argmax(corr)]
                if best_lag > 0:
                    aligned[r, best_lag:] = seeds[r, : len(seeds[r]) - best_lag]
                    aligned[r, :best_lag] = seeds[r, 0]
                elif best_lag < 0:
                    end = len(seeds[r]) + best_lag
                    aligned[r, :end] = seeds[r, -best_lag:]
                    aligned[r, end:] = seeds[r, -1]
            avg = np.mean(aligned, axis=0)
        else:
            avg = seeds[0]

        # Extract the central spike_template_width-sized window centered on
        # the peak within the middle region
        center_region = avg[stw + np.arange(-narrow_half, narrow_half + 1)]
        peak_offset = np.argmax(center_region)
        peak_in_avg = stw - narrow_half + peak_offset

        template_window = np.arange(
            peak_in_avg - narrow_half, peak_in_avg + narrow_half + 1
        )
        # Clamp to valid range
        template_window = template_window[
            (template_window >= 0) & (template_window < len(avg))
        ]
        template = avg[template_window]

        return template
