"""Interactive filter parameter tuning GUI.

Ports the MATLAB ``filter_sliderGUI.m`` function. Provides sliders for
high-pass cutoff, low-pass cutoff, peak threshold, derivative order, and
a polarity toggle button. Filtered data and detected peaks update live.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox

from spikedetect.gui._widgets import (
    raster_ticks, blocking_wait, install_finish_handlers, UserCancelled,
)
from spikedetect.models import SpikeDetectionParams
from spikedetect.pipeline.filtering import filter_data
from spikedetect.pipeline.peaks import find_spike_locations


class FilterGUI:
    """Interactive GUI for tuning filter and peak-detection parameters.

    Args:
        unfiltered_data: Raw 1-D voltage trace.
        params: Initial detection parameters.

    Attributes:
        params: Current parameters (updated by sliders).
        fig: The GUI figure.
    """

    def __init__(
        self,
        unfiltered_data: np.ndarray,
        params: SpikeDetectionParams,
    ) -> None:
        self._unfiltered = np.asarray(unfiltered_data, dtype=np.float64).ravel()
        self.params = deepcopy(params)
        self.fig = None
        self._filtered = None
        self._locs = None
        self._finished = False
        self._cancelled = False
        self._disconnect_handlers: Callable[[], None] | None = None
        # User-supplied callbacks for non-blocking (e.g. Qt) embedding.
        self.on_finished: Callable[[SpikeDetectionParams], None] | None = None
        self.on_params_changed: Callable[[SpikeDetectionParams], None] | None = None

    def setup(self):
        """Build the figure and install handlers without starting an event loop.

        Use this when embedding the GUI in a host application that
        drives its own event loop (e.g. a PyQt window). The host
        should set ``MPLBACKEND=QtAgg`` (or call
        ``matplotlib.use("QtAgg")``) before importing matplotlib so
        that the returned canvas is a ``FigureCanvasQTAgg`` ready
        to embed in a Qt layout.

        Set ``self.on_finished`` (and optionally ``self.on_params_changed``)
        before or after this call to receive the result.

        Returns:
            The figure's canvas (e.g. ``FigureCanvasQTAgg``).
        """
        if self.fig is not None:
            return self.fig.canvas
        self._apply_filter()
        self._build_figure()
        self._center_window()
        self._update_plots()
        self._disconnect_handlers = install_finish_handlers(
            self.fig, self._on_key, self._finish,
        )
        return self.fig.canvas

    def run(self) -> SpikeDetectionParams:
        """Display the GUI and block until the user presses Enter.

        Returns:
            Updated parameters reflecting the user's slider
            choices.

        Raises:
            UserCancelled: If the user presses Esc.
        """
        self.setup()

        # Block until keypress
        while True:
            key = blocking_wait(self.fig)
            if key is None or key in ("enter", "return", "escape"):
                break

        self.close()
        if self._cancelled:
            raise UserCancelled("FilterGUI cancelled by user (Esc)")
        return self.params

    def finish(self) -> None:
        """Programmatically signal that the user is done.

        Useful from a host's "Finish" button. Idempotent.
        """
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
        elif key == "escape":
            self._cancelled = True
            self._finish()

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_filter(self) -> None:
        """Re-filter the data with current params."""
        self._filtered = filter_data(
            self._unfiltered,
            fs=self.params.fs,
            hp_cutoff=self.params.hp_cutoff,
            lp_cutoff=self.params.lp_cutoff,
            diff_order=self.params.diff_order,
            polarity=self.params.polarity,
        )
        self._locs = find_spike_locations(
            self._filtered,
            peak_threshold=self.params.peak_threshold,
            fs=self.params.fs,
            spike_template_width=self.params.spike_template_width,
        )

    def _build_figure(self) -> None:
        """Create the Matplotlib figure with axes, sliders, and buttons."""
        self.fig, (self._ax_unfilt, self._ax_filt) = plt.subplots(
            2, 1, figsize=(12, 9),
            gridspec_kw={"height_ratios": [1, 6], "hspace": 0.15},
        )
        self.fig.subplots_adjust(bottom=0.28, left=0.08, right=0.95)
        self.fig.set_facecolor("white")

        # Unfiltered trace (top)
        self._ax_unfilt.set_title(
            "Adjust filter parameters, then press Enter to accept"
        )

        # Slider axes
        ax_hp = self.fig.add_axes([0.15, 0.02, 0.25, 0.03])
        ax_lp = self.fig.add_axes([0.15, 0.07, 0.25, 0.03])
        ax_thresh = self.fig.add_axes([0.55, 0.02, 0.30, 0.03])

        # HP slider (0.5 -- 1000 Hz)
        self._sl_hp = Slider(
            ax_hp, "HP (Hz)", 0.5, 1000.0,
            valinit=self.params.hp_cutoff, valstep=0.5,
        )
        # LP slider (0.11 -- 1000 Hz)
        self._sl_lp = Slider(
            ax_lp, "LP (Hz)", 0.11, 1000.0,
            valinit=self.params.lp_cutoff, valstep=0.5,
        )
        # Peak threshold on log10 scale
        log_thresh = np.log10(max(self.params.peak_threshold, 1e-10))
        self._sl_thresh = Slider(
            ax_thresh, "Peak thresh (log10)", -10, -1,
            valinit=log_thresh,
        )

        # Diff radio buttons
        ax_diff = self.fig.add_axes([0.02, 0.12, 0.06, 0.10])
        self._radio_diff = RadioButtons(
            ax_diff, ("0", "1", "2"),
            active=self.params.diff_order,
        )
        ax_diff.set_title("Diff", fontsize=9)

        # Polarity toggle button
        ax_pol = self.fig.add_axes([0.02, 0.02, 0.06, 0.04])
        pol_label = f"Pol: {self.params.polarity:+d}"
        self._btn_pol = Button(ax_pol, pol_label)

        # Peak threshold precision controls (arrow buttons + text box)
        ax_arrow_left = self.fig.add_axes([0.88, 0.02, 0.03, 0.03])
        self._btn_arrow_left = Button(ax_arrow_left, u"\u25C4")  # left arrow
        ax_arrow_right = self.fig.add_axes([0.93, 0.02, 0.03, 0.03])
        self._btn_arrow_right = Button(ax_arrow_right, u"\u25BA")  # right arrow
        ax_text = self.fig.add_axes([0.88, 0.07, 0.08, 0.03])
        self._txt_thresh = TextBox(ax_text, "Value:", initial="")

        # Connect callbacks
        self._sl_hp.on_changed(self._on_slider_change)
        self._sl_lp.on_changed(self._on_slider_change)
        self._sl_thresh.on_changed(self._on_slider_change)
        self._radio_diff.on_clicked(self._on_diff_change)
        self._btn_pol.on_clicked(self._on_polarity_toggle)
        self._btn_arrow_left.on_clicked(self._on_arrow_left)
        self._btn_arrow_right.on_clicked(self._on_arrow_right)
        self._txt_thresh.on_submit(self._on_text_submit)

    def _on_slider_change(self, _val) -> None:
        """Callback for any slider change."""
        self.params.hp_cutoff = self._sl_hp.val
        self.params.lp_cutoff = self._sl_lp.val
        self.params.peak_threshold = 10 ** self._sl_thresh.val
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_diff_change(self, label: str) -> None:
        """Callback for diff order radio button."""
        self.params.diff_order = int(label)
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_polarity_toggle(self, _event) -> None:
        """Callback for polarity toggle button."""
        self.params.polarity *= -1
        self._btn_pol.label.set_text(f"Pol: {self.params.polarity:+d}")
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_arrow_left(self, _event) -> None:
        """Decrease peak threshold by 10% when left arrow is clicked."""
        current = self.params.peak_threshold
        new_val = current * 0.9  # decrease by 10%
        self.params.peak_threshold = new_val
        # Update slider to match (convert to log10 scale)
        log_thresh = np.log10(max(new_val, 1e-10))
        self._sl_thresh.set_val(log_thresh)
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_arrow_right(self, _event) -> None:
        """Increase peak threshold by 10% when right arrow is clicked."""
        current = self.params.peak_threshold
        new_val = current * 1.1  # increase by 10%
        self.params.peak_threshold = new_val
        # Update slider to match (convert to log10 scale)
        log_thresh = np.log10(max(new_val, 1e-10))
        self._sl_thresh.set_val(log_thresh)
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_text_submit(self, text: str) -> None:
        """Apply threshold value from text box when Enter is pressed."""
        try:
            val = float(text)
            if val > 0:
                self.params.peak_threshold = val
                # Update slider to match (convert to log10 scale)
                log_thresh = np.log10(max(val, 1e-10))
                self._sl_thresh.set_val(log_thresh)
                self._apply_filter()
                self._update_plots()
                self._notify_params_changed()
        except ValueError:
            pass  # Ignore invalid input

    def _center_window(self) -> None:
        """Center the figure window on the screen."""
        try:
            # Get the backend's window
            manager = plt.get_current_fig_manager()
            # Try different backends
            if hasattr(manager, 'window'):
                # TkAgg backend
                window = manager.window
                window.update_idletasks()
                # Get screen dimensions
                screen_width = window.winfo_screenwidth()
                screen_height = window.winfo_screenheight()
                # Get window dimensions
                win_width = window.winfo_width()
                win_height = window.winfo_height()
                # Calculate center position
                x = (screen_width - win_width) // 2
                y = (screen_height - win_height) // 2
                window.geometry(f"+{x}+{y}")
            elif hasattr(manager, 'canvas'):
                # Try Qt backends
                canvas = manager.canvas
                if hasattr(canvas, 'parent'):
                    parent = canvas.parent()
                    if parent is not None:
                        parent.move(
                            parent.x() + (parent.width() - self.fig.get_figwidth() * self.fig.dpi) // 2,
                            parent.y() + (parent.height() - self.fig.get_figheight() * self.fig.dpi) // 2
                        )
        except Exception:
            # Silently fail if window centering is not supported
            pass

    def _notify_params_changed(self) -> None:
        if self.on_params_changed is not None:
            self.on_params_changed(self.params)

    def _update_plots(self) -> None:
        """Redraw the filtered data and peak markers."""
        n = len(self._unfiltered)

        # Top axis: unfiltered with raster ticks
        self._ax_unfilt.cla()
        self._ax_unfilt.plot(
            np.arange(n), self._unfiltered,
            color=(0.85, 0.325, 0.098),
        )
        if len(self._locs) > 0:
            y_top = np.max(self._unfiltered) + 0.02 * np.ptp(self._unfiltered)
            raster_ticks(self._ax_unfilt, self._locs, y_top)
        self._ax_unfilt.set_xlim(0, n)
        self._ax_unfilt.set_title(
            "Adjust filter parameters, then press Enter to accept"
        )

        # Bottom axis: filtered with peaks
        self._ax_filt.cla()
        filt = self._filtered
        self._ax_filt.plot(np.arange(n), filt, "k", linewidth=0.5)
        if len(self._locs) > 0:
            self._ax_filt.plot(self._locs, filt[self._locs], "ro", markersize=4)
        # Threshold line
        self._ax_filt.axhline(
            self.params.peak_threshold, color=(0.8, 0.8, 0.8),
            linestyle="--", linewidth=1,
        )
        self._ax_filt.set_xlim(0, n)
        if len(filt) > 0:
            self._ax_filt.set_ylim(np.min(filt), np.max(filt))

        self.fig.canvas.draw_idle()
