"""Qt-based spot-check GUI for spike-by-spike review.

Wraps :class:`SpotCheckGUI`'s multi-panel figure in a QDialog with a
button bar for the most common actions (accept / reject / prev / skip /
finish). All original keyboard shortcuts are preserved via
``keyPressEvent``. The heavy plotting and state-management logic
(``_setup``, ``_run_template_matching``, ``_show_current_spike``,
``_handle_key``, etc.) lives in :class:`SpotCheckGUI`. We avoid multiple
inheritance with PySide6 (its cooperative super chain calls into our
non-cooperative bases with the wrong arg shape) and instead bind those
helpers as instance methods so ``self.method()`` resolves correctly inside
delegated code.
"""

from __future__ import annotations

import types
from copy import deepcopy
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from spikedetect.gui._qt_imports import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, Qt,
)
from spikedetect.gui._widgets import raster_ticks, UserCancelled
from spikedetect.gui.spotcheck_gui import SpotCheckGUI
from spikedetect.models import (
    Recording, SpikeDetectionParams, SpikeDetectionResult,
)


# SpotCheckGUI helpers we need reachable as ``self._method`` from inside
# delegated code (e.g. _build_scatter connects pick events via
# ``self._on_scatter_pick``; _show_current_spike calls
# ``self._update_scatter_dot``; etc.). Any new self-call inside SpotCheckGUI
# helpers needs adding here.
_DELEGATED_METHODS = (
    "_setup",
    "_compute_mean_waveform",
    "_run_template_matching",
    "_build_scatter",
    "_show_current_spike",
    "_handle_key",
    "_advance",
    "_update_scatter_dot",
    "_update_scatter_colors",
    "_on_scatter_pick",
    "_on_raster_pick",
)


class SpotCheckGUIQt(QDialog):
    """Qt dialog for reviewing detected spikes one at a time.

    Args:
        recording: The electrophysiology recording.
        result: Detection result to review (a deep copy is made).

    Attributes:
        result: The (possibly modified) detection result.
        canvas: The embedded ``FigureCanvasQTAgg``.

    Keyboard shortcuts (preserved from SpotCheckGUI):
        y / n        accept / reject current spike, advance
        Tab          skip forward
        Shift+Tab    skip backward
        Right/Left   shift spike +10 / -10 samples
        Shift+ArrowKeys  shift +1 / -1 sample
        Enter        finish review
        Esc          cancel
    """

    def __init__(
        self,
        recording: Recording,
        result: SpikeDetectionResult,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # SpotCheckGUI.__init__ is pure attribute assignment (no matplotlib
        # side effects), so we can call it as a function to set up state.
        SpotCheckGUI.__init__(self, recording, result)
        # Bind SpotCheckGUI's helper methods to this instance so any
        # ``self._foo()`` call inside delegated code resolves to it.
        for name in _DELEGATED_METHODS:
            setattr(
                self, name,
                types.MethodType(getattr(SpotCheckGUI, name), self),
            )

        self._setup_ui()
        self._setup()
        self._build_plot()
        if self.result.n_spikes > 0:
            self._show_current_spike()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Spot-check spikes")
        self.setMinimumSize(1200, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self._fig = Figure(figsize=(16, 7))
        self.canvas = FigureCanvasQTAgg(self._fig)
        self.canvas.setMinimumHeight(550)
        layout.addWidget(self.canvas, stretch=1)

        # SpotCheckGUI delegates draw_idle through self.fig.canvas
        self.fig = self._fig

        # Button bar
        bar = QHBoxLayout()
        self._btn_accept_spike = QPushButton("Accept (y)")
        self._btn_accept_spike.clicked.connect(lambda: self._dispatch("y"))
        self._btn_reject_spike = QPushButton("Reject (n)")
        self._btn_reject_spike.clicked.connect(lambda: self._dispatch("n"))
        self._btn_prev = QPushButton("◄ Prev (Shift+Tab)")
        self._btn_prev.clicked.connect(lambda: self._dispatch("shift+tab"))
        self._btn_skip = QPushButton("Skip ► (Tab)")
        self._btn_skip.clicked.connect(lambda: self._dispatch("tab"))
        bar.addWidget(self._btn_accept_spike)
        bar.addWidget(self._btn_reject_spike)
        bar.addWidget(self._btn_prev)
        bar.addWidget(self._btn_skip)
        bar.addStretch()
        self._btn_cancel = QPushButton("Cancel (Esc)")
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_finish = QPushButton("Finish (Enter)")
        self._btn_finish.setDefault(True)
        self._btn_finish.clicked.connect(self._on_finish)
        bar.addWidget(self._btn_cancel)
        bar.addWidget(self._btn_finish)
        layout.addLayout(bar)

    def _build_plot(self) -> None:
        # Mirror SpotCheckGUI._build_figure with a raw Figure. Constructs
        # the same axes the delegated update methods expect, then plots
        # the static traces and populates the scatter.
        self._fig.set_facecolor("white")
        gs = GridSpec(
            4, 3, figure=self._fig, hspace=0.4, wspace=0.3,
            height_ratios=[2, 1, 1, 4],
        )

        self._ax_trace = self._fig.add_subplot(gs[0, :])
        self._ax_trace.set_title("Unfiltered trace with spike ticks")
        self._ax_filt = self._fig.add_subplot(gs[1, :])
        self._ax_current = self._fig.add_subplot(gs[2, :])
        self._ax_hist = self._fig.add_subplot(gs[3, 0])
        self._ax_hist.set_xlabel("DTW Distance")
        self._ax_hist.set_ylabel("Amplitude")
        self._ax_spike = self._fig.add_subplot(gs[3, 1])
        self._ax_spike.set_title("Is this a spike? (y/n) Arrows to adjust")
        self._ax_squig = self._fig.add_subplot(gs[3, 2])
        self._ax_squig.set_title("Filtered context")

        voltage = self._recording.voltage
        n = len(voltage)
        fs = self.result.params.fs
        t = np.arange(n) / fs

        self._ax_trace.plot(
            t, voltage,
            color=(0.85, 0.325, 0.098), linewidth=0.5,
        )
        if len(self._spikes) > 0:
            y_top = np.max(voltage) + 0.02 * np.ptp(voltage)
            spike_t = self._spikes / fs
            tick_lines = raster_ticks(
                self._ax_trace, spike_t, y_top, picker=5,
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
                t, self._recording.current,
                color=(0.74, 0, 0), linewidth=0.5,
            )
            self._ax_current.set_xlim(t[0], t[-1])
        else:
            self._ax_current.set_visible(False)

        # Build the scatter (pick connections happen inside)
        self._build_scatter()

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_finish()
        elif key == Qt.Key.Key_Escape:
            self._cancelled = True
            self.reject()
        elif key == Qt.Key.Key_Y:
            self._dispatch("y")
        elif key == Qt.Key.Key_N:
            self._dispatch("n")
        elif key == Qt.Key.Key_Right:
            self._dispatch("shift+right" if shift else "right")
        elif key == Qt.Key.Key_Left:
            self._dispatch("shift+left" if shift else "left")
        elif key == Qt.Key.Key_Tab:
            self._dispatch("shift+tab" if shift else "tab")
        elif key == Qt.Key.Key_Backtab:
            # Some platforms deliver Shift+Tab as Key_Backtab
            self._dispatch("shift+tab")
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_centered", False):
            return
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            frame = self.frameGeometry()
            frame.moveCenter(geom.center())
            self.move(frame.topLeft())
        self._centered = True

    def _dispatch(self, key: str) -> None:
        """Run a keyboard-equivalent action and notify embedders if needed."""
        if self._spikes is None or len(self._spikes) == 0:
            return
        prev_idx = self._spike_idx
        self._handle_key(key)
        if (self.on_spike_index_changed is not None
                and self._spike_idx != prev_idx):
            self.on_spike_index_changed(self._spike_idx)

    def _on_finish(self) -> None:
        # Build final result from accepted mask (same logic as
        # SpotCheckGUI._finish, minus the matplotlib stop_event_loop).
        if (self._spikes is not None and self._accepted is not None
                and len(self._spikes) > 0):
            self.result.spike_times = np.sort(
                self._spikes[self._accepted]
            ).astype(np.int64)
        self.result.spot_checked = True
        if self.on_finished is not None:
            self.on_finished(self.result)
        self.accept()

    def setup(self):
        """Return the dialog itself (for embedding API parity)."""
        return self

    def run(self) -> SpikeDetectionResult:
        """Display modally and return the (possibly modified) result.

        Raises:
            UserCancelled: If the user presses Esc.
        """
        if self.result.n_spikes == 0:
            self._on_finish()
            return self.result
        self.exec()
        if self._cancelled:
            raise UserCancelled("SpotCheckGUIQt cancelled by user (Esc)")
        return self.result

    def finish(self) -> None:
        """Programmatically accept the dialog. Idempotent."""
        self._on_finish()
