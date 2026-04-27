"""Qt-based threshold tuning GUI.

Wraps :class:`ThresholdGUI`'s multi-panel scatter+waveform plot in a
QDialog with native action buttons. Keyboard 'b' still toggles the active
threshold; the same logic is reachable via a button. The panel-update
algorithm lives in ``ThresholdGUI._update_panels`` and is reused via class
delegation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from spikedetect.gui._qt_imports import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, Qt,
)
from spikedetect.gui.threshold_gui import ThresholdGUI
from spikedetect.models import SpikeDetectionParams
from spikedetect.pipeline.template import TemplateMatchResult


class ThresholdGUIQt(QDialog):
    """Qt dialog for adjusting DTW distance and amplitude thresholds.

    Args:
        match_result: Output of template matching (distances, amplitudes,
            normalized waveforms).
        params: Current detection parameters.

    Attributes:
        params: Parameters (deep-copied; updated by user interaction).
        canvas: The embedded ``FigureCanvasQTAgg``.
    """

    def __init__(
        self,
        match_result: TemplateMatchResult,
        params: SpikeDetectionParams,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._match = match_result
        self.params = deepcopy(params)
        self._active_threshold = "distance"

        self.on_finished: Callable[[SpikeDetectionParams], None] | None = None
        self.on_thresholds_changed: Callable[[float, float], None] | None = None

        self._setup_ui()
        self._build_plot()
        # Initial render of all panels
        ThresholdGUI._update_panels(self)

    def _setup_ui(self) -> None:
        self.setWindowTitle("Tune thresholds")
        self.setMinimumSize(1100, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self._fig = Figure(figsize=(14, 9))
        self.canvas = FigureCanvasQTAgg(self._fig)
        self.canvas.setMinimumHeight(550)
        layout.addWidget(self.canvas, stretch=1)

        bar = QHBoxLayout()
        self._btn_toggle = QPushButton("Toggle active threshold (b)")
        self._btn_toggle.clicked.connect(self._on_toggle)
        self._btn_use_mean = QPushButton("Use mean as template")
        self._btn_use_mean.clicked.connect(self._on_use_mean_template)
        bar.addWidget(self._btn_toggle)
        bar.addWidget(self._btn_use_mean)
        bar.addStretch()
        self._btn_cancel = QPushButton("Cancel (Esc)")
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_accept = QPushButton("Accept (Enter)")
        self._btn_accept.setDefault(True)
        self._btn_accept.clicked.connect(self._on_accept)
        bar.addWidget(self._btn_cancel)
        bar.addWidget(self._btn_accept)
        layout.addLayout(bar)

    def _build_plot(self) -> None:
        # Mirror ThresholdGUI._build_figure but using a raw Figure (no pyplot)
        # so the canvas lives only in our QDialog, not in a pyplot manager.
        self._fig.set_facecolor("white")
        gs = GridSpec(3, 3, figure=self._fig, hspace=0.35, wspace=0.3)

        self._ax_scatter = self._fig.add_subplot(gs[0:2, 0])
        self._ax_scatter.set_xlabel("DTW Distance")
        self._ax_scatter.set_ylabel("Amplitude")
        self._ax_scatter.set_title(
            "Click to move distance threshold (use Toggle for amplitude)"
        )

        self._ax_good_filt = self._fig.add_subplot(gs[0, 1])
        self._ax_good_filt.set_title("Good + Weird (filtered)")
        self._ax_bad_filt = self._fig.add_subplot(gs[0, 2])
        self._ax_bad_filt.set_title("Bad + Weirdbad (filtered)")
        self._ax_good_uf = self._fig.add_subplot(gs[1, 1])
        self._ax_good_uf.set_title("Good + Weird (unfiltered)")
        self._ax_bad_uf = self._fig.add_subplot(gs[1, 2])
        self._ax_bad_uf.set_title("Bad + Weirdbad (unfiltered)")
        self._ax_mean = self._fig.add_subplot(gs[2, :])
        self._ax_mean.set_title("Click to use mean waveform as new template")

        # Used by ThresholdGUI._update_panels for canvas.draw_idle()
        self.fig = self._fig

        self.canvas.mpl_connect("button_press_event", self._on_click)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()
        elif event.key() == Qt.Key.Key_B:
            self._on_toggle()
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

    def _on_click(self, event) -> None:
        # Delegate to ThresholdGUI._on_click — it reads self._match,
        # self.params, self._active_threshold, the axes refs, and calls
        # self._update_panels (which we route through ThresholdGUI too).
        ThresholdGUI._on_click(self, event)

    def _update_panels(self) -> None:
        ThresholdGUI._update_panels(self)

    def _on_toggle(self) -> None:
        ThresholdGUI._toggle_active(self)

    def _on_use_mean_template(self) -> None:
        ThresholdGUI._update_template_from_mean(self)

    def _on_accept(self) -> None:
        if self.on_finished is not None:
            self.on_finished(self.params)
        self.accept()

    def setup(self):
        """Return the dialog itself (for embedding API parity)."""
        return self

    def run(self) -> SpikeDetectionParams:
        """Display modally and return the (possibly updated) params."""
        self.exec()
        return self.params

    def finish(self) -> None:
        """Programmatically accept the dialog. Idempotent."""
        self._on_accept()
