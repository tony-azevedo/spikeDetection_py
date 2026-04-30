"""Qt-based spike template selection GUI.

Wraps the matplotlib pick-driven plot in a QDialog with Accept / Cancel /
Clear buttons and centers the window on first show. The plot interaction
is unchanged from :class:`TemplateSelectionGUI`; the underlying template
build helper is reused via class-method delegation, so there is a single
source of truth for that algorithm.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from spikedetect.gui._qt_imports import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, Qt,
)
from spikedetect.gui._widgets import UserCancelled
from spikedetect.gui.template_gui import (
    TemplateSelectionGUI, _format_template_age,
)
from spikedetect.models import SpikeDetectionParams
from spikedetect.pipeline.peaks import find_spike_locations


class TemplateSelectionGUIQt(QDialog):
    """Qt dialog for selecting seed spikes and building a template.

    Args:
        filtered_data: 1-D bandpass-filtered voltage trace.
        params: Current detection parameters (must have ``fs`` and
            ``spike_template_width`` set).

    Attributes:
        params: Detection parameters (deep-copied).
        canvas: The embedded ``FigureCanvasQTAgg``.
    """

    def __init__(
        self,
        filtered_data: np.ndarray,
        params: SpikeDetectionParams,
        parent: QWidget | None = None,
        template_ttl_hours: float = 24.0,
    ) -> None:
        super().__init__(parent)
        self._filtered = np.asarray(filtered_data, dtype=np.float64).ravel()
        self.params = deepcopy(params)
        self._selected_indices: list[int] = []
        self._template: np.ndarray | None = None
        self._cancelled = False
        self._template_ttl_hours = template_ttl_hours
        self._reuse_existing_template = (
            self.params.spike_template is not None
        )
        self._existing_template_is_fresh = self.params.template_is_fresh(
            template_ttl_hours,
        )
        self.template_updated_at: datetime | None = None

        self.on_finished: Callable[[np.ndarray | None], None] | None = None
        self.on_selection_changed: Callable[[int], None] | None = None

        self._setup_ui()
        self._build_plot()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Select template spikes")
        self.setMinimumSize(900, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self._fig = Figure(figsize=(12, 7))
        self.canvas = FigureCanvasQTAgg(self._fig)
        self.canvas.setMinimumHeight(450)
        layout.addWidget(self.canvas, stretch=1)

        self._ax_main = self._fig.add_subplot(2, 1, 1)
        self._ax_squig = self._fig.add_subplot(2, 1, 2)
        self._fig.subplots_adjust(
            left=0.06, right=0.98, top=0.94, bottom=0.08, hspace=0.3,
        )

        bar = QHBoxLayout()
        bar.addStretch()
        self._btn_clear = QPushButton("Clear selection")
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_cancel = QPushButton("Cancel (Esc)")
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_accept = QPushButton("Accept (Enter)")
        self._btn_accept.setDefault(True)
        self._btn_accept.clicked.connect(self._on_accept)
        bar.addWidget(self._btn_clear)
        bar.addWidget(self._btn_cancel)
        bar.addWidget(self._btn_accept)
        layout.addLayout(bar)

    def _build_plot(self) -> None:
        stw = self.params.spike_template_width
        self._peak_locs = find_spike_locations(
            self._filtered,
            peak_threshold=self.params.peak_threshold,
            fs=self.params.fs,
            spike_template_width=stw,
        )

        self._ax_main.clear()
        self._ax_squig.clear()

        n = len(self._filtered)
        self._ax_main.plot(np.arange(n), self._filtered, "k", linewidth=0.5)
        if len(self._peak_locs) > 0:
            self._ax_main.plot(
                self._peak_locs, self._filtered[self._peak_locs], "ro",
                markersize=4, picker=5,
            )
        self._ax_main.set_xlim(0, n)
        if self._reuse_existing_template:
            if self._existing_template_is_fresh:
                age_note = _format_template_age(
                    self.params.template_updated_at,
                )
                color = (0, 0.3, 1.0)  # fresh: blue
            elif self.params.template_updated_at is not None:
                age_note = (
                    f"stale, "
                    f"{_format_template_age(self.params.template_updated_at)}"
                )
                color = (1.0, 0.55, 0.0)  # stale: orange
            else:
                age_note = "no timestamp"
                color = (1.0, 0.55, 0.0)  # untimestamped: orange
            self._ax_main.set_title(
                f"Existing template ({age_note}). Accept to keep "
                "(clock resets), or click peaks to rebuild."
            )
            half = self.params.spike_template_width // 2
            window = np.arange(-half, half + 1)
            tmpl = self.params.spike_template
            window = window[: len(tmpl)]
            self._ax_squig.plot(
                window, tmpl, color=color, linewidth=2,
            )
            self._ax_squig.set_title(
                f"Existing template ({age_note}, will be kept on Accept)"
            )
        else:
            self._ax_main.set_title(
                "Click peaks to select seed spikes; Accept to build template"
            )
            self._ax_squig.set_title("Selected waveforms")

        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.draw_idle()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_accept()
        elif event.key() == Qt.Key.Key_Escape:
            self._cancelled = True
            self.reject()
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

    def _on_pick(self, event) -> None:
        if event.artist is None:
            return
        ind = event.ind
        if ind is None or len(ind) == 0:
            return
        x_click = event.mouseevent.xdata
        if x_click is None:
            return

        locs = self._peak_locs
        distances = np.abs(locs[ind] - x_click)
        best = ind[np.argmin(distances)]
        loc = int(locs[best])

        if loc in self._selected_indices:
            return

        # First click while an existing template (fresh or stale) was on
        # display: clear it so the user sees only the new selection.
        if (self._reuse_existing_template
                and len(self._selected_indices) == 0):
            self._ax_squig.cla()

        self._selected_indices.append(loc)

        self._ax_main.plot(
            loc, self._filtered[loc], "go", markersize=8,
        )

        stw = self.params.spike_template_width
        half = stw // 2
        window = np.arange(-half, half + 1)
        if loc - half >= 0 and loc + half < len(self._filtered):
            snippet = self._filtered[loc + window]
            self._ax_squig.plot(window, snippet, alpha=0.6)
            self._ax_squig.set_title(
                f"Selected waveforms ({len(self._selected_indices)})"
            )

        self.canvas.draw_idle()
        if self.on_selection_changed is not None:
            self.on_selection_changed(len(self._selected_indices))

    def _on_clear(self) -> None:
        self._selected_indices.clear()
        self._build_plot()
        if self.on_selection_changed is not None:
            self.on_selection_changed(0)

    def _on_accept(self) -> None:
        # Reuse TemplateSelectionGUI._build_template — same implementation,
        # single source of truth. It only reads self._filtered, self.params,
        # and self._selected_indices, all of which we have.
        self._template = TemplateSelectionGUI._build_template(self)
        if self.on_finished is not None:
            self.on_finished(self._template)
        self.accept()

    def setup(self):
        """Return the dialog itself (for embedding API parity)."""
        return self

    def run(self) -> np.ndarray | None:
        """Display modally and return the template (or None if no selection).

        Raises:
            UserCancelled: If the user presses Esc.
        """
        self.exec()
        if self._cancelled:
            raise UserCancelled(
                "TemplateSelectionGUIQt cancelled by user (Esc)"
            )
        return self._template

    def finish(self) -> None:
        """Programmatically accept the dialog. Idempotent."""
        self._on_accept()
