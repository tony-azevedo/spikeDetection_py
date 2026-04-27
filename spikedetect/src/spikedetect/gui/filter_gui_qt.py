"""Qt-based filter parameter tuning GUI.

This is a PyQt/PySide version of FilterGUI that provides better
window control and native-looking widgets.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from spikedetect.gui._widgets import raster_ticks
from spikedetect.models import SpikeDetectionParams
from spikedetect.pipeline.filtering import filter_data
from spikedetect.pipeline.peaks import find_spike_locations

# Try to import Qt - support both PyQt and PySide
try:
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
        QPushButton, QLineEdit, QButtonGroup, QRadioButton,
        QWidget, QGridLayout, QGroupBox,
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QTimer
    from PyQt6.QtGui import QKeyEvent
    QT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
            QPushButton, QLineEdit, QButtonGroup, QRadioButton,
            QWidget, QGridLayout, QGroupBox,
        )
        from PyQt5.QtCore import Qt, pyqtSignal, QTimer
        from PyQt5.QtGui import QKeyEvent
        QT_VERSION = 5
    except ImportError:
        try:
            from PySide6.QtWidgets import (
                QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                QPushButton, QLineEdit, QButtonGroup, QRadioButton,
                QWidget, QGridLayout, QGroupBox,
            )
            from PySide6.QtCore import Qt, Signal as pyqtSignal, QTimer
            from PySide6.QtGui import QKeyEvent
            QT_VERSION = 6
        except ImportError:
            try:
                from PySide2.QtWidgets import (
                    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                    QPushButton, QLineEdit, QButtonGroup, QRadioButton,
                    QWidget, QGridLayout, QGroupBox,
                )
                from PySide2.QtCore import Qt, Signal as pyqtSignal, QTimer
                from PySide2.QtGui import QKeyEvent
                QT_VERSION = 2
            except ImportError:
                raise ImportError(
                    "No Qt binding found. Install PyQt6, PyQt5, PySide6, or PySide2."
                )


class FilterGUIQt(QDialog):
    """Qt-based interactive GUI for tuning filter and peak-detection parameters.

    Args:
        unfiltered_data: Raw 1-D voltage trace.
        params: Initial detection parameters.

    Attributes:
        params: Current parameters (updated by controls).
        canvas: The matplotlib canvas (for embedding in other Qt apps).
    """

    # Signal emitted when params change (for live updates)
    params_changed = pyqtSignal(object)

    def __init__(
        self,
        unfiltered_data: np.ndarray,
        params: SpikeDetectionParams,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._unfiltered = np.asarray(unfiltered_data, dtype=np.float64).ravel()
        self.params = deepcopy(params)
        self._filtered = None
        self._locs = None
        self._finished = False

        # User-supplied callbacks for non-blocking embedding
        self.on_finished: Callable[[SpikeDetectionParams], None] | None = None
        self.on_params_changed: Callable[[SpikeDetectionParams], None] | None = None

        self._setup_ui()
        self._apply_filter()
        self._update_plots()

    def _setup_ui(self) -> None:
        """Build the Qt UI with matplotlib canvas and controls."""
        self.setWindowTitle("Filter Parameters")
        self.setMinimumSize(900, 700)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- Matplotlib figure area ---
        self._fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasQTAgg(self._fig)
        self.canvas.setMinimumHeight(400)

        # Create axes
        self._ax_unfilt = self._fig.add_subplot(211)
        self._ax_filt = self._fig.add_subplot(212)

        self._fig.subplots_adjust(
            left=0.08, right=0.95, top=0.95, bottom=0.25, hspace=0.25
        )

        main_layout.addWidget(self.canvas, stretch=3)

        # --- Control panel ---
        controls_group = QGroupBox("Filter Parameters")
        controls_layout = QGridLayout()

        # High-pass cutoff slider
        hp_label = QLabel("HP Cutoff (Hz):")
        self._slider_hp = QSlider(Qt.Orientation.Horizontal)
        self._slider_hp.setRange(1, 2000)  # 0.5 to 1000 Hz (scaled by 1/2)
        self._slider_hp.setValue(int(self.params.hp_cutoff * 2))
        self._slider_hp.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_hp.setTickInterval(100)
        self._label_hp_value = QLabel(f"{self.params.hp_cutoff:.1f}")
        self._label_hp_value.setMinimumWidth(50)
        self._slider_hp.valueChanged.connect(self._on_hp_changed)

        controls_layout.addWidget(hp_label, 0, 0)
        controls_layout.addWidget(self._slider_hp, 0, 1)
        controls_layout.addWidget(self._label_hp_value, 0, 2)

        # Low-pass cutoff slider
        lp_label = QLabel("LP Cutoff (Hz):")
        self._slider_lp = QSlider(Qt.Orientation.Horizontal)
        self._slider_lp.setRange(1, 2000)  # 0.5 to 1000 Hz (scaled by 1/2)
        self._slider_lp.setValue(int(self.params.lp_cutoff * 2))
        self._slider_lp.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_lp.setTickInterval(100)
        self._label_lp_value = QLabel(f"{self.params.lp_cutoff:.1f}")
        self._label_lp_value.setMinimumWidth(50)
        self._slider_lp.valueChanged.connect(self._on_lp_changed)

        controls_layout.addWidget(lp_label, 1, 0)
        controls_layout.addWidget(self._slider_lp, 1, 1)
        controls_layout.addWidget(self._label_lp_value, 1, 2)

        # Peak threshold slider (log scale)
        thresh_label = QLabel("Peak Threshold:")
        self._slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self._slider_thresh.setRange(0, 100)  # 0-100 representing log scale
        log_thresh = np.log10(max(self.params.peak_threshold, 1e-10))
        # Map log10 range [-10, -1] to [0, 100]
        slider_val = int((log_thresh + 10) / 9 * 100)
        self._slider_thresh.setValue(slider_val)
        self._slider_thresh.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_thresh.setTickInterval(25)
        self._label_thresh_value = QLabel(f"{self.params.peak_threshold:.2e}")
        self._label_thresh_value.setMinimumWidth(80)
        self._slider_thresh.valueChanged.connect(self._on_thresh_changed)

        controls_layout.addWidget(thresh_label, 2, 0)
        controls_layout.addWidget(self._slider_thresh, 2, 1)
        controls_layout.addWidget(self._label_thresh_value, 2, 2)

        # Derivative order radio buttons
        diff_label = QLabel("Diff Order:")
        self._diff_group = QButtonGroup(self)
        self._radio_diff0 = QRadioButton("0")
        self._radio_diff1 = QRadioButton("1")
        self._radio_diff2 = QRadioButton("2")
        self._diff_group.addButton(self._radio_diff0, 0)
        self._diff_group.addButton(self._radio_diff1, 1)
        self._diff_group.addButton(self._radio_diff2, 2)

        if self.params.diff_order == 0:
            self._radio_diff0.setChecked(True)
        elif self.params.diff_order == 1:
            self._radio_diff1.setChecked(True)
        else:
            self._radio_diff2.setChecked(True)

        # Connect using id() to get button ID - works across PyQt/PySide
        self._diff_group.buttonClicked.connect(
            lambda btn: self._on_diff_changed(self._diff_group.id(btn))
        )

        diff_layout = QHBoxLayout()
        diff_layout.addWidget(self._radio_diff0)
        diff_layout.addWidget(self._radio_diff1)
        diff_layout.addWidget(self._radio_diff2)
        diff_layout.addStretch()

        controls_layout.addWidget(diff_label, 3, 0)
        controls_layout.addWidget(QLabel(), 3, 1)
        controls_layout.addLayout(diff_layout, 3, 1, 1, 2)

        # Polarity toggle
        pol_label = QLabel("Polarity:")
        self._btn_pol = QPushButton(f"Pol: {self.params.polarity:+d}")
        self._btn_pol.setCheckable(False)
        self._btn_pol.clicked.connect(self._on_polarity_toggle)

        controls_layout.addWidget(pol_label, 4, 0)
        controls_layout.addWidget(self._btn_pol, 4, 1)

        # Precision controls: arrow buttons + text box
        precision_label = QLabel("Threshold Precision:")
        self._btn_arrow_left = QPushButton("◄")
        self._btn_arrow_left.setMaximumWidth(40)
        self._btn_arrow_left.clicked.connect(self._on_arrow_left)
        self._btn_arrow_right = QPushButton("►")
        self._btn_arrow_right.setMaximumWidth(40)
        self._btn_arrow_right.clicked.connect(self._on_arrow_right)
        self._edit_thresh = QLineEdit()
        self._edit_thresh.setPlaceholderText("Enter value")
        self._edit_thresh.setMaximumWidth(100)
        self._edit_thresh.returnPressed.connect(self._on_text_submit)

        precision_layout = QHBoxLayout()
        precision_layout.addWidget(self._btn_arrow_left)
        precision_layout.addWidget(self._btn_arrow_right)
        precision_layout.addWidget(self._edit_thresh)
        precision_layout.addStretch()

        controls_layout.addWidget(precision_label, 5, 0)
        controls_layout.addLayout(precision_layout, 5, 1, 1, 2)

        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group, stretch=1)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._btn_accept = QPushButton("Accept (Enter)")
        self._btn_accept.clicked.connect(self._on_accept)
        self._btn_cancel = QPushButton("Cancel (Esc)")
        self._btn_cancel.clicked.connect(self.reject)

        button_layout.addWidget(self._btn_accept)
        button_layout.addWidget(self._btn_cancel)

        main_layout.addLayout(button_layout)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key presses."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        """Center the dialog on the active screen the first time it's shown."""
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

    def _on_accept(self) -> None:
        """Handle accept button."""
        self._finished = True
        self.accept()
        if self.on_finished is not None:
            self.on_finished(self.params)

    def _on_hp_changed(self, value: int) -> None:
        """Handle HP cutoff slider change."""
        self.params.hp_cutoff = value / 2  # Scale back to Hz
        self._label_hp_value.setText(f"{self.params.hp_cutoff:.1f}")
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_lp_changed(self, value: int) -> None:
        """Handle LP cutoff slider change."""
        self.params.lp_cutoff = value / 2  # Scale back to Hz
        self._label_lp_value.setText(f"{self.params.lp_cutoff:.1f}")
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_thresh_changed(self, value: int) -> None:
        """Handle threshold slider change."""
        # Map [0, 100] back to log10 range [-10, -1]
        log_thresh = value / 100 * 9 - 10
        self.params.peak_threshold = 10 ** log_thresh
        self._label_thresh_value.setText(f"{self.params.peak_threshold:.2e}")
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_diff_changed(self, id: int) -> None:
        """Handle derivative order radio button change."""
        self.params.diff_order = id
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_polarity_toggle(self) -> None:
        """Handle polarity toggle button."""
        self.params.polarity *= -1
        self._btn_pol.setText(f"Pol: {self.params.polarity:+d}")
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_arrow_left(self) -> None:
        """Decrease peak threshold by 10%."""
        current = self.params.peak_threshold
        new_val = current * 0.9
        self.params.peak_threshold = new_val
        self._update_slider_from_value()
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_arrow_right(self) -> None:
        """Increase peak threshold by 10%."""
        current = self.params.peak_threshold
        new_val = current * 1.1
        self.params.peak_threshold = new_val
        self._update_slider_from_value()
        self._apply_filter()
        self._update_plots()
        self._notify_params_changed()

    def _on_text_submit(self) -> None:
        """Apply threshold value from text box."""
        try:
            text = self._edit_thresh.text()
            val = float(text)
            if val > 0:
                self.params.peak_threshold = val
                self._update_slider_from_value()
                self._apply_filter()
                self._update_plots()
                self._notify_params_changed()
        except ValueError:
            pass  # Ignore invalid input

    def _update_slider_from_value(self) -> None:
        """Update slider position to match current threshold value."""
        log_thresh = np.log10(max(self.params.peak_threshold, 1e-10))
        slider_val = int((log_thresh + 10) / 9 * 100)
        self._slider_thresh.setValue(slider_val)
        self._label_thresh_value.setText(f"{self.params.peak_threshold:.2e}")

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
        self._ax_unfilt.set_title("Adjust filter parameters")

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

        self.canvas.draw()

    def _notify_params_changed(self) -> None:
        if self.on_params_changed is not None:
            self.on_params_changed(self.params)
        self.params_changed.emit(self.params)

    def setup(self):
        """Build the UI (for non-blocking use).

        Returns:
            The dialog itself (for embedding).
        """
        return self

    def run(self) -> SpikeDetectionParams:
        """Display the GUI and block until the user accepts or cancels.

        Returns:
            Updated parameters reflecting the user's choices.
        """
        self.exec()
        return self.params

    def finish(self) -> None:
        """Programmatically signal that the user is done."""
        self._on_accept()

    def close(self) -> None:
        """Close the dialog. Idempotent."""
        self.reject()