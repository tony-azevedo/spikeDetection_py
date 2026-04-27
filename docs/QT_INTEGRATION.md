# Embedding spikedetect GUIs in a Qt application

This guide shows how to embed `spikedetect`'s interactive GUIs inside a host Qt application (PyQt5/6 or PySide2/6) — for example, adding a "Detect spikes" button to a Qt-based browser, viewer, or analysis tool that you've built separately.

## The blocking problem (and why this works)

Each GUI's `run()` method blocks: it calls `fig.canvas.start_event_loop()` and waits for the user to press Enter. If your host app is already running its own Qt event loop, calling `run()` from inside it either freezes your app or fails outright — you cannot run two event loops at once.

The fix: each GUI also exposes a non-blocking `setup()` method that builds the figure and installs handlers but **does not** start an event loop. The host's existing Qt loop drives everything. When the user is done, an `on_finished` callback fires with the result.

This works because matplotlib's `QtAgg` backend produces a `FigureCanvasQTAgg` object that is a real `QWidget` — you can drop it into any `QLayout`, `QDockWidget`, or `QStackedWidget` like any other widget.

## Setup

### 1. Pick a Qt backend before importing matplotlib

```python
import os
os.environ.setdefault("MPLBACKEND", "QtAgg")  # MUST be set before importing matplotlib
```

Or equivalently, the very first time you touch matplotlib in your app:

```python
import matplotlib
matplotlib.use("QtAgg")  # before any pyplot import
```

`QtAgg` works with PyQt6, PyQt5, PySide6, and PySide2 — matplotlib auto-selects whichever you have installed.

### 2. Install dependencies

The host app needs a Qt binding installed; spikedetect itself does not import Qt. For example:

```bash
pip install PyQt6        # or PyQt5, PySide6, PySide2
pip install -e .         # spikedetect from the spikedetect/ subdirectory
```

## Per-GUI API

All four GUIs follow the same pattern:

```python
gui = SomeGUI(...)
canvas = gui.setup()              # FigureCanvasQTAgg, ready to embed
gui.on_finished = my_callback     # fires when user accepts (Enter / close / finish())
host_layout.addWidget(canvas)     # host's Qt loop drives interaction
# ...later:
gui.close()                       # disconnects handlers, closes the figure (idempotent)
```

| GUI | `on_finished` payload | Live signals |
|---|---|---|
| `FilterGUI` | `params: SpikeDetectionParams` | `on_params_changed(params)` on every slider tick |
| `TemplateSelectionGUI` | `template: np.ndarray \| None` | `on_selection_changed(n_selected: int)` |
| `ThresholdGUI` | `params: SpikeDetectionParams` | `on_thresholds_changed(distance, amplitude)` |
| `SpotCheckGUI` | `result: SpikeDetectionResult` | `on_spike_index_changed(idx: int)` |

`finish()` programmatically signals "done" — useful from a host's Finish button. `close()` is always safe to call multiple times.

## Minimal example: one GUI in a QMainWindow

```python
import os
os.environ["MPLBACKEND"] = "QtAgg"

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

import spikedetect as sd
from spikedetect.gui import FilterGUI


class MyApp(QMainWindow):
    def __init__(self, voltage, fs):
        super().__init__()
        self.voltage = voltage
        self.params = sd.SpikeDetectionParams.default(fs=fs)

        central = QWidget()
        self.layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        btn = QPushButton("Tune filter parameters")
        btn.clicked.connect(self.open_filter_gui)
        self.layout.addWidget(btn)

        self.filter_gui = None
        self.canvas = None

    def open_filter_gui(self):
        # Keep refs as instance attrs — otherwise garbage collection kills them
        self.filter_gui = FilterGUI(self.voltage, self.params)
        self.canvas = self.filter_gui.setup()
        self.filter_gui.on_finished = self.on_filter_done
        self.layout.addWidget(self.canvas)

    def on_filter_done(self, new_params):
        self.params = new_params
        print(f"Got params: hp={new_params.hp_cutoff}, lp={new_params.lp_cutoff}")
        self.layout.removeWidget(self.canvas)
        self.canvas.setParent(None)
        self.filter_gui.close()
        self.filter_gui = None
        self.canvas = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    voltage = np.random.randn(50000)
    win = MyApp(voltage, fs=10000)
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())
```

## Full workflow: chaining all four GUIs

The host orchestrates the sequence. This replicates what the blocking [`InteractiveWorkflow`](../spikedetect/src/spikedetect/gui/workflow.py) does, but driven by Qt callbacks instead of a `while` loop.

```python
import os
os.environ["MPLBACKEND"] = "QtAgg"

from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QMessageBox

import spikedetect as sd
from spikedetect.gui import (
    FilterGUI, TemplateSelectionGUI, ThresholdGUI, SpotCheckGUI,
)
from spikedetect.pipeline.filtering import SignalFilter
from spikedetect.pipeline.peaks import PeakFinder
from spikedetect.pipeline.template import TemplateMatcher


class SpikeDetectionWorkflow(QMainWindow):
    def __init__(self, recording, params=None):
        super().__init__()
        self.recording = recording
        self.params = params or sd.SpikeDetectionParams.default(fs=recording.sample_rate)
        self.result = None
        self._gui = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self._step_filter()

    def _show(self, gui):
        # Replace any previous GUI in the stack
        if self._gui is not None:
            self._gui.close()
        self._gui = gui
        canvas = gui.setup()
        self.stack.addWidget(canvas)
        self.stack.setCurrentWidget(canvas)

    # --- Step 1: filter tuning ---
    def _step_filter(self):
        gui = FilterGUI(self.recording.voltage, self.params)
        gui.on_finished = self._after_filter
        self._show(gui)

    def _after_filter(self, params):
        self.params = params
        self._step_template()

    # --- Step 2: template selection ---
    def _step_template(self):
        filtered = SignalFilter.filter_data(
            self.recording.voltage, fs=self.params.fs,
            hp_cutoff=self.params.hp_cutoff, lp_cutoff=self.params.lp_cutoff,
            diff_order=self.params.diff_order, polarity=self.params.polarity,
        )
        gui = TemplateSelectionGUI(filtered, self.params)
        gui.on_finished = self._after_template
        self._show(gui)

    def _after_template(self, template):
        if template is None:
            QMessageBox.warning(self, "No template", "Select at least one spike.")
            self._step_template()
            return
        self.params.spike_template = template
        self.result = sd.detect_spikes(self.recording, self.params)
        self._step_threshold()

    # --- Step 3: threshold adjustment ---
    def _step_threshold(self):
        start = round(0.01 * self.params.fs)
        unfiltered = self.recording.voltage[start:]
        filt = SignalFilter.filter_data(
            unfiltered, fs=self.params.fs,
            hp_cutoff=self.params.hp_cutoff, lp_cutoff=self.params.lp_cutoff,
            diff_order=self.params.diff_order, polarity=self.params.polarity,
        )
        locs = PeakFinder.find_spike_locations(
            filt, peak_threshold=self.params.peak_threshold,
            fs=self.params.fs, spike_template_width=self.params.spike_template_width,
        )
        match = TemplateMatcher.match(
            locs, self.params.spike_template, filt, unfiltered,
            self.params.spike_template_width, self.params.fs,
        )
        gui = ThresholdGUI(match, self.params)
        gui.on_finished = self._after_threshold
        self._show(gui)

    def _after_threshold(self, params):
        self.params = params
        self.result = sd.detect_spikes(self.recording, self.params)

        # Native "Done?" prompt instead of matplotlib popup
        reply = QMessageBox.question(
            self, "Done?", "Done with spike detection?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            self._step_filter()
        else:
            self._step_spotcheck()

    # --- Step 4: spot-check ---
    def _step_spotcheck(self):
        if self.result.n_spikes == 0:
            self._done()
            return
        gui = SpotCheckGUI(self.recording, self.result)
        gui.on_finished = self._after_spotcheck
        self._show(gui)

    def _after_spotcheck(self, result):
        self.result = result
        self._done()

    def _done(self):
        if self._gui is not None:
            self._gui.close()
            self._gui = None
        print(f"Final: {self.result.n_spikes} spikes")
        self.close()
```

Launch it like any `QMainWindow`:

```python
from PyQt6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
win = SpikeDetectionWorkflow(my_recording)
win.resize(1400, 900)
win.show()
sys.exit(app.exec())
```

## Common pitfalls

**1. Garbage collection eats your GUI**

If you write `gui = FilterGUI(...); canvas = gui.setup(); layout.addWidget(canvas)` as locals in a function, the `gui` Python object goes out of scope when the function returns. The canvas stays in the layout but the matplotlib callbacks on `gui` may stop firing because their owner is gone. **Always store the GUI as an instance attribute** (`self.filter_gui = ...`) for as long as it's on screen.

**2. Forgetting to set the backend**

If `MPLBACKEND` is not set to `QtAgg` before matplotlib is imported, you get TkAgg by default — the canvas returned by `setup()` is then a `FigureCanvasTkAgg`, which is **not** a `QWidget` and cannot be added to a Qt layout. Set the env var at the top of your entry-point script.

**3. Mixing `plt.show()` with the host loop**

Don't call `plt.show()` anywhere in your host app. The Qt event loop is already running; `plt.show()` will try to start its own and freeze things. Stick to `gui.setup()` + adding the canvas to a layout.

**4. Calling `gui.run()` instead of `gui.setup()`**

`run()` is the standalone blocking entry point — it spins up its own event loop. Useful when running spikedetect from a plain Python script with no host, but **not** for embedding. Use `setup()` instead.

**5. Re-using a GUI after `close()`**

Once `close()` has been called, the figure is destroyed and the GUI cannot be reopened. Create a fresh instance for each interaction.

## Forwarding to PyQt signals (optional)

The callbacks are plain Python functions, but you can wrap them in a Qt signal if you want signal/slot semantics elsewhere in your app:

```python
from PyQt6.QtCore import QObject, pyqtSignal


class FilterGuiBridge(QObject):
    finished = pyqtSignal(object)  # SpikeDetectionParams

    def __init__(self, voltage, params):
        super().__init__()
        self.gui = FilterGUI(voltage, params)
        self.gui.on_finished = self.finished.emit  # forward callback to signal

    def setup(self):
        return self.gui.setup()

    def close(self):
        self.gui.close()


# Usage:
bridge = FilterGuiBridge(voltage, params)
canvas = bridge.setup()
bridge.finished.connect(self.on_filter_done)  # idiomatic Qt
layout.addWidget(canvas)
```

A future Level 2 in [`spikedetect/gui/qt/`](../spikedetect/src/spikedetect/gui/) would provide these `*Widget` wrappers (with `pyqtSignal`s, native toolbars, and a Finish button) out of the box. For now, the wrapper is short enough to write per-app as needed.

## Reference

- Public GUI API: [`spikedetect/src/spikedetect/gui/`](../spikedetect/src/spikedetect/gui/)
- Standalone blocking workflow (for non-Qt scripts): [`InteractiveWorkflow`](../spikedetect/src/spikedetect/gui/workflow.py)
- Non-blocking smoke tests: [`test_gui.py::TestNonBlockingAPI`](../spikedetect/tests/test_gui.py)
