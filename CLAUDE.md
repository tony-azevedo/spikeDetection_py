# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python package (`spikedetect`) for detecting and sorting neural/EMG spikes from electrophysiology recordings. Ported from a MATLAB codebase using DTW template-matching with amplitude thresholds to classify candidate spikes, with interactive Matplotlib GUIs for parameter tuning and manual spot-checking.

The original MATLAB code is at [tony-azevedo/spikeDetection](https://github.com/tony-azevedo/spikeDetection) for reference.

## Installation and Test Commands

The Python package lives in the `spikedetect/` subdirectory — all `pip` and `pytest` commands must be run from there. See [README.md](README.md) for the full install matrix (extras: `[io]` for `.abf`/HDF5, `[fast]` for numba, `[dev]` for pytest+ruff, `[all]`) and common installation issues.

```bash
cd spikedetect
pip install -e ".[dev]"

# Tests (115 currently pass; 28 cross-validation tests skip without the .mat file)
python -m pytest tests/ -v
python -m pytest tests/test_dtw.py -v
python -m pytest tests/test_pipeline.py::TestDetectSpikes::test_detects_embedded_spikes -v
python -m pytest tests/ --cov=spikedetect --cov-report=term-missing
```

### Non-obvious gotchas

- **numba/numpy version mismatch warnings** (`_ARRAY_API not found`) are harmless — the pipeline falls back to pure numpy. If you must silence: `pip uninstall numba` or pin `numpy<2`.
- **Jupyter interactivity** requires `pip install ipympl` plus `%matplotlib widget` at the top of the notebook.
- **`pip install -e .` from the repo root fails** — `pyproject.toml` is at `spikedetect/pyproject.toml`, not the root.

## Architecture

### Package Structure

```
spikedetect/src/spikedetect/
├── __init__.py          # Public API: all classes + detect_spikes alias
├── models.py            # SpikeDetectionParams, Recording, SpikeDetectionResult dataclasses
├── utils.py             # WaveformProcessor (smooth, smooth_and_differentiate, differentiate)
├── io/
│   ├── config.py        # JSON param persistence (~/.spikedetect/)
│   ├── mat.py           # .mat loading (v5/v7 via scipy, v7.3 via h5py)
│   ├── abf.py           # ABF loading via pyabf
│   └── native.py        # HDF5 native format via h5py
├── pipeline/
│   ├── filtering.py     # SignalFilter — Butterworth bandpass + differentiation
│   ├── peaks.py         # PeakFinder — find candidate spike peaks
│   ├── dtw.py           # DTW — squared-Euclidean cost DTW (optional numba JIT)
│   ├── template.py      # TemplateMatcher — DTW template matching + amplitude projection
│   ├── inflection.py    # InflectionPointDetector — 2nd derivative spike time correction
│   ├── classify.py      # SpikeClassifier — quantile-based 4-category classification
│   └── detect.py        # SpikeDetector — full pipeline orchestrator
└── gui/
    ├── _widgets.py      # Shared helpers (raster_ticks, blocking_wait, install_finish_handlers)
    ├── filter_gui.py    # FilterGUI — interactive filter parameter tuning
    ├── template_gui.py  # TemplateSelectionGUI — click to select seed spikes
    ├── threshold_gui.py # ThresholdGUI — DTW/amplitude scatter threshold adjustment
    ├── spotcheck_gui.py # SpotCheckGUI — spike-by-spike review
    └── workflow.py      # InteractiveWorkflow — orchestrates Filter→Template→Threshold→SpotCheck
```

Each GUI exposes both a blocking `run()` (for standalone use) and a non-blocking `setup()` + `on_finished` callback pair (for embedding in a host Qt event loop). See [docs/QT_INTEGRATION.md](docs/QT_INTEGRATION.md). `InteractiveWorkflow.run()` is blocking-only; embedding the chained workflow in Qt is the host's job (a worked example is in the integration doc).

### Detection Pipeline (SpikeDetector.detect)

1. **SignalFilter.filter_data** — Causal Butterworth bandpass (`scipy.signal.lfilter`, NOT `filtfilt`) with optional differentiation and polarity flip
2. **PeakFinder.find_spike_locations** — `scipy.signal.find_peaks` with height and distance constraints
3. **TemplateMatcher.match** — Extract windows, normalize, compute DTW distance against template, compute amplitude via projection
4. **Threshold** — Accept spikes where `DTW_distance < threshold` AND `amplitude > threshold`
5. **InflectionPointDetector.estimate_spike_times** — Correct timing using peak of smoothed 2nd derivative

### Class-Based Design

All pipeline functions are organized as static methods within classes. Each pipeline module also exports backwards-compatible function aliases (e.g., `filter_data = SignalFilter.filter_data`), but **only the classes are re-exported from the top-level package**. The function aliases are reachable only through the submodule path:

```python
# Class-based (preferred — top-level import works)
from spikedetect import SignalFilter, SpikeDetector
filtered = SignalFilter.filter_data(voltage, fs=10000, hp_cutoff=200, lp_cutoff=800)

# Function alias (must import from the submodule, NOT from spikedetect directly)
from spikedetect.pipeline.filtering import filter_data
filtered = filter_data(voltage, fs=10000, hp_cutoff=200, lp_cutoff=800)
```

### Critical Porting Details

- MATLAB `smooth(x, n)` → `scipy.ndimage.uniform_filter1d(x, size=n, mode='nearest')`
- MATLAB `filter(b, a, x)` → `scipy.signal.lfilter(b, a, x)` (causal, NOT `filtfilt`)
- DTW uses squared-Euclidean cost `(r[i] - t[j])^2`, not absolute difference
- MATLAB 1-based indexing converted to Python 0-based throughout — especially tricky in `inflection.py` where MATLAB `(idx_i:end-idx_f)` (1-based inclusive) becomes `[idx_i-1:len-idx_f]` (0-based)
- `SpikeDetectionParams` dataclass replaces the MATLAB global `vars` struct
- JSON persistence in `~/.spikedetect/` replaces MATLAB `setacqpref`/`getacqpref`

### Cross-Validation Against MATLAB

The Python pipeline has been rigorously cross-validated against the original MATLAB code using a step-by-step comparison of all intermediate variables (see `CROSS_VALIDATION_REPORT.md`). Key findings:

- **Filtering, peak detection, DTW distances, thresholds**: exact or near-exact match
- **Spike detection (which spikes, where)**: identical (296/296 peaks at the same sample indices)
- **Spike time correction**: median 2-sample (40 us) jitter, max 11 samples (220 us) — inherent to the smoothed-2nd-derivative algorithm, not a porting bug
- **Systematic bias**: -1.29 samples mean offset (26 us), constant across all spikes, caused by a 1-sample inflection point difference. Scientifically negligible.

The cross-validation test suite (`test_cross_validation.py`, 18 tests) loads MATLAB intermediates from `cross_validation_intermediates.mat` and verifies each pipeline stage independently.

### Test Data

- `LEDFlashTriggerPiezoControl_Raw_240430_F1_C1_5.mat` — Real recording (50kHz, 400k samples, 296 MATLAB-detected spikes) used as golden standard
- `cross_validation_intermediates.mat` — 36 intermediate variables exported from the MATLAB pipeline for step-by-step comparison

## Directory Layout

- `spikedetect/` — Python package (installable via pip)
  - `examples/` — Runnable scripts: `batch_detection.py`, `gui_workflow.py`, `gui_demo.ipynb`. Fastest way to see the package end-to-end.
  - `DATA_FORMAT_SPEC.md` — Formal IO specification: data models, file formats, and how to write a translator from any acquisition frontend
  - `MIGRATION_GUIDE.md` — MATLAB-to-Python function/parameter mapping
- `docs/` — User-facing documentation:
  - `GETTING_STARTED.md` — Install + first-spike walkthrough
  - `USER_GUIDE.md` — Full parameter/pipeline/GUI reference
  - `QT_INTEGRATION.md` — How to embed the GUIs in a host PyQt/PySide app (non-blocking API)
  - `GUI_PERFORMANCE.md` — Design doc for GUI perf optimization (forward-looking)
  - `CHANGELOG.md` — What's changed and what's new vs MATLAB
- `matlab_reference/` — Original MATLAB code for reference
  - `functions/` — MATLAB pipeline functions
  - `utils/` — MATLAB utility functions
  - `scripts/cross_validate_pipeline.m` — Script to regenerate cross-validation intermediates
- `cross_validation_intermediates.mat` — MATLAB pipeline intermediates for cross-validation tests
- `CROSS_VALIDATION_REPORT.md` — Detailed cross-validation findings
- `README.md`, `LICENSE` — Project README and license
