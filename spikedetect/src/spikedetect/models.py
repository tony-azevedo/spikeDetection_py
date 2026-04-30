"""Data models for spike detection.

Converted from MATLAB structs: ``vars`` -> SpikeDetectionParams,
``trial`` -> Recording, detection outputs -> SpikeDetectionResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpikeDetectionParams:
    """Persistent detection parameters (replaces MATLAB ``vars`` struct).

    Only persistent parameters live here. Transient data (filtered_data,
    locs, etc.) exist as local variables in pipeline functions.

    Original MATLAB fields -> Python attributes:
        fs                  -> fs
        spikeTemplateWidth  -> spike_template_width
        hp_cutoff           -> hp_cutoff
        lp_cutoff           -> lp_cutoff
        diff                -> diff_order
        peak_threshold      -> peak_threshold
        Distance_threshold  -> distance_threshold
        Amplitude_threshold -> amplitude_threshold
        spikeTemplate       -> spike_template
        polarity            -> polarity
        likelyiflpntpeak    -> likely_inflection_point_peak
        lastfilename        -> last_filename
    """

    fs: float
    spike_template_width: int = 0
    hp_cutoff: float = 200.0
    lp_cutoff: float = 800.0
    diff_order: int = 1
    peak_threshold: float = 5.0
    distance_threshold: float = 15.0
    amplitude_threshold: float = 0.2
    spike_template: np.ndarray | None = None
    polarity: int = 1
    likely_inflection_point_peak: int | None = None
    last_filename: str = ""
    # Minimum spacing between accepted spikes (in samples). When None,
    # the detector derives it from the template's full-width-half-max.
    # Set to 0 to disable refractory filtering. Catches the common
    # double-detection failure mode where two near-simultaneous local
    # peaks both pass threshold for a single underlying spike.
    min_isi_samples: int | None = None
    # When the spike_template was last built/saved (naive local time).
    # GUIs use this to skip the click-to-rebuild step when the template
    # is still fresh; bypassed for direct ``params.spike_template = ...``
    # assignment, so set it manually if you write a template by hand.
    template_updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.spike_template_width == 0:
            self.spike_template_width = round(0.005 * self.fs) + 1
        if self.spike_template is not None:
            self.spike_template = np.asarray(
                self.spike_template, dtype=np.float64
            )
            if self.spike_template.ndim != 1:
                raise ValueError(
                    f"spike_template must be a 1-D array, got shape "
                    f"{self.spike_template.shape}. Flatten it with "
                    f"template.ravel() before passing."
                )

    @classmethod
    def default(cls, fs: float = 10000) -> SpikeDetectionParams:
        """Create params with sensible defaults for a given sample rate.

        The high-pass and low-pass cutoffs are automatically scaled to
        stay below the Nyquist frequency when using low sample rates.

        Args:
            fs: Sample rate in Hz (default 10000).
        """
        nyquist = fs / 2.0
        hp = min(200.0, nyquist * 0.08)
        lp = min(800.0, nyquist * 0.32)
        if lp <= hp:
            lp = hp * 2
        return cls(fs=fs, hp_cutoff=hp, lp_cutoff=lp)

    def validate(self) -> SpikeDetectionParams:
        """Validate and clean parameters (replaces cleanUpSpikeVarsStruct).

        Returns self for chaining.
        """
        if self.fs <= 0:
            raise ValueError(
                f"Sample rate (fs) must be positive, got {self.fs}. "
                "Check that the recording was loaded correctly."
            )
        if self.hp_cutoff <= 0:
            raise ValueError(
                f"High-pass cutoff must be positive, got {self.hp_cutoff} Hz. "
                "Typical values are 100-500 Hz for spike detection."
            )
        if self.lp_cutoff <= 0:
            raise ValueError(
                f"Low-pass cutoff must be positive, got {self.lp_cutoff} Hz. "
                "Typical values are 500-3000 Hz for spike detection."
            )
        if self.hp_cutoff >= self.fs / 2:
            raise ValueError(
                f"High-pass cutoff ({self.hp_cutoff} Hz) must be below the "
                f"Nyquist frequency ({self.fs / 2} Hz). Try lowering hp_cutoff "
                f"or using a higher sample rate."
            )
        if self.lp_cutoff >= self.fs / 2:
            raise ValueError(
                f"Low-pass cutoff ({self.lp_cutoff} Hz) must be below the "
                f"Nyquist frequency ({self.fs / 2} Hz). Try lowering lp_cutoff "
                f"or using a higher sample rate."
            )
        if self.diff_order not in (0, 1, 2):
            raise ValueError(
                f"diff_order must be 0, 1, or 2, got {self.diff_order}. "
                "Use 0 for no differentiation, "
                "1 for first derivative (recommended), "
                "or 2 for second derivative."
            )
        if self.polarity not in (-1, 1):
            raise ValueError(
                f"polarity must be -1 or 1, got {self.polarity}. "
                "Use 1 for upward spikes, -1 for downward spikes."
            )
        if self.spike_template is not None:
            self.spike_template_width = len(self.spike_template)
        return self

    def template_is_fresh(self, ttl_hours: float = 24.0) -> bool:
        """Whether ``spike_template`` exists and was saved within ``ttl_hours``.

        Used by the GUIs to decide whether to skip the click-to-rebuild step.
        Returns False if there's no template, no timestamp, or the
        timestamp is older than the TTL.
        """
        if self.spike_template is None or self.template_updated_at is None:
            return False
        age = datetime.now() - self.template_updated_at
        return age < timedelta(hours=ttl_hours)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = {
            "fs": self.fs,
            "spike_template_width": self.spike_template_width,
            "hp_cutoff": self.hp_cutoff,
            "lp_cutoff": self.lp_cutoff,
            "diff_order": self.diff_order,
            "peak_threshold": self.peak_threshold,
            "distance_threshold": self.distance_threshold,
            "amplitude_threshold": self.amplitude_threshold,
            "polarity": self.polarity,
            "last_filename": self.last_filename,
        }
        if self.spike_template is not None:
            d["spike_template"] = self.spike_template.tolist()
        if self.likely_inflection_point_peak is not None:
            d["likely_inflection_point_peak"] = (
                self.likely_inflection_point_peak
            )
        if self.min_isi_samples is not None:
            d["min_isi_samples"] = self.min_isi_samples
        if self.template_updated_at is not None:
            d["template_updated_at"] = self.template_updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpikeDetectionParams:
        """Deserialize from a dict (e.g., loaded from JSON)."""
        template = d.get("spike_template")
        if template is not None:
            template = np.asarray(template, dtype=np.float64)
        ts = d.get("template_updated_at")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            fs=d["fs"],
            spike_template_width=d.get("spike_template_width", 0),
            hp_cutoff=d.get("hp_cutoff", 200.0),
            lp_cutoff=d.get("lp_cutoff", 800.0),
            diff_order=d.get("diff_order", 1),
            peak_threshold=d.get("peak_threshold", 5.0),
            distance_threshold=d.get("distance_threshold", 15.0),
            amplitude_threshold=d.get("amplitude_threshold", 0.2),
            spike_template=template,
            polarity=d.get("polarity", 1),
            likely_inflection_point_peak=d.get("likely_inflection_point_peak"),
            last_filename=d.get("last_filename", ""),
            min_isi_samples=d.get("min_isi_samples"),
            template_updated_at=ts,
        )


@dataclass
class Recording:
    """A single electrophysiology recording (replaces MATLAB ``trial`` struct).

    Original MATLAB fields -> Python attributes:
        trial.name      -> name
        trial.voltage_1 -> voltage
        trial.params.sampratein -> sample_rate
        trial.current_2 -> current (optional)
    """

    name: str
    voltage: np.ndarray
    sample_rate: float
    current: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    result: SpikeDetectionResult | None = None

    def __post_init__(self) -> None:
        self.voltage = np.asarray(self.voltage, dtype=np.float64).ravel()
        if self.current is not None:
            self.current = np.asarray(self.current, dtype=np.float64).ravel()
        if len(self.voltage) == 0:
            raise ValueError(
                "Voltage array is empty. Check that the recording file "
                "loaded correctly."
            )
        if self.sample_rate <= 0:
            raise ValueError(
                f"Sample rate must be positive, got {self.sample_rate}. "
                "Check that the recording file has a valid sample rate."
            )

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return len(self.voltage) / self.sample_rate

    @property
    def n_samples(self) -> int:
        """Number of voltage samples."""
        return len(self.voltage)

    def plot(
        self, show_spikes: bool = True
    ) -> "matplotlib.figure.Figure":
        """Plot the voltage trace with optional spike markers.

        Args:
            show_spikes: If True and a detection result
                exists, mark spike times on the trace.

        Returns:
            The figure object. Call ``plt.show()`` to
            display interactively.
        """
        import matplotlib.pyplot as plt

        time = np.arange(len(self.voltage)) / self.sample_rate
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, self.voltage, linewidth=0.5, color="0.3")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage")
        ax.set_title(self.name)

        if show_spikes and self.result is not None and self.result.n_spikes > 0:
            spike_times_sec = self.result.spike_times / self.sample_rate
            spike_voltages = self.voltage[self.result.spike_times]
            ax.plot(spike_times_sec, spike_voltages, "r|", markersize=10,
                    label=f"{self.result.n_spikes} spikes")
            ax.legend()

        fig.tight_layout()
        return fig


@dataclass
class SpikeDetectionResult:
    """Output of the spike detection pipeline.

    Original MATLAB fields -> Python attributes:
        trial.spikes              -> spike_times (sample indices)
        trial.spikes_uncorrected  -> spike_times_uncorrected
        trial.spikeDetectionParams -> params
        trial.spikeSpotChecked    -> spot_checked
    """

    spike_times: np.ndarray
    spike_times_uncorrected: np.ndarray
    params: SpikeDetectionParams
    spot_checked: bool = False

    def __post_init__(self) -> None:
        self.spike_times = np.asarray(self.spike_times, dtype=np.int64)
        self.spike_times_uncorrected = np.asarray(
            self.spike_times_uncorrected, dtype=np.int64
        )

    @property
    def n_spikes(self) -> int:
        return len(self.spike_times)

    @property
    def spike_times_seconds(self) -> np.ndarray:
        """Spike times in seconds."""
        return self.spike_times.astype(np.float64) / self.params.fs

    def summary(self) -> str:
        """Return a human-readable summary of the detection results."""
        lines = [
            f"Spike Detection Result",
            f"  Spikes found: {self.n_spikes}",
        ]
        if self.n_spikes > 0:
            times = self.spike_times_seconds
            lines.append(f"  Time range: {times[0]:.3f} - {times[-1]:.3f} s")
            if self.n_spikes >= 2:
                isis = np.diff(times)
                lines.append(
                    f"  Mean ISI: "
                    f"{np.mean(isis)*1000:.1f} ms "
                    f"(range "
                    f"{np.min(isis)*1000:.1f} - "
                    f"{np.max(isis)*1000:.1f} ms)"
                )
                lines.append(
                    f"  Mean firing rate: "
                    f"{1.0 / np.mean(isis):.1f} Hz"
                )
        lines.append(f"  Spot-checked: {'yes' if self.spot_checked else 'no'}")
        return "\n".join(lines)

    def to_dataframe(self) -> "pandas.DataFrame":
        """Convert spike times to a pandas DataFrame.

        Returns:
            DataFrame with columns: spike_index,
            spike_time_s, spike_index_uncorrected.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            )
        return pd.DataFrame({
            "spike_index": self.spike_times,
            "spike_time_s": self.spike_times_seconds,
            "spike_index_uncorrected": self.spike_times_uncorrected,
        })
