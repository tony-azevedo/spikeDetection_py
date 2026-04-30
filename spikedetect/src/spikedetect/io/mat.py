"""Load and save MATLAB .mat trial structs.

Supports both MATLAB v5/v7 (via scipy.io.loadmat) and v7.3 HDF5
format (via h5py). The MATLAB trial struct fields are mapped to
the :class:`~spikedetect.models.Recording` and
:class:`~spikedetect.models.SpikeDetectionResult` data models.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from spikedetect.models import (
    Recording,
    SpikeDetectionParams,
    SpikeDetectionResult,
)


def _parse_iso_datetime(s: str) -> datetime | None:
    """Parse an ISO-8601 timestamp written by :meth:`SpikeDetectionParams.to_dict`,
    returning None on empty / unparseable input rather than raising.
    """
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _read_h5_string(dataset: "h5py.Dataset") -> str:
    """Read a MATLAB string stored as uint16 codes in HDF5."""
    return "".join(chr(c) for c in np.asarray(dataset).ravel())


def _read_h5_scalar(dataset: "h5py.Dataset") -> float:
    """Read a scalar value from an HDF5 dataset."""
    return float(np.asarray(dataset).ravel()[0])


def _read_h5_array_or_empty(
    dataset: "h5py.Dataset", dtype=np.int64
) -> np.ndarray:
    """Read a 1-D array, honoring MATLAB's empty-array marker.

    ``hdf5storage`` (and MATLAB v7.3 ``save``) encode an empty array by
    storing its shape vector as the dataset payload and setting an
    ``MATLAB_empty=1`` attribute on the dataset. A naive
    ``np.asarray(ds).ravel()`` therefore returns the shape vector (e.g.
    ``[0, 1]``) as if it were real data — yielding phantom values at the
    start of the trace. We detect the marker and return a true empty
    array of the requested dtype.
    """
    if int(dataset.attrs.get("MATLAB_empty", 0)):
        return np.array([], dtype=dtype)
    return np.asarray(dataset).ravel().astype(dtype)


def _load_params_h5(
    params_group: "h5py.Group",
) -> SpikeDetectionParams:
    """Build SpikeDetectionParams from an HDF5 group.

    Args:
        params_group: The ``spikeDetectionParams`` group
            from the .mat file.

    Returns:
        Populated detection parameters.
    """
    def scalar(name: str, default: float = 0.0) -> float:
        if name in params_group:
            return _read_h5_scalar(params_group[name])
        return default

    template = None
    if "spikeTemplate" in params_group:
        template = (
            np.asarray(params_group["spikeTemplate"])
            .ravel()
            .astype(np.float64)
        )

    last_filename = ""
    if "lastfilename" in params_group:
        last_filename = _read_h5_string(
            params_group["lastfilename"]
        )

    likelyiflpntpeak = None
    if "likelyiflpntpeak" in params_group:
        likelyiflpntpeak = int(scalar("likelyiflpntpeak"))

    template_updated_at = None
    if "templateUpdatedAt" in params_group:
        template_updated_at = _parse_iso_datetime(
            _read_h5_string(params_group["templateUpdatedAt"])
        )

    min_isi_samples = None
    if "min_isi_samples" in params_group:
        min_isi_samples = int(scalar("min_isi_samples"))

    return SpikeDetectionParams(
        fs=scalar("fs", 10000.0),
        spike_template_width=int(
            scalar("spikeTemplateWidth", 0)
        ),
        hp_cutoff=scalar("hp_cutoff", 200.0),
        lp_cutoff=scalar("lp_cutoff", 800.0),
        diff_order=int(scalar("diff", 1)),
        peak_threshold=scalar("peak_threshold", 5.0),
        distance_threshold=scalar(
            "Distance_threshold", 15.0
        ),
        amplitude_threshold=scalar(
            "Amplitude_threshold", 0.2
        ),
        spike_template=template,
        polarity=int(scalar("polarity", 1)),
        likely_inflection_point_peak=likelyiflpntpeak,
        last_filename=last_filename,
        min_isi_samples=min_isi_samples,
        template_updated_at=template_updated_at,
    )


def _load_params_scipy(d: dict) -> SpikeDetectionParams:
    """Build SpikeDetectionParams from a scipy struct dict.

    Args:
        d: The ``spikeDetectionParams`` dict from
            scipy.io.loadmat.

    Returns:
        Populated detection parameters.
    """
    def scalar(name: str, default: float = 0.0) -> float:
        val = d.get(name, default)
        if isinstance(val, np.ndarray):
            return float(val.ravel()[0])
        return float(val)

    template = d.get("spikeTemplate")
    if template is not None:
        template = np.asarray(
            template, dtype=np.float64
        ).ravel()

    last_filename = str(d.get("lastfilename", ""))

    likelyiflpntpeak = None
    if "likelyiflpntpeak" in d:
        likelyiflpntpeak = int(scalar("likelyiflpntpeak"))

    template_updated_at = None
    if "templateUpdatedAt" in d:
        template_updated_at = _parse_iso_datetime(
            str(d["templateUpdatedAt"])
        )

    min_isi_samples = None
    if "min_isi_samples" in d:
        min_isi_samples = int(scalar("min_isi_samples"))

    return SpikeDetectionParams(
        fs=scalar("fs", 10000.0),
        spike_template_width=int(
            scalar("spikeTemplateWidth", 0)
        ),
        hp_cutoff=scalar("hp_cutoff", 200.0),
        lp_cutoff=scalar("lp_cutoff", 800.0),
        diff_order=int(scalar("diff", 1)),
        peak_threshold=scalar("peak_threshold", 5.0),
        distance_threshold=scalar(
            "Distance_threshold", 15.0
        ),
        amplitude_threshold=scalar(
            "Amplitude_threshold", 0.2
        ),
        spike_template=template,
        polarity=int(scalar("polarity", 1)),
        likely_inflection_point_peak=likelyiflpntpeak,
        last_filename=last_filename,
        min_isi_samples=min_isi_samples,
        template_updated_at=template_updated_at,
    )


def _load_h5(path: Path) -> Recording:
    """Load a MATLAB v7.3 (HDF5) .mat file.

    Args:
        path: Path to the .mat file.

    Returns:
        The loaded recording with any existing spike
        detection results.
    """
    import h5py

    with h5py.File(path, "r") as f:
        # Name
        name = (
            _read_h5_string(f["name"])
            if "name" in f
            else str(path)
        )

        # Sample rate
        sample_rate = _read_h5_scalar(
            f["params"]["sampratein"]
        )

        # Voltage (required)
        voltage = (
            np.asarray(f["voltage_1"])
            .ravel()
            .astype(np.float64)
        )

        # Current (optional)
        current = None
        if "current_2" in f:
            current = (
                np.asarray(f["current_2"])
                .ravel()
                .astype(np.float64)
            )

        # Spike detection results (optional)
        result = None
        if "spikeDetectionParams" in f:
            params = _load_params_h5(
                f["spikeDetectionParams"]
            )

            spike_times = np.array([], dtype=np.int64)
            if "spikes" in f:
                spike_times = _read_h5_array_or_empty(
                    f["spikes"], dtype=np.int64
                )

            spike_times_uncorrected = np.array(
                [], dtype=np.int64
            )
            if "spikes_uncorrected" in f:
                spike_times_uncorrected = (
                    _read_h5_array_or_empty(
                        f["spikes_uncorrected"],
                        dtype=np.int64,
                    )
                )

            spot_checked = False
            if "spikeSpotChecked" in f:
                spot_checked = bool(
                    _read_h5_scalar(f["spikeSpotChecked"])
                )

            result = SpikeDetectionResult(
                spike_times=spike_times,
                spike_times_uncorrected=(
                    spike_times_uncorrected
                ),
                params=params,
                spot_checked=spot_checked,
            )

    return Recording(
        name=name,
        voltage=voltage,
        sample_rate=sample_rate,
        current=current,
        result=result,
    )


def _load_scipy(path: Path) -> Recording:
    """Load a MATLAB v5/v7 .mat file using scipy.

    Args:
        path: Path to the .mat file.

    Returns:
        The loaded recording with any existing spike
        detection results.
    """
    import scipy.io

    data = scipy.io.loadmat(
        str(path), squeeze_me=True, struct_as_record=True
    )

    # Name
    name = str(data.get("name", str(path)))

    # Sample rate - try params.sampratein
    params_struct = data.get("params", {})
    if isinstance(params_struct, np.ndarray):
        # struct_as_record=True with squeeze_me gives
        # a 0-d structured array
        names = params_struct.dtype.names
        if names and "sampratein" in names:
            sample_rate = float(
                np.asarray(
                    params_struct["sampratein"]
                ).ravel()[0]
            )
        else:
            sample_rate = 10000.0
    elif (
        isinstance(params_struct, dict)
        and "sampratein" in params_struct
    ):
        sample_rate = float(
            np.asarray(
                params_struct["sampratein"]
            ).ravel()[0]
        )
    else:
        sample_rate = 10000.0

    # Voltage (required)
    voltage = np.asarray(
        data["voltage_1"], dtype=np.float64
    ).ravel()

    # Current (optional)
    current = None
    if "current_2" in data:
        current = np.asarray(
            data["current_2"], dtype=np.float64
        ).ravel()

    # Spike detection results (optional)
    result = None
    if "spikeDetectionParams" in data:
        sdp = data["spikeDetectionParams"]
        # Convert structured array to dict
        if isinstance(sdp, np.ndarray) and sdp.dtype.names:
            sdp_dict = {
                name: sdp[name].item()
                for name in sdp.dtype.names
            }
        elif isinstance(sdp, dict):
            sdp_dict = sdp
        else:
            sdp_dict = {}

        if sdp_dict:
            params = _load_params_scipy(sdp_dict)

            spike_times = np.array([], dtype=np.int64)
            if "spikes" in data:
                spike_times = (
                    np.asarray(data["spikes"])
                    .ravel()
                    .astype(np.int64)
                )

            spike_times_uncorrected = np.array(
                [], dtype=np.int64
            )
            if "spikes_uncorrected" in data:
                spike_times_uncorrected = (
                    np.asarray(data["spikes_uncorrected"])
                    .ravel()
                    .astype(np.int64)
                )

            spot_checked = False
            if "spikeSpotChecked" in data:
                spot_checked = bool(
                    np.asarray(
                        data["spikeSpotChecked"]
                    ).ravel()[0]
                )

            result = SpikeDetectionResult(
                spike_times=spike_times,
                spike_times_uncorrected=(
                    spike_times_uncorrected
                ),
                params=params,
                spot_checked=spot_checked,
            )

    return Recording(
        name=name,
        voltage=voltage,
        sample_rate=sample_rate,
        current=current,
        result=result,
    )


def load_recording(path: str | Path) -> Recording:
    """Load a MATLAB trial .mat file into a Recording.

    Automatically detects whether the file is MATLAB v5/v7
    (scipy) or v7.3 HDF5 format and uses the appropriate
    reader.

    Args:
        path: Path to the .mat file.

    Returns:
        The loaded recording, including any existing spike
        detection results stored in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read by either
            method.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"MAT file not found: {path}\n"
            "Check that the file path is correct "
            "and the file exists."
        )
    if path.suffix.lower() not in (".mat",):
        raise ValueError(
            f"Expected a .mat file, got '{path.suffix}'. "
            "For ABF files, use load_abf() instead."
        )

    # Try scipy first (v5/v7), fall back to h5py (v7.3)
    try:
        import scipy.io

        scipy.io.loadmat(
            str(path), variable_names=[]
        )  # quick header check
        return _load_scipy(path)
    except (NotImplementedError, ValueError):
        # v7.3 HDF5 format, or file written by h5py
        return _load_h5(path)


def save_result(
    path: str | Path, recording: Recording
) -> None:
    """Save spike detection results back to a .mat file.

    Writes a MATLAB-compatible v7.3 (HDF5) file containing
    the recording data and any detection results.

    Args:
        path: Output file path.
        recording: The recording with results to save.

    Raises:
        ValueError: If the recording has no detection
            result to save.
    """
    import h5py

    path = Path(path)

    with h5py.File(path, "w") as f:
        # Name
        name_codes = np.array(
            [ord(c) for c in recording.name],
            dtype=np.uint16,
        )
        f.create_dataset(
            "name", data=name_codes.reshape(-1, 1)
        )

        # Voltage
        f.create_dataset(
            "voltage_1",
            data=recording.voltage.reshape(-1, 1),
        )

        # Current
        if recording.current is not None:
            f.create_dataset(
                "current_2",
                data=recording.current.reshape(-1, 1),
            )

        # Params group
        pg = f.create_group("params")
        pg.create_dataset(
            "sampratein",
            data=np.array(
                [[recording.sample_rate]],
                dtype=np.float64,
            ),
        )

        # Spike detection results
        if recording.result is not None:
            res = recording.result

            f.create_dataset(
                "spikes",
                data=(
                    res.spike_times
                    .astype(np.float64)
                    .reshape(-1, 1)
                ),
            )
            f.create_dataset(
                "spikes_uncorrected",
                data=(
                    res.spike_times_uncorrected
                    .astype(np.float64)
                    .reshape(-1, 1)
                ),
            )
            f.create_dataset(
                "spikeSpotChecked",
                data=np.array(
                    [[float(res.spot_checked)]],
                    dtype=np.float64,
                ),
            )

            # spikeDetectionParams group
            sdp = f.create_group(
                "spikeDetectionParams"
            )
            p = res.params
            for field_name, matlab_name in [
                ("fs", "fs"),
                (
                    "spike_template_width",
                    "spikeTemplateWidth",
                ),
                ("hp_cutoff", "hp_cutoff"),
                ("lp_cutoff", "lp_cutoff"),
                ("diff_order", "diff"),
                ("peak_threshold", "peak_threshold"),
                (
                    "distance_threshold",
                    "Distance_threshold",
                ),
                (
                    "amplitude_threshold",
                    "Amplitude_threshold",
                ),
                ("polarity", "polarity"),
            ]:
                val = getattr(p, field_name)
                sdp.create_dataset(
                    matlab_name,
                    data=np.array(
                        [[float(val)]],
                        dtype=np.float64,
                    ),
                )

            if p.likely_inflection_point_peak is not None:
                sdp.create_dataset(
                    "likelyiflpntpeak",
                    data=np.array(
                        [[float(
                            p.likely_inflection_point_peak
                        )]],
                        dtype=np.float64,
                    ),
                )

            if p.spike_template is not None:
                sdp.create_dataset(
                    "spikeTemplate",
                    data=p.spike_template.reshape(-1, 1),
                )

            if p.last_filename:
                lf_codes = np.array(
                    [ord(c) for c in p.last_filename],
                    dtype=np.uint16,
                )
                sdp.create_dataset(
                    "lastfilename",
                    data=lf_codes.reshape(-1, 1),
                )

            if p.template_updated_at is not None:
                ts_codes = np.array(
                    [ord(c) for c in p.template_updated_at.isoformat()],
                    dtype=np.uint16,
                )
                sdp.create_dataset(
                    "templateUpdatedAt",
                    data=ts_codes.reshape(-1, 1),
                )

            if p.min_isi_samples is not None:
                sdp.create_dataset(
                    "min_isi_samples",
                    data=np.array(
                        [[float(p.min_isi_samples)]],
                        dtype=np.float64,
                    ),
                )
