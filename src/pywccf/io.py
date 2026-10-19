"""Readers for the combined 3D/1D WCCF HDF5 files written by ``pywccf.pipeline``.

(First part of the former ``func2.py``; the amplitude-ratio part lives in
``pywccf.ratio``.)

Original ``func2`` notes on the assembly-bias summaries:

The scalar amplitude is a projection, not an average of pointwise ratios.
For sample ``s`` and total-sample template ``t`` over a declared radial range,
the default estimator is

    B_s = sum_k dr_k t_k xi_s,k / sum_k dr_k t_k^2.

Thus each linear separation interval has equal measure, isolated zeros of the
template cause no divergence, and scale-dependent shape differences are
retained in a reported residual statistic. When covariance matrices are
available, ``fit_scale_averaged_B(..., averaging="gls")`` uses the generalized
least-squares analogue. Current production HDF5 files save only marginal
standard deviations, so the high-level analysis uses the linear-radius
projection and labels all propagated errors as diagonal approximations.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D


COMBINED_FORMAT_NAME = "halo_web_combined_results"
COMBINED_SCHEMA_VERSION = "1.0"
WEB_TYPES = ("knot", "filament", "sheet", "void")
DIRECTIONS = ("e1", "e2", "e3")
DIRECTION_COLORS = {
    "e1": '#f33b1f',
    "e2": '#00ac38',
    "e3": '#5e76ff',
}

PROPERTY_SPECS = {
    "age": {
        "display": "Formation redshift",
        "low_display": "young",
        "high_display": "old",
    },
    "concentration": {
        "display": "Concentration",
        "low_display": "low concentration",
        "high_display": "high concentration",
    },
    "spin": {
        "display": "Spin",
        "low_display": "low spin",
        "high_display": "high spin",
    },
}


class WeakTemplateError(ValueError):
    """Raised when a WCCF template is too weak for a stable relative fit."""

    def __init__(self, message, *, template_rms):
        super().__init__(message)
        self.template_rms = float(template_rms)


def load_xi_3d_h5(file_path):
    """Load and validate a 3D-WCCF HDF5 result."""
    file_path = Path(file_path)
    required = (
        "xi",
        "xi_sd",
        "r1_centers",
        "r2_centers",
        "r3_centers",
    )
    with h5py.File(file_path, "r") as stream:
        missing = [name for name in required if name not in stream]
        if missing:
            raise KeyError(f"{file_path} is missing datasets: {missing}")
        xi = np.asarray(stream["xi"], dtype=np.float64)
        xi_sd = np.asarray(stream["xi_sd"], dtype=np.float64)
        centers = tuple(
            np.asarray(stream[f"r{axis}_centers"], dtype=np.float64)
            for axis in (1, 2, 3)
        )
        attrs = dict(stream.attrs)

    expected_shape = tuple(len(axis) for axis in centers)
    if xi.ndim != 3 or xi.shape != expected_shape:
        raise ValueError(
            f"{file_path}: xi shape {xi.shape} != centers {expected_shape}"
        )
    if xi_sd.shape != xi.shape:
        raise ValueError(
            f"{file_path}: xi_sd shape {xi_sd.shape} != xi {xi.shape}"
        )
    return xi, xi_sd, centers, attrs


def load_xi_1d_h5(file_path):
    """Load and validate an isotropic 1D-WCCF HDF5 result."""
    file_path = Path(file_path)
    required = ("xi", "xi_sd", "r_centers", "r_edges")
    with h5py.File(file_path, "r") as stream:
        missing = [name for name in required if name not in stream]
        if missing:
            raise KeyError(f"{file_path} is missing datasets: {missing}")
        xi = np.asarray(stream["xi"], dtype=np.float64)
        xi_sd = np.asarray(stream["xi_sd"], dtype=np.float64)
        centers = np.asarray(stream["r_centers"], dtype=np.float64)
        edges = np.asarray(stream["r_edges"], dtype=np.float64)
        attrs = dict(stream.attrs)

    if xi.ndim != 1 or xi_sd.shape != xi.shape or centers.shape != xi.shape:
        raise ValueError(
            f"{file_path}: incompatible 1D shapes "
            f"xi={xi.shape}, xi_sd={xi_sd.shape}, r={centers.shape}"
        )
    return xi, xi_sd, centers, edges, attrs


def _validate_combined_stream(
    stream, file_path, *, require_complete=True
):
    if stream.attrs.get("format_name") != COMBINED_FORMAT_NAME:
        raise ValueError(
            f"{file_path} is not a {COMBINED_FORMAT_NAME!r} file"
        )
    if stream.attrs.get("schema_version") != COMBINED_SCHEMA_VERSION:
        raise ValueError(
            f"{file_path}: unsupported schema version "
            f"{stream.attrs.get('schema_version')!r}"
        )
    if (
        require_complete
        and stream.attrs.get("validation_status") != "complete"
    ):
        run_status = stream.attrs.get("run_status", "unknown")
        failure_message = stream.attrs.get("failure_message", "")
        detail = f"; run_status={run_status!r}"
        if failure_message:
            detail += f", failure={failure_message!r}"
        raise ValueError(
            f"{file_path}: combined-file validation is not complete"
            f"{detail}"
        )


def _combined_entry_path(
    dimension,
    sample,
    tail,
    web,
    mass_bin_index,
):
    dimension = str(dimension).lower()
    sample = str(sample).lower()
    tail = str(tail).lower()
    web = str(web).lower()
    if dimension not in ("1d", "3d"):
        raise ValueError("dimension must be '1d' or '3d'")
    if web not in WEB_TYPES:
        raise ValueError(f"web must be one of {WEB_TYPES}")
    if sample == "total" and tail != "all":
        raise ValueError("The total sample uses tail='all'")
    if sample != "total" and tail not in ("low", "high"):
        raise ValueError("Property samples use tail='low' or 'high'")
    return (
        f"data/{dimension}/{sample}/{tail}/{web}/"
        f"mass_bin_{int(mass_bin_index):03d}"
    )


def _load_combined_entry(
    stream,
    *,
    dimension,
    sample,
    tail,
    web,
    mass_bin_index,
):
    entry_path = _combined_entry_path(
        dimension,
        sample,
        tail,
        web,
        mass_bin_index,
    )
    if entry_path not in stream:
        absolute_path = "/" + entry_path
        if "index/planned_record_paths" in stream:
            planned_paths = [
                value.decode("utf-8")
                if isinstance(value, bytes)
                else str(value)
                for value in stream["index/planned_record_paths"][()]
            ]
            if absolute_path in planned_paths:
                plan_index = planned_paths.index(absolute_path)
                status = stream["index/planned_status"][plan_index]
                error = stream["index/planned_error"][plan_index]
                if isinstance(status, bytes):
                    status = status.decode("utf-8")
                if isinstance(error, bytes):
                    error = error.decode("utf-8")
                detail = f"planned status={status!r}"
                if error:
                    detail += f", error={error!r}"
            else:
                detail = "this record was not requested by the run plan"
            raise KeyError(
                f"{stream.filename} has no combined result at "
                f"{absolute_path}; {detail}"
            )
        raise KeyError(
            f"{stream.filename} has no combined result at /{entry_path}"
        )
    entry = stream[entry_path]
    original = entry["original"]
    datasets = {
        name: np.asarray(value)
        for name, value in original.items()
        if isinstance(value, h5py.Dataset)
    }
    required = {"xi", "xi_sd"}
    missing = required - set(datasets)
    if missing:
        raise KeyError(
            f"{stream.filename}: /{entry_path}/original is missing "
            f"{sorted(missing)}"
        )

    xi = np.asarray(datasets["xi"], dtype=np.float64)
    xi_sd = np.asarray(datasets["xi_sd"], dtype=np.float64)
    if xi_sd.shape != xi.shape:
        raise ValueError(
            f"{stream.filename}: /{entry_path} has incompatible xi arrays"
        )

    if dimension == "3d":
        center_names = tuple(
            f"r{axis}_centers" for axis in (1, 2, 3)
        )
        edge_names = tuple(f"r{axis}_edges" for axis in (1, 2, 3))
        missing_grid = (
            set(center_names) | set(edge_names)
        ) - set(datasets)
        if missing_grid:
            raise KeyError(
                f"{stream.filename}: /{entry_path} is missing "
                f"{sorted(missing_grid)}"
            )
        centers = tuple(
            np.asarray(datasets[name], dtype=np.float64)
            for name in center_names
        )
        edges = tuple(
            np.asarray(datasets[name], dtype=np.float64)
            for name in edge_names
        )
        expected_shape = tuple(len(axis) for axis in centers)
        if xi.ndim != 3 or xi.shape != expected_shape:
            raise ValueError(
                f"{stream.filename}: /{entry_path} has xi shape "
                f"{xi.shape}, expected {expected_shape}"
            )
    else:
        missing_grid = {"r_centers", "r_edges"} - set(datasets)
        if missing_grid:
            raise KeyError(
                f"{stream.filename}: /{entry_path} is missing "
                f"{sorted(missing_grid)}"
            )
        centers = np.asarray(
            datasets["r_centers"], dtype=np.float64
        )
        edges = np.asarray(datasets["r_edges"], dtype=np.float64)
        if xi.ndim != 1 or xi.shape != centers.shape:
            raise ValueError(
                f"{stream.filename}: /{entry_path} has incompatible "
                "1D arrays"
            )

    return {
        "xi": xi,
        "xi_sd": xi_sd,
        "centers": centers,
        "edges": edges,
        "datasets": datasets,
        "attrs": dict(original.attrs),
        "container_attrs": dict(entry.attrs),
        "entry_path": "/" + entry_path,
    }


def load_combined_h5(
    file_path,
    *,
    dimension,
    sample,
    tail,
    web,
    mass_bin_index,
):
    """Load one normalized record from a combined result file.

    Examples
    --------
    ``load_combined_h5(path, dimension="3d", sample="age", tail="high",
    web="filament", mass_bin_index=1)``.

    The returned dictionary includes the standard arrays, every copied source
    dataset under ``datasets``, original HDF5 attributes under ``attrs``, and
    combined-container provenance under ``container_attrs``.
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as stream:
        _validate_combined_stream(stream, file_path)
        return _load_combined_entry(
            stream,
            dimension=dimension,
            sample=sample,
            tail=tail,
            web=web,
            mass_bin_index=mass_bin_index,
        )


def combined_h5_inventory(file_path):
    """Return the main metadata and source/log indices of a combined file."""
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as stream:
        _validate_combined_stream(
            stream, file_path, require_complete=False
        )

        def strings(name, default=None):
            if name not in stream:
                return [] if default is None else default
            return [
                value.decode("utf-8")
                if isinstance(value, bytes)
                else str(value)
                for value in stream[name][()]
            ]

        samples = strings("metadata/samples")
        contains_subproperties = bool(
            stream["metadata"].attrs.get(
                "contains_subproperty_samples",
                any(sample != "total" for sample in samples),
            )
        )
        return {
            "file_path": str(file_path),
            "run_status": str(
                stream.attrs.get("run_status", "converted")
            ),
            "validation_status": str(
                stream.attrs["validation_status"]
            ),
            "smoothing_scale_mpc_h": float(
                stream.attrs["smoothing_scale_mpc_h"]
            ),
            "lambda_threshold": (
                float(stream.attrs["lambda_threshold"])
                if "lambda_threshold" in stream.attrs
                else None
            ),
            "field_lambda_threshold": (
                float(stream.attrs["field_lambda_threshold"])
                if "field_lambda_threshold" in stream.attrs
                else None
            ),
            "log_discovery_status": str(
                stream.attrs.get(
                    "log_discovery_status",
                    "complete"
                    if int(stream.attrs.get("log_count", 0)) > 0
                    else "not_recorded",
                )
            ),
            "mass_bin_edges": np.asarray(
                stream["metadata/mass_bin_edges"], dtype=np.float64
            ),
            # median log10 M of each total mass bin (run.py >= open-bin
            # support); None for older files
            "mass_bin_medians": (
                np.asarray(
                    stream["metadata/mass_bin_median_log10_msun_h"],
                    dtype=np.float64,
                )
                if "metadata/mass_bin_median_log10_msun_h" in stream
                else None
            ),
            "radial_bins": np.asarray(
                stream["metadata/radial_bins"], dtype=np.float64
            ),
            "samples": samples,
            "contains_subproperty_samples": contains_subproperties,
            "sample_scope": str(
                stream["metadata"].attrs.get(
                    "sample_scope",
                    "total_and_subproperties"
                    if contains_subproperties
                    else "total_only",
                )
            ),
            "dimensions": strings("metadata/dimensions"),
            "record_paths": strings("index/record_paths"),
            "source_paths": strings("index/source_paths"),
            "source_sha256": strings("index/source_sha256"),
            "log_paths": strings("index/log_paths"),
            "sidecar_paths": strings("index/sidecar_paths"),
            "planned_record_paths": strings(
                "index/planned_record_paths"
            ),
            "planned_status": strings("index/planned_status"),
            "planned_error": strings("index/planned_error"),
        }


def load_combined_log(file_path, log_path):
    """Return one embedded log as exact bytes.

    ``log_path`` can be the stored relative path or its unique basename.
    Decode the returned bytes with ``.decode("utf-8", errors="replace")``
    when text is desired.
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as stream:
        _validate_combined_stream(
            stream, file_path, require_complete=False
        )
        stored_paths = [
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
            for value in stream["index/log_paths"][()]
        ]
        matches = [
            index
            for index, stored in enumerate(stored_paths)
            if stored == str(log_path)
            or Path(stored).name == str(log_path)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected one log matching {log_path!r}; "
                f"found {[stored_paths[index] for index in matches]}"
            )
        dataset = stream[
            f"provenance/logs/file_{matches[0]:03d}/raw_bytes"
        ]
        return bytes(np.asarray(dataset, dtype=np.uint8))
