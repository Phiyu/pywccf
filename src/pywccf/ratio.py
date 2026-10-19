"""Scale-averaged amplitudes and amplitude-ratio (AR) statistics of the 3D-WCCF.

(Second part of the former ``func2.py``; the HDF5 readers live in ``pywccf.io``.)
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

from .io import (  # noqa: F401  (re-exported: constants and readers)
    DIRECTIONS,
    DIRECTION_COLORS,
    PROPERTY_SPECS,
    WEB_TYPES,
    WeakTemplateError,
    _combined_entry_path,
    _load_combined_entry,
    _validate_combined_stream,
    combined_h5_inventory,
    load_combined_h5,
    load_xi_3d_h5,
)


def _directional_slice(xi, xi_sd, direction):
    if direction == "e1":
        return xi[:, 0, 0], xi_sd[:, 0, 0]
    if direction == "e2":
        return xi[0, :, 0], xi_sd[0, :, 0]
    if direction == "e3":
        return xi[0, 0, :], xi_sd[0, 0, :]
    raise ValueError(f"Unknown direction: {direction}")


def _matching_grid(first, second, *, name):
    # The legacy total 3D files retain full-precision logarithmic bin centers,
    # while the YAML-driven split runs store the same grid rounded to about
    # five decimal places.
    if len(first) != len(second) or not np.allclose(
        first, second, rtol=5.0e-6, atol=5.0e-8
    ):
        raise ValueError(f"Mismatched {name} grids")


def _profile_mask(r, sample, sample_sd, template, template_sd, r_range):
    r = np.asarray(r, dtype=np.float64)
    sample = np.asarray(sample, dtype=np.float64)
    sample_sd = np.asarray(sample_sd, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    template_sd = np.asarray(template_sd, dtype=np.float64)
    if not (
        r.shape
        == sample.shape
        == sample_sd.shape
        == template.shape
        == template_sd.shape
    ):
        raise ValueError("Profile arrays must have identical shapes")

    r_lo, r_hi = r_range
    mask = (
        np.isfinite(r)
        & np.isfinite(sample)
        & np.isfinite(sample_sd)
        & np.isfinite(template)
        & np.isfinite(template_sd)
        & (sample_sd > 0.0)
        & (template_sd > 0.0)
        & (r >= float(r_lo))
        & (r <= float(r_hi))
    )
    return mask


def _trapezoid_weights_log_radius(radius):
    """Return trapezoidal quadrature weights for integration over ln(r)."""
    radius = np.asarray(radius, dtype=np.float64)
    if radius.ndim != 1 or len(radius) < 2:
        raise ValueError("At least two radial centers are required")
    if np.any(radius <= 0.0) or np.any(np.diff(radius) <= 0.0):
        raise ValueError("Radial centers must be positive and increasing")
    intervals = np.diff(np.log(radius))
    weights = np.empty_like(radius)
    weights[0] = 0.5 * intervals[0]
    weights[-1] = 0.5 * intervals[-1]
    weights[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
    return weights


def _trapezoid_weights_radius(radius):
    """Return trapezoidal quadrature weights for integration over r."""
    radius = np.asarray(radius, dtype=np.float64)
    if radius.ndim != 1 or len(radius) < 2:
        raise ValueError("At least two radial centers are required")
    if np.any(~np.isfinite(radius)) or np.any(np.diff(radius) <= 0.0):
        raise ValueError("Radial centers must be finite and increasing")
    intervals = np.diff(radius)
    weights = np.empty_like(radius)
    weights[0] = 0.5 * intervals[0]
    weights[-1] = 0.5 * intervals[-1]
    weights[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
    return weights


def fit_scale_averaged_B(
    sample,
    sample_sd,
    template,
    template_sd,
    r,
    *,
    r_range=(5.0, 20.0),
    averaging="linear_projection",
    covariance_sample=None,
    covariance_template=None,
    min_template_rms=None,
    min_points=3,
    max_iterations=50,
    tolerance=1.0e-8,
):
    """Fit a single effective WCCF amplitude across a radial interval.

    Parameters
    ----------
    averaging : {"linear_projection", "log_projection", "diagonal_eiv", "gls"}
        ``linear_projection`` is the production default. It gives equal
        measure per linear separation interval and uses the saved diagonal
        errors only for uncertainty propagation. ``log_projection`` retains
        the previous equal-logarithmic-interval definition for comparisons.
        ``diagonal_eiv`` iteratively uses inverse effective variance. ``gls``
        requires full sample and template covariance matrices.
    min_template_rms : float or None
        Optional absolute floor on the linearly weighted RMS of the
        total-sample template. Fits below the floor are returned as weak
        templates by the high-level interface instead of producing an
        arbitrarily large relative amplitude. No universal floor is imposed
        by default because WCCF amplitudes depend on web and sample choices.

    Returns
    -------
    dict
        ``B``, ``B_err_diag``, ``n_scale``, ``shape_residual``,
        ``chi2_red_diag``, and the selected radial limits.
    """
    sample = np.asarray(sample, dtype=np.float64)
    sample_sd = np.asarray(sample_sd, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    template_sd = np.asarray(template_sd, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    mask = _profile_mask(
        r, sample, sample_sd, template, template_sd, r_range
    )
    if int(mask.sum()) < int(min_points):
        raise ValueError(
            f"Only {int(mask.sum())} valid scale bins in r={r_range}; "
            f"need at least {min_points}"
        )

    y = sample[mask]
    sy = sample_sd[mask]
    t = template[mask]
    st = template_sd[mask]
    radius = r[mask]
    linear_weights = _trapezoid_weights_radius(radius)
    log_weights = _trapezoid_weights_log_radius(radius)
    template_rms = np.sqrt(
        np.sum(linear_weights * t**2) / np.sum(linear_weights)
    )
    template_noise_rms = np.sqrt(
        np.sum(linear_weights * st**2) / np.sum(linear_weights)
    )
    template_snr_diag = np.sqrt(np.sum((t / st) ** 2))
    if min_template_rms is not None:
        min_template_rms = float(min_template_rms)
        if not np.isfinite(min_template_rms) or min_template_rms < 0.0:
            raise ValueError("min_template_rms must be finite and non-negative")
        if template_rms < min_template_rms:
            raise WeakTemplateError(
                "The selected total-sample template is too weak for a "
                f"stable relative amplitude: RMS={template_rms:.6g} < "
                f"min_template_rms={min_template_rms:.6g}",
                template_rms=template_rms,
            )

    if averaging in ("linear_projection", "log_projection"):
        weights = (
            linear_weights
            if averaging == "linear_projection"
            else log_weights
        )
        denominator = np.sum(weights * t**2)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("The selected total-sample template has no power")
        amplitude = np.sum(weights * t * y) / denominator
        variance_eff = sy**2 + amplitude**2 * st**2
        amplitude_variance = (
            np.sum((weights * t) ** 2 * variance_eff) / denominator**2
        )
    elif averaging == "diagonal_eiv":
        denominator = np.sum(linear_weights * t**2)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("The selected total-sample template has no power")
        amplitude = np.sum(linear_weights * t * y) / denominator
        for _ in range(int(max_iterations)):
            variance_eff = sy**2 + amplitude**2 * st**2
            weights = 1.0 / variance_eff
            denominator = np.sum(weights * t**2)
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("Invalid diagonal EIV denominator")
            updated = np.sum(weights * t * y) / denominator
            if abs(updated - amplitude) <= tolerance * max(
                1.0, abs(updated)
            ):
                amplitude = updated
                break
            amplitude = updated
        variance_eff = sy**2 + amplitude**2 * st**2
        amplitude_variance = 1.0 / denominator
    elif averaging == "gls":
        if covariance_sample is None or covariance_template is None:
            raise ValueError(
                "averaging='gls' requires both covariance matrices"
            )
        selected_ids = np.flatnonzero(mask)
        covariance_sample = np.asarray(
            covariance_sample, dtype=np.float64
        )[np.ix_(selected_ids, selected_ids)]
        covariance_template = np.asarray(
            covariance_template, dtype=np.float64
        )[np.ix_(selected_ids, selected_ids)]
        amplitude = np.sum(linear_weights * t * y) / np.sum(
            linear_weights * t**2
        )
        for _ in range(int(max_iterations)):
            covariance = (
                covariance_sample + amplitude**2 * covariance_template
            )
            solved_t = np.linalg.solve(covariance, t)
            solved_y = np.linalg.solve(covariance, y)
            denominator = np.dot(t, solved_t)
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("Invalid GLS denominator")
            updated = np.dot(t, solved_y) / denominator
            if abs(updated - amplitude) <= tolerance * max(
                1.0, abs(updated)
            ):
                amplitude = updated
                break
            amplitude = updated
        variance_eff = sy**2 + amplitude**2 * st**2
        amplitude_variance = 1.0 / denominator
        weights = linear_weights
    else:
        raise ValueError(
            "averaging must be linear_projection, log_projection, "
            "diagonal_eiv, or gls"
        )

    residual = y - amplitude * t
    residual_weights = (
        log_weights if averaging == "log_projection" else linear_weights
    )
    residual_denominator = np.sum(residual_weights * y**2)
    if residual_denominator > 0.0:
        shape_residual = np.sqrt(
            np.sum(residual_weights * residual**2)
            / residual_denominator
        )
    else:
        shape_residual = np.nan
    chi2 = np.sum(residual**2 / variance_eff)
    dof = max(len(y) - 1, 1)

    return {
        "B": float(amplitude),
        "B_err_diag": float(np.sqrt(max(amplitude_variance, 0.0))),
        "n_scale": int(len(y)),
        "shape_residual": float(shape_residual),
        "chi2_red_diag": float(chi2 / dof),
        "r_min": float(radius.min()),
        "r_max": float(radius.max()),
        "averaging": averaging,
        "template_rms": float(template_rms),
        "template_noise_rms": float(template_noise_rms),
        "template_snr_diag": float(template_snr_diag),
        "projection_denominator": float(denominator),
    }


def _result_path(folder, web, dimension, mass_bin_index, lab):
    return (
        Path(folder)
        / f"{web}_{lab}_xi_{dimension}_{int(mass_bin_index)}.h5"
    )


def _available_indices(folder, dimension, lab):
    indices = set()
    for file_path in Path(folder).glob(
        f"knot_{lab}_xi_{dimension}_*.h5"
    ):
        try:
            indices.add(int(file_path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return sorted(indices)


def mass_bin_centers(inventory_or_file):
    """x positions of the mass bins: the midpoints of the edges, except for
    an open-ended bin (upper edge +inf, e.g. log M >= 14), which takes the
    median log10 mass of its haloes stored by run.py (or, for a file
    without medians, its lower edge)."""
    inventory = (
        inventory_or_file
        if isinstance(inventory_or_file, dict)
        else combined_h5_inventory(inventory_or_file)
    )
    edges = np.asarray(inventory["mass_bin_edges"], dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    open_bins = ~np.isfinite(centers)
    if open_bins.any():
        medians = inventory.get("mass_bin_medians")
        if medians is not None:
            centers[open_bins] = np.asarray(medians, dtype=np.float64)[
                open_bins
            ]
        else:
            centers[open_bins] = edges[:-1][open_bins]
    return centers


def mass_range_tex(mass_lo, mass_hi, fmt=".1f"):
    """'a <= log10 M < b', or 'log10 M >= a' for an open-ended bin."""
    if np.isfinite(mass_hi):
        return (
            rf"{mass_lo:{fmt}}\leq\log_{{10}}M<{mass_hi:{fmt}}"
        )
    return rf"\log_{{10}}M\geq{mass_lo:{fmt}}"


def _mass_interval(attrs, mass_bin_index):
    lo = attrs.get("mass_bin_lower_log10_msun_h")
    hi = attrs.get("mass_bin_upper_log10_msun_h")
    if lo is not None and hi is not None:
        return float(lo), float(hi)

    resolved = attrs.get("resolved_config_yaml")
    if isinstance(resolved, bytes):
        resolved = resolved.decode("utf-8")
    if resolved is not None:
        try:
            config = yaml.safe_load(resolved)
            edges = config["sample"]["mass_bin_edges"]
            return float(edges[mass_bin_index]), float(
                edges[mass_bin_index + 1]
            )
        except (KeyError, IndexError, TypeError, yaml.YAMLError):
            pass
    raise ValueError(
        f"Cannot determine mass interval for index {mass_bin_index}"
    )


def _store_fit(record, sample_label, component, fit):
    prefix = f"B_{sample_label}_{component}"
    record[prefix] = fit["B"]
    record[f"{prefix}_err_diag"] = fit["B_err_diag"]
    record[f"{prefix}_n_scale"] = fit["n_scale"]
    record[f"{prefix}_shape_residual"] = fit["shape_residual"]
    record[f"{prefix}_chi2_red_diag"] = fit["chi2_red_diag"]
    record[f"{prefix}_template_rms"] = fit["template_rms"]
    record[f"{prefix}_template_noise_rms"] = fit[
        "template_noise_rms"
    ]
    record[f"{prefix}_template_snr_diag"] = fit[
        "template_snr_diag"
    ]
    record[f"{prefix}_projection_denominator"] = fit[
        "projection_denominator"
    ]
    record[f"{prefix}_fit_status"] = fit.get("fit_status", "ok")
    record[f"{prefix}_fit_error"] = fit.get("fit_error", "")


def _fit_or_nan(*args, strict=False, **kwargs):
    try:
        fit = fit_scale_averaged_B(*args, **kwargs)
        fit["fit_status"] = "ok"
        fit["fit_error"] = ""
        return fit
    except (ValueError, np.linalg.LinAlgError) as error:
        if strict:
            raise
        fit_status = (
            "weak_template"
            if isinstance(error, WeakTemplateError)
            else "invalid"
        )
        return {
            "B": np.nan,
            "B_err_diag": np.nan,
            "n_scale": 0,
            "shape_residual": np.nan,
            "chi2_red_diag": np.nan,
            "r_min": np.nan,
            "r_max": np.nan,
            "averaging": kwargs.get("averaging", "linear_projection"),
            "template_rms": float(
                getattr(error, "template_rms", np.nan)
            ),
            "template_noise_rms": np.nan,
            "template_snr_diag": np.nan,
            "projection_denominator": np.nan,
            "fit_status": fit_status,
            "fit_error": str(error),
        }


def _log_ratio(
    value_high,
    error_high,
    value_low,
    error_low,
    *,
    min_amplitude_snr=2.0,
):
    if (
        not np.isfinite(value_high)
        or not np.isfinite(value_low)
        or value_high <= 0.0
        or value_low <= 0.0
    ):
        return np.nan, np.nan
    if min_amplitude_snr is not None:
        min_amplitude_snr = float(min_amplitude_snr)
        if min_amplitude_snr < 0.0 or not np.isfinite(min_amplitude_snr):
            raise ValueError(
                "min_amplitude_snr must be finite and non-negative"
            )
        if (
            not np.isfinite(error_high)
            or not np.isfinite(error_low)
            or error_high <= 0.0
            or error_low <= 0.0
            or value_high / error_high < min_amplitude_snr
            or value_low / error_low < min_amplitude_snr
        ):
            return np.nan, np.nan
    value = np.log(value_high / value_low)
    error = np.sqrt(
        (error_high / value_high) ** 2 + (error_low / value_low) ** 2
    )
    return float(value), float(error)


def calculate_assembly_metrics(
    combined_file,
    *,
    properties=("age", "concentration", "spin"),
    property_specs=None,
    web_types=WEB_TYPES,
    mass_bin_indices=None,
    r_range=(5.0, 20.0),
    averaging="linear_projection",
    min_template_rms=None,
    min_log_amplitude_snr=2.0,
    strict=False,
):
    """Calculate assembly metrics from one combined HDF5 file.

    Definitions
    -----------
    ``B_s^alpha`` projects property sample ``s`` onto the matching total-sample
    WCCF template. ``alpha=iso`` uses the spherical 1D-WCCF; ``e1/e2/e3`` use
    directional slices of the 3D-WCCF with the other coordinates fixed in the
    underflow bin.

    ``S_iso = ln(B_high^iso / B_low^iso)``.

    ``S_alpha = ln(B_high^alpha / B_low^alpha)`` and
    ``E_alpha = S_alpha - S_iso``.

    The default projection uses trapezoidal ``Delta r`` weights. Set
    ``averaging="log_projection"`` to reproduce the previous ``Delta ln r``
    definition. ``min_template_rms`` can mask a nearly vanishing total-sample
    reference, while ``min_log_amplitude_snr`` prevents logarithmic contrasts
    from diverging when either fitted amplitude is consistent with zero.

    Notes
    -----
    Current files contain no cross-scale or cross-sample covariance. Keys
    ending in ``_err_diag`` therefore ignore those correlations. They are
    diagnostic uncertainties, not final publication covariance errors.
    If a sparse profile has fewer than three valid scale bins, its amplitude
    is returned as ``NaN`` with ``*_fit_status="invalid"``. Set
    ``strict=True`` to raise instead.

    Returns
    -------
    list of dict
        One record per property, web type, and mass bin.
    """
    combined_file = Path(combined_file)
    specs = PROPERTY_SPECS if property_specs is None else property_specs
    records = []

    with h5py.File(combined_file, "r") as stream:
        _validate_combined_stream(stream, combined_file)
        smoothing_scale = float(
            stream.attrs["smoothing_scale_mpc_h"]
        )
        mass_edges = np.asarray(
            stream["metadata/mass_bin_edges"], dtype=np.float64
        )
        mass_medians = (
            np.asarray(
                stream["metadata/mass_bin_median_log10_msun_h"],
                dtype=np.float64,
            )
            if "metadata/mass_bin_median_log10_msun_h" in stream
            else None
        )
        available_indices = list(range(len(mass_edges) - 1))
        if mass_bin_indices is None:
            indices = available_indices
        else:
            indices = sorted({int(index) for index in mass_bin_indices})
            invalid = set(indices) - set(available_indices)
            if invalid:
                raise ValueError(
                    f"Mass bins {sorted(invalid)} are unavailable; "
                    f"choose from {available_indices}"
                )

        available_samples = set(stream["data/3d"].keys())
        for property_name in properties:
            if property_name not in specs:
                raise KeyError(
                    f"Unknown property specification: {property_name}"
                )
            if property_name not in available_samples:
                raise KeyError(
                    f"{combined_file} has no sample {property_name!r}; "
                    f"available: {sorted(available_samples)}"
                )
            spec = specs[property_name]

            for web in web_types:
                if web not in WEB_TYPES:
                    raise ValueError(f"web must be one of {WEB_TYPES}")
                for mass_bin_index in indices:
                    total_3d = _load_combined_entry(
                        stream,
                        dimension="3d",
                        sample="total",
                        tail="all",
                        web=web,
                        mass_bin_index=mass_bin_index,
                    )
                    total_1d = _load_combined_entry(
                        stream,
                        dimension="1d",
                        sample="total",
                        tail="all",
                        web=web,
                        mass_bin_index=mass_bin_index,
                    )

                    sample_data = {}
                    for sample_label in ("low", "high"):
                        data_3d = _load_combined_entry(
                            stream,
                            dimension="3d",
                            sample=property_name,
                            tail=sample_label,
                            web=web,
                            mass_bin_index=mass_bin_index,
                        )
                        data_1d = _load_combined_entry(
                            stream,
                            dimension="1d",
                            sample=property_name,
                            tail=sample_label,
                            web=web,
                            mass_bin_index=mass_bin_index,
                        )
                        if data_3d["xi"].shape != total_3d["xi"].shape:
                            raise ValueError(
                                f"{property_name}/{web}/"
                                f"{mass_bin_index}: 3D shape mismatch"
                            )
                        for direction_idx, direction in enumerate(
                            DIRECTIONS
                        ):
                            _matching_grid(
                                data_3d["centers"][direction_idx],
                                total_3d["centers"][direction_idx],
                                name=f"{direction} centers",
                            )
                        _matching_grid(
                            data_1d["centers"],
                            total_1d["centers"],
                            name="1D centers",
                        )
                        sample_data[sample_label] = {
                            "cube": data_3d["xi"],
                            "cube_sd": data_3d["xi_sd"],
                            "axes": data_3d["centers"],
                            "iso": data_1d["xi"],
                            "iso_sd": data_1d["xi_sd"],
                            "attrs": data_3d["attrs"],
                            "attrs_1d": data_1d["attrs"],
                        }

                    mass_lo = float(mass_edges[mass_bin_index])
                    mass_hi = float(mass_edges[mass_bin_index + 1])
                    low_attrs = sample_data["low"]["attrs"]
                    high_attrs = sample_data["high"]["attrs"]
                    record = {
                        "combined_file": str(combined_file),
                        "smoothing_scale_mpc_h": smoothing_scale,
                        "property": property_name,
                        "property_display": spec["display"],
                        "low_label": str(
                            low_attrs.get("assembly_label", "low")
                        ),
                        "high_label": str(
                            high_attrs.get("assembly_label", "high")
                        ),
                        "low_display": spec["low_display"],
                        "high_display": spec["high_display"],
                        "web": web,
                        "mass_bin_index": int(mass_bin_index),
                        "mass_lo": mass_lo,
                        "mass_hi": mass_hi,
                        "mass_center": (
                            0.5 * (mass_lo + mass_hi)
                            if np.isfinite(mass_hi)
                            else float(mass_medians[mass_bin_index])
                            if mass_medians is not None
                            else mass_lo
                        ),
                        "r_min_requested": float(r_range[0]),
                        "r_max_requested": float(r_range[1]),
                        "averaging": averaging,
                        "min_template_rms": min_template_rms,
                        "min_log_amplitude_snr": min_log_amplitude_snr,
                        "uncertainty": "diagonal approximation",
                        "low_sample_size": int(
                            low_attrs.get("sample_size", -1)
                        ),
                        "high_sample_size": int(
                            high_attrs.get("sample_size", -1)
                        ),
                    }

                    for sample_label in ("low", "high"):
                        sample = sample_data[sample_label]
                        fit_iso = _fit_or_nan(
                            sample["iso"],
                            sample["iso_sd"],
                            total_1d["xi"],
                            total_1d["xi_sd"],
                            total_1d["centers"],
                            r_range=r_range,
                            averaging=averaging,
                            min_template_rms=min_template_rms,
                            strict=strict,
                        )
                        _store_fit(
                            record, sample_label, "iso", fit_iso
                        )

                        for direction_idx, direction in enumerate(
                            DIRECTIONS
                        ):
                            profile, profile_sd = _directional_slice(
                                sample["cube"],
                                sample["cube_sd"],
                                direction,
                            )
                            template, template_sd = _directional_slice(
                                total_3d["xi"],
                                total_3d["xi_sd"],
                                direction,
                            )
                            fit_direction = _fit_or_nan(
                                profile,
                                profile_sd,
                                template,
                                template_sd,
                                sample["axes"][direction_idx],
                                r_range=r_range,
                                averaging=averaging,
                                min_template_rms=min_template_rms,
                                strict=strict,
                            )
                            _store_fit(
                                record,
                                sample_label,
                                direction,
                                fit_direction,
                            )

                    record[
                        "S_iso"
                    ], record["S_iso_err_diag"] = _log_ratio(
                        record["B_high_iso"],
                        record["B_high_iso_err_diag"],
                        record["B_low_iso"],
                        record["B_low_iso_err_diag"],
                        min_amplitude_snr=min_log_amplitude_snr,
                    )
                    for direction in DIRECTIONS:
                        s_value, s_error = _log_ratio(
                            record[f"B_high_{direction}"],
                            record[f"B_high_{direction}_err_diag"],
                            record[f"B_low_{direction}"],
                            record[f"B_low_{direction}_err_diag"],
                            min_amplitude_snr=min_log_amplitude_snr,
                        )
                        record[f"S_{direction}"] = s_value
                        record[f"S_{direction}_err_diag"] = s_error
                        record[f"E_{direction}"] = (
                            s_value - record["S_iso"]
                        )
                        record[f"E_{direction}_err_diag"] = np.sqrt(
                            s_error**2
                            + record["S_iso_err_diag"] ** 2
                        )
                    records.append(record)
    return records


def plot_all_B_mass_relation(
    combined_file,
    *,
    properties=("age", "concentration", "spin"),
    property_specs=None,
    web_types=WEB_TYPES,
    mass_bin_indices=None,
    r_range=(5.0, 20.0),
    averaging="linear_projection",
    min_template_rms=None,
    min_log_amplitude_snr=2.0,
    strict=False,
    show_uncertainty=True,
    figsize=None,
    save_path=None,
    show=True,
):
    """Calculate and plot every ``B_s^gamma`` in a web-by-property grid.

    Rows are web environments and columns are secondary properties. Each
    panel contains the high- and low-tail amplitudes for
    ``gamma=iso,e1,e2,e3``. High tails use filled markers and solid lines;
    low tails use open markers and dashed lines. A horizontal line at one
    marks equality with the matching total-sample WCCF template.

    Returns
    -------
    figure, axes, records
        The Matplotlib objects and the records returned by
        :func:`calculate_assembly_metrics`.
    """
    properties = tuple(properties)
    web_types = tuple(web_types)
    if not properties:
        raise ValueError("properties must contain at least one selection")
    if not web_types:
        raise ValueError("web_types must contain at least one environment")

    records = calculate_assembly_metrics(
        combined_file,
        properties=properties,
        property_specs=property_specs,
        web_types=web_types,
        mass_bin_indices=mass_bin_indices,
        r_range=r_range,
        averaging=averaging,
        min_template_rms=min_template_rms,
        min_log_amplitude_snr=min_log_amplitude_snr,
        strict=strict,
    )
    if figsize is None:
        figsize = (4.6 * len(properties), 3.1 * len(web_types))
    figure, axes = plt.subplots(
        len(web_types),
        len(properties),
        figsize=figsize,
        sharex="col",
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )

    component_styles = {
        "iso": ("0.15", "o", r"{\rm iso}"),
        "e1": (DIRECTION_COLORS["e1"], "s", r"e_1"),
        "e2": (DIRECTION_COLORS["e2"], "^", r"e_2"),
        "e3": (DIRECTION_COLORS["e3"], "D", r"e_3"),
    }
    sample_styles = {
        "high": ("-", "full"),
        "low": ("--", "none"),
    }

    for row, web in enumerate(web_types):
        for column, property_name in enumerate(properties):
            axis = axes[row, column]
            panel_records = sorted(
                (
                    record
                    for record in records
                    if record["web"] == web
                    and record["property"] == property_name
                ),
                key=lambda record: record["mass_bin_index"],
            )
            if not panel_records:
                axis.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                continue

            mass = np.asarray(
                [record["mass_center"] for record in panel_records]
            )
            omitted = 0
            for component, (
                color,
                marker,
                component_label,
            ) in component_styles.items():
                for sample_label, (
                    linestyle,
                    fillstyle,
                ) in sample_styles.items():
                    key = f"B_{sample_label}_{component}"
                    values = np.asarray(
                        [record[key] for record in panel_records]
                    )
                    errors = np.asarray(
                        [
                            record[f"{key}_err_diag"]
                            for record in panel_records
                        ]
                    )
                    statuses = [
                        record[f"{key}_fit_status"]
                        for record in panel_records
                    ]
                    omitted += sum(status != "ok" for status in statuses)
                    axis.errorbar(
                        mass,
                        values,
                        yerr=errors if show_uncertainty else None,
                        color=color,
                        marker=marker,
                        fillstyle=fillstyle,
                        markerfacecolor=(
                            color if fillstyle == "full" else "white"
                        ),
                        markeredgecolor=color,
                        ms=4.5,
                        lw=1.25,
                        ls=linestyle,
                        capsize=2.0,
                        label=(
                            rf"$B_{{{sample_label}}}^{{{component_label}}}$"
                        ),
                    )

            axis.axhline(1.0, color="0.4", ls=":", lw=0.9)
            axis.grid(alpha=0.22)
            if omitted:
                axis.text(
                    0.02,
                    0.03,
                    f"{omitted} weak/invalid fits omitted",
                    ha="left",
                    va="bottom",
                    fontsize=7.5,
                    color="0.35",
                    transform=axis.transAxes,
                )
            if row == 0:
                axis.set_title(
                    panel_records[0]["property_display"],
                    fontweight="bold",
                )
            if column == 0:
                axis.set_ylabel(
                    f"{web.capitalize()}\n" + r"$B_s^{\gamma}$"
                )
            if row == len(web_types) - 1:
                axis.set_xlabel(
                    r"$\log_{10}(M_{200{\rm m}}/h^{-1}M_\odot)$"
                )

    legend_handles = []
    for component, (color, marker, component_label) in (
        component_styles.items()
    ):
        for sample_label, (linestyle, fillstyle) in sample_styles.items():
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker=marker,
                    fillstyle=fillstyle,
                    markerfacecolor=(
                        color if fillstyle == "full" else "white"
                    ),
                    markeredgecolor=color,
                    ls=linestyle,
                    lw=1.25,
                    label=rf"$B_{{{sample_label}}}^{{{component_label}}}$",
                )
            )
    figure.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=4,
        frameon=True,
        fontsize=8.5,
    )

    first = records[0]
    threshold_text = ""
    if min_template_rms is not None:
        threshold_text = rf", template RMS $\geq {min_template_rms:g}$"
    figure.suptitle(
        rf"All scale-averaged $B_s^\gamma$ amplitudes, "
        rf"$r={r_range[0]:g}$--${r_range[1]:g}\,h^{{-1}}{{\rm Mpc}}$, "
        rf"$R_{{\rm s}}={first['smoothing_scale_mpc_h']:g}"
        rf"\,h^{{-1}}{{\rm Mpc}}${threshold_text}",
        fontsize=13,
        fontweight="bold",
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes, records


def plot_assembly_mass_relation(
    records,
    property_name,
    *,
    web_types=WEB_TYPES,
    show_uncertainty=True,
    figsize=(16, 8),
    save_path=None,
    show=True,
):
    """Plot isotropic/directional assembly signals and directional excess.

    The top row shows ``S_iso`` and ``S_e1/e2/e3``. The bottom row shows
    ``E_e1/e2/e3 = S_e1/e2/e3 - S_iso``. Columns are web types.
    """
    selected = [
        record
        for record in records
        if record["property"] == property_name
    ]
    if not selected:
        raise ValueError(f"No records for property={property_name!r}")

    figure, axes = plt.subplots(
        2,
        len(web_types),
        figsize=figsize,
        sharex="col",
        constrained_layout=True,
        squeeze=False,
        sharey = True
    )
    for column, web in enumerate(web_types):
        web_records = sorted(
            (
                record
                for record in selected
                if record["web"] == web
            ),
            key=lambda record: record["mass_bin_index"],
        )
        if not web_records:
            continue
        mass = np.asarray(
            [record["mass_center"] for record in web_records]
        )

        top = axes[0, column]
        bottom = axes[1, column]
        series = (
            ("iso", "0.1", "o", r"$S_{\rm iso}$"),
            ("e1", DIRECTION_COLORS["e1"], "s", r"$S_{e_1}$"),
            ("e2", DIRECTION_COLORS["e2"], "^", r"$S_{e_2}$"),
            ("e3", DIRECTION_COLORS["e3"], "D", r"$S_{e_3}$"),
        )
        for component, color, marker, label in series:
            key = f"S_{component}"
            values = np.asarray([record[key] for record in web_records])
            errors = np.asarray(
                [
                    record[f"{key}_err_diag"]
                    for record in web_records
                ]
            )
            top.errorbar(
                mass,
                values,
                yerr=errors if show_uncertainty else None,
                color=color,
                marker=marker,
                ms=4.5,
                lw=1.5,
                capsize=2.5,
                label=label,
            )

        for direction, marker in zip(DIRECTIONS, ("s", "^", "D")):
            key = f"E_{direction}"
            values = np.asarray([record[key] for record in web_records])
            errors = np.asarray(
                [
                    record[f"{key}_err_diag"]
                    for record in web_records
                ]
            )
            bottom.errorbar(
                mass,
                values,
                yerr=errors if show_uncertainty else None,
                color=DIRECTION_COLORS[direction],
                marker=marker,
                ms=4.5,
                lw=1.5,
                capsize=2.5,
                label=rf"$E_{{{direction.replace('e', 'e_')}}}$",
            )

        for axis in (top, bottom):
            axis.axhline(0.0, color="0.35", ls=":", lw=0.9)
            axis.grid(alpha=0.25)
        top.set_title(web.capitalize(), fontweight="bold")
        bottom.set_xlabel(
            r"$\log_{10}(M_{200{\rm m}}/h^{-1}M_\odot)$"
        )
        if column == 0:
            top.set_ylabel(r"$S=\ln(B_{\rm high}/B_{\rm low})$")
            bottom.set_ylabel(r"$E_\alpha=S_\alpha-S_{\rm iso}$")

        top.set_ylim(-1.5, 1.5)
        bottom.set_ylim(-1.5, 1.5)

    handles_top = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            lw=1.5,
            label=label,
        )
        for _, color, marker, label in series
    ]
    handles_bottom = [
        Line2D(
            [0],
            [0],
            color=DIRECTION_COLORS[direction],
            marker=marker,
            lw=1.5,
            label=rf"$E_{{{idx}}}$",
        )
        for direction, marker, idx in zip(DIRECTIONS, ("s", "^", "D"), (1, 2, 3))
    ]
    figure.legend(
        handles=handles_top + handles_bottom,
        loc="outside lower center",
        ncol=7,
        frameon=True,
        fontsize=9,
    )
    first = selected[0]
    smoothing_text = ""
    if "smoothing_scale_mpc_h" in first:
        smoothing_text = (
            rf", $R_{{\rm s}}={first['smoothing_scale_mpc_h']:g}"
            r"\,h^{-1}{\rm Mpc}$"
        )
    figure.suptitle(
        f"{first['property_display']} assembly bias: "
        f"{first['high_display']} versus {first['low_display']}"
        f"{smoothing_text}",
        fontsize=13,
        fontweight="bold",
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def calculate_brel_cube(
    numerator_file,
    reference_file,
    *,
    min_reference_snr=2.0,
    max_fractional_error=1.0,
):
    """Calculate a diagnostic pointwise ``xi_num / xi_ref`` cube.

    This pointwise map is not the scale-averaged ``B`` used in the mass
    relation. Voxels are masked when the reference signal is less than
    ``min_reference_snr`` times its saved standard deviation or when the
    propagated fractional error exceeds ``max_fractional_error``.
    """
    numerator, numerator_sd, axes_num, attrs_num = load_xi_3d_h5(
        numerator_file
    )
    reference, reference_sd, axes_ref, attrs_ref = load_xi_3d_h5(
        reference_file
    )
    if numerator.shape != reference.shape:
        raise ValueError("Numerator and reference cubes have different shapes")
    for direction, first, second in zip(DIRECTIONS, axes_num, axes_ref):
        _matching_grid(first, second, name=f"{direction} centers")

    valid = (
        np.isfinite(numerator)
        & np.isfinite(numerator_sd)
        & np.isfinite(reference)
        & np.isfinite(reference_sd)
        & (reference_sd > 0.0)
        & (np.abs(reference) >= float(min_reference_snr) * reference_sd)
    )
    ratio = np.full_like(numerator, np.nan, dtype=np.float64)
    ratio_sd = np.full_like(numerator, np.nan, dtype=np.float64)
    ratio[valid] = numerator[valid] / reference[valid]
    ratio_sd[valid] = np.sqrt(
        (numerator_sd[valid] / reference[valid]) ** 2
        + (
            numerator[valid]
            * reference_sd[valid]
            / reference[valid] ** 2
        )
        ** 2
    )
    if max_fractional_error is not None:
        fractional_error = np.full_like(ratio, np.inf)
        nonzero = valid & (np.abs(ratio) > 0.0)
        fractional_error[nonzero] = (
            ratio_sd[nonzero] / np.abs(ratio[nonzero])
        )
        valid &= fractional_error <= float(max_fractional_error)
        ratio[~valid] = np.nan
        ratio_sd[~valid] = np.nan

    return {
        "B_rel": ratio,
        "B_rel_sd_diag": ratio_sd,
        "valid": valid,
        "centers": axes_num,
        "numerator_attrs": attrs_num,
        "reference_attrs": attrs_ref,
        "numerator_file": str(Path(numerator_file)),
        "reference_file": str(Path(reference_file)),
    }


def calculate_combined_brel_cube(
    combined_file,
    *,
    sample,
    web,
    mass_bin_index=1,
    numerator_tail="high",
    reference_tail="low",
    min_reference_snr=2.0,
    max_fractional_error=1.0,
):
    """Calculate a property-tail ``B_rel`` cube from one combined file."""
    combined_file = Path(combined_file)
    numerator = load_combined_h5(
        combined_file,
        dimension="3d",
        sample=sample,
        tail=numerator_tail,
        web=web,
        mass_bin_index=mass_bin_index,
    )
    reference = load_combined_h5(
        combined_file,
        dimension="3d",
        sample=sample,
        tail=reference_tail,
        web=web,
        mass_bin_index=mass_bin_index,
    )
    if numerator["xi"].shape != reference["xi"].shape:
        raise ValueError(
            "Numerator and reference cubes have different shapes"
        )
    for direction, first, second in zip(
        DIRECTIONS,
        numerator["centers"],
        reference["centers"],
    ):
        _matching_grid(first, second, name=f"{direction} centers")

    xi_num = numerator["xi"]
    sd_num = numerator["xi_sd"]
    xi_ref = reference["xi"]
    sd_ref = reference["xi_sd"]
    valid = (
        np.isfinite(xi_num)
        & np.isfinite(sd_num)
        & np.isfinite(xi_ref)
        & np.isfinite(sd_ref)
        & (sd_ref > 0.0)
        & (np.abs(xi_ref) >= float(min_reference_snr) * sd_ref)
    )
    ratio = np.full_like(xi_num, np.nan, dtype=np.float64)
    ratio_sd = np.full_like(xi_num, np.nan, dtype=np.float64)
    ratio[valid] = xi_num[valid] / xi_ref[valid]
    ratio_sd[valid] = np.sqrt(
        (sd_num[valid] / xi_ref[valid]) ** 2
        + (xi_num[valid] * sd_ref[valid] / xi_ref[valid] ** 2) ** 2
    )
    if max_fractional_error is not None:
        fractional_error = np.full_like(ratio, np.inf)
        nonzero = valid & (np.abs(ratio) > 0.0)
        fractional_error[nonzero] = (
            ratio_sd[nonzero] / np.abs(ratio[nonzero])
        )
        valid &= fractional_error <= float(max_fractional_error)
        ratio[~valid] = np.nan
        ratio_sd[~valid] = np.nan

    container_attrs = numerator["container_attrs"]
    return {
        "B_rel": ratio,
        "B_rel_sd_diag": ratio_sd,
        "valid": valid,
        "centers": numerator["centers"],
        "numerator_attrs": numerator["attrs"],
        "reference_attrs": reference["attrs"],
        "numerator_entry": numerator["entry_path"],
        "reference_entry": reference["entry_path"],
        "combined_file": str(combined_file),
        "sample": str(sample),
        "web": str(web),
        "mass_bin_index": int(mass_bin_index),
        "mass_lo": float(
            container_attrs["mass_bin_lower_log10_msun_h"]
        ),
        "mass_hi": float(
            container_attrs["mass_bin_upper_log10_msun_h"]
        ),
    }


def _mirror_plane(values, x, y):
    nx, ny = values.shape
    mirrored = np.empty((2 * nx, 2 * ny), dtype=values.dtype)
    mirrored[:nx, :ny] = values[::-1, ::-1]
    mirrored[:nx, ny:] = values[::-1, :]
    mirrored[nx:, :ny] = values[:, ::-1]
    mirrored[nx:, ny:] = values
    x_mirrored = np.concatenate([-x[::-1], x])
    y_mirrored = np.concatenate([-y[::-1], y])
    return mirrored, x_mirrored, y_mirrored


def plot_brel_3d_heatmap_from_folders(
    numerator_folder,
    reference_folder,
    *,
    web,
    mass_bin_index=1,
    numerator_label="high",
    reference_label="low",
    lab="halo",
    min_reference_snr=2.0,
    max_fractional_error=1.0,
    skip_underflow=True,
    mirror=True,
    cmap="coolwarm",
    vlim=None,
    figsize=(11, 10),
    save_path=None,
    show=True,
):
    """Legacy folder-based plot of diagnostic pointwise ``B_rel``.

    The layout matches the xi heatmap in ``func.py``. The unshown coordinate
    is fixed at the underflow-bin index. Colors are centered on ``B_rel=1``;
    unreliable voxels are masked.
    """
    if web not in WEB_TYPES:
        raise ValueError(f"web must be one of {WEB_TYPES}")
    numerator_file = _result_path(
        numerator_folder, web, "3d", mass_bin_index, lab
    )
    reference_file = _result_path(
        reference_folder, web, "3d", mass_bin_index, lab
    )
    result = calculate_brel_cube(
        numerator_file,
        reference_file,
        min_reference_snr=min_reference_snr,
        max_fractional_error=max_fractional_error,
    )
    cube = result["B_rel"]
    r1, r2, r3 = result["centers"]

    planes = [
        (cube[:, :, 0], r1, r2, r"$r_1$", r"$r_2$", r"$(r_1,r_2)$"),
        (cube[:, 0, :], r1, r3, r"$r_1$", r"$r_3$", r"$(r_1,r_3)$"),
        (cube[0, :, :], r2, r3, r"$r_2$", r"$r_3$", r"$(r_2,r_3)$"),
    ]
    prepared = []
    for plane, x, y, xlabel, ylabel, title in planes:
        if skip_underflow:
            plane = plane[1:, 1:]
            x = x[1:]
            y = y[1:]
        if mirror:
            plane, x, y = _mirror_plane(plane, x, y)
        prepared.append((plane, x, y, xlabel, ylabel, title))

    finite_values = np.concatenate(
        [plane[np.isfinite(plane)] for plane, *_ in prepared]
    )
    if finite_values.size == 0:
        raise ValueError(
            "No reliable B_rel voxels remain after uncertainty masking"
        )
    if vlim is None:
        low, high = np.nanpercentile(finite_values, [5.0, 95.0])
        half_range = max(abs(low - 1.0), abs(high - 1.0), 0.05)
        vmin, vmax = 1.0 - half_range, 1.0 + half_range
    else:
        vmin, vmax = map(float, vlim)
        if vmax <= vmin:
            raise ValueError("vlim must be (vmin, vmax) with vmax > vmin")

    figure, axes = plt.subplots(
        2, 2, figsize=figsize, constrained_layout=True
    )
    plot_axes = (axes[0, 0], axes[1, 0], axes[1, 1])
    image = None
    for axis, (plane, x, y, xlabel, ylabel, title) in zip(
        plot_axes, prepared
    ):
        image = axis.pcolormesh(
            x,
            y,
            np.ma.masked_invalid(plane).T,
            shading="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_xlabel(xlabel + r" [$h^{-1}{\rm Mpc}$]")
        axis.set_ylabel(ylabel + r" [$h^{-1}{\rm Mpc}$]")
        axis.set_title(title)
        axis.set_aspect("equal")

    axes[0, 1].set_axis_off()
    color_axis = axes[0, 1].inset_axes([0.40, 0.10, 0.18, 0.80])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label(
        rf"$B_{{\rm rel}}=\xi_{{\rm {numerator_label}}}/"
        rf"\xi_{{\rm {reference_label}}}$"
    )
    mass_lo, mass_hi = _mass_interval(
        result["numerator_attrs"], mass_bin_index
    )
    valid_fraction = float(np.mean(result["valid"]))
    figure.suptitle(
        f"{web.capitalize()}, "
        f"${mass_range_tex(mass_lo, mass_hi)}$: "
        f"{numerator_label}/{reference_label} "
        f"(reliable voxels {100.0 * valid_fraction:.1f}%)",
        fontsize=13,
        fontweight="bold",
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes, result


def _plot_combined_brel_result(
    result,
    *,
    numerator_label,
    reference_label,
    skip_underflow,
    mirror,
    cmap,
    vlim,
    figsize,
    save_path,
    show,
):
    cube = result["B_rel"]
    r1, r2, r3 = result["centers"]
    planes = [
        (cube[:, :, 0], r1, r2, r"$r_1$", r"$r_2$", r"$(r_1,r_2)$"),
        (cube[:, 0, :], r1, r3, r"$r_1$", r"$r_3$", r"$(r_1,r_3)$"),
        (cube[0, :, :], r2, r3, r"$r_2$", r"$r_3$", r"$(r_2,r_3)$"),
    ]
    prepared = []
    for plane, x, y, xlabel, ylabel, title in planes:
        if skip_underflow:
            plane = plane[1:, 1:]
            x = x[1:]
            y = y[1:]
        if mirror:
            plane, x, y = _mirror_plane(plane, x, y)
        prepared.append((plane, x, y, xlabel, ylabel, title))

    finite_values = np.concatenate(
        [plane[np.isfinite(plane)] for plane, *_ in prepared]
    )
    if finite_values.size == 0:
        raise ValueError(
            "No reliable B_rel voxels remain after uncertainty masking"
        )
    if vlim is None:
        low, high = np.nanpercentile(finite_values, [5.0, 95.0])
        half_range = max(abs(low - 1.0), abs(high - 1.0), 0.05)
        vmin, vmax = 1.0 - half_range, 1.0 + half_range
    else:
        vmin, vmax = map(float, vlim)
        if vmax <= vmin:
            raise ValueError("vlim must have vmax > vmin")

    figure, axes = plt.subplots(
        2, 2, figsize=figsize, constrained_layout=True
    )
    plot_axes = (axes[0, 0], axes[1, 0], axes[1, 1])
    image = None
    for axis, (plane, x, y, xlabel, ylabel, title) in zip(
        plot_axes, prepared
    ):
        image = axis.pcolormesh(
            x,
            y,
            np.ma.masked_invalid(plane).T,
            shading="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_xlabel(xlabel + r" [$h^{-1}{\rm Mpc}$]")
        axis.set_ylabel(ylabel + r" [$h^{-1}{\rm Mpc}$]")
        axis.set_title(title)
        axis.set_aspect("equal")

    axes[0, 1].set_axis_off()
    color_axis = axes[0, 1].inset_axes([0.40, 0.10, 0.18, 0.80])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label(
        rf"$B_{{\rm rel}}=\xi_{{\rm {numerator_label}}}/"
        rf"\xi_{{\rm {reference_label}}}$"
    )
    valid_fraction = float(np.mean(result["valid"]))
    figure.suptitle(
        f"{result['web'].capitalize()}, "
        f"${mass_range_tex(result['mass_lo'], result['mass_hi'])}$: "
        f"{numerator_label}/{reference_label} "
        f"(reliable voxels {100.0 * valid_fraction:.1f}%)",
        fontsize=13,
        fontweight="bold",
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes, result


def plot_brel_3d_heatmap(
    combined_file,
    *,
    sample,
    web,
    mass_bin_index=1,
    numerator_tail="high",
    reference_tail="low",
    numerator_label=None,
    reference_label=None,
    min_reference_snr=2.0,
    max_fractional_error=1.0,
    skip_underflow=True,
    mirror=True,
    cmap="coolwarm",
    vlim=None,
    figsize=(11, 10),
    save_path=None,
    show=True,
):
    """Plot a high/low 3D ``B_rel`` heatmap from one combined file."""
    result = calculate_combined_brel_cube(
        combined_file,
        sample=sample,
        web=web,
        mass_bin_index=mass_bin_index,
        numerator_tail=numerator_tail,
        reference_tail=reference_tail,
        min_reference_snr=min_reference_snr,
        max_fractional_error=max_fractional_error,
    )
    if numerator_label is None:
        numerator_label = f"{numerator_tail} {sample}"
    if reference_label is None:
        reference_label = f"{reference_tail} {sample}"
    return _plot_combined_brel_result(
        result,
        numerator_label=numerator_label,
        reference_label=reference_label,
        skip_underflow=skip_underflow,
        mirror=mirror,
        cmap=cmap,
        vlim=vlim,
        figsize=figsize,
        save_path=save_path,
        show=show,
    )


def plot_property_brel_3d_heatmap(
    combined_file,
    property_name,
    web,
    *,
    mass_bin_index=1,
    property_specs=None,
    save_path=None,
    show=True,
    **heatmap_kwargs,
):
    """Plot the high/low property-tail 3D ``B_rel`` heatmap.

    This convenience wrapper supplies publication labels for age,
    concentration, or spin. All numerical data come from ``combined_file``.
    """
    specs = PROPERTY_SPECS if property_specs is None else property_specs
    if property_name not in specs:
        raise KeyError(f"Unknown property specification: {property_name}")
    spec = specs[property_name]
    return plot_brel_3d_heatmap(
        combined_file,
        sample=property_name,
        web=web,
        mass_bin_index=mass_bin_index,
        numerator_label=spec["high_display"],
        reference_label=spec["low_display"],
        save_path=save_path,
        show=show,
        **heatmap_kwargs,
    )


def _lambda_threshold_from_combined_file(file_path):
    """Read the effective web-classification threshold from HDF5 or name."""
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as stream:
        if "lambda_threshold" in stream.attrs:
            return float(stream.attrs["lambda_threshold"])

    match = re.search(
        r"_lth(?P<tag>m?\d+(?:p\d+)?)_combined\.h5$",
        file_path.name,
    )
    if match is None:
        raise ValueError(
            f"Cannot determine lambda_threshold from {file_path}"
        )
    tag = match.group("tag")
    sign = -1.0 if tag.startswith("m") else 1.0
    if tag.startswith("m"):
        tag = tag[1:]
    return sign * float(tag.replace("p", "."))


def plot_lambda_threshold_robustness(
    data_folder,
    *,
    mass_bin_index=1,
    smoothing_scale=3.0,
    lambda_thresholds=None,
    file_pattern="*s{smoothing_scale:.0f}_lth*_combined.h5",
    skip_underflow=True,
    xscale="linear",
    yscale="linear",
    xlim=None,
    ratio_ylim=None,
    show_uncertainty=False,
    figsize=(14, 14),
    save_path=None,
    show=True,
):
    """Plot 1D/3D WCCF robustness to the web threshold ``lambda_th``.

    The four outer panels are knot, filament, sheet, and void. Each contains
    a main WCCF axis and a smaller ratio axis. The main axis overlays the
    three local-eigenframe 3D slices and the isotropic 1D WCCF for every
    discovered threshold. The ratio axis shows

        ``(1 + xi_i) / (1 + xi_1D)``,  i = 1, 2, 3.

    Direction/statistic is encoded by color and marker; ``lambda_th`` is
    encoded by line style. All data are read directly from combined HDF5
    records for ``sample='total'`` and ``tail='all'``.

    Parameters
    ----------
    data_folder : path-like
        Folder containing files such as ``*_s3_lthm0p2_combined.h5``.
    mass_bin_index : int
        Fixed halo-mass-bin index.
    smoothing_scale : float
        Select files at this smoothing scale in Mpc/h.
    lambda_thresholds : sequence of float or None
        Thresholds to include. ``None`` discovers and sorts all matching
        thresholds from the HDF5 attributes, falling back to the ``lth``
        filename tag for legacy files.
    file_pattern : str
        Glob pattern with a named ``smoothing_scale`` formatting field.
    ratio_ylim : (float, float) or None
        Optional common limits for the four ratio axes.

    Returns
    -------
    figure : matplotlib.figure.Figure
    axes : ndarray, shape (2, 2, 2)
        The last index is 0 for the WCCF axis and 1 for its ratio axis.
    """
    data_folder = Path(data_folder)
    search_pattern = file_pattern.format(
        smoothing_scale=float(smoothing_scale)
    )
    candidate_files = sorted(data_folder.glob(search_pattern))
    if not candidate_files:
        raise FileNotFoundError(
            f"No combined HDF5 files match {search_pattern!r} in "
            f"{data_folder}"
        )

    runs_by_threshold = {}
    inventories = {}
    for file_path in candidate_files:
        inventory = combined_h5_inventory(file_path)
        file_scale = float(inventory["smoothing_scale_mpc_h"])
        if not np.isclose(file_scale, smoothing_scale):
            continue
        threshold = _lambda_threshold_from_combined_file(file_path)
        duplicate = next(
            (
                known
                for known in runs_by_threshold
                if np.isclose(known, threshold, rtol=0.0, atol=1.0e-12)
            ),
            None,
        )
        if duplicate is not None:
            raise ValueError(
                f"Duplicate lambda_threshold={threshold:g}: "
                f"{runs_by_threshold[duplicate]} and {file_path}"
            )
        runs_by_threshold[threshold] = file_path
        inventories[threshold] = inventory

    if lambda_thresholds is None:
        thresholds = tuple(sorted(runs_by_threshold))
    else:
        requested = tuple(float(value) for value in lambda_thresholds)
        thresholds = []
        for value in requested:
            match = next(
                (
                    known
                    for known in runs_by_threshold
                    if np.isclose(
                        known, value, rtol=0.0, atol=1.0e-12
                    )
                ),
                None,
            )
            if match is None:
                raise FileNotFoundError(
                    f"No combined run for lambda_threshold={value:g}"
                )
            thresholds.append(match)
        thresholds = tuple(thresholds)
    if not thresholds:
        raise FileNotFoundError(
            f"No usable lambda-threshold runs in {data_folder}"
        )

    nonzero_styles = [
        ":",
        "--",
        "-.",
        (0, (5, 1, 1, 1)),
        (0, (8, 2)),
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (1, 1, 4, 1)),
    ]
    line_styles = []
    nonzero_index = 0
    for threshold in thresholds:
        if np.isclose(threshold, 0.0, rtol=0.0, atol=1.0e-12):
            line_styles.append("-")
        else:
            line_styles.append(
                nonzero_styles[nonzero_index % len(nonzero_styles)]
            )
            nonzero_index += 1

    statistic_colors = [
        DIRECTION_COLORS["e1"],
        DIRECTION_COLORS["e2"],
        DIRECTION_COLORS["e3"],
        "0.08",
    ]
    statistic_markers = ["o", "s", "^", "x"]
    statistic_labels = [
        r"3D along $\mathbf{e}_1$ ($\lambda_1$)",
        r"3D along $\mathbf{e}_2$ ($\lambda_2$)",
        r"3D along $\mathbf{e}_3$ ($\lambda_3$)",
        "1D isotropic shell",
    ]

    figure = plt.figure(figsize=figsize)
    outer_grid = figure.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.065,
        top=0.77,
        hspace=0.30,
        wspace=0.16,
    )
    axes = np.empty((2, 2, 2), dtype=object)
    for web_index in range(4):
        row, column = divmod(web_index, 2)
        inner_grid = outer_grid[row, column].subgridspec(
            2, 1, height_ratios=(3.0, 1.25), hspace=0.04
        )
        axes[row, column, 0] = figure.add_subplot(inner_grid[0])
        axes[row, column, 1] = figure.add_subplot(
            inner_grid[1], sharex=axes[row, column, 0]
        )

    pair_count = 0
    mass_lower = None
    mass_upper = None
    for web_index, web in enumerate(WEB_TYPES):
        row, column = divmod(web_index, 2)
        main_axis = axes[row, column, 0]
        ratio_axis = axes[row, column, 1]

        for threshold_index, threshold in enumerate(thresholds):
            file_path = runs_by_threshold[threshold]
            try:
                result_1d = load_combined_h5(
                    file_path,
                    dimension="1d",
                    sample="total",
                    tail="all",
                    web=web,
                    mass_bin_index=mass_bin_index,
                )
                result_3d = load_combined_h5(
                    file_path,
                    dimension="3d",
                    sample="total",
                    tail="all",
                    web=web,
                    mass_bin_index=mass_bin_index,
                )
            except (KeyError, ValueError) as error:
                print(
                    f"[skip] {file_path.name}, {web}, "
                    f"mass bin {mass_bin_index}: {error}"
                )
                continue

            if mass_lower is None:
                mass_lower = float(
                    result_1d["container_attrs"][
                        "mass_bin_lower_log10_msun_h"
                    ]
                )
                mass_upper = float(
                    result_1d["container_attrs"][
                        "mass_bin_upper_log10_msun_h"
                    ]
                )

            xi_1d = np.asarray(result_1d["xi"], dtype=np.float64)
            sd_1d = np.asarray(result_1d["xi_sd"], dtype=np.float64)
            r_1d = np.asarray(result_1d["centers"], dtype=np.float64)
            if skip_underflow:
                xi_1d = xi_1d[1:]
                sd_1d = sd_1d[1:]
                r_1d = r_1d[1:]

            style = line_styles[threshold_index]
            cube = np.asarray(result_3d["xi"], dtype=np.float64)
            cube_sd = np.asarray(result_3d["xi_sd"], dtype=np.float64)
            centers_3d = result_3d["centers"]
            for direction_index, direction in enumerate(DIRECTIONS):
                xi_direction, sd_direction = _directional_slice(
                    cube, cube_sd, direction
                )
                r_direction = np.asarray(
                    centers_3d[direction_index], dtype=np.float64
                )
                if skip_underflow:
                    xi_direction = xi_direction[1:]
                    sd_direction = sd_direction[1:]
                    r_direction = r_direction[1:]

                if len(r_1d) == len(r_direction) and np.allclose(
                    r_1d,
                    r_direction,
                    rtol=5.0e-6,
                    atol=5.0e-8,
                ):
                    xi_1d_at_r = xi_1d
                    sd_1d_at_r = sd_1d
                else:
                    xi_1d_at_r = np.interp(
                        r_direction, r_1d, xi_1d
                    )
                    sd_1d_at_r = np.interp(
                        r_direction, r_1d, sd_1d
                    )

                main_axis.plot(
                    r_direction,
                    xi_direction,
                    color=statistic_colors[direction_index],
                    linestyle=style,
                    marker=statistic_markers[direction_index],
                    markevery=3,
                    ms=3.0,
                    lw=1.35,
                    alpha=0.90,
                )
                if show_uncertainty:
                    main_axis.fill_between(
                        r_direction,
                        xi_direction - sd_direction,
                        xi_direction + sd_direction,
                        color=statistic_colors[direction_index],
                        alpha=0.018,
                        linewidth=0,
                    )

                denominator = 1.0 + xi_1d_at_r
                valid = (
                    np.isfinite(xi_direction)
                    & np.isfinite(denominator)
                    & (np.abs(denominator) > 1.0e-10)
                )
                ratio = np.full_like(
                    xi_direction, np.nan, dtype=np.float64
                )
                ratio[valid] = (
                    1.0 + xi_direction[valid]
                ) / denominator[valid]
                ratio_axis.plot(
                    r_direction,
                    ratio,
                    color=statistic_colors[direction_index],
                    linestyle=style,
                    marker=statistic_markers[direction_index],
                    markevery=3,
                    ms=2.8,
                    lw=1.25,
                    alpha=0.90,
                )
                if show_uncertainty:
                    ratio_sd = np.full_like(
                        ratio, np.nan, dtype=np.float64
                    )
                    ratio_sd[valid] = np.sqrt(
                        (sd_direction[valid] / denominator[valid]) ** 2
                        + (
                            (1.0 + xi_direction[valid])
                            * sd_1d_at_r[valid]
                            / denominator[valid] ** 2
                        )
                        ** 2
                    )
                    ratio_axis.fill_between(
                        r_direction,
                        ratio - ratio_sd,
                        ratio + ratio_sd,
                        color=statistic_colors[direction_index],
                        alpha=0.018,
                        linewidth=0,
                    )

            main_axis.plot(
                r_1d,
                xi_1d,
                color=statistic_colors[3],
                linestyle=style,
                marker=statistic_markers[3],
                markevery=3,
                ms=3.5,
                lw=1.55,
                alpha=0.95,
                zorder=5,
            )
            if show_uncertainty:
                main_axis.fill_between(
                    r_1d,
                    xi_1d - sd_1d,
                    xi_1d + sd_1d,
                    color=statistic_colors[3],
                    alpha=0.025,
                    linewidth=0,
                )
            pair_count += 1

        main_axis.axhline(
            0.0, color="k", lw=0.8, ls=":", alpha=0.65
        )
        ratio_axis.axhline(
            1.0, color="k", lw=0.9, ls=":", alpha=0.75
        )
        main_axis.set_xscale(xscale)
        ratio_axis.set_xscale(xscale)
        main_axis.set_yscale(yscale)
        if xlim is not None:
            main_axis.set_xlim(*xlim)
        if ratio_ylim is not None:
            ratio_axis.set_ylim(*ratio_ylim)
        main_axis.grid(alpha=0.23)
        ratio_axis.grid(alpha=0.23)
        main_axis.set_title(web.capitalize(), fontweight="bold")
        main_axis.set_ylabel(r"$\xi$")
        ratio_axis.set_xlabel(r"$r$ [$h^{-1}{\rm Mpc}$]")
        ratio_axis.set_ylabel(
            r"$(1+\xi_i)/(1+\xi_{\rm 1D})$", fontsize=9
        )
        ratio_axis.tick_params(axis="both", labelsize=8)
        plt.setp(main_axis.get_xticklabels(), visible=False)

    if pair_count == 0:
        plt.close(figure)
        raise FileNotFoundError(
            f"No plottable total-sample records for mass bin "
            f"{mass_bin_index}"
        )

    statistic_handles = [
        Line2D(
            [0],
            [0],
            color=statistic_colors[index],
            marker=statistic_markers[index],
            linestyle="-",
            lw=1.8,
            ms=4,
            label=statistic_labels[index],
        )
        for index in range(4)
    ]
    threshold_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=line_styles[index],
            lw=1.8,
            label=rf"$\lambda_{{\rm th}}={threshold:g}$",
        )
        for index, threshold in enumerate(thresholds)
    ]
    figure.legend(
        handles=statistic_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        fontsize=9,
        frameon=True,
    )
    figure.legend(
        handles=threshold_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.900),
        ncol=min(len(threshold_handles), 6),
        fontsize=9,
        frameon=True,
    )

    mass_text = (
        rf"${mass_lower:.1f}\leq\log_{{10}}"
        rf"(M_{{200\rm m}}/h^{{-1}}{{\rm M_\odot}})"
        rf"<{mass_upper:.1f}$"
    )
    figure.suptitle(
        "3D and 1D halo-web correlations across classification "
        rf"thresholds: {mass_text}, "
        rf"$R_{{\rm s}}={float(smoothing_scale):g}\,h^{{-1}}{{\rm Mpc}}$",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


# ---------------------------------------------------------------------------
# Anisotropic relative assembly bias  A = b_high / b_low
# ---------------------------------------------------------------------------

def _projection_amplitude(y, sy, t, st, weights):
    """Weighted projection of ``y`` onto ``t``: A = sum w t y / sum w t^2.

    The point estimate is the plain weighted projection; ``st`` enters only
    through the effective variance used for the uncertainty and chi2.
    """
    y, sy, t, st, weights = (
        np.asarray(value, dtype=np.float64)
        for value in (y, sy, t, st, weights)
    )
    denominator = float(np.sum(weights * t**2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("The template has no power over the selected bins")
    amplitude = float(np.sum(weights * t * y) / denominator)
    variance_eff = sy**2 + amplitude**2 * st**2
    amplitude_variance = float(
        np.sum((weights * t) ** 2 * variance_eff) / denominator**2
    )
    residual = y - amplitude * t
    residual_denominator = float(np.sum(weights * y**2))
    shape_residual = (
        float(np.sqrt(np.sum(weights * residual**2) / residual_denominator))
        if residual_denominator > 0.0
        else np.nan
    )
    chi2 = float(np.sum(residual**2 / variance_eff))
    return {
        "B": amplitude,
        "B_err_diag": float(np.sqrt(max(amplitude_variance, 0.0))),
        "n_scale": int(y.size),
        "shape_residual": shape_residual,
        "chi2_red_diag": chi2 / max(int(y.size) - 1, 1),
    }


def _directional_slice_at(xi, xi_sd, direction, fixed_index=0):
    """``_directional_slice`` with the two fixed axes held at an arbitrary
    index. Index 0 is the underflow bin (see the ``axes`` attribute of the
    stored records); ``fixed_index=1`` is the first resolved radial bin, the
    convention used by the contour plots with ``skip_underflow=True``."""
    j = int(fixed_index)
    if direction == "e1":
        return xi[:, j, j], xi_sd[:, j, j]
    if direction == "e2":
        return xi[j, :, j], xi_sd[j, :, j]
    if direction == "e3":
        return xi[j, j, :], xi_sd[j, j, :]
    raise ValueError(f"Unknown direction: {direction}")


def _cube_amplitude(high, low, r_range, min_points=8):
    """One amplitude from every 3D bin whose |r| lies inside ``r_range``.

    Bins carry equal weight, so this is the direction-averaged reference
    against which the three directional amplitudes are compared.
    """
    for axis_high, axis_low in zip(high["centers"], low["centers"]):
        _matching_grid(axis_high, axis_low, name="3D radial")
    r1, r2, r3 = high["centers"]
    radius = np.sqrt(
        r1[:, None, None] ** 2
        + r2[None, :, None] ** 2
        + r3[None, None, :] ** 2
    )
    y, sy = high["xi"], high["xi_sd"]
    t, st = low["xi"], low["xi_sd"]
    mask = (
        np.isfinite(y)
        & np.isfinite(sy)
        & np.isfinite(t)
        & np.isfinite(st)
        & (sy > 0.0)
        & (st > 0.0)
        & (radius >= float(r_range[0]))
        & (radius <= float(r_range[1]))
    )
    selected = int(mask.sum())
    if selected < int(min_points):
        raise ValueError(
            f"Only {selected} valid 3D bins with |r| in {r_range}; "
            f"need at least {min_points}"
        )
    fit = _projection_amplitude(
        y[mask], sy[mask], t[mask], st[mask], np.ones(selected)
    )
    fit["r_min"] = float(r_range[0])
    fit["r_max"] = float(r_range[1])
    fit["averaging"] = "cube_uniform"
    return fit


def _amplitude_grid(
    h5_file,
    *,
    numerator_sample,
    numerator_tail,
    denominator_sample,
    denominator_tail,
    n_mass,
    webs,
    keys,
    directions,
    r_range,
    averaging,
    fixed_index,
    strict,
):
    """Fit ``xi_numerator = A xi_denominator`` on every (web, mass, key) cell.

    Shared engine of ``fit_anisotropic_A`` and ``fit_anisotropic_A2``.
    Returns ``{field: {web: {key: array over mass bins}}}``.
    """
    fields = ("A", "A_err", "shape_residual", "chi2_red", "n_scale")
    grid = {
        field: {
            web: {key: np.full(n_mass, np.nan) for key in keys}
            for web in webs
        }
        for field in fields
    }

    for web in webs:
        for mass_bin_index in range(n_mass):
            try:
                numerator = load_combined_h5(
                    h5_file, dimension="3d", sample=numerator_sample,
                    tail=numerator_tail, web=web,
                    mass_bin_index=mass_bin_index,
                )
                denominator = load_combined_h5(
                    h5_file, dimension="3d", sample=denominator_sample,
                    tail=denominator_tail, web=web,
                    mass_bin_index=mass_bin_index,
                )
            except (KeyError, ValueError):
                if strict:
                    raise
                continue

            radii = dict(zip(directions, numerator["centers"]))
            for key in keys:
                try:
                    if key == "cube":
                        fit = _cube_amplitude(numerator, denominator, r_range)
                    else:
                        y, sy = _directional_slice_at(
                            numerator["xi"], numerator["xi_sd"],
                            key, fixed_index,
                        )
                        t, st = _directional_slice_at(
                            denominator["xi"], denominator["xi_sd"],
                            key, fixed_index,
                        )
                        fit = fit_scale_averaged_B(
                            y, sy, t, st, radii[key],
                            r_range=r_range, averaging=averaging,
                        )
                except (ValueError, WeakTemplateError):
                    if strict:
                        raise
                    continue
                grid["A"][web][key][mass_bin_index] = fit["B"]
                grid["A_err"][web][key][mass_bin_index] = fit["B_err_diag"]
                grid["shape_residual"][web][key][mass_bin_index] = fit[
                    "shape_residual"
                ]
                grid["chi2_red"][web][key][mass_bin_index] = fit[
                    "chi2_red_diag"
                ]
                grid["n_scale"][web][key][mass_bin_index] = fit["n_scale"]
    return grid


def _amplitude_setup(h5_file, webs, directions, include_cube):
    inventory = combined_h5_inventory(h5_file)
    mass_bin_edges = np.asarray(inventory["mass_bin_edges"], dtype=np.float64)
    centers = mass_bin_centers(inventory)
    webs = tuple(webs)
    directions = tuple(directions)
    keys = directions + (("cube",) if include_cube else ())
    return mass_bin_edges, centers, webs, directions, keys


def fit_anisotropic_A(
    h5_file,
    sample="age",
    *,
    webs=WEB_TYPES,
    directions=DIRECTIONS,
    r_range=(5.0, 20.0),
    averaging="linear_projection",
    fixed_index=0,
    include_cube=True,
    strict=False,
):
    """Fit the anisotropic relative assembly bias A = b_high / b_low.

    For every web type, mass bin and tidal eigen-direction the high-tail
    profile is projected onto the low-tail profile of the same mass bin,

        A = sum_k w_k xi_k^low xi_k^high / sum_k w_k (xi_k^low)^2,

    the least-squares amplitude of the model xi^high = A xi^low. Under linear
    biasing xi^S = b^S xi_Wm the unmeasured xi_Wm cancels exactly, so A is
    b^high / b^low with no reference to the matter field.

    Being a projection rather than an average of pointwise ratios, A stays
    finite where xi^low crosses zero. How well a single constant describes
    the data is reported per fit as ``shape_residual`` -- the weighted RMS of
    xi^high - A xi^low divided by the weighted RMS of xi^high, so 0 means the
    constant-A model is exact -- and as ``chi2_red``.

    ``include_cube`` adds the key ``"cube"`` alongside the direction keys:
    the same amplitude fitted to every 3D bin with |r| inside ``r_range``.
    Anisotropy of the assembly bias is the spread of e1/e2/e3 about it.

    Parameters
    ----------
    r_range : (float, float)
        Separation window of the fit. The default matches
        ``fit_scale_averaged_B``. Physically the frame-coherence analysis in
        ``pywccf.geometry`` argues for R_s <~ r <~ L_m, e.g.
        (3.0, 9.0) for R_s = 3, outside which the eigen-frame no longer
        labels the local structure.
    fixed_index : int
        Index at which the two non-varying separation components are held.
        0 (default) reproduces ``_directional_slice`` and the profile figures;
        note that index 0 is the underflow bin, so ``fixed_index=1`` matches
        the ``skip_underflow=True`` convention of the contour figures.
    strict : bool
        Re-raise instead of leaving NaN when one record is missing or its fit
        is degenerate.

    Returns
    -------
    dict
        ``A``, ``A_err``, ``shape_residual``, ``chi2_red`` and ``n_scale``,
        each a ``{web: {direction: array over mass bins}}`` mapping, plus the
        mass grid and the fit settings.

    Notes
    -----
    Errors are diagonal. The combined files store marginal standard
    deviations only, and the high and low tails are measured against the same
    web cells, so their errors are strongly correlated and the quoted A
    uncertainty is conservative. A bootstrap forming xi^high - xi^low inside
    each realization would cancel the shared cosmic variance and be tighter.
    """
    (mass_bin_edges, mass_bin_centers, webs, directions,
     keys) = _amplitude_setup(h5_file, webs, directions, include_cube)

    result = _amplitude_grid(
        h5_file,
        numerator_sample=sample, numerator_tail="high",
        denominator_sample=sample, denominator_tail="low",
        n_mass=len(mass_bin_centers), webs=webs, keys=keys,
        directions=directions, r_range=r_range, averaging=averaging,
        fixed_index=fixed_index, strict=strict,
    )
    result.update({
        "sample": sample,
        "webs": webs,
        "directions": directions,
        "keys": keys,
        "include_cube": bool(include_cube),
        "r_range": (float(r_range[0]), float(r_range[1])),
        "averaging": averaging,
        "fixed_index": int(fixed_index),
        "mass_bin_edges": mass_bin_edges,
        "mass_bin_centers": mass_bin_centers,
        "h5_file": str(h5_file),
    })
    return result


def fit_anisotropic_A2(
    h5_file,
    sample="age",
    *,
    tails=("high", "low"),
    reference_sample="total",
    reference_tail="all",
    webs=WEB_TYPES,
    directions=DIRECTIONS,
    r_range=(5.0, 20.0),
    averaging="linear_projection",
    fixed_index=0,
    include_cube=True,
    strict=False,
):
    """Fit A^S = b^S / b^total for each tail, anisotropically.

    The tail-vs-total counterpart of ``fit_anisotropic_A``, standing to it as
    ``plot_ratio2`` stands to ``plot_ratio``. For each tail S the projection

        A^S = sum_k w_k xi_k^total xi_k^S / sum_k w_k (xi_k^total)^2

    is the least-squares amplitude of xi^S = A^S xi^total, i.e. b^S / b^total
    under linear biasing. A^S > 1 means the tail clusters around the cosmic
    web more strongly than the full mass-matched population.

    Splitting A = b^high/b^low into A^high and A^low says *which side* of the
    population moves: a signal carried entirely by A^high > 1 with A^low ~ 1
    (old halos pushed up) is a different physical statement from A^high ~ 1
    with A^low < 1 (young halos pushed down), and the high/low ratio alone
    cannot distinguish them. Note A^high / A^low only approximates the A of
    ``fit_anisotropic_A``: the two use different templates, hence different
    effective weights, so they agree exactly only for a perfectly constant
    amplitude.

    Returns
    -------
    dict
        Same fields as ``fit_anisotropic_A`` but with a tail level on top:
        ``result["A"]["high"][web][direction]`` is an array over mass bins.
        ``result["tails"]`` lists the tails present.
    """
    (mass_bin_edges, mass_bin_centers, webs, directions,
     keys) = _amplitude_setup(h5_file, webs, directions, include_cube)
    tails = tuple(tails)

    grids = {
        tail: _amplitude_grid(
            h5_file,
            numerator_sample=sample, numerator_tail=tail,
            denominator_sample=reference_sample,
            denominator_tail=reference_tail,
            n_mass=len(mass_bin_centers), webs=webs, keys=keys,
            directions=directions, r_range=r_range, averaging=averaging,
            fixed_index=fixed_index, strict=strict,
        )
        for tail in tails
    }

    result = {
        field: {tail: grids[tail][field] for tail in tails}
        for field in ("A", "A_err", "shape_residual", "chi2_red", "n_scale")
    }
    result.update({
        "sample": sample,
        "tails": tails,
        "reference": (reference_sample, reference_tail),
        "webs": webs,
        "directions": directions,
        "keys": keys,
        "include_cube": bool(include_cube),
        "r_range": (float(r_range[0]), float(r_range[1])),
        "averaging": averaging,
        "fixed_index": int(fixed_index),
        "mass_bin_edges": mass_bin_edges,
        "mass_bin_centers": mass_bin_centers,
        "h5_file": str(h5_file),
    })
    return result
