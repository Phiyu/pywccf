"""Unified YAML runner for total and assembly-bias halo-web correlations.

This file replaces the orchestration performed by ``3d.py``, ``3d_total.py``,
``1d.py``, and ``1d_total.py``.  It does not import any of those runners or
their helper modules.  The scientific 3D estimator comes from
``pywccf.estimator.halo_web`` (engine ``legacy``) or
``pywccf.estimator.halo_web_fast`` (engine ``fast``), while the 1D estimator is implemented
locally below.

Every requested result and the complete run log are written into one combined
HDF5 file using the schema consumed by ``pywccf.io``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.spatial import cKDTree


FORMAT_NAME = "halo_web_combined_results"
SCHEMA_VERSION = "1.0"
WEB_N_LAMBDA = {"knot": 3, "filament": 2, "sheet": 1, "void": 0}
PROPERTY_NAMES = {
    "zf": "age",
    "age": "age",
    "formation_redshift": "age",
    "concentration": "concentration",
    "lambda": "spin",
    "spin": "spin",
}
TERMINAL_PLAN_STATUSES = {
    "complete",
    "skipped_empty",
    "skipped_empty_web",
}

DEFAULT_RADIAL_BINS = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
    5.5,
    6.5,
    7.5,
    8.5,
    9.5,
    10.5,
    11.6904,
    13.0158,
    14.4914,
    16.1343,
    17.9635,
    20.0,
]

DEFAULT_CONFIG = {
    "paths": {
        "library_dir": None,
        "base_dir": "/data14/Fhong/ELUCID",
        "field_file": None,
        "field_file_template": (
            "field_{snapshot:03d}.{smoothing_scale:.0f}.hdf5"
        ),
        "halo_file": None,
    },
    "simulation": {
        "snapshot": 99,
        "smoothing_scale": 3.0,
    },
    "tidal_field": {
        # null preserves the lambda_th saved in the field file.
        "lambda_threshold": None,
    },
    "run": {
        # total | subproperty | both, or a list of the first two values.
        "samples": "both",
        # 1d | 3d | both, or a list containing 1d and/or 3d.
        "dimensions": "both",
    },
    "sample": {
        "label": "halo",
        "mass_bin_edges": [
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
        ],
        "formation_redshift_range": [0.01, 20.0],
        "exclude_backsplash": True,
    },
    "assembly_bias": {
        # These legacy keys define one selection when selections is null.
        "subproperty": "zf",
        "cutoff_percent": 20.0,
        "lower_label": "young",
        "upper_label": "old",
        # A list can calculate several subproperties in the same run.
        "selections": None,
        # Mass matching of the two tails inside every mass bin (all
        # selections).  Off by default: results are then unchanged.
        "mass_weight": {
            "enabled": False,
            # parent: both tails get the mass distribution of the whole
            #         mass bin (the "total" sample), so total, low and high
            #         are compared at the same masses;
            # tails:  the distribution of the two tails together.
            "target": "parent",
            # equal-count sub-bins of the target inside each mass bin
            "sub_bins": 10,
            # cap on a single weight (mean weight = 1); null = no cap
            "max_weight": 10.0,
        },
    },
    "correlation": {
        "radial_bins": DEFAULT_RADIAL_BINS,
        "n_bootstrap": 100,
        "max_centers": 500000,
        "web_types": ["knot", "filament", "sheet", "void"],
        "use_density_weight": True,
        "chunk_size": 500,
        "n_processes": 4,
        "use_disk_cache": False,
        # 3D estimator: "legacy" = halo_web.HaloWeb3DCross, one neighbour
        # search per record; "fast" = halo_web_fast.HaloWeb3DCross, one
        # neighbour search per web type shared by every sample, tail and
        # mass bin (same results to floating-point rounding).
        "engine": "fast",
        # errors: "jackknife" (delete-one over n_jk_side^3 sub-cubes of the
        # web cells; needs engine fast for 3D) or "bootstrap" (n_bootstrap
        # resamples of the web cells, as before)
        "error": "jackknife",
        "n_jk_side": 4,
        "center_subsample_seed": 0,
        "bootstrap_seed": 42,
    },
    "output": {
        "run_name": "halo-web",
        "combined_file": None,
        "overwrite": False,
        "compression": "gzip",
        "compression_level": 4,
    },
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _deep_update(base, update):
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _absolute_path(value, config_dir):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _safe_key(value, label):
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()
    if not key or key in {".", ".."}:
        raise ValueError(f"{label} cannot be converted to a safe HDF5 key")
    return key


def _number_token(value):
    if value is None:
        return "file"
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _normalize_choice(value, *, label, choices):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "both":
            values = list(choices)
        else:
            values = [lowered]
    elif isinstance(value, (list, tuple)):
        values = [str(item).strip().lower() for item in value]
    else:
        raise ValueError(f"{label} must be a string or list")
    values = list(dict.fromkeys(values))
    if not values or any(item not in choices for item in values):
        raise ValueError(
            f"{label} must contain only {', '.join(choices)}, or 'both'"
        )
    return values


def _validate_selection(raw_selection, defaults, index):
    if not isinstance(raw_selection, dict):
        raise ValueError(
            f"assembly_bias.selections[{index}] must be a mapping"
        )
    selection = copy.deepcopy(defaults)
    selection.update(raw_selection)

    subproperty = str(selection["subproperty"]).strip()
    if not subproperty:
        raise ValueError(
            f"assembly_bias.selections[{index}].subproperty is empty"
        )
    selection["subproperty"] = subproperty

    name = selection.get("name")
    if name is None:
        name = PROPERTY_NAMES.get(subproperty.lower(), subproperty.lower())
    selection["name"] = _safe_key(
        name, f"assembly_bias.selections[{index}].name"
    )
    if selection["name"] == "total":
        raise ValueError("An assembly selection cannot be named 'total'")

    cutoff = float(selection["cutoff_percent"])
    if not 0.0 < cutoff < 50.0:
        raise ValueError(
            f"assembly_bias.selections[{index}].cutoff_percent must be "
            "between 0 and 50"
        )
    selection["cutoff_percent"] = cutoff

    labels = []
    for key in ("lower_label", "upper_label"):
        value = str(selection[key]).strip()
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError(
                f"assembly_bias.selections[{index}].{key} must be a "
                "single non-empty label"
            )
        selection[key] = value
        labels.append(value)
    if labels[0] == labels[1]:
        raise ValueError(
            f"assembly_bias.selections[{index}] lower/upper labels differ"
        )
    return selection


def _validate_config(config):
    simulation = config["simulation"]
    tidal_cfg = config["tidal_field"]
    run_cfg = config["run"]
    sample = config["sample"]
    assembly = config["assembly_bias"]
    corr = config["correlation"]
    output = config["output"]

    simulation["snapshot"] = int(simulation["snapshot"])
    if simulation["snapshot"] < 0:
        raise ValueError("simulation.snapshot must be non-negative")
    simulation["smoothing_scale"] = float(simulation["smoothing_scale"])
    if simulation["smoothing_scale"] <= 0:
        raise ValueError("simulation.smoothing_scale must be positive")

    threshold = tidal_cfg.get("lambda_threshold")
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError(
                "tidal_field.lambda_threshold must be finite or null"
            )
    tidal_cfg["lambda_threshold"] = threshold

    run_cfg["samples"] = _normalize_choice(
        run_cfg["samples"],
        label="run.samples",
        choices=("total", "subproperty"),
    )
    run_cfg["dimensions"] = _normalize_choice(
        run_cfg["dimensions"],
        label="run.dimensions",
        choices=("1d", "3d"),
    )

    mass_edges = [float(value) for value in sample["mass_bin_edges"]]
    if (
        len(mass_edges) < 2
        or any(b <= a for a, b in zip(mass_edges[:-1], mass_edges[1:]))
    ):
        raise ValueError(
            "sample.mass_bin_edges must be strictly increasing"
        )
    sample["mass_bin_edges"] = mass_edges

    zf_range = [
        float(value) for value in sample["formation_redshift_range"]
    ]
    if len(zf_range) != 2 or zf_range[1] <= zf_range[0]:
        raise ValueError(
            "sample.formation_redshift_range must be [minimum, maximum]"
        )
    sample["formation_redshift_range"] = zf_range
    sample["exclude_backsplash"] = bool(sample["exclude_backsplash"])
    if str(sample["label"]).strip() != "halo":
        raise ValueError("The current estimators require sample.label=halo")
    sample["label"] = "halo"

    radial_bins = [float(value) for value in corr["radial_bins"]]
    if (
        len(radial_bins) < 2
        or radial_bins[0] <= 0
        or any(b <= a for a, b in zip(radial_bins[:-1], radial_bins[1:]))
    ):
        raise ValueError(
            "correlation.radial_bins must be positive and increasing"
        )
    corr["radial_bins"] = radial_bins
    corr["n_bootstrap"] = int(corr["n_bootstrap"])
    if corr["n_bootstrap"] < 0:
        raise ValueError("correlation.n_bootstrap must be non-negative")
    max_centers = corr["max_centers"]
    if max_centers is not None:
        max_centers = int(max_centers)
        if max_centers <= 0:
            raise ValueError(
                "correlation.max_centers must be positive or null"
            )
    corr["max_centers"] = max_centers
    for key in ("chunk_size", "n_processes"):
        corr[key] = int(corr[key])
        if corr[key] <= 0:
            raise ValueError(f"correlation.{key} must be positive")
    for key in ("center_subsample_seed", "bootstrap_seed"):
        corr[key] = int(corr[key])
    corr["use_density_weight"] = bool(corr["use_density_weight"])
    corr["use_disk_cache"] = bool(corr["use_disk_cache"])
    corr["engine"] = str(corr.get("engine", "fast")).strip().lower()
    if corr["engine"] not in ("legacy", "fast"):
        raise ValueError("correlation.engine must be legacy or fast")
    corr["error"] = str(corr.get("error", "jackknife")).strip().lower()
    if corr["error"] not in ("jackknife", "bootstrap"):
        raise ValueError("correlation.error must be jackknife or bootstrap")
    corr["n_jk_side"] = int(corr.get("n_jk_side", 4))
    if corr["n_jk_side"] < 2:
        raise ValueError("correlation.n_jk_side must be >= 2")
    if (corr["error"] == "jackknife" and corr["engine"] == "legacy"
            and "3d" in run_cfg["dimensions"]):
        raise ValueError(
            "correlation.error: jackknife needs correlation.engine: fast for "
            "3D (halo_web.py only bootstraps); use error: bootstrap with the "
            "legacy engine"
        )
    web_types = list(dict.fromkeys(str(web) for web in corr["web_types"]))
    if (
        not web_types
        or any(web not in WEB_N_LAMBDA for web in web_types)
    ):
        raise ValueError(
            "correlation.web_types must contain knot, filament, sheet, "
            "and/or void"
        )
    corr["web_types"] = web_types
    if "3d" in run_cfg["dimensions"] and corr["center_subsample_seed"] != 0:
        raise ValueError(
            "HaloWeb3DCross fixes the center subsample seed at 0; set "
            "correlation.center_subsample_seed: 0 when running 3d"
        )

    legacy_defaults = {
        "subproperty": assembly["subproperty"],
        "cutoff_percent": assembly["cutoff_percent"],
        "lower_label": assembly["lower_label"],
        "upper_label": assembly["upper_label"],
    }
    raw_selections = assembly.get("selections")
    if raw_selections is None:
        raw_selections = [{}]
    if not isinstance(raw_selections, list) or not raw_selections:
        raise ValueError(
            "assembly_bias.selections must be a non-empty list or null"
        )
    selections = [
        _validate_selection(raw, legacy_defaults, index)
        for index, raw in enumerate(raw_selections)
    ]
    names = [selection["name"] for selection in selections]
    if len(names) != len(set(names)):
        raise ValueError(
            "assembly_bias.selections must have unique semantic names"
        )
    assembly["selections"] = selections

    mass_weight = assembly.get("mass_weight")
    if mass_weight is None or mass_weight is False:
        mass_weight = {"enabled": False}
    if not isinstance(mass_weight, dict):
        raise ValueError("assembly_bias.mass_weight must be a mapping or null")
    mass_weight = {
        **DEFAULT_CONFIG["assembly_bias"]["mass_weight"],
        **mass_weight,
    }
    mass_weight["enabled"] = bool(mass_weight["enabled"])
    mass_weight["target"] = str(mass_weight["target"]).strip().lower()
    if mass_weight["target"] not in ("parent", "tails"):
        raise ValueError(
            "assembly_bias.mass_weight.target must be parent or tails"
        )
    mass_weight["sub_bins"] = int(mass_weight["sub_bins"])
    if mass_weight["sub_bins"] < 1:
        raise ValueError("assembly_bias.mass_weight.sub_bins must be >= 1")
    max_weight = mass_weight["max_weight"]
    if max_weight is not None:
        max_weight = float(max_weight)
        if not max_weight > 1.0:
            raise ValueError(
                "assembly_bias.mass_weight.max_weight must be > 1 or null"
            )
    mass_weight["max_weight"] = max_weight
    assembly["mass_weight"] = mass_weight

    output["run_name"] = _safe_key(output["run_name"], "output.run_name")
    output["overwrite"] = bool(output["overwrite"])
    compression = output.get("compression")
    if compression is not None:
        compression = str(compression).strip().lower()
        if compression not in {"gzip", "lzf"}:
            raise ValueError(
                "output.compression must be gzip, lzf, or null"
            )
    output["compression"] = compression
    output["compression_level"] = int(output["compression_level"])
    if not 0 <= output["compression_level"] <= 9:
        raise ValueError("output.compression_level must be between 0 and 9")


def load_config(config_file):
    """Load, validate, and fully resolve a unified runner configuration."""
    config_path = Path(config_file).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        user_config = yaml.safe_load(stream) or {}
    if not isinstance(user_config, dict):
        raise ValueError("The top level of the YAML file must be a mapping")

    config = _deep_update(copy.deepcopy(DEFAULT_CONFIG), user_config)
    _validate_config(config)
    config_dir = config_path.parent
    paths = config["paths"]
    simulation = config["simulation"]
    output = config["output"]

    paths["base_dir"] = str(_absolute_path(paths["base_dir"], config_dir))
    if paths["library_dir"] is not None:
        paths["library_dir"] = str(
            _absolute_path(paths["library_dir"], config_dir)
        )
    if paths["field_file"] is None:
        field_name = paths["field_file_template"].format(
            snapshot=simulation["snapshot"],
            smoothing_scale=simulation["smoothing_scale"],
        )
        paths["field_file"] = str(Path(paths["base_dir"]) / field_name)
    else:
        paths["field_file"] = str(
            _absolute_path(paths["field_file"], config_dir)
        )
    if paths["halo_file"] is None:
        paths["halo_file"] = str(Path(paths["base_dir"]) / "halo_0701.hdf5")
    else:
        paths["halo_file"] = str(
            _absolute_path(paths["halo_file"], config_dir)
        )

    if output["combined_file"] is None:
        scale_tag = _number_token(simulation["smoothing_scale"])
        threshold_tag = _number_token(
            config["tidal_field"]["lambda_threshold"]
        )
        weight_tag = (
            "_mw" if config["assembly_bias"]["mass_weight"]["enabled"]
            else ""
        )
        output["combined_file"] = str(
            Path(paths["base_dir"])
            / "combined"
            / (
                f"{output['run_name']}_s{scale_tag}_"
                f"lth{threshold_tag}{weight_tag}_combined.h5"
            )
        )
    else:
        output["combined_file"] = str(
            _absolute_path(output["combined_file"], config_dir)
        )

    config["_meta"] = {
        "config_file": str(config_path),
        "resolved_for": "unified-runner",
    }
    return config


def dump_config(config):
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def print_config(config):
    print("=== BEGIN RESOLVED YAML CONFIG ===", flush=True)
    print(dump_config(config).rstrip(), flush=True)
    print("=== END RESOLVED YAML CONFIG ===", flush=True)


def _import_runtime(library_dir=None, engine="legacy"):
    """Import the catalog and 3D estimator classes lazily.

    ``library_dir`` is kept for backward compatibility with old YAML files
    (``paths.library_dir``); everything is now imported from ``pywccf``.
    """
    if library_dir is not None and str(library_dir) not in sys.path:
        sys.path.append(str(library_dir))
    from ..samples import HaloSample, TidalField
    from ..estimator.halo_web import Edges

    if engine == "fast":
        from ..estimator.halo_web_fast import HaloWeb3DCross
    else:
        from ..estimator.halo_web import HaloWeb3DCross

    return HaloSample, TidalField, Edges, HaloWeb3DCross


def _sample_positions(sample):
    if "position" in sample:
        positions = np.asarray(sample["position"], dtype=np.float64)
    elif all(key in sample for key in ("x", "y", "z")):
        positions = np.column_stack(
            [sample["x"], sample["y"], sample["z"]]
        ).astype(np.float64, copy=False)
    else:
        raise ValueError("Halo sample needs position or x/y/z fields")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Halo positions must have shape (N, 3), got {positions.shape}"
        )
    return positions


def _sample_true_flat_indices(mask, max_items, seed, chunk_size=5_000_000):
    """Uniformly sample true cells without materializing every true index."""
    flat_mask = np.asarray(mask, dtype=bool).ravel()
    n_true = int(flat_mask.sum())
    if n_true == 0:
        raise ValueError("The requested web mask is empty")
    if max_items is None or n_true <= max_items:
        return np.flatnonzero(flat_mask)

    rng = np.random.default_rng(seed)
    selected_ranks = np.sort(
        rng.choice(n_true, size=max_items, replace=False)
    )
    selected_flat = np.empty(max_items, dtype=np.int64)
    rank_offset = 0
    out_offset = 0
    for start in range(0, flat_mask.size, chunk_size):
        stop = min(start + chunk_size, flat_mask.size)
        local_ids = np.flatnonzero(flat_mask[start:stop])
        next_rank_offset = rank_offset + len(local_ids)
        lo = np.searchsorted(selected_ranks, rank_offset, side="left")
        hi = np.searchsorted(
            selected_ranks, next_rank_offset, side="left"
        )
        if hi > lo:
            local_ranks = selected_ranks[lo:hi] - rank_offset
            n_take = hi - lo
            selected_flat[out_offset : out_offset + n_take] = (
                start + local_ids[local_ranks]
            )
            out_offset += n_take
        rank_offset = next_rank_offset
    if out_offset != max_items:
        raise RuntimeError(
            f"Selected {out_offset} web cells, expected {max_items}"
        )
    return selected_flat


def _bootstrap_worker_1d(args):
    per_center_xi, center_prob, n_iterations, seed = args
    rng = np.random.default_rng(seed)
    samples = np.zeros(
        (n_iterations, per_center_xi.shape[1]), dtype=np.float64
    )
    n_centers = len(per_center_xi)
    for iteration in range(n_iterations):
        ids = rng.choice(
            n_centers, size=n_centers, replace=True, p=center_prob
        )
        samples[iteration] = per_center_xi[ids].mean(axis=0)
    return samples


class HaloWeb1DCross:
    """Spherical, orientation-averaged halo-web cross-correlation."""

    def __init__(
        self,
        center_positions,
        halo_sample,
        box_size,
        radial_edges,
        *,
        center_weights=None,
        halo_weights=None,
        chunk_size=500,
        n_processes=4,
        bootstrap_seed=42,
        verbose=True,
    ):
        self.box_size = float(box_size)
        self.center_positions = (
            np.asarray(center_positions, dtype=np.float64) % self.box_size
        )
        self.halo_positions = (
            _sample_positions(halo_sample) % self.box_size
        )
        self.radial_edges = np.asarray(radial_edges, dtype=np.float64)
        self.chunk_size = int(chunk_size)
        self.n_processes = min(int(n_processes), cpu_count())
        self.bootstrap_seed = int(bootstrap_seed)
        self.verbose = bool(verbose)
        if len(self.center_positions) == 0 or len(self.halo_positions) == 0:
            raise ValueError("Web centers and halo sample must be non-empty")
        if (
            self.radial_edges.ndim != 1
            or len(self.radial_edges) < 2
            or self.radial_edges[0] <= 0
            or np.any(np.diff(self.radial_edges) <= 0)
        ):
            raise ValueError(
                "radial_edges must be strictly increasing and positive"
            )

        if center_weights is None:
            weights = np.ones(len(self.center_positions), dtype=np.float64)
        else:
            weights = np.asarray(center_weights, dtype=np.float64).clip(0.0)
            if len(weights) != len(self.center_positions):
                raise ValueError("center_weights length mismatch")
        if weights.sum() <= 0:
            raise ValueError("All web-center weights are zero")
        self.center_prob = weights / weights.sum()
        # halo weights enter both the pair counts and the mean density, so
        # xi is the cross-correlation of the weighted halo field
        self.halo_weights = _check_halo_weights(
            halo_weights, len(self.halo_positions)
        )

        inner = np.concatenate([[0.0], self.radial_edges[:-1]])
        self.shell_volumes = (
            4.0
            * np.pi
            / 3.0
            * (self.radial_edges**3 - inner**3)
        )
        self.radial_centers = 0.5 * (inner + self.radial_edges)
        self.tree = cKDTree(self.halo_positions, boxsize=self.box_size)
        if self.verbose:
            print(
                f"[1DWeb] N_centers={len(self.center_positions):,}, "
                f"N_halos={len(self.halo_positions):,}, "
                f"N_shells={len(self.radial_edges)}",
                flush=True,
            )

    @classmethod
    def from_field_data(
        cls,
        tidal_field,
        halo_sample,
        radial_edges,
        web_type,
        *,
        use_density_weight=True,
        max_centers=500000,
        center_subsample_seed=0,
        **kwargs,
    ):
        if web_type not in WEB_N_LAMBDA:
            raise ValueError(f"Unknown web type: {web_type}")
        mesh = tidal_field.mesh
        ng = int(mesh.n_grids)
        mask = np.asarray(
            tidal_field.n_lams == WEB_N_LAMBDA[web_type], dtype=bool
        )
        recon_mask = getattr(tidal_field, "recon_mask", None)
        if recon_mask is not None:
            mask &= np.asarray(recon_mask, dtype=bool)
        flat_ids = _sample_true_flat_indices(
            mask,
            max_items=max_centers,
            seed=center_subsample_seed,
        )
        grid_indices = np.column_stack(
            np.unravel_index(flat_ids, (ng, ng, ng))
        )
        center_positions = (
            grid_indices.astype(np.float64) * float(mesh.l_grid)
        )

        center_weights = None
        if use_density_weight:
            delta = np.asarray(tidal_field.delta)
            if delta.shape != (ng, ng, ng):
                raise ValueError(
                    f"delta shape {delta.shape} != {(ng, ng, ng)}"
                )
            center_weights = (
                1.0 + delta.ravel()[flat_ids].astype(np.float64)
            ).clip(0.0)
        return cls(
            center_positions,
            halo_sample,
            mesh.l_box,
            radial_edges,
            center_weights=center_weights,
            **kwargs,
        )

    def _counts_per_center(self):
        n_centers = len(self.center_positions)
        n_shells = len(self.radial_edges)
        counts = np.zeros((n_centers, n_shells), dtype=np.float64)
        r_max = float(self.radial_edges[-1])
        for start in range(0, n_centers, self.chunk_size):
            stop = min(start + self.chunk_size, n_centers)
            neighbor_lists = self.tree.query_ball_point(
                self.center_positions[start:stop], r=r_max
            )
            for local_idx, neighbor_ids in enumerate(neighbor_lists):
                if not neighbor_ids:
                    continue
                center_idx = start + local_idx
                neighbor_ids = np.asarray(neighbor_ids, dtype=np.int64)
                delta = (
                    self.halo_positions[neighbor_ids]
                    - self.center_positions[center_idx]
                )
                delta -= self.box_size * np.round(
                    delta / self.box_size
                )
                distance = np.linalg.norm(delta, axis=1)
                keep = (distance > 1.0e-12) & (distance <= r_max)
                distance = distance[keep]
                if len(distance) == 0:
                    continue
                bin_ids = np.searchsorted(
                    self.radial_edges, distance, side="right"
                )
                np.clip(bin_ids, 0, n_shells - 1, out=bin_ids)
                pair_weights = (
                    None
                    if self.halo_weights is None
                    else self.halo_weights[neighbor_ids[keep]]
                )
                counts[center_idx] = np.bincount(
                    bin_ids, weights=pair_weights, minlength=n_shells
                )[:n_shells]
            if self.verbose and (
                stop == n_centers or start == 0 or stop % 10000 == 0
            ):
                print(
                    f"[1DWeb] accumulated {stop:,}/{n_centers:,} centers",
                    flush=True,
                )
        return counts

    def _jk_regions(self, n_jk_side):
        side = int(n_jk_side)
        cells = np.clip(
            (self.center_positions / self.box_size * side).astype(np.int64),
            0, side - 1,
        )
        return (cells[:, 0] * side + cells[:, 1]) * side + cells[:, 2], side**3

    def corrs(self, n_bootstrap=100, error="bootstrap", n_jk_side=4):
        counts = self._counts_per_center()
        n_halo = (
            len(self.halo_positions)
            if self.halo_weights is None
            else float(self.halo_weights.sum())
        )
        mean_number_density = n_halo / self.box_size**3
        denominator = (
            mean_number_density * self.shell_volumes
        ).clip(1.0e-12)
        per_center_xi = counts / denominator[None, :] - 1.0
        if error == "jackknife":
            # delete-one over sub-cubes of the web cells (same rule as the
            # fast 3D estimator)
            region, n_region = self._jk_regions(n_jk_side)
            prob = self.center_prob
            w_k = np.bincount(region, weights=prob, minlength=n_region)
            s_k = np.zeros((n_region, per_center_xi.shape[1]))
            np.add.at(s_k, region, prob[:, None] * per_center_xi)
            total, w_tot = s_k.sum(axis=0), w_k.sum()
            mean = total / w_tot
            use = (w_k > 0) & (w_tot - w_k > 0)
            n_jk = int(use.sum())
            if n_jk >= 2:
                theta = (total[None, :] - s_k[use]) / (w_tot - w_k[use])[:, None]
                sd = np.sqrt(
                    (n_jk - 1.0) / n_jk
                    * ((theta - theta.mean(axis=0)) ** 2).sum(axis=0)
                )
            else:
                sd = np.zeros_like(mean)
            return {
                "mean": mean,
                "sd": sd,
                "n_jk": n_jk,
                "r_edges": self.radial_edges,
                "r_centers": self.radial_centers,
            }
        n_bootstrap = int(n_bootstrap)
        if n_bootstrap <= 0:
            mean = (
                per_center_xi * self.center_prob[:, None]
            ).sum(axis=0)
            sd = np.zeros_like(mean)
        elif n_bootstrap >= 20 and self.n_processes > 1:
            n_each = n_bootstrap // self.n_processes
            remainder = n_bootstrap % self.n_processes
            seed_rng = np.random.default_rng(self.bootstrap_seed)
            tasks = []
            for process_idx in range(self.n_processes):
                n_iterations = n_each + (process_idx < remainder)
                if n_iterations:
                    tasks.append(
                        (
                            per_center_xi,
                            self.center_prob,
                            n_iterations,
                            int(seed_rng.integers(0, 2**31)),
                        )
                    )
            if self.verbose:
                print(
                    f"[1DWeb] bootstrap: {n_bootstrap} iterations / "
                    f"{len(tasks)} processes",
                    flush=True,
                )
            with Pool(len(tasks)) as pool:
                bootstrap_samples = np.vstack(
                    pool.map(_bootstrap_worker_1d, tasks)
                )
            mean = bootstrap_samples.mean(axis=0)
            sd = bootstrap_samples.std(axis=0)
        else:
            bootstrap_samples = _bootstrap_worker_1d(
                (
                    per_center_xi,
                    self.center_prob,
                    n_bootstrap,
                    self.bootstrap_seed,
                )
            )
            mean = bootstrap_samples.mean(axis=0)
            sd = bootstrap_samples.std(axis=0)
        return {
            "mean": mean,
            "sd": sd,
            "r_edges": self.radial_edges,
            "r_centers": self.radial_centers,
        }


MASS_WEIGHT_KEY = "w_mass_match"
ROW_KEY = "_row_valid"


def _check_halo_weights(halo_weights, n_halo):
    if halo_weights is None:
        return None
    weights = np.asarray(halo_weights, dtype=np.float64)
    if weights.shape != (n_halo,):
        raise ValueError(
            f"halo_weights shape {weights.shape} != ({n_halo},)"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("halo_weights must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("All halo weights are zero")
    return weights


def _mass_match_weights(target_mass, tail_masses, *, sub_bins, max_weight):
    """Weights that give every tail the same mass distribution.

    The target's mass range (inside one mass bin) is cut into ``sub_bins``
    equal-count sub-bins.  Only sub-bins populated by EVERY tail can be
    matched, so the target is restricted to them (``coverage`` = fraction of
    the target kept; 1 means the tails reproduce the full target).  A halo of
    a tail in sub-bin j then gets w = P_target(j) / P_tail(j), normalized to
    a mean of 1, so all tails share one weighted mass distribution (exactly,
    at the sub-bin level, unless weights are capped by ``max_weight``).
    Haloes in sub-bins that some tail lacks get w = 0.
    """
    target_mass = np.asarray(target_mass, dtype=np.float64)
    tail_masses = [np.asarray(m, dtype=np.float64) for m in tail_masses]
    nan_diag = {
        "mass_weight_n_eff_frac": np.nan,
        "mass_weight_max": np.nan,
        "mass_weight_coverage": np.nan,
        "mass_weight_n_capped": 0,
        "mass_weight_n_zero": 0,
        "mean_log10_mass_weighted": np.nan,
        "mean_log10_mass_target": np.nan,
    }
    if len(target_mass) == 0 or any(len(m) == 0 for m in tail_masses):
        return [
            (
                np.ones(len(m)),
                {
                    **nan_diag,
                    "mass_weight_n_eff": float(len(m)),
                    "mean_log10_mass_raw": (
                        float(m.mean()) if len(m) else np.nan
                    ),
                },
            )
            for m in tail_masses
        ]
    inner = np.unique(
        np.quantile(target_mass, np.linspace(0.0, 1.0, sub_bins + 1)[1:-1])
    )
    n_sub = len(inner) + 1
    target_count = np.bincount(
        np.searchsorted(inner, target_mass, side="right"), minlength=n_sub
    ).astype(np.float64)
    subs = [np.searchsorted(inner, m, side="right") for m in tail_masses]
    tail_fracs = [
        np.bincount(sub, minlength=n_sub) / len(sub) for sub in subs
    ]
    covered = np.logical_and.reduce([f > 0 for f in tail_fracs])
    coverage = float(target_count[covered].sum() / target_count.sum())
    target_frac = np.where(covered, target_count, 0.0)
    target_frac /= target_frac.sum()
    target_mean = float(
        np.average(
            target_mass,
            weights=covered[
                np.searchsorted(inner, target_mass, side="right")
            ].astype(np.float64),
        )
    )
    out = []
    for tail_mass, sub, tail_frac in zip(tail_masses, subs, tail_fracs):
        ratio = np.zeros(n_sub)
        ratio[covered] = target_frac[covered] / tail_frac[covered]
        weights = ratio[sub]
        weights /= weights.mean()
        n_capped = 0
        if max_weight is not None:
            for _ in range(20):
                over = weights > max_weight * (1.0 + 1.0e-12)
                if not over.any():
                    break
                n_capped = max(n_capped, int(over.sum()))
                weights = np.minimum(weights, max_weight)
                weights /= weights.mean()
        n_eff = float(weights.sum() ** 2 / (weights**2).sum())
        out.append(
            (
                weights,
                {
                    "mass_weight_n_eff": n_eff,
                    "mass_weight_n_eff_frac": n_eff / len(tail_mass),
                    "mass_weight_max": float(weights.max()),
                    "mass_weight_coverage": coverage,
                    "mass_weight_n_capped": n_capped,
                    "mass_weight_n_zero": int((weights == 0).sum()),
                    "mean_log10_mass_raw": float(tail_mass.mean()),
                    "mean_log10_mass_weighted": float(
                        np.average(tail_mass, weights=weights)
                    ),
                    "mean_log10_mass_target": target_mean,
                },
            )
        )
    return out


def _attach_mass_weights(config, mass_sample, lower, upper):
    """Add MASS_WEIGHT_KEY to both tails; return their weight diagnostics."""
    cfg = config["assembly_bias"]["mass_weight"]
    lower_mass = np.asarray(lower["mass"], dtype=np.float64)
    upper_mass = np.asarray(upper["mass"], dtype=np.float64)
    if cfg["target"] == "parent":
        target_mass = np.asarray(mass_sample["mass"], dtype=np.float64)
    else:
        target_mass = np.concatenate([lower_mass, upper_mass])
    matched = _mass_match_weights(
        target_mass,
        [lower_mass, upper_mass],
        sub_bins=cfg["sub_bins"],
        max_weight=cfg["max_weight"],
    )
    out = []
    for tail, (weights, diag) in zip((lower, upper), matched):
        tail.add_property(
            lambda sample, data=weights: {MASS_WEIGHT_KEY: data}
        )
        out.append(
            {
                "mass_weight": True,
                "mass_weight_key": MASS_WEIGHT_KEY,
                "mass_weight_target": cfg["target"],
                "mass_weight_sub_bins": cfg["sub_bins"],
                "mass_weight_cap": cfg["max_weight"],
                **diag,
            }
        )
    return out


def _scalar_property(halo_sample, key):
    values = np.asarray(halo_sample[key])
    if values.ndim == 1:
        scalar_key = key
        scalar_values = values
    elif values.ndim >= 2:
        scalar_key = f"{key}_mag"
        flattened_width = int(np.prod(values.shape[1:]))
        scalar_values = np.linalg.norm(
            values.reshape(values.shape[0], flattened_width), axis=1
        )
        if scalar_key not in halo_sample:
            halo_sample.add_property(
                lambda sample, name=scalar_key, data=scalar_values: {
                    name: data
                }
            )
    else:
        raise ValueError(f"Bad shape for halo property {key!r}: {values.shape}")
    scalar_values = np.asarray(scalar_values, dtype=np.float64)
    if not np.all(np.isfinite(scalar_values)):
        n_bad = int((~np.isfinite(scalar_values)).sum())
        raise ValueError(
            f"Halo property {key!r} contains {n_bad} non-finite values"
        )
    return scalar_key, scalar_values


def _split_mass_bin(halo_sample, selection):
    scalar_key, values = _scalar_property(
        halo_sample, selection["subproperty"]
    )
    if halo_sample.n_objs == 0:
        return halo_sample, halo_sample, scalar_key, np.nan, np.nan
    tail_fraction = selection["cutoff_percent"] / 100.0
    lower_threshold, upper_threshold = np.quantile(
        values, [tail_fraction, 1.0 - tail_fraction]
    )
    lower = halo_sample.subset_by_p(scalar_key, p_hi=tail_fraction)
    upper = halo_sample.subset_by_p(
        scalar_key, p_lo=1.0 - tail_fraction
    )
    return (
        lower,
        upper,
        scalar_key,
        float(lower_threshold),
        float(upper_threshold),
    )


def _select_mass_samples(config, HaloSample, return_parent=False):
    sample_cfg = config["sample"]
    halo_total = HaloSample.from_file(config["paths"]["halo_file"])
    zf_lo, zf_hi = sample_cfg["formation_redshift_range"]
    halo_valid = halo_total.subset_by_val("zf", lo=zf_lo, hi=zf_hi)
    if sample_cfg["exclude_backsplash"]:
        halo_valid = halo_valid.subset_by_val("is_backsplash", eq=False)
    # row id in halo_valid, carried by every subsample (for the shared 3D
    # neighbour search, which treats subsamples as 0/1 weights)
    halo_valid.add_property(
        lambda sample: {ROW_KEY: np.arange(sample.n_objs, dtype=np.int64)}
    )
    mass_edges = sample_cfg["mass_bin_edges"]
    mass_samples = [
        halo_valid.subset_by_val("mass", lo=lo, hi=hi)
        for lo, hi in zip(mass_edges[:-1], mass_edges[1:])
    ]
    print(f"Total valid halos: {halo_valid.n_objs}", flush=True)
    print("| idx |     log10(M)     | total N |", flush=True)
    print("|-----|------------------|---------|", flush=True)
    for idx, (lo, hi, sample) in enumerate(
        zip(mass_edges[:-1], mass_edges[1:], mass_samples)
    ):
        print(
            f"| {idx:^3d} | {lo:4.1f} <= M < {hi:4.1f} | "
            f"{sample.n_objs:^7d} |",
            flush=True,
        )
    if return_parent:
        parent = halo_valid.subset_by_val(
            "mass", lo=mass_edges[0], hi=mass_edges[-1]
        )
        return mass_samples, parent
    return mass_samples


def _median_log_mass(sample):
    """Median log10 mass of a sample (NaN when empty): the representative
    mass of a bin, finite also for an open-ended bin (upper edge .inf)."""
    if sample.n_objs == 0:
        return float("nan")
    return float(np.median(np.asarray(sample["mass"], dtype=np.float64)))


def _write_mass_medians(output_path, mass_samples):
    medians = np.array([_median_log_mass(s) for s in mass_samples])
    with h5py.File(output_path, "r+", locking=False) as stream:
        name = "metadata/mass_bin_median_log10_msun_h"
        if name in stream:
            del stream[name]
        stream.create_dataset(name, data=medians)


def _total_group(config, mass_samples):
    edges = config["sample"]["mass_bin_edges"]
    metadata = [
        {
            "mass_bin_index": index,
            "mass_bin_lower_log10_msun_h": edges[index],
            "mass_bin_upper_log10_msun_h": edges[index + 1],
            "mass_median_log10_msun_h": _median_log_mass(sample),
            "sample_size": sample.n_objs,
        }
        for index, sample in enumerate(mass_samples)
    ]
    return {
        "sample": "total",
        "tail": "all",
        "label": "all",
        "samples": mass_samples,
        "metadata": metadata,
    }


def _assembly_groups(config, selection, mass_samples):
    edges = config["sample"]["mass_bin_edges"]
    weighted = config["assembly_bias"]["mass_weight"]["enabled"]
    weight_rows = []
    lower_samples = []
    upper_samples = []
    lower_metadata = []
    upper_metadata = []
    rows = []
    for index, (lo, hi, mass_sample) in enumerate(
        zip(edges[:-1], edges[1:], mass_samples)
    ):
        lower, upper, scalar_key, lower_cut, upper_cut = _split_mass_bin(
            mass_sample, selection
        )
        if weighted:
            lower_weight, upper_weight = _attach_mass_weights(
                config, mass_sample, lower, upper
            )
            weight_rows.append(
                (index, lo, hi, lower_weight, upper_weight)
            )
        else:
            lower_weight = upper_weight = {"mass_weight": False}
        common = {
            "assembly_subproperty": selection["subproperty"],
            "assembly_scalar_property": scalar_key,
            "assembly_cutoff_percent": selection["cutoff_percent"],
            "mass_bin_index": index,
            "mass_bin_lower_log10_msun_h": lo,
            "mass_bin_upper_log10_msun_h": hi,
        }
        lower_samples.append(lower)
        upper_samples.append(upper)
        lower_metadata.append(
            {
                **common,
                "assembly_label": selection["lower_label"],
                "assembly_tail": "lower",
                "assembly_threshold": lower_cut,
                "sample_size": lower.n_objs,
                "mass_median_log10_msun_h": _median_log_mass(lower),
                **lower_weight,
            }
        )
        upper_metadata.append(
            {
                **common,
                "assembly_label": selection["upper_label"],
                "assembly_tail": "upper",
                "assembly_threshold": upper_cut,
                "sample_size": upper.n_objs,
                "mass_median_log10_msun_h": _median_log_mass(upper),
                **upper_weight,
            }
        )
        rows.append(
            (
                index,
                lo,
                hi,
                mass_sample.n_objs,
                lower.n_objs,
                lower_cut,
                upper.n_objs,
                upper_cut,
            )
        )

    print(
        f"\nAssembly split {selection['name']!r}: "
        f"{selection['cutoff_percent']:g}% tails of "
        f"{selection['subproperty']!r}",
        flush=True,
    )
    print(
        "| idx |     log10(M)     | all | low N | low cut | "
        "high N | high cut |",
        flush=True,
    )
    for row in rows:
        index, lo, hi, n_all, n_low, cut_low, n_high, cut_high = row
        print(
            f"| {index:^3d} | {lo:4.1f} <= M < {hi:4.1f} | "
            f"{n_all:^3d} | {n_low:^5d} | {cut_low:^7.4g} | "
            f"{n_high:^6d} | {cut_high:^8.4g} |",
            flush=True,
        )
    if weighted:
        _print_mass_weights(config, selection, weight_rows)
    weight_key = MASS_WEIGHT_KEY if weighted else None
    return [
        {
            "sample": selection["name"],
            "tail": "low",
            "label": selection["lower_label"],
            "samples": lower_samples,
            "metadata": lower_metadata,
            "weight_key": weight_key,
        },
        {
            "sample": selection["name"],
            "tail": "high",
            "label": selection["upper_label"],
            "samples": upper_samples,
            "metadata": upper_metadata,
            "weight_key": weight_key,
        },
    ]


def _print_mass_weights(config, selection, weight_rows):
    cfg = config["assembly_bias"]["mass_weight"]
    cap = "none" if cfg["max_weight"] is None else f"{cfg['max_weight']:g}"
    print(
        f"Mass matching ({selection['name']!r}): target={cfg['target']}, "
        f"{cfg['sub_bins']} sub-bins, weight cap={cap}",
        flush=True,
    )
    print(
        "| idx |     log10(M)     | <logM> target | tail  | <logM> raw "
        "-> weighted | N_eff/N | w_max | coverage | capped | w=0 |",
        flush=True,
    )
    for index, lo, hi, lower_weight, upper_weight in weight_rows:
        for name, diag in (
            (selection["lower_label"], lower_weight),
            (selection["upper_label"], upper_weight),
        ):
            print(
                f"| {index:^3d} | {lo:4.1f} <= M < {hi:4.1f} | "
                f"{diag['mean_log10_mass_target']:^13.4f} | {name:^5s} | "
                f"{diag['mean_log10_mass_raw']:.4f} -> "
                f"{diag['mean_log10_mass_weighted']:.4f}   | "
                f"{diag['mass_weight_n_eff_frac']:^7.3f} | "
                f"{diag['mass_weight_max']:^5.2f} | "
                f"{diag['mass_weight_coverage']:^8.3f} | "
                f"{diag['mass_weight_n_capped']:^6d} | "
                f"{diag['mass_weight_n_zero']:^3d} |",
                flush=True,
            )


def _load_tidal_data(config, TidalField, need_eigenvectors):
    field_file = Path(config["paths"]["field_file"])
    with h5py.File(field_file, "r", locking=False) as stream:
        required = {
            "n_grids",
            "l_box",
            "smoothing_scale",
        }
        missing = required - set(stream)
        if missing:
            raise KeyError(
                f"{field_file} is missing field datasets {sorted(missing)}"
            )
        ng = int(np.asarray(stream["n_grids"][()]).item())
        box_size = float(np.asarray(stream["l_box"][()]).item())
        field_scale = float(
            np.asarray(stream["smoothing_scale"][()]).item()
        )
        stored_threshold = (
            float(np.asarray(stream["lambda_th"][()]).item())
            if "lambda_th" in stream
            else 0.0
        )
        evecs = None
        if need_eigenvectors:
            if "evec" not in stream:
                raise KeyError(f"{field_file} is missing the evec dataset")
            evecs = np.asarray(stream["evec"], dtype=np.float32)
            expected_shape = (ng, ng, ng, 3, 3)
            if evecs.shape != expected_shape:
                raise ValueError(
                    f"{field_file}: evec shape {evecs.shape} != "
                    f"{expected_shape}"
                )

    requested_scale = config["simulation"]["smoothing_scale"]
    if not np.isclose(field_scale, requested_scale):
        raise ValueError(
            f"YAML smoothing scale {requested_scale} does not match "
            f"field-file value {field_scale}"
        )
    requested_threshold = config["tidal_field"]["lambda_threshold"]
    used_threshold = (
        stored_threshold
        if requested_threshold is None
        else float(requested_threshold)
    )

    print(
        f"[load] ng={ng}, L_box={box_size:g}, "
        f"R_smooth={field_scale:g}, lambda_th(file)={stored_threshold:g}, "
        f"lambda_th(used)={used_threshold:g}",
        flush=True,
    )
    print(f"[load] reading TidalField from {field_file} ...", flush=True)
    tidal_field = TidalField.from_file(field_file, recon_only=False)
    if not hasattr(tidal_field, "lams"):
        raise AttributeError(
            "TidalField must expose eigenvalues as .lams to override "
            "lambda_threshold"
        )
    tidal_field.lam_off = used_threshold
    tidal_field.n_lams = (
        np.asarray(tidal_field.lams) >= used_threshold
    ).sum(axis=-1)
    if tidal_field.n_lams.shape != (ng, ng, ng):
        raise ValueError(
            f"{field_file}: tidal eigenvalue grid produces "
            f"{tidal_field.n_lams.shape}, expected {(ng, ng, ng)}"
        )
    cache = getattr(tidal_field, "_xs_web_t_cache", None)
    if cache is not None:
        cache.clear()

    web_masks = {}
    recon_mask = getattr(tidal_field, "recon_mask", None)
    for web, n_lambda in WEB_N_LAMBDA.items():
        mask = np.asarray(tidal_field.n_lams == n_lambda, dtype=bool)
        if recon_mask is not None:
            mask &= np.asarray(recon_mask, dtype=bool)
        web_masks[web] = mask
    print(
        "[load] classified cells: "
        + "  ".join(
            f"{web}={int(web_masks[web].sum()):,}"
            for web in WEB_N_LAMBDA
        ),
        flush=True,
    )
    return (
        tidal_field,
        evecs,
        web_masks,
        field_scale,
        stored_threshold,
        used_threshold,
    )


def _entry_path(dimension, sample, tail, web, mass_bin_index):
    return (
        f"/data/{dimension}/{sample}/{tail}/{web}/"
        f"mass_bin_{mass_bin_index:03d}"
    )


def build_run_plan(config):
    """Return the exact set of result records requested by the YAML."""
    selections = []
    if "total" in config["run"]["samples"]:
        selections.append(("total", "all"))
    if "subproperty" in config["run"]["samples"]:
        for selection in config["assembly_bias"]["selections"]:
            selections.extend(
                [
                    (selection["name"], "low"),
                    (selection["name"], "high"),
                ]
            )

    n_mass = len(config["sample"]["mass_bin_edges"]) - 1
    plan = []
    for dimension in config["run"]["dimensions"]:
        for sample, tail in selections:
            for web in config["correlation"]["web_types"]:
                for mass_bin_index in range(n_mass):
                    plan.append(
                        {
                            "dimension": dimension,
                            "sample": sample,
                            "tail": tail,
                            "web": web,
                            "mass_bin_index": mass_bin_index,
                            "path": _entry_path(
                                dimension,
                                sample,
                                tail,
                                web,
                                mass_bin_index,
                            ),
                        }
                    )
    return plan


def _create_string_dataset(
    group, name, values, *, resizable=False
):
    dtype = h5py.string_dtype(encoding="utf-8")
    values = list(values)
    kwargs = {}
    if resizable:
        kwargs = {"maxshape": (None,), "chunks": True}
    return group.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=dtype,
        **kwargs,
    )


def _append_string(dataset, value):
    old_size = dataset.shape[0]
    dataset.resize((old_size + 1,))
    dataset[old_size] = str(value)


def _set_attr(attrs, key, value):
    if value is None:
        attrs[key] = "null"
    elif isinstance(value, (str, bytes, bool, int, float, np.number)):
        attrs[key] = value
    else:
        attrs[key] = str(value)


def _initialize_output(config, plan):
    output_path = Path(config["output"]["combined_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not config["output"]["overwrite"]:
            raise FileExistsError(
                f"{output_path} already exists; set output.overwrite: true "
                "or pass --overwrite to replace it"
            )
        output_path.unlink()

    mass_edges = config["sample"]["mass_bin_edges"]
    radial_bins = config["correlation"]["radial_bins"]
    samples = list(
        dict.fromkeys(record["sample"] for record in plan)
    )
    dimensions = list(
        dict.fromkeys(record["dimension"] for record in plan)
    )
    with h5py.File(output_path, "w", locking=False) as stream:
        stream.attrs["format_name"] = FORMAT_NAME
        stream.attrs["schema_version"] = SCHEMA_VERSION
        stream.attrs["validation_status"] = "pending"
        stream.attrs["run_status"] = "running"
        stream.attrs["created_utc"] = _utc_now()
        stream.attrs["generator"] = "code/run.py"
        stream.attrs["source_root"] = config["paths"]["base_dir"]
        stream.attrs["smoothing_scale_mpc_h"] = config["simulation"][
            "smoothing_scale"
        ]
        requested_threshold = config["tidal_field"]["lambda_threshold"]
        stream.attrs["lambda_threshold_requested"] = (
            np.nan
            if requested_threshold is None
            else requested_threshold
        )
        stream.attrs["record_count"] = 0
        stream.attrs["planned_record_count"] = len(plan)
        stream.attrs["log_count"] = 0
        stream.attrs["sidecar_count"] = 0

        metadata = stream.create_group("metadata")
        metadata.create_dataset("mass_bin_edges", data=mass_edges)
        metadata.create_dataset("radial_bins", data=radial_bins)
        _create_string_dataset(
            metadata, "web_types", config["correlation"]["web_types"]
        )
        _create_string_dataset(metadata, "samples", samples)
        _create_string_dataset(metadata, "dimensions", dimensions)
        metadata.attrs["mass_unit"] = "log10(M_200m / (h^-1 Msun))"
        metadata.attrs["radial_unit"] = "h^-1 Mpc"

        data = stream.create_group("data")
        data.create_group("1d")
        data.create_group("3d")

        index = stream.create_group("index")
        _create_string_dataset(
            index, "record_paths", [], resizable=True
        )
        _create_string_dataset(
            index, "source_paths", [], resizable=True
        )
        _create_string_dataset(
            index, "source_sha256", [], resizable=True
        )
        _create_string_dataset(
            index,
            "planned_record_paths",
            [record["path"] for record in plan],
        )
        _create_string_dataset(
            index, "planned_status", ["pending"] * len(plan)
        )
        _create_string_dataset(
            index, "planned_error", [""] * len(plan)
        )
        _create_string_dataset(
            index, "log_paths", [], resizable=True
        )
        _create_string_dataset(
            index, "sidecar_paths", [], resizable=True
        )

        provenance = stream.create_group("provenance")
        provenance.create_group("logs")
        provenance.create_group("sidecars")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        provenance.create_dataset(
            "resolved_config_yaml",
            data=dump_config(config),
            dtype=string_dtype,
        )
    return output_path


def _plan_index(plan):
    return {record["path"]: index for index, record in enumerate(plan)}


def _update_plan_status(
    output_path, plan_lookup, record_path, status, error=""
):
    index = plan_lookup[record_path]
    with h5py.File(output_path, "r+", locking=False) as stream:
        stream["index/planned_status"][index] = status
        stream["index/planned_error"][index] = str(error)


def _dataset_options(config):
    compression = config["output"]["compression"]
    if compression is None:
        return {}
    options = {"compression": compression}
    if compression == "gzip":
        options["compression_opts"] = config["output"]["compression_level"]
    return options


def _save_result(
    output_path,
    config,
    plan_lookup,
    *,
    dimension,
    sample,
    tail,
    web,
    mass_bin_index,
    result,
    sample_metadata,
    selection_label,
    used_threshold,
):
    path = _entry_path(
        dimension, sample, tail, web, mass_bin_index
    )
    xi = np.asarray(result["mean"], dtype=np.float64)
    xi_sd = np.asarray(result["sd"], dtype=np.float64)
    if xi.shape != xi_sd.shape:
        raise ValueError(f"{path}: mean/sd shapes do not match")
    options = _dataset_options(config)
    corr = config["correlation"]
    label = config["sample"]["label"]
    weighted = bool(sample_metadata.get("mass_weight", False))
    weight_word = "Mass-matched (halo-weighted)" if weighted else "Unweighted"

    with h5py.File(output_path, "r+", locking=False) as stream:
        entry = stream.require_group(path.lstrip("/"))
        if "original" in entry:
            raise ValueError(f"{path} has already been written")
        entry.attrs["dimension"] = dimension
        entry.attrs["sample"] = sample
        entry.attrs["tail"] = tail
        entry.attrs["web"] = web
        entry.attrs["tracer"] = label
        entry.attrs["mass_bin_index"] = mass_bin_index
        entry.attrs["mass_bin_lower_log10_msun_h"] = sample_metadata[
            "mass_bin_lower_log10_msun_h"
        ]
        entry.attrs["mass_bin_upper_log10_msun_h"] = sample_metadata[
            "mass_bin_upper_log10_msun_h"
        ]
        entry.attrs["source_kind"] = "direct_calculation"
        entry.attrs["source_relative_path"] = ""
        entry.attrs["source_sha256"] = ""
        entry.attrs["source_byte_count"] = -1

        original = entry.create_group("original")
        if dimension == "3d":
            corr_label = (
                f"{weight_word} {selection_label} {label} sample "
                f"({mass_bin_index}) x {web} 3D cross-correlation in the "
                "local tidal eigenframe"
            )
            original.attrs["axes"] = (
                "axis 0 = e1 (lambda_1, most-compressing); axis 1 = e2; "
                "axis 2 = e3 (lambda_3, most-extending). Index 0 along "
                "each axis is the underflow bin."
            )
        else:
            corr_label = (
                f"{weight_word} {selection_label} {label} sample "
                f"({mass_bin_index}) x {web} isotropic 1D "
                "cross-correlation"
            )
            original.attrs["geometry"] = (
                "isotropic spherical shells; no local tidal eigenframe"
            )
        original.attrs["label"] = corr_label
        original.attrs["halo_weight"] = (
            sample_metadata.get("mass_weight_key", MASS_WEIGHT_KEY)
            if weighted
            else "none"
        )
        original.attrs["smoothing_scale_mpc_h"] = config["simulation"][
            "smoothing_scale"
        ]
        original.attrs["lambda_threshold"] = used_threshold
        original.attrs["n_bootstrap"] = corr["n_bootstrap"]
        original.attrs["error_method"] = corr["error"]
        original.attrs["n_jk_side"] = corr["n_jk_side"]
        original.attrs["n_jk"] = (
            int(result["n_jk"]) if "n_jk" in result else -1
        )
        original.attrs["max_centers"] = (
            -1 if corr["max_centers"] is None else corr["max_centers"]
        )
        original.attrs["resolved_config_yaml"] = dump_config(config)
        for key, value in sample_metadata.items():
            _set_attr(original.attrs, key, value)

        original.create_dataset("xi", data=xi, **options)
        original.create_dataset("xi_sd", data=xi_sd, **options)
        original.create_dataset("xi_p16", data=xi - xi_sd, **options)
        original.create_dataset("xi_p84", data=xi + xi_sd, **options)
        if dimension == "3d":
            for axis in (1, 2, 3):
                original.create_dataset(
                    f"r{axis}_edges", data=result[f"r{axis}_edges"]
                )
                original.create_dataset(
                    f"r{axis}_centers",
                    data=result[f"r{axis}_centers"],
                )
        else:
            original.create_dataset("r_edges", data=result["r_edges"])
            original.create_dataset(
                "r_centers", data=result["r_centers"]
            )

        _append_string(stream["index/record_paths"], path)
        _append_string(
            stream["index/source_paths"], "direct:code/run.py"
        )
        _append_string(stream["index/source_sha256"], "")
        plan_id = plan_lookup[path]
        stream["index/planned_status"][plan_id] = "complete"
        stream["index/planned_error"][plan_id] = ""
        stream.attrs["record_count"] = (
            int(stream.attrs["record_count"]) + 1
        )


def _set_runtime_field_metadata(
    output_path, stored_threshold, used_threshold
):
    with h5py.File(output_path, "r+", locking=False) as stream:
        stream.attrs["field_lambda_threshold"] = stored_threshold
        stream.attrs["lambda_threshold"] = used_threshold
        stream["metadata"].attrs["field_lambda_threshold"] = (
            stored_threshold
        )
        stream["metadata"].attrs["lambda_threshold"] = used_threshold


def _decode_strings(dataset):
    return [
        value.decode("utf-8")
        if isinstance(value, bytes)
        else str(value)
        for value in dataset[()]
    ]


def _validate_completed_output(output_path):
    """Validate the direct output before marking it complete."""
    with h5py.File(output_path, "r+", locking=False) as stream:
        if stream.attrs.get("format_name") != FORMAT_NAME:
            raise ValueError("Output format_name is invalid")
        if stream.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Output schema_version is invalid")
        plan_paths = _decode_strings(
            stream["index/planned_record_paths"]
        )
        statuses = _decode_strings(stream["index/planned_status"])
        errors = _decode_strings(stream["index/planned_error"])
        if not (
            len(plan_paths) == len(statuses) == len(errors)
        ):
            raise ValueError("Run-plan index arrays have different lengths")
        bad_status = [
            (path, status, error)
            for path, status, error in zip(plan_paths, statuses, errors)
            if status not in TERMINAL_PLAN_STATUSES
        ]
        if bad_status:
            raise ValueError(
                f"Run plan has unfinished records: {bad_status[:3]}"
            )

        record_paths = _decode_strings(stream["index/record_paths"])
        if len(record_paths) != len(set(record_paths)):
            raise ValueError("Output record_paths contains duplicates")
        expected = {
            path
            for path, status in zip(plan_paths, statuses)
            if status == "complete"
        }
        if set(record_paths) != expected:
            raise ValueError(
                "Completed plan records and output record_paths differ"
            )
        for path in record_paths:
            original_path = path.lstrip("/") + "/original"
            if original_path not in stream:
                raise KeyError(f"Missing {original_path}")
            original = stream[original_path]
            for name in ("xi", "xi_sd"):
                if name not in original:
                    raise KeyError(f"{original.name} is missing {name}")
            if original["xi"].shape != original["xi_sd"].shape:
                raise ValueError(
                    f"{original.name} has incompatible xi shapes"
                )
            dimension = original.parent.attrs["dimension"]
            expected_ndim = 1 if dimension == "1d" else 3
            if original["xi"].ndim != expected_ndim:
                raise ValueError(
                    f"{original.name}: expected {expected_ndim}D xi"
                )
        if int(stream.attrs["record_count"]) != len(record_paths):
            raise ValueError("Output record_count is inconsistent")

        n_skipped = sum(status.startswith("skipped_") for status in statuses)
        stream.attrs["validation_status"] = "complete"
        stream.attrs["run_status"] = (
            "complete_with_skips" if n_skipped else "complete"
        )
        stream.attrs["skipped_record_count"] = n_skipped
        stream.attrs["validated_utc"] = _utc_now()
        stream.attrs["finished_utc"] = _utc_now()


def _mark_failed(output_path, error, interrupted=False):
    if not Path(output_path).is_file():
        return
    with h5py.File(output_path, "r+", locking=False) as stream:
        stream.attrs["validation_status"] = "incomplete"
        stream.attrs["run_status"] = (
            "interrupted" if interrupted else "failed"
        )
        stream.attrs["failure_type"] = type(error).__name__
        stream.attrs["failure_message"] = str(error)
        stream.attrs["finished_utc"] = _utc_now()


def _embed_log(output_path, log_path):
    raw = Path(log_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with h5py.File(output_path, "r+", locking=False) as stream:
        logs = stream["provenance/logs"]
        if "file_000" in logs:
            del logs["file_000"]
        entry = logs.create_group("file_000")
        entry.attrs["source_relative_path"] = "run.log"
        entry.attrs["source_sha256"] = digest
        entry.attrs["source_byte_count"] = len(raw)
        entry.create_dataset(
            "raw_bytes",
            data=np.frombuffer(raw, dtype=np.uint8),
            compression="gzip",
            compression_opts=6,
        )
        index = stream["index"]
        if "log_paths" in index:
            del index["log_paths"]
        _create_string_dataset(
            index, "log_paths", ["run.log"], resizable=True
        )
        stream.attrs["log_count"] = 1


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(
            getattr(stream, "isatty", lambda: False)()
            for stream in self.streams
        )


def _run_group(
    config,
    output_path,
    plan_lookup,
    group,
    dimension,
    web,
    tidal_field,
    evecs,
    web_masks,
    edges_3d,
    HaloWeb3DCross,
    used_threshold,
):
    corr = config["correlation"]
    for mass_bin_index, halo_sample in enumerate(group["samples"]):
        path = _entry_path(
            dimension,
            group["sample"],
            group["tail"],
            web,
            mass_bin_index,
        )
        if not web_masks[web].any():
            print(f"[skip] empty {web} web mask for {path}", flush=True)
            _update_plan_status(
                output_path,
                plan_lookup,
                path,
                "skipped_empty_web",
                (
                    f"no {web} cells at lambda_threshold="
                    f"{used_threshold:g}"
                ),
            )
            continue
        if halo_sample.n_objs == 0:
            print(f"[skip] empty halo sample for {path}", flush=True)
            _update_plan_status(
                output_path,
                plan_lookup,
                path,
                "skipped_empty",
                "halo sample contains zero objects",
            )
            continue
        print(
            f"\n=== [{dimension.upper()}Web] "
            f"{group['sample']}/{group['tail']}/{web} "
            f"idx={mass_bin_index} (N={halo_sample.n_objs}) ===",
            flush=True,
        )
        weight_key = group.get("weight_key")
        halo_weights = (
            None
            if weight_key is None
            else np.asarray(halo_sample[weight_key], dtype=np.float64)
        )
        try:
            if dimension == "3d":
                cross = HaloWeb3DCross.from_field_data(
                    evecs=evecs,
                    halo_sample=halo_sample,
                    tidal_field=tidal_field,
                    edges=edges_3d,
                    web_mask=web_masks[web],
                    web_type=web,
                    use_density_weight=corr["use_density_weight"],
                    chunk_size=corr["chunk_size"],
                    use_disk_cache=corr["use_disk_cache"],
                    n_processes=corr["n_processes"],
                    max_centers=corr["max_centers"],
                    rng=corr["bootstrap_seed"],
                    test_mode=False,
                    verbose=True,
                )
                if halo_weights is None:
                    result = cross.corrs(
                        n_bootstrap=corr["n_bootstrap"],
                        bootstrap=corr["n_bootstrap"] > 0,
                        mass_weighted=False,
                        weight_key=None,
                    )
                else:
                    # explicit per-halo values (no silent fallback to 1)
                    result = cross.corrs(
                        field_vals=halo_weights,
                        n_bootstrap=corr["n_bootstrap"],
                        bootstrap=corr["n_bootstrap"] > 0,
                        mass_weighted=False,
                        weight_key=None,
                    )
            else:
                cross = HaloWeb1DCross.from_field_data(
                    tidal_field,
                    halo_sample,
                    np.asarray(corr["radial_bins"], dtype=np.float64),
                    web,
                    use_density_weight=corr["use_density_weight"],
                    max_centers=corr["max_centers"],
                    center_subsample_seed=corr[
                        "center_subsample_seed"
                    ],
                    chunk_size=corr["chunk_size"],
                    n_processes=corr["n_processes"],
                    bootstrap_seed=corr["bootstrap_seed"],
                    halo_weights=halo_weights,
                    verbose=True,
                )
                result = cross.corrs(
                    n_bootstrap=corr["n_bootstrap"],
                    error=corr["error"],
                    n_jk_side=corr["n_jk_side"],
                )
            _save_result(
                output_path,
                config,
                plan_lookup,
                dimension=dimension,
                sample=group["sample"],
                tail=group["tail"],
                web=web,
                mass_bin_index=mass_bin_index,
                result=result,
                sample_metadata=group["metadata"][mass_bin_index],
                selection_label=group["label"],
                used_threshold=used_threshold,
            )
            print(f"[saved] {path}", flush=True)
            del cross
        except BaseException as error:
            _update_plan_status(
                output_path,
                plan_lookup,
                path,
                "failed",
                f"{type(error).__name__}: {error}",
            )
            raise


def _run_3d_shared(
    config,
    output_path,
    plan_lookup,
    groups,
    parent,
    tidal_field,
    evecs,
    web_masks,
    edges_3d,
    HaloWeb3DCross,
    used_threshold,
):
    """All 3D records of one web type from one neighbour search.

    Every (sample, tail, mass bin) is a subsample of ``parent``; it enters
    ``HaloWeb3DCross.corrs_many`` as a weight vector over the parent
    (membership, times the mass-matching weight when enabled).  The
    estimator normalises by the sum of the weights, so each result equals
    the one of a separate run on that subsample.
    """
    corr = config["correlation"]
    parent_rows = np.asarray(parent[ROW_KEY])
    n_parent = len(parent_rows)
    for web in corr["web_types"]:
        records = []
        for group in groups:
            for mass_bin_index, halo_sample in enumerate(group["samples"]):
                path = _entry_path(
                    "3d", group["sample"], group["tail"], web, mass_bin_index
                )
                if not web_masks[web].any():
                    print(f"[skip] empty {web} web mask for {path}",
                          flush=True)
                    _update_plan_status(
                        output_path, plan_lookup, path, "skipped_empty_web",
                        f"no {web} cells at lambda_threshold="
                        f"{used_threshold:g}",
                    )
                    continue
                if halo_sample.n_objs == 0:
                    print(f"[skip] empty halo sample for {path}", flush=True)
                    _update_plan_status(
                        output_path, plan_lookup, path, "skipped_empty",
                        "halo sample contains zero objects",
                    )
                    continue
                rows = np.asarray(halo_sample[ROW_KEY])
                where = np.searchsorted(parent_rows, rows)
                where = np.clip(where, 0, n_parent - 1)
                if not np.array_equal(parent_rows[where], rows):
                    raise RuntimeError(f"{path}: sample is not a subsample "
                                       "of the parent catalogue")
                weights = np.zeros(n_parent, dtype=np.float64)
                weight_key = group.get("weight_key")
                weights[where] = (
                    1.0 if weight_key is None
                    else np.asarray(halo_sample[weight_key], dtype=np.float64)
                )
                records.append((group, mass_bin_index, path, weights))
        if not records:
            continue
        print(
            f"\n=== [3DWeb shared] {web}: {len(records)} records from one "
            f"neighbour search over {n_parent} haloes ===",
            flush=True,
        )
        saved = set()
        try:
            cross = HaloWeb3DCross.from_field_data(
                evecs=evecs,
                halo_sample=parent,
                tidal_field=tidal_field,
                edges=edges_3d,
                web_mask=web_masks[web],
                web_type=web,
                use_density_weight=corr["use_density_weight"],
                chunk_size=corr["chunk_size"],
                n_processes=corr["n_processes"],
                max_centers=corr["max_centers"],
                rng=corr["bootstrap_seed"],
                test_mode=False,
                verbose=True,
                cache_pairs=False,
                error=corr["error"],
                n_jk_side=corr["n_jk_side"],
            )
            results = cross.corrs_many(
                [record[3] for record in records],
                n_bootstrap=corr["n_bootstrap"],
                bootstrap=corr["n_bootstrap"] > 0,
            )
            for (group, mass_bin_index, path, _), result in zip(
                records, results
            ):
                saved.add(path)
                _save_result(
                    output_path,
                    config,
                    plan_lookup,
                    dimension="3d",
                    sample=group["sample"],
                    tail=group["tail"],
                    web=web,
                    mass_bin_index=mass_bin_index,
                    result=result,
                    sample_metadata=group["metadata"][mass_bin_index],
                    selection_label=group["label"],
                    used_threshold=used_threshold,
                )
                print(f"[saved] {path}", flush=True)
            del cross
        except BaseException as error:
            for _, _, path, _ in records:
                if path in plan_lookup and path not in saved:
                    _update_plan_status(
                        output_path, plan_lookup, path, "failed",
                        f"{type(error).__name__}: {error}",
                    )
            raise


def _run_calculations(config, output_path, plan, runtime=None):
    engine = config["correlation"].get("engine", "legacy")
    if runtime is None:
        runtime = _import_runtime(config["paths"]["library_dir"], engine)
    HaloSample, TidalField, Edges, HaloWeb3DCross = runtime
    need_3d = "3d" in config["run"]["dimensions"]
    (
        tidal_field,
        evecs,
        web_masks,
        _,
        stored_threshold,
        used_threshold,
    ) = _load_tidal_data(config, TidalField, need_3d)
    _set_runtime_field_metadata(
        output_path, stored_threshold, used_threshold
    )
    shared_3d = engine == "fast" and need_3d
    if shared_3d:
        mass_samples, parent = _select_mass_samples(
            config, HaloSample, return_parent=True
        )
    else:
        mass_samples = _select_mass_samples(config, HaloSample)
    _write_mass_medians(output_path, mass_samples)
    edges_3d = (
        Edges(
            np.asarray(
                config["correlation"]["radial_bins"], dtype=np.float64
            )
        )
        if need_3d
        else None
    )
    plan_lookup = _plan_index(plan)

    if shared_3d:
        groups = []
        if "total" in config["run"]["samples"]:
            groups.append(_total_group(config, mass_samples))
        if "subproperty" in config["run"]["samples"]:
            for selection in config["assembly_bias"]["selections"]:
                groups.extend(_assembly_groups(config, selection, mass_samples))
        _run_3d_shared(
            config, output_path, plan_lookup, groups, parent, tidal_field,
            evecs, web_masks, edges_3d, HaloWeb3DCross, used_threshold,
        )
        if "1d" in config["run"]["dimensions"]:
            for group in groups:
                for web in config["correlation"]["web_types"]:
                    _run_group(
                        config, output_path, plan_lookup, group, "1d", web,
                        tidal_field, evecs, web_masks, edges_3d,
                        HaloWeb3DCross, used_threshold,
                    )
        return

    if "total" in config["run"]["samples"]:
        groups = [_total_group(config, mass_samples)]
        for group in groups:
            for dimension in config["run"]["dimensions"]:
                for web in config["correlation"]["web_types"]:
                    _run_group(
                        config,
                        output_path,
                        plan_lookup,
                        group,
                        dimension,
                        web,
                        tidal_field,
                        evecs,
                        web_masks,
                        edges_3d,
                        HaloWeb3DCross,
                        used_threshold,
                    )

    if "subproperty" in config["run"]["samples"]:
        for selection in config["assembly_bias"]["selections"]:
            groups = _assembly_groups(
                config, selection, mass_samples
            )
            for group in groups:
                for dimension in config["run"]["dimensions"]:
                    for web in config["correlation"]["web_types"]:
                        _run_group(
                            config,
                            output_path,
                            plan_lookup,
                            group,
                            dimension,
                            web,
                            tidal_field,
                            evecs,
                            web_masks,
                            edges_3d,
                            HaloWeb3DCross,
                            used_threshold,
                        )
            del groups


def run(config, *, runtime=None):
    """Execute one resolved config and return the combined HDF5 path.

    ``runtime`` is an optional four-class tuple used by lightweight tests:
    ``(HaloSample, TidalField, Edges, HaloWeb3DCross)``.
    """
    plan = build_run_plan(config)
    output_path = _initialize_output(config, plan)
    log_path = output_path.with_name(output_path.name + ".running.log")
    failure = None
    failure_traceback = None

    with log_path.open("w", encoding="utf-8", buffering=1) as log_stream:
        tee_stdout = _Tee(sys.stdout, log_stream)
        tee_stderr = _Tee(sys.stderr, log_stream)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            try:
                print_config(config)
                print(
                    f"Run plan: {len(plan)} records -> {output_path}",
                    flush=True,
                )
                _run_calculations(
                    config, output_path, plan, runtime=runtime
                )
                _validate_completed_output(output_path)
                print(
                    f"\nCompleted and validated: {output_path}",
                    flush=True,
                )
            except BaseException as error:
                failure = error
                failure_traceback = error.__traceback__
                traceback.print_exc()
                _mark_failed(
                    output_path,
                    error,
                    interrupted=isinstance(error, KeyboardInterrupt),
                )

    try:
        _embed_log(output_path, log_path)
    except BaseException as log_error:
        if failure is None:
            failure = log_error
            failure_traceback = log_error.__traceback__
            _mark_failed(output_path, log_error)
    else:
        log_path.unlink()

    if failure is not None:
        raise failure.with_traceback(failure_traceback)
    return output_path


def _print_plan(config):
    plan = build_run_plan(config)
    print_config(config)
    print(f"Planned result records: {len(plan)}")
    by_dimension = {
        dimension: sum(
            record["dimension"] == dimension for record in plan
        )
        for dimension in config["run"]["dimensions"]
    }
    for dimension, count in by_dimension.items():
        print(f"  {dimension}: {count}")
    samples = list(dict.fromkeys(record["sample"] for record in plan))
    print(f"Samples: {', '.join(samples)}")
    print(f"Output: {config['output']['combined_file']}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run total/subproperty and 1D/3D halo-web correlations into "
            "one combined HDF5 file."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Unified YAML configuration",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="validate and print the resolved YAML, then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved YAML and run-plan size, then exit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace output.combined_file if it already exists",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.overwrite:
        config["output"]["overwrite"] = True
    if args.dry_run:
        _print_plan(config)
        return 0
    if args.print_config:
        print_config(config)
        return 0
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
