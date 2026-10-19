"""Combine one smoothing scale of halo-web results into one HDF5 file.

The converter is deliberately lossless at the scientific-data level:

* every source HDF5 dataset, dataset attribute, and root attribute is copied;
* every source HDF5 file path and SHA-256 digest is stored;
* matching log files and text sidecars are embedded as exact raw bytes;
* the completed container is validated against every source before it replaces
  the temporary output.

The normalized result path is

    /data/{1d|3d}/{total|age|concentration|spin}/{all|low|high}/
          {web}/mass_bin_NNN/original

The ``original`` group is a faithful copy of the corresponding source-file
root. Container metadata lives on its parent group, so it cannot overwrite an
original attribute.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml


FORMAT_NAME = "halo_web_combined_results"
SCHEMA_VERSION = "1.0"
WEB_TYPES = ("knot", "filament", "sheet", "void")
PROPERTY_NAMES = {
    "zf": "age",
    "age": "age",
    "formation_redshift": "age",
    "concentration": "concentration",
    "lambda": "spin",
    "spin": "spin",
}
TAIL_NAMES = {
    "lower": "low",
    "low": "low",
    "upper": "high",
    "high": "high",
}
RESULT_PATTERN = re.compile(
    r"^(?P<web>knot|filament|sheet|void)_"
    r"(?P<tracer>.+)_xi_(?P<dimension>1d|3d)_"
    r"(?P<mass_bin>\d+)\.h5$"
)


@dataclass(frozen=True)
class SourceRecord:
    path: Path
    relative_path: str
    dimension: str
    sample: str
    tail: str
    web: str
    tracer: str
    mass_bin_index: int
    mass_lo: float
    mass_hi: float
    mass_edges: tuple[float, ...]
    radial_bins: tuple[float, ...]

    @property
    def key(self):
        return (
            self.dimension,
            self.sample,
            self.tail,
            self.web,
            self.mass_bin_index,
        )

    @property
    def container_path(self):
        return (
            f"/data/{self.dimension}/{self.sample}/{self.tail}/"
            f"{self.web}/mass_bin_{self.mass_bin_index:03d}"
        )


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _scale_token(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_key(value):
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    if not key:
        raise ValueError(f"Cannot make an HDF5 key from {value!r}")
    return key


def _read_resolved_config(attrs, path):
    raw = attrs.get("resolved_config_yaml")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        raise ValueError(f"{path}: missing resolved_config_yaml")
    config = yaml.safe_load(raw)
    try:
        mass_edges = tuple(
            float(value) for value in config["sample"]["mass_bin_edges"]
        )
        radial_bins = tuple(
            float(value) for value in config["correlation"]["radial_bins"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{path}: resolved config lacks mass/radial bins"
        ) from error
    return mass_edges, radial_bins


def _semantic_sample(attrs):
    raw_property = attrs.get("assembly_subproperty")
    if raw_property is None:
        return "total", "all"
    if isinstance(raw_property, bytes):
        raw_property = raw_property.decode("utf-8")
    property_key = str(raw_property).strip().lower()
    sample = PROPERTY_NAMES.get(property_key, _safe_key(property_key))

    raw_tail = attrs.get("assembly_tail")
    if isinstance(raw_tail, bytes):
        raw_tail = raw_tail.decode("utf-8")
    tail_key = str(raw_tail).strip().lower()
    if tail_key not in TAIL_NAMES:
        raise ValueError(f"Unknown assembly tail {raw_tail!r}")
    return sample, TAIL_NAMES[tail_key]


def discover_records(source_root, smoothing_scale):
    """Discover and validate all source HDF5 records for one scale."""
    source_root = Path(source_root).resolve()
    target_scale = float(smoothing_scale)
    records = []
    keys = {}

    for path in sorted(source_root.rglob("*.h5")):
        with h5py.File(path, "r") as source:
            if source.attrs.get("format_name") == FORMAT_NAME:
                continue
            if "xi" not in source:
                continue
            file_scale = source.attrs.get("smoothing_scale_mpc_h")
            if file_scale is None or not np.isclose(
                float(file_scale), target_scale
            ):
                continue

            match = RESULT_PATTERN.match(path.name)
            if match is None:
                raise ValueError(
                    f"Cannot parse result filename for {path}"
                )
            dimension = match.group("dimension")
            expected_ndim = 1 if dimension == "1d" else 3
            if source["xi"].ndim != expected_ndim:
                raise ValueError(
                    f"{path}: filename says {dimension}, "
                    f"but xi is {source['xi'].ndim}D"
                )

            sample, tail = _semantic_sample(source.attrs)
            filename_bin = int(match.group("mass_bin"))
            attr_bin = source.attrs.get("mass_bin_index")
            if attr_bin is not None and int(attr_bin) != filename_bin:
                raise ValueError(
                    f"{path}: filename mass bin {filename_bin} conflicts "
                    f"with attribute {attr_bin}"
                )

            mass_edges, radial_bins = _read_resolved_config(
                source.attrs, path
            )
            if filename_bin + 1 >= len(mass_edges):
                raise ValueError(
                    f"{path}: mass bin {filename_bin} is outside "
                    f"{mass_edges}"
                )
            mass_lo = float(
                source.attrs.get(
                    "mass_bin_lower_log10_msun_h",
                    mass_edges[filename_bin],
                )
            )
            mass_hi = float(
                source.attrs.get(
                    "mass_bin_upper_log10_msun_h",
                    mass_edges[filename_bin + 1],
                )
            )
            expected_interval = (
                mass_edges[filename_bin],
                mass_edges[filename_bin + 1],
            )
            if not np.allclose((mass_lo, mass_hi), expected_interval):
                raise ValueError(
                    f"{path}: mass attributes {(mass_lo, mass_hi)} "
                    f"conflict with config {expected_interval}"
                )

        record = SourceRecord(
            path=path.resolve(),
            relative_path=path.relative_to(source_root).as_posix(),
            dimension=dimension,
            sample=sample,
            tail=tail,
            web=match.group("web"),
            tracer=match.group("tracer"),
            mass_bin_index=filename_bin,
            mass_lo=mass_lo,
            mass_hi=mass_hi,
            mass_edges=mass_edges,
            radial_bins=radial_bins,
        )
        if record.key in keys:
            raise ValueError(
                f"Duplicate semantic record {record.key}: "
                f"{keys[record.key]} and {path}"
            )
        keys[record.key] = path
        records.append(record)

    if not records:
        raise FileNotFoundError(
            f"No source HDF5 results at smoothing scale {target_scale:g} "
            f"under {source_root}"
        )
    _validate_inventory(records)
    return sorted(records, key=lambda record: record.key)


def _validate_inventory(records):
    mass_variants = {record.mass_edges for record in records}
    radial_variants = {record.radial_bins for record in records}
    tracer_variants = {record.tracer for record in records}
    if len(mass_variants) != 1:
        raise ValueError(f"Multiple mass grids found: {mass_variants}")
    if len(radial_variants) != 1:
        raise ValueError(f"Multiple radial grids found: {radial_variants}")
    if len(tracer_variants) != 1:
        raise ValueError(f"Multiple tracer labels found: {tracer_variants}")

    by_selection = {}
    for record in records:
        selection = (record.dimension, record.sample, record.tail)
        by_selection.setdefault(selection, {}).setdefault(
            record.web, set()
        ).add(record.mass_bin_index)

    reference_bins = None
    for selection, web_bins in sorted(by_selection.items()):
        missing_webs = set(WEB_TYPES) - set(web_bins)
        if missing_webs:
            raise ValueError(
                f"{selection} is missing web types {sorted(missing_webs)}"
            )
        variants = {tuple(sorted(values)) for values in web_bins.values()}
        if len(variants) != 1:
            raise ValueError(
                f"{selection} has inconsistent web mass bins: {web_bins}"
            )
        bins = next(iter(variants))
        if reference_bins is None:
            reference_bins = bins
        elif bins != reference_bins:
            raise ValueError(
                f"{selection} mass bins {bins} differ from "
                f"{reference_bins}"
            )


def _infer_log_scale(path):
    name = Path(path).name
    match = re.search(r"-s(?P<scale>\d+(?:\.\d+)?)\.log$", name)
    if match is None:
        match = re.match(
            r"^total(?P<scale>\d+(?:\.\d+)?)(?:-1d)?\.log$", name
        )
    return None if match is None else float(match.group("scale"))


def discover_logs(source_root, smoothing_scale):
    target_scale = float(smoothing_scale)
    return [
        path.resolve()
        for path in sorted(Path(source_root).resolve().rglob("*.log"))
        if (
            _infer_log_scale(path) is not None
            and np.isclose(_infer_log_scale(path), target_scale)
        )
    ]


def discover_sidecars(records):
    sidecars = []
    for record in records:
        for suffix in (".txt",):
            candidate = record.path.with_suffix(suffix)
            if candidate.is_file():
                sidecars.append(candidate.resolve())
    return sorted(set(sidecars))


def _create_string_dataset(group, name, values):
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(
        name,
        data=np.asarray(list(values), dtype=object),
        dtype=dtype,
    )


def _copy_original(source, destination):
    for key, value in source.attrs.items():
        destination.attrs[key] = value
    for key in source.keys():
        source.copy(key, destination, name=key, without_attrs=False)


def _store_raw_files(parent, paths, source_root):
    stored_paths = []
    source_root = Path(source_root).resolve()
    for index, path in enumerate(paths):
        path = Path(path).resolve()
        entry = parent.create_group(f"file_{index:03d}")
        try:
            relative_path = path.relative_to(source_root).as_posix()
        except ValueError:
            relative_path = path.as_posix()
        raw = path.read_bytes()
        entry.attrs["source_relative_path"] = relative_path
        entry.attrs["source_sha256"] = hashlib.sha256(raw).hexdigest()
        entry.attrs["source_byte_count"] = len(raw)
        entry.create_dataset(
            "raw_bytes",
            data=np.frombuffer(raw, dtype=np.uint8),
            compression="gzip",
            compression_opts=6,
        )
        stored_paths.append(relative_path)
    return stored_paths


def write_combined_file(
    source_root,
    output_path,
    smoothing_scale,
    *,
    overwrite=False,
):
    """Create and validate one combined file, then atomically install it."""
    source_root = Path(source_root).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )

    records = discover_records(source_root, smoothing_scale)
    logs = discover_logs(source_root, smoothing_scale)
    sidecars = discover_sidecars(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    record_paths = []
    source_paths = []
    source_hashes = []
    mass_edges = records[0].mass_edges
    radial_bins = records[0].radial_bins

    try:
        with h5py.File(temporary_path, "w") as destination:
            destination.attrs["format_name"] = FORMAT_NAME
            destination.attrs["schema_version"] = SCHEMA_VERSION
            destination.attrs["validation_status"] = "pending"
            destination.attrs["created_utc"] = _utc_now()
            destination.attrs["source_root"] = source_root.as_posix()
            destination.attrs["smoothing_scale_mpc_h"] = float(
                smoothing_scale
            )
            destination.attrs["record_count"] = len(records)
            destination.attrs["log_count"] = len(logs)
            destination.attrs["sidecar_count"] = len(sidecars)
            destination.attrs["log_discovery_status"] = (
                "complete" if logs else "not_found"
            )

            metadata = destination.create_group("metadata")
            metadata.create_dataset("mass_bin_edges", data=mass_edges)
            metadata.create_dataset("radial_bins", data=radial_bins)
            _create_string_dataset(metadata, "web_types", WEB_TYPES)
            _create_string_dataset(
                metadata,
                "samples",
                sorted({record.sample for record in records}),
            )
            contains_subproperties = any(
                record.sample != "total" for record in records
            )
            metadata.attrs["contains_subproperty_samples"] = (
                contains_subproperties
            )
            metadata.attrs["sample_scope"] = (
                "total_and_subproperties"
                if contains_subproperties
                else "total_only"
            )
            metadata.attrs["mass_unit"] = (
                "log10(M_200m / (h^-1 Msun))"
            )
            metadata.attrs["radial_unit"] = "h^-1 Mpc"

            for record in records:
                entry = destination.require_group(
                    record.container_path.lstrip("/")
                )
                entry.attrs["dimension"] = record.dimension
                entry.attrs["sample"] = record.sample
                entry.attrs["tail"] = record.tail
                entry.attrs["web"] = record.web
                entry.attrs["tracer"] = record.tracer
                entry.attrs["mass_bin_index"] = record.mass_bin_index
                entry.attrs[
                    "mass_bin_lower_log10_msun_h"
                ] = record.mass_lo
                entry.attrs[
                    "mass_bin_upper_log10_msun_h"
                ] = record.mass_hi
                entry.attrs[
                    "source_relative_path"
                ] = record.relative_path
                source_hash = _sha256(record.path)
                entry.attrs["source_sha256"] = source_hash
                entry.attrs["source_byte_count"] = record.path.stat().st_size

                original = entry.create_group("original")
                with h5py.File(record.path, "r") as source:
                    _copy_original(source, original)

                record_paths.append(record.container_path)
                source_paths.append(record.relative_path)
                source_hashes.append(source_hash)

            index = destination.create_group("index")
            _create_string_dataset(index, "record_paths", record_paths)
            _create_string_dataset(index, "source_paths", source_paths)
            _create_string_dataset(index, "source_sha256", source_hashes)

            provenance = destination.create_group("provenance")
            log_group = provenance.create_group("logs")
            log_group.attrs["discovery_status"] = (
                "complete" if logs else "not_found"
            )
            if not logs:
                log_group.attrs["note"] = (
                    "No matching run-log file was present in source_root; "
                    "no log content was fabricated."
                )
            log_paths = _store_raw_files(log_group, logs, source_root)
            sidecar_paths = _store_raw_files(
                provenance.create_group("sidecars"),
                sidecars,
                source_root,
            )
            _create_string_dataset(index, "log_paths", log_paths)
            _create_string_dataset(index, "sidecar_paths", sidecar_paths)

        validate_combined_file(
            temporary_path,
            source_root,
            records,
            logs,
            sidecars,
        )
        with h5py.File(temporary_path, "r+") as destination:
            destination.attrs["validation_status"] = "complete"
            destination.attrs["validated_utc"] = _utc_now()
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return {
        "output_path": output_path,
        "record_count": len(records),
        "log_count": len(logs),
        "sidecar_count": len(sidecars),
    }


def _attr_equal(first, second):
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape or first.dtype.kind != second.dtype.kind:
        return False
    if first.dtype.kind in "fc":
        return np.array_equal(first, second, equal_nan=True)
    return np.array_equal(first, second)


def _compare_attrs(source, copied, label):
    if set(source.attrs) != set(copied.attrs):
        raise ValueError(f"{label}: attribute keys differ")
    for key in source.attrs:
        if not _attr_equal(source.attrs[key], copied.attrs[key]):
            raise ValueError(f"{label}: attribute {key!r} differs")


def _compare_h5_tree(source, copied, label):
    _compare_attrs(source, copied, label)
    if set(source.keys()) != set(copied.keys()):
        raise ValueError(f"{label}: object keys differ")
    for key in source.keys():
        source_object = source[key]
        copied_object = copied[key]
        child_label = f"{label}/{key}"
        if isinstance(source_object, h5py.Group):
            if not isinstance(copied_object, h5py.Group):
                raise ValueError(f"{child_label}: object type differs")
            _compare_h5_tree(source_object, copied_object, child_label)
            continue
        if not isinstance(copied_object, h5py.Dataset):
            raise ValueError(f"{child_label}: object type differs")
        _compare_attrs(source_object, copied_object, child_label)
        if (
            source_object.shape != copied_object.shape
            or source_object.dtype != copied_object.dtype
        ):
            raise ValueError(f"{child_label}: shape or dtype differs")
        source_values = source_object[()]
        copied_values = copied_object[()]
        if source_object.dtype.kind in "fc":
            equal = np.array_equal(
                source_values, copied_values, equal_nan=True
            )
        else:
            equal = np.array_equal(source_values, copied_values)
        if not equal:
            raise ValueError(f"{child_label}: dataset values differ")
        properties = (
            "chunks",
            "compression",
            "compression_opts",
            "shuffle",
            "fletcher32",
            "scaleoffset",
        )
        for property_name in properties:
            if getattr(source_object, property_name) != getattr(
                copied_object, property_name
            ):
                raise ValueError(
                    f"{child_label}: {property_name} differs"
                )


def _validate_raw_files(group, paths, source_root, label):
    entries = sorted(group.keys())
    if len(entries) != len(paths):
        raise ValueError(
            f"{label}: stored {len(entries)} files, expected {len(paths)}"
        )
    source_root = Path(source_root).resolve()
    for entry_name, source_path in zip(entries, paths):
        entry = group[entry_name]
        raw = bytes(np.asarray(entry["raw_bytes"], dtype=np.uint8))
        expected = Path(source_path).read_bytes()
        if raw != expected:
            raise ValueError(f"{label}/{entry_name}: raw bytes differ")
        try:
            relative = Path(source_path).resolve().relative_to(
                source_root
            ).as_posix()
        except ValueError:
            relative = Path(source_path).resolve().as_posix()
        if entry.attrs["source_relative_path"] != relative:
            raise ValueError(f"{label}/{entry_name}: source path differs")


def validate_combined_file(
    combined_path,
    source_root,
    records,
    logs,
    sidecars,
):
    """Verify copied objects and embedded raw files against every source."""
    with h5py.File(combined_path, "r") as combined:
        if combined.attrs.get("format_name") != FORMAT_NAME:
            raise ValueError("Combined format name is missing or invalid")
        if combined.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Combined schema version is invalid")
        if int(combined.attrs.get("record_count", -1)) != len(records):
            raise ValueError("Combined record count is invalid")

        for record in records:
            copied = combined[
                record.container_path.lstrip("/") + "/original"
            ]
            parent = copied.parent
            if parent.attrs["source_sha256"] != _sha256(record.path):
                raise ValueError(
                    f"{record.relative_path}: source digest differs"
                )
            with h5py.File(record.path, "r") as source:
                _compare_h5_tree(
                    source,
                    copied,
                    record.relative_path,
                )

        _validate_raw_files(
            combined["provenance/logs"],
            logs,
            source_root,
            "logs",
        )
        _validate_raw_files(
            combined["provenance/sidecars"],
            sidecars,
            source_root,
            "sidecars",
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Combine all 1D/3D total and assembly HDF5 results for each "
            "requested smoothing scale."
        )
    )
    parser.add_argument(
        "source_root",
        type=Path,
        help="Folder containing the existing result folders and logs.",
    )
    parser.add_argument(
        "--smoothing-scales",
        type=float,
        nargs="+",
        required=True,
        help="One or more smoothing scales, for example: 3 5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder; defaults to source_root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing combined file after validation.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    source_root = args.source_root.resolve()
    output_dir = (
        source_root
        if args.output_dir is None
        else args.output_dir.resolve()
    )

    for smoothing_scale in args.smoothing_scales:
        token = _scale_token(smoothing_scale)
        output_path = (
            output_dir
            / f"{source_root.name}_s{token}_combined.h5"
        )
        result = write_combined_file(
            source_root,
            output_path,
            smoothing_scale,
            overwrite=args.overwrite,
        )
        size_mib = result["output_path"].stat().st_size / 1024**2
        print(
            f"Created {result['output_path']} ({size_mib:.1f} MiB): "
            f"{result['record_count']} HDF5 records, "
            f"{result['log_count']} logs, "
            f"{result['sidecar_count']} sidecars; validation complete."
        )


if __name__ == "__main__":
    main()
