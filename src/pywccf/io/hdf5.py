"""Nested ``dict`` <-> HDF5 helpers.

* dict values become groups, arrays / scalars become datasets;
* strings are stored as UTF-8 variable-length strings;
* ``None`` is skipped on writing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

__all__ = ["dump_dict", "load_dict", "write_dict", "read_dict", "list_keys"]

_STR = h5py.string_dtype(encoding="utf-8")


def _to_dataset(group: h5py.Group, key: str, value, compression):
    arr = np.asarray(value)
    if arr.dtype.kind in ("U", "S", "O"):
        group.create_dataset(key, data=arr.astype(str).astype(object),
                             dtype=_STR)
    elif arr.ndim == 0 or compression is None:
        group.create_dataset(key, data=arr)
    else:
        group.create_dataset(key, data=arr, compression=compression)


def dump_dict(group: h5py.Group, data: Mapping[str, Any], *,
              compression: str | None = None, overwrite: bool = False):
    """Write ``data`` recursively into an open HDF5 group."""
    for key, value in data.items():
        key = str(key)
        if value is None:
            continue
        if key in group:
            if not overwrite:
                raise KeyError(f"{group.name}/{key} exists (overwrite=False)")
            del group[key]
        if isinstance(value, Mapping):
            dump_dict(group.create_group(key), value,
                      compression=compression, overwrite=overwrite)
        else:
            _to_dataset(group, key, value, compression)


def _from_dataset(ds: h5py.Dataset):
    if h5py.check_string_dtype(ds.dtype) is not None:
        val = ds.asstr()[()]
        return val if isinstance(val, str) else np.asarray(val, dtype=object)
    val = ds[()]
    return val.item() if isinstance(val, np.generic) else val


def load_dict(group: h5py.Group, keys=None) -> dict:
    """Read a group recursively (optionally only ``keys`` at the top)."""
    out = {}
    for key in (group.keys() if keys is None else keys):
        obj = group[key]
        out[key] = (load_dict(obj) if isinstance(obj, h5py.Group)
                    else _from_dataset(obj))
    return out


def write_dict(path, data: Mapping[str, Any], *, group: str | None = None,
               mode: str = "w", compression: str | None = "gzip",
               overwrite: bool = False):
    """Write a nested dict to ``path`` (optionally under ``group``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, mode) as f:
        g = f if group is None else f.require_group(group)
        dump_dict(g, data, compression=compression, overwrite=overwrite)
    return path


def read_dict(path, *, group: str | None = None, keys=None) -> dict:
    """Read a nested dict from ``path`` (optionally only ``group``)."""
    with h5py.File(path, "r", locking=False) as f:
        g = f if group is None else f[group]
        return load_dict(g, keys)


def list_keys(path, group: str | None = None) -> list[str]:
    with h5py.File(path, "r", locking=False) as f:
        return list((f if group is None else f[group]).keys())
