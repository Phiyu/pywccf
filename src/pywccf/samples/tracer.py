"""Discrete tracers (halos, subhalos, galaxies).

A :class:`Tracer` is a column table (``dict`` of equal-length arrays) with
one set of key conventions shared by every tracer class:

=============  ==========  ==================================================
key            type        meaning
=============  ==========  ==================================================
``position``   (N, 3) f8   comoving position [h^-1 Mpc]           (required)
``mass``       (N,) f8     log10(M / h^-1 Msun): M_h for halos, m_sub for
                           subhalos, M_* for galaxies    (required, except
                           for the base class)
``id``         (N,) int    object id                             (optional)
``central``    (N,) bool   central object of its host     (subhalo / galaxy)
``host_id``    (N,) int    ``id`` of the host halo         (subhalo / galaxy)
=============  ==========  ==================================================

Any other column is kept as is.  Converting a simulation or survey output
to these keys is left to the user (one dict of arrays).

Selections (``subset*``) return new objects of the same class; masks over
the parent (``quantile_masks``) are what the WCCF estimators take to measure
several sub-samples in one pass.  :meth:`Tracer.to_field` assigns the tracer
to a mesh (NGP / CIC / TSC, number or mass weighted).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Mapping

import h5py
import numpy as np

from ..core.assignment import assign
from ..core.log import HasLog
from ..core.mesh import Mesh
from ..io.hdf5 import dump_dict

__all__ = ["Tracer"]


class Tracer(HasLog):
    """Column table of point-like objects.

    Parameters
    ----------
    data : mapping of column name -> array (first axis = object); must
        contain :attr:`required_keys`.
    l_box : box size [h^-1 Mpc]; needed by :meth:`to_field` without a mesh.
    meta : free-form metadata (simulation, snapshot, cuts ...).
    copy : copy the arrays.
    """

    required_keys: tuple[str, ...] = ("position",)

    def __init__(self, data: Mapping[str, np.ndarray], *,
                 l_box: float | None = None, meta: Mapping | None = None,
                 copy: bool = False, verbose: bool = True):
        super().__init__(verbose=verbose)
        cols = {str(k): (np.array(v) if copy else np.asarray(v))
                for k, v in dict(data).items()}
        missing = [k for k in self.required_keys if k not in cols]
        if missing:
            raise KeyError(f"{type(self).__name__} needs keys {missing}")
        lens = {k: len(v) for k, v in cols.items()}
        n = next(iter(lens.values()))
        bad = {k: m for k, m in lens.items() if m != n}
        if bad:
            raise ValueError(f"columns differ in length ({n}): {bad}")
        pos = cols["position"]
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"'position' must be (N, 3), got {pos.shape}")
        if "central" in cols:
            cols["central"] = cols["central"].astype(bool)
        if "host_id" in cols and cols["host_id"].dtype.kind not in "iu":
            raise TypeError("'host_id' must be an integer array")
        self.data = cols
        self.l_box = None if l_box is None else float(l_box)
        self.meta = dict(meta or {})
        self.log(f"{n:,} objects, keys={tuple(cols)}")

    # -- table interface ---------------------------------------------------
    @property
    def n_objs(self) -> int:
        return len(self.data["position"])

    def __len__(self) -> int:
        return self.n_objs

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __contains__(self, key) -> bool:
        return key in self.data

    def __getitem__(self, key: str | tuple[str, ...]):
        if isinstance(key, tuple):
            return tuple(self.data[k] for k in key)
        return self.data[key]

    def __setitem__(self, key: str, value):
        value = np.asarray(value)
        if len(value) != self.n_objs:
            raise ValueError(f"{key!r}: length {len(value)} != {self.n_objs}")
        self.data[key] = value

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(n_objs={self.n_objs:,}, "
                f"keys={tuple(self.keys())})")

    def add_property(self, func_or_dict: Callable | Mapping) -> None:
        """Add columns from a dict, or from ``func(self) -> dict``."""
        add = func_or_dict(self) if callable(func_or_dict) else func_or_dict
        for k, v in add.items():
            self[k] = v

    def _need(self, key: str) -> np.ndarray:
        if key not in self.data:
            raise KeyError(f"{type(self).__name__} has no {key!r} column")
        return np.asarray(self.data[key])

    # -- common properties -------------------------------------------------
    @property
    def positions(self) -> np.ndarray:
        """(N, 3) float64 positions."""
        return np.asarray(self.data["position"], dtype=np.float64)

    @property
    def log_masses(self) -> np.ndarray:
        """``mass`` column, log10(M / h^-1 Msun)."""
        return self._need("mass").astype(np.float64)

    @property
    def masses(self) -> np.ndarray:
        """Linear masses [h^-1 Msun]."""
        return 10.0 ** self.log_masses

    @property
    def ids(self) -> np.ndarray:
        """``id`` column (row numbers when absent)."""
        return (np.asarray(self.data["id"]) if "id" in self.data
                else np.arange(self.n_objs))

    def number_density(self) -> float:
        if self.l_box is None:
            raise ValueError("l_box is not set")
        return self.n_objs / self.l_box ** 3

    def weights(self, weight=None) -> np.ndarray:
        """Per-object weights.

        weight : None / 'number' (1), 'mass' (linear mass), a column name,
        an array of length N or a callable ``f(self) -> array``.
        """
        if weight is None or (isinstance(weight, str) and weight == "number"):
            return np.ones(self.n_objs)
        if isinstance(weight, str):
            return (self.masses if weight == "mass"
                    else np.asarray(self.data[weight], dtype=np.float64))
        w = weight(self) if callable(weight) else weight
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (self.n_objs,):
            raise ValueError(f"weights must have shape ({self.n_objs},)")
        return w

    # -- hosts (samples with 'central' and 'host_id') -----------------------
    @property
    def is_central(self) -> np.ndarray:
        return self._need("central").astype(bool)

    @property
    def host_ids(self) -> np.ndarray:
        return self._need("host_id")

    def centrals(self) -> "Tracer":
        return self.subset(self.is_central)

    def satellites(self) -> "Tracer":
        return self.subset(~self.is_central)

    def n_per_host(self) -> tuple[np.ndarray, np.ndarray]:
        """Unique host ids and the number of members of each."""
        return np.unique(self.host_ids, return_counts=True)

    def host_index(self, hosts: "Tracer") -> np.ndarray:
        """Row of every object's host in ``hosts`` (matched on
        ``self['host_id'] == hosts['id']``; ``-1`` if absent)."""
        hid = hosts._need("id")
        order = np.argsort(hid, kind="stable")
        sorted_ids = hid[order]
        pos = np.searchsorted(sorted_ids, self.host_ids)
        pos_c = np.clip(pos, 0, len(sorted_ids) - 1)
        found = (pos < len(sorted_ids)) & (sorted_ids[pos_c] == self.host_ids)
        return np.where(found, order[pos_c], -1)

    def attach_host(self, hosts: "Tracer", keys, *, prefix: str = "host_",
                    fill=np.nan) -> None:
        """Copy host columns ``keys`` to every member as ``prefix + key``."""
        idx = self.host_index(hosts)
        ok = idx >= 0
        for k in ([keys] if isinstance(keys, str) else keys):
            src = np.asarray(hosts[k])
            out = np.full((self.n_objs,) + src.shape[1:], fill,
                          dtype=np.result_type(src.dtype, np.asarray(fill).dtype))
            out[ok] = src[idx[ok]]
            self[prefix + k] = out

    # -- selections --------------------------------------------------------
    def _new(self, data: dict) -> "Tracer":
        new = copy.copy(self)
        new.data = data
        new.meta = dict(self.meta)
        return new

    def subset(self, sel, copy_data: bool = False) -> "Tracer":
        """Row selection by boolean mask, index array or slice."""
        return self._new({k: (np.array(v[sel]) if copy_data else v[sel])
                          for k, v in self.data.items()})

    def subset_by_val(self, key: str, lo=None, hi=None, eq=None) -> "Tracer":
        """Rows with ``lo <= v < hi`` and/or ``v == eq``."""
        return self.subset(self.mask_by_val(key, lo=lo, hi=hi, eq=eq))

    def subset_by_p(self, key: str, p_lo=None, p_hi=None) -> "Tracer":
        """Rows between the ``p_lo`` and ``p_hi`` quantiles of ``key``."""
        val = np.asarray(self.data[key])
        lo = None if p_lo is None else np.quantile(val, p_lo)
        hi = None if p_hi is None else np.quantile(val, p_hi)
        return self.subset_by_val(key, lo=lo, hi=hi)

    def subset_if(self, key: str) -> "Tracer":
        return self.subset_by_val(key, eq=True)

    def subset_if_not(self, key: str) -> "Tracer":
        return self.subset_by_val(key, eq=False)

    def mask_by_val(self, key: str, lo=None, hi=None, eq=None) -> np.ndarray:
        val = np.asarray(self.data[key])
        sel = np.ones(self.n_objs, dtype=bool)
        if lo is not None:
            sel &= val >= lo
        if hi is not None:
            sel &= val < hi
        if eq is not None:
            sel &= val == eq
        return sel

    def mass_bins(self, edges) -> list["Tracer"]:
        """Sub-samples with ``lo <= mass < hi``."""
        edges = np.asarray(edges, dtype=np.float64)
        return [self.subset_by_val("mass", lo=lo, hi=hi)
                for lo, hi in zip(edges[:-1], edges[1:])]

    def count_mass_bins(self, edges) -> np.ndarray:
        m = self.log_masses
        edges = np.asarray(edges, dtype=np.float64)
        return np.array([int(((m >= lo) & (m < hi)).sum())
                         for lo, hi in zip(edges[:-1], edges[1:])])

    def quantile_masks(self, key: str, p: float = 0.2, *,
                       bin_key: str | None = None, bin_edges=None):
        """Lower / upper ``p`` tails of ``key``, optionally inside every bin
        ``[lo, hi)`` of ``bin_key`` (e.g. formation redshift at fixed mass).

        Returns two boolean masks over this sample, ``(low, high)``:
        ``v < q_p`` and ``v >= q_{1-p}`` within each bin (the selection of
        the paper's old / young halos).
        """
        val = np.asarray(self.data[key], dtype=np.float64)
        low = np.zeros(self.n_objs, dtype=bool)
        high = np.zeros(self.n_objs, dtype=bool)
        if bin_key is None:
            groups = [np.ones(self.n_objs, dtype=bool)]
        else:
            b = np.asarray(self.data[bin_key])
            edges = np.asarray(bin_edges, dtype=np.float64)
            groups = [(b >= lo) & (b < hi) for lo, hi in zip(edges[:-1], edges[1:])]
        for g in groups:
            if not g.any():
                continue
            q_lo, q_hi = np.quantile(val[g], [p, 1.0 - p])
            low |= g & (val < q_lo)
            high |= g & (val >= q_hi)
        return low, high

    # -- conversion to a field ---------------------------------------------
    def to_field(self, mesh: Mesh | None = None, *, n_grids: int | None = None,
                 scheme: str = "cic", weight=None, field_cls=None,
                 verbose: bool | None = None):
        """Assign the tracer to a mesh.

        Parameters
        ----------
        mesh : target mesh, or give ``n_grids`` (box size from ``l_box``).
        scheme : 'ngp', 'cic' or 'tsc'.
        weight : see :meth:`weights`; 'number' (default) gives the number
            density, 'mass' the mass density.
        field_cls : class of the result (default :class:`MatterField`).

        Returns a field with ``rho`` (weight per unit volume) and
        ``delta = rho / <rho> - 1``.
        """
        from .field import MatterField
        if mesh is None:
            if n_grids is None or self.l_box is None:
                raise ValueError("give mesh, or n_grids together with l_box")
            mesh = Mesh(n_grids, self.l_box)
        w = self.weights(weight)
        grid = assign(self.positions, mesh, weights=w, scheme=scheme)
        rho = grid / mesh.cell_volume
        mean = w.sum() / mesh.volume
        if mean <= 0:
            raise ValueError("total weight must be positive")
        cls = field_cls or MatterField
        meta = {"source": type(self).__name__, "n_objs": self.n_objs,
                "scheme": scheme,
                "weight": weight if isinstance(weight, str) or weight is None
                else "custom"}
        return cls({"rho": rho, "delta": rho / mean - 1.0}, mesh, meta=meta,
                   verbose=self.verbose if verbose is None else verbose)

    # -- io ----------------------------------------------------------------
    @classmethod
    def from_file(cls, path, *, keys=None, group: str | None = None,
                  l_box: float | None = None, **init_kw):
        """Read every (or the selected) top-level dataset of an HDF5 file or
        group as a column; ``l_box`` falls back to the file attribute.  The
        file must already use the key conventions of this module."""
        with h5py.File(path, "r", locking=False) as f:
            g = f if group is None else f[group]
            names = [k for k in (g.keys() if keys is None else keys)
                     if isinstance(g[k], h5py.Dataset) and g[k].ndim >= 1]
            data = {k: g[k][()] for k in names}
            if l_box is None:
                for k in ("l_box", "BoxSize", "box_size"):
                    if k in g.attrs:
                        l_box = float(np.asarray(g.attrs[k]).ravel()[0])
                        break
        return cls(data, l_box=l_box, **init_kw)

    def to_file(self, path, *, group: str | None = None, mode: str = "w",
                compression: str | None = "gzip"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, mode) as f:
            g = f if group is None else f.require_group(group)
            dump_dict(g, self.data, compression=compression, overwrite=True)
            if self.l_box is not None:
                g.attrs["l_box"] = self.l_box
            g.attrs["class"] = type(self).__name__
        return path
