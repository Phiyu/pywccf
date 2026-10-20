"""Discrete tracers (halos, subhalos, galaxies).

A :class:`Tracer` is a column table (``dict`` of equal-length arrays) with a
few conventions:

* positions in ``data['position']`` (N, 3) or in ``x``, ``y``, ``z``
  [h^-1 Mpc, comoving];
* masses in ``data['mass']``, by default as ``log10(M / h^-1 Msun)``;
* an optional object id (``id_key``) and, for :class:`HostedTracer`, the id
  of the host (``host_key``).

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

__all__ = ["Tracer", "HostedTracer"]


class Tracer(HasLog):
    """Column table of point-like objects.

    Parameters
    ----------
    data : mapping of column name -> array (first axis = object).
    l_box : box size [h^-1 Mpc]; needed by :meth:`to_field` without a mesh.
    meta : free-form metadata (simulation, snapshot, cuts ...).
    copy : copy the arrays.
    position_key, mass_key, log_mass, id_key : override the class defaults.
    """

    position_key: str = "position"
    mass_key: str = "mass"
    log_mass: bool = True
    id_key: str | None = None

    def __init__(self, data: Mapping[str, np.ndarray], *, l_box: float | None = None,
                 meta: Mapping | None = None, copy: bool = False,
                 verbose: bool = True, position_key: str | None = None,
                 mass_key: str | None = None, log_mass: bool | None = None,
                 id_key: str | None = None):
        super().__init__(verbose=verbose)
        if position_key is not None:
            self.position_key = position_key
        if mass_key is not None:
            self.mass_key = mass_key
        if log_mass is not None:
            self.log_mass = bool(log_mass)
        if id_key is not None:
            self.id_key = id_key
        cols = {str(k): (np.array(v) if copy else np.asarray(v))
                for k, v in dict(data).items()}
        if not cols:
            raise ValueError("empty data")
        lens = {k: len(v) for k, v in cols.items()}
        n = next(iter(lens.values()))
        bad = {k: m for k, m in lens.items() if m != n}
        if bad:
            raise ValueError(f"columns differ in length ({n}): {bad}")
        self.data = cols
        self.l_box = None if l_box is None else float(l_box)
        self.meta = dict(meta or {})
        self.log(f"{n:,} objects, keys={tuple(cols)}")

    # -- table interface ---------------------------------------------------
    @property
    def n_objs(self) -> int:
        return len(next(iter(self.data.values())))

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

    # -- common properties -------------------------------------------------
    @property
    def positions(self) -> np.ndarray:
        """(N, 3) float64 positions."""
        if self.position_key in self.data:
            xs = np.asarray(self.data[self.position_key], dtype=np.float64)
            if xs.ndim != 2 or xs.shape[1] != 3:
                raise ValueError(f"{self.position_key!r} must be (N, 3)")
            return xs
        if all(k in self.data for k in ("x", "y", "z")):
            return np.column_stack([self.data[k] for k in ("x", "y", "z")]
                                   ).astype(np.float64)
        raise KeyError(f"no positions: need {self.position_key!r} or x, y, z")

    @property
    def masses(self) -> np.ndarray:
        """Linear masses [h^-1 Msun]."""
        m = np.asarray(self.data[self.mass_key], dtype=np.float64)
        return 10.0 ** m if self.log_mass else m

    @property
    def log_masses(self) -> np.ndarray:
        m = np.asarray(self.data[self.mass_key], dtype=np.float64)
        return m if self.log_mass else np.log10(m)

    @property
    def ids(self) -> np.ndarray:
        if self.id_key is None or self.id_key not in self.data:
            return np.arange(self.n_objs)
        return np.asarray(self.data[self.id_key])

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
        group as a column; ``l_box`` falls back to the file attribute."""
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


class HostedTracer(Tracer):
    """A tracer that lives in a host (subhalos in FoF groups, galaxies in
    halos / groups).

    ``host_key`` holds the id of the host; ``central_key`` an optional
    boolean central flag.  When the flag is missing, an object is central
    iff its own id equals its host id (the ELUCID / SUBFIND convention
    ``f_in_grp == subhalo_id``).
    """

    host_key: str = "host_id"
    central_key: str = "is_c"

    def __init__(self, data, *, host_key: str | None = None,
                 central_key: str | None = None, **kw):
        if host_key is not None:
            self.host_key = host_key
        if central_key is not None:
            self.central_key = central_key
        super().__init__(data, **kw)

    @property
    def host_ids(self) -> np.ndarray:
        return np.asarray(self.data[self.host_key])

    @property
    def is_central(self) -> np.ndarray:
        if self.central_key in self.data:
            return np.asarray(self.data[self.central_key], dtype=bool)
        if self.id_key is not None and self.id_key in self.data:
            return self.ids == self.host_ids
        raise KeyError(f"need {self.central_key!r}, or {self.id_key!r} and "
                       f"{self.host_key!r}")

    def centrals(self) -> "HostedTracer":
        return self.subset(self.is_central)

    def satellites(self) -> "HostedTracer":
        return self.subset(~self.is_central)

    def n_per_host(self) -> tuple[np.ndarray, np.ndarray]:
        """Unique host ids and the number of members of each."""
        return np.unique(self.host_ids, return_counts=True)

    def host_index(self, hosts: Tracer, host_id_key: str | None = None) -> np.ndarray:
        """Row of every object's host in ``hosts`` (``-1`` if absent)."""
        key = host_id_key or hosts.id_key
        if key is None:
            raise ValueError("hosts has no id_key; pass host_id_key")
        hid = np.asarray(hosts[key])
        order = np.argsort(hid, kind="stable")
        sorted_ids = hid[order]
        pos = np.searchsorted(sorted_ids, self.host_ids)
        pos_c = np.clip(pos, 0, len(sorted_ids) - 1)
        found = (pos < len(sorted_ids)) & (sorted_ids[pos_c] == self.host_ids)
        return np.where(found, order[pos_c], -1)

    def attach_host(self, hosts: Tracer, keys, *, prefix: str = "host_",
                    host_id_key: str | None = None, fill=np.nan) -> None:
        """Copy host columns ``keys`` to every member as ``prefix + key``."""
        idx = self.host_index(hosts, host_id_key)
        ok = idx >= 0
        for k in ([keys] if isinstance(keys, str) else keys):
            src = np.asarray(hosts[k])
            out = np.full((self.n_objs,) + src.shape[1:], fill,
                          dtype=np.result_type(src.dtype, np.asarray(fill).dtype))
            out[ok] = src[idx[ok]]
            self[prefix + k] = out
