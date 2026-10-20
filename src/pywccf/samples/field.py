"""Gridded fields on a periodic :class:`~pywccf.core.Mesh`.

* :class:`Field` -- base class: a dict of node arrays sharing one mesh,
  optional boolean mask (e.g. a reconstruction volume), interpolation,
  Gaussian smoothing, HDF5 io, conversion to a weighted node tracer.
* :class:`MatterField` -- matter (or tracer) density, key ``delta``;
  builds the :class:`~pywccf.samples.tidal.TidalField`.
* :class:`GasField` -- gas quantities (``rho``, ``temperature``, ...).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from ..core.assignment import interpolate, window_k
from ..core.log import HasLog
from ..core.mesh import Mesh
from ..io.hdf5 import dump_dict

__all__ = ["Field", "MatterField", "GasField", "gaussian_smooth"]


def gaussian_smooth(grid: np.ndarray, mesh: Mesh, R: float,
                    deconvolve: str | None = None) -> np.ndarray:
    """Gaussian smoothing ``W(k) = exp(-k^2 R^2 / 2)`` by FFT, optionally
    deconvolving the assignment window of scheme ``deconvolve``."""
    grid = mesh.check_shape(grid, "grid")
    fk = np.fft.rfftn(grid)
    if deconvolve:
        fk /= window_k(mesh, deconvolve)
    if R and R > 0:
        fk *= np.exp(-0.5 * mesh.k2() * R * R)
    return np.fft.irfftn(fk, s=mesh.shape, axes=(0, 1, 2))


class Field(HasLog):
    """Named node arrays on one mesh.

    Parameters
    ----------
    data : mapping name -> array whose first three axes are ``mesh.shape``
        (trailing axes allowed, e.g. eigenvalues), or a single array
        (stored under ``default_key``).
    mesh : the :class:`Mesh`.
    mask : optional boolean node mask; statistics and conversions use only
        masked-in nodes.
    meta : free-form metadata.
    """

    default_key: str = "value"

    def __init__(self, data, mesh: Mesh, *, mask: np.ndarray | None = None,
                 meta: Mapping | None = None, verbose: bool = True):
        super().__init__(verbose=verbose)
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a pywccf.core.Mesh")
        if not isinstance(data, Mapping):
            data = {self.default_key: data}
        self.mesh = mesh
        self.data = {}
        for k, v in data.items():
            self[k] = v
        self.mask = None if mask is None else np.asarray(
            mesh.check_shape(mask, "mask"), dtype=bool)
        self.meta = dict(meta or {})
        self.log(f"{mesh}, keys={tuple(self.data)}"
                 + ("" if self.mask is None else
                    f", mask={self.mask.mean():.3f} of the volume"))

    # -- container ---------------------------------------------------------
    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def __contains__(self, key) -> bool:
        return key in self.data

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def __setitem__(self, key: str, value):
        value = np.asarray(value)
        if value.shape[:3] != self.mesh.shape:
            raise ValueError(f"{key!r} has shape {value.shape}, expected "
                             f"{self.mesh.shape} + (...)")
        self.data[str(key)] = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.mesh}, keys={tuple(self.data)})"

    @property
    def n_grids(self) -> int:
        return self.mesh.n_grids

    @property
    def l_box(self) -> float:
        return self.mesh.l_box

    def value(self, key: str | None = None) -> np.ndarray:
        return self.data[key or self.default_key]

    # -- statistics / operations ------------------------------------------
    def mean(self, key: str | None = None) -> float:
        v = self.value(key)
        return float(v[self.mask].mean() if self.mask is not None else v.mean())

    def overdensity(self, key: str | None = None) -> np.ndarray:
        """``v / <v> - 1`` (mean over the mask)."""
        v = self.value(key).astype(np.float64)
        return v / self.mean(key) - 1.0

    def values_at(self, xs, key: str | None = None, scheme: str = "ngp"):
        """Field values at positions ``xs`` (NGP or CIC/TSC interpolation)."""
        return interpolate(self.value(key), xs, self.mesh, scheme=scheme)

    def smoothed(self, R: float, key: str | None = None,
                 deconvolve: str | None = None) -> np.ndarray:
        return gaussian_smooth(self.value(key).astype(np.float64), self.mesh,
                               R, deconvolve)

    def node_positions(self, sel: np.ndarray | None = None) -> np.ndarray:
        if sel is None:
            sel = self.mask
        return self.mesh.node_positions(sel)

    def to_tracer(self, key: str | None = None, *, weight: str = "1+value",
                  stride: int = 1, mask: np.ndarray | None = None):
        """Nodes as a weighted point set, for pair-count (PH) estimators.

        weight : '1+value' (the field is an overdensity; weight = 1 + delta,
            clipped at 0), 'value' (the field is already a density) or None.
        stride : keep every ``stride``-th node along each axis (coarser,
            cheaper sampling; weights are not rescaled because only their
            ratio to the mean enters the estimators).
        """
        from .tracer import Tracer
        sel = np.zeros(self.mesh.shape, dtype=bool)
        s = max(1, int(stride))
        sel[::s, ::s, ::s] = True
        if mask is not None:
            sel &= np.asarray(mask, dtype=bool)
        if self.mask is not None:
            sel &= self.mask
        v = self.value(key)[sel].astype(np.float64)
        if weight == "1+value":
            w = (1.0 + v).clip(0.0)
        elif weight == "value":
            w = v
        elif weight is None:
            w = np.ones_like(v)
        else:
            raise ValueError("weight must be '1+value', 'value' or None")
        return Tracer({"position": self.mesh.node_positions(sel), "weight": w},
                      l_box=self.l_box, verbose=self.verbose,
                      meta={"source": type(self).__name__, "stride": s})

    # -- io ----------------------------------------------------------------
    @staticmethod
    def _read_mesh(g: h5py.Group) -> Mesh:
        n = int(np.asarray(g["n_grids"][()]).item())
        l = float(np.asarray(g["l_box"][()]).item())
        return Mesh(n, l)

    @classmethod
    def from_file(cls, path, keys=None, *, group: str | None = None,
                  key_map: Mapping[str, str] | None = None,
                  mask_key: str | None = None, **init_kw):
        """Read node arrays from an HDF5 file with scalar datasets
        ``n_grids`` and ``l_box``.

        keys : datasets to read (default: every dataset with the mesh
            shape); ``key_map`` renames file datasets to field keys.
        mask_key : dataset holding a boolean mask.
        """
        key_map = dict(key_map or {})
        with h5py.File(path, "r", locking=False) as f:
            g = f if group is None else f[group]
            mesh = cls._read_mesh(g)
            names = keys if keys is not None else [
                k for k in g.keys() if isinstance(g[k], h5py.Dataset)
                and g[k].shape[:3] == mesh.shape and k != mask_key]
            data = {key_map.get(k, k): g[k][()] for k in names}
            mask = None if mask_key is None else g[mask_key][()].astype(bool)
            meta = {k: (v.decode() if isinstance(v, bytes) else v)
                    for k, v in g.attrs.items()}
        meta["path"] = str(path)
        return cls(data, mesh, mask=mask, meta=meta, **init_kw)

    def to_file(self, path, keys=None, *, group: str | None = None,
                mode: str = "w", compression: str | None = None,
                key_map: Mapping[str, str] | None = None):
        key_map = dict(key_map or {})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, mode) as f:
            g = f if group is None else f.require_group(group)
            out = {"n_grids": self.n_grids, "l_box": self.l_box}
            out.update({key_map.get(k, k): self.data[k]
                        for k in (keys or self.data)})
            if self.mask is not None:
                out["mask"] = self.mask
            dump_dict(g, out, compression=compression, overwrite=True)
            g.attrs["class"] = type(self).__name__
        return path


class MatterField(Field):
    """Matter density contrast ``delta`` (and optionally ``rho``)."""

    default_key = "delta"

    @classmethod
    def from_tracer(cls, tracer, mesh: Mesh | None = None, *,
                    n_grids: int | None = None, scheme: str = "cic",
                    weight="mass", **kw) -> "MatterField":
        """Density of a :class:`~pywccf.samples.tracer.Tracer` (e.g. dark
        matter particles: ``weight='number'`` for equal-mass particles,
        ``'mass'`` otherwise)."""
        return tracer.to_field(mesh, n_grids=n_grids, scheme=scheme,
                               weight=weight, field_cls=cls, **kw)

    @property
    def delta(self) -> np.ndarray:
        return self.data["delta"]

    @property
    def rho_over_mean(self) -> np.ndarray:
        """``1 + delta``."""
        return 1.0 + self.data["delta"]

    def tidal_field(self, R_s: float, *, lam_th: float = 0.0,
                    deconvolve: str | None = None, with_evecs: bool = True,
                    **kw):
        """Tidal field of the Gaussian-smoothed density (see
        :meth:`TidalField.from_density`)."""
        from .tidal import TidalField
        return TidalField.from_density(
            self.delta, self.mesh, R_s, lam_th=lam_th, deconvolve=deconvolve,
            with_evecs=with_evecs, mask=self.mask, verbose=self.verbose, **kw)


class GasField(Field):
    """Gas quantities on the mesh, e.g. ``rho`` (gas density),
    ``temperature``, ``pressure``, ``metallicity``."""

    default_key = "rho"

    @property
    def delta(self) -> np.ndarray:
        """Gas overdensity ``rho / <rho> - 1``."""
        return self.overdensity("rho")

    def mass_weighted_mean(self, key: str) -> float:
        rho = self.data["rho"].astype(np.float64)
        v = self.data[key].astype(np.float64)
        if self.mask is not None:
            rho, v = rho[self.mask], v[self.mask]
        return float((rho * v).sum() / rho.sum())
