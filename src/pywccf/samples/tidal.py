"""Tidal field and T-web classification.

At every node, ``T_ij = d_i d_j Phi`` with ``nabla^2 Phi = delta_s`` (the
density contrast smoothed by a Gaussian of scale ``R_s``); its eigenvalues
``lambda_1 >= lambda_2 >= lambda_3`` and eigenvectors ``e_1, e_2, e_3``
define the web type (number of eigenvalues ``>= lambda_th``: 3 knot,
2 filament, 1 sheet, 0 void) and the local eigenframe of the 3D-WCCF.

Array layout (also the HDF5 layout of the ELUCID field files):
``lam`` (n, n, n, 3), descending; ``evec`` (n, n, n, 3, 3) with
``evec[i, j, k, :, m] = e_{m+1}``; ``delta`` (file dataset ``delta_sm_x``)
is the density contrast used for the cell weights ``w_c = (1 + delta)_+``.
"""
from __future__ import annotations

from functools import cached_property
from typing import Mapping

import h5py
import numpy as np

from ..core.assignment import window_k
from ..core.mesh import Mesh
from .field import Field

__all__ = ["WEB_TYPES", "WEB_N_LAM", "TidalField", "tidal_eigensystem"]

WEB_TYPES = ("void", "sheet", "filament", "knot")
WEB_N_LAM = {w: i for i, w in enumerate(WEB_TYPES)}

#: file dataset name -> TidalField key
FILE_KEYS = {"lam": "lam", "evec": "evec", "delta_sm_x": "delta"}


def _n_lam(web) -> int:
    if isinstance(web, str):
        if web not in WEB_N_LAM:
            raise ValueError(f"unknown web type {web!r}; use {WEB_TYPES}")
        return WEB_N_LAM[web]
    n = int(web)
    if n not in (0, 1, 2, 3):
        raise ValueError("web index must be 0..3")
    return n


def tidal_eigensystem(delta: np.ndarray, mesh: Mesh, R_s: float, *,
                      deconvolve: str | None = None, with_evecs: bool = True,
                      dtype=np.float32, slab: int = 16):
    """Smoothed density, tidal eigenvalues and (optionally) eigenvectors.

    ``T_ij(k) = k_i k_j / k^2 * delta(k) * exp(-k^2 R_s^2 / 2)`` (zero mode
    removed), optionally divided by the assignment window of
    ``deconvolve`` ('ngp', 'cic', 'tsc'); this is the same tensor as the
    ELUCID field files (pyhipp), up to the eigenvalue ordering.

    Returns ``(delta_s, lam, evec)``; ``lam`` descending, ``evec`` None
    unless ``with_evecs``.  Memory: six real tensor components are kept
    (6 n^3 values of ``dtype``); the eigen-decomposition runs in slabs.
    """
    delta = mesh.check_shape(delta, "delta").astype(np.float64)
    dk = np.fft.rfftn(delta)
    if deconvolve:
        dk /= window_k(mesh, deconvolve)
    dk *= np.exp(-0.5 * mesh.k2() * R_s * R_s)
    delta_s = np.fft.irfftn(dk, s=mesh.shape, axes=(0, 1, 2)).astype(dtype)
    k = mesh.k_axes()
    k2 = mesh.k2()
    k2[0, 0, 0] = 1.0
    comps = {}
    for i in range(3):
        for j in range(i, 3):
            tk = k[i] * k[j] / k2 * dk
            tk[0, 0, 0] = 0.0
            comps[i, j] = np.fft.irfftn(tk, s=mesh.shape, axes=(0, 1, 2)).astype(dtype)
    del dk
    n = mesh.n_grids
    lam = np.empty(mesh.shape + (3,), dtype=dtype)
    evec = np.empty(mesh.shape + (3, 3), dtype=dtype) if with_evecs else None
    for a in range(0, n, slab):
        b = min(a + slab, n)
        T = np.empty((b - a, n, n, 3, 3), dtype=np.float64)
        for (i, j), c in comps.items():
            T[..., i, j] = c[a:b]
            T[..., j, i] = c[a:b]
        if with_evecs:
            w, v = np.linalg.eigh(T)
            lam[a:b] = w[..., ::-1]
            evec[a:b] = v[..., ::-1]
        else:
            lam[a:b] = np.linalg.eigvalsh(T)[..., ::-1]
    return delta_s, lam, evec


class TidalField(Field):
    """Tidal eigenvalues / eigenvectors and the T-web classification.

    Parameters
    ----------
    data : must contain ``lam`` (n, n, n, 3); optional ``evec``
        (n, n, n, 3, 3) and ``delta`` (n, n, n).
    mesh : the :class:`Mesh`.
    lam_th : classification threshold.
    R_s : smoothing scale [h^-1 Mpc] (metadata).
    mask : optional node mask (e.g. reconstruction volume).
    evec_path : HDF5 file from which ``evec`` is read on demand when it is
        not in ``data`` (see :meth:`from_file`, ``load_evec='lazy'``).
    """

    default_key = "delta"

    def __init__(self, data: Mapping[str, np.ndarray], mesh: Mesh, *,
                 lam_th: float = 0.0, R_s: float | None = None,
                 mask=None, meta=None, verbose=True,
                 evec_path=None, evec_dataset: str = "evec"):
        if "lam" not in data:
            raise KeyError("TidalField needs 'lam'")
        super().__init__(data, mesh, mask=mask, meta=meta, verbose=verbose)
        mesh.check_shape(self.data["lam"], "lam", (3,))
        if "evec" in self.data:
            mesh.check_shape(self.data["evec"], "evec", (3, 3))
        self.lam_th = float(lam_th)
        self.R_s = None if R_s is None else float(R_s)
        self.evec_path = evec_path
        self.evec_dataset = evec_dataset
        self.log(f"R_s={self.R_s}, lambda_th={self.lam_th:g}, "
                 + ", ".join(f"{w}={f:.3f}" for w, f in
                             self.volume_fractions().items()))

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_density(cls, delta: np.ndarray, mesh: Mesh, R_s: float, *,
                     lam_th: float = 0.0, deconvolve: str | None = None,
                     with_evecs: bool = True, mask=None, verbose=True,
                     **kw) -> "TidalField":
        """Tidal field of an (unsmoothed) density contrast on ``mesh``; the
        stored ``delta`` is the smoothed contrast, as in the field files."""
        delta_s, lam, evec = tidal_eigensystem(
            delta, mesh, R_s, deconvolve=deconvolve, with_evecs=with_evecs,
            **kw)
        data = {"lam": lam, "delta": delta_s}
        if evec is not None:
            data["evec"] = evec
        return cls(data, mesh, lam_th=lam_th, R_s=R_s, mask=mask,
                   verbose=verbose, meta={"deconvolve": deconvolve or ""})

    @classmethod
    def from_file(cls, path, *, load_evec="lazy", recon_only: bool = False,
                  lam_th: float | None = None, delta_key: str = "delta_sm_x",
                  **init_kw) -> "TidalField":
        """Read an ELUCID-style field file (``lam``, ``evec``,
        ``delta_sm_x``, ``n_grids``, ``l_box``, optional
        ``smoothing_scale``, ``lambda_th``, ``reconstruction_mask``).

        load_evec : True (read now), False (never) or 'lazy' (read the
            eigenvectors of the requested cells on demand, slab by slab).
        recon_only : apply ``reconstruction_mask``.
        lam_th : threshold; default the file value (or 0).
        """
        with h5py.File(path, "r", locking=False) as f:
            mesh = cls._read_mesh(f)
            lam = np.asarray(f["lam"][()], dtype=np.float32)
            data = {"lam": lam}
            if delta_key in f:
                data["delta"] = f[delta_key][()]
            has_evec = "evec" in f
            if load_evec is True:
                if not has_evec:
                    raise KeyError(f"{path} has no 'evec'")
                data["evec"] = np.asarray(f["evec"][()], dtype=np.float32)
            R_s = (float(np.asarray(f["smoothing_scale"][()]).item())
                   if "smoothing_scale" in f else None)
            if lam_th is None:
                lam_th = (float(np.asarray(f["lambda_th"][()]).item())
                          if "lambda_th" in f else 0.0)
            mask = None
            if recon_only:
                mask = f["reconstruction_mask"][()].astype(bool)
        if lam.shape[-1] == 3:
            probe = lam.reshape(-1, 3)[:: max(1, lam.size // 3 // 10000)]
            if np.mean(probe[:, 0] < probe[:, 2]) > 0.5:
                lam = lam[..., ::-1]           # stored ascending
                data["lam"] = lam
        return cls(data, mesh, lam_th=lam_th, R_s=R_s, mask=mask,
                   evec_path=(path if (load_evec == "lazy" and has_evec)
                              else None),
                   meta={"path": str(path)}, **init_kw)

    def to_file(self, path, *, with_evec: bool = True, compression=None):
        """Write in the ELUCID field-file layout."""
        keys = [k for k in ("lam", "delta") if k in self.data]
        if with_evec and "evec" in self.data:
            keys.append("evec")
        out = super().to_file(path, keys, compression=compression,
                              key_map={"delta": "delta_sm_x"})
        with h5py.File(path, "a") as f:
            f["lambda_th"] = self.lam_th
            if self.R_s is not None:
                f["smoothing_scale"] = self.R_s
            if self.mask is not None:
                f["reconstruction_mask"] = self.mask
        return out

    # -- eigen-system ------------------------------------------------------
    @property
    def lams(self) -> np.ndarray:
        return self.data["lam"]

    @property
    def delta(self) -> np.ndarray:
        return self.data["delta"]

    @property
    def has_evecs(self) -> bool:
        return "evec" in self.data or self.evec_path is not None

    @property
    def evecs(self) -> np.ndarray:
        """Full eigenvector array (read from file on first access when
        lazy)."""
        if "evec" not in self.data:
            if self.evec_path is None:
                raise KeyError("no eigenvectors (load_evec=False?)")
            with h5py.File(self.evec_path, "r", locking=False) as f:
                self.data["evec"] = np.asarray(f[self.evec_dataset][()],
                                               dtype=np.float32)
        return self.data["evec"]

    def evecs_in(self, sel: np.ndarray) -> np.ndarray:
        """Eigenvectors of the nodes in boolean ``sel`` (C order),
        shape (N, 3, 3), without loading the full array when lazy."""
        sel = self.mesh.check_shape(sel, "sel").astype(bool)
        if "evec" in self.data:
            return self.data["evec"][sel].astype(np.float64)
        if self.evec_path is None:
            raise KeyError("no eigenvectors (load_evec=False?)")
        out = []
        with h5py.File(self.evec_path, "r", locking=False) as f:
            ds = f[self.evec_dataset]
            for i in np.flatnonzero(sel.any(axis=(1, 2))):
                out.append(ds[i][sel[i]].astype(np.float64))
        return (np.concatenate(out) if out
                else np.empty((0, 3, 3), dtype=np.float64))

    # -- classification ----------------------------------------------------
    @cached_property
    def n_lams(self) -> np.ndarray:
        """Number of eigenvalues ``>= lambda_th`` (0 void ... 3 knot)."""
        return (self.lams >= self.lam_th).sum(-1).astype(np.int8)

    def with_threshold(self, lam_th: float) -> "TidalField":
        """Same arrays, another classification threshold."""
        new = TidalField(self.data, self.mesh, lam_th=lam_th, R_s=self.R_s,
                         mask=self.mask, meta=self.meta, verbose=self.verbose,
                         evec_path=self.evec_path,
                         evec_dataset=self.evec_dataset)
        return new

    def web_mask(self, web) -> np.ndarray:
        """Boolean node mask of web type ``web`` (name or 0..3), within
        :attr:`mask`."""
        m = self.n_lams == _n_lam(web)
        if self.mask is not None:
            m &= self.mask
        return m

    def volume_fractions(self) -> dict:
        n = self.n_lams if self.mask is None else self.n_lams[self.mask]
        counts = np.bincount(n.ravel(), minlength=4)
        return {w: counts[i] / counts.sum() for i, w in enumerate(WEB_TYPES)}

    def web_type_at(self, xs) -> np.ndarray:
        """Web index (0 void ... 3 knot) at positions ``xs`` (NGP)."""
        idx = self.mesh.x_to_index(xs)
        return self.n_lams[idx[:, 0], idx[:, 1], idx[:, 2]]

    def lam_at(self, xs) -> np.ndarray:
        idx = self.mesh.x_to_index(xs)
        return self.lams[idx[:, 0], idx[:, 1], idx[:, 2]]
