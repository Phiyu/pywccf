"""Mass assignment (NGP / CIC / TSC) onto a periodic :class:`Mesh`, the
matching interpolation, and the Fourier windows used for deconvolution.

All schemes use the node convention of :class:`Mesh` (node ``i`` at
``i * l_grid``).  The implementation is pure NumPy: the ``p^3``
contributions of each point (p = 1, 2, 3) are accumulated with
``np.bincount`` over flat node indices, in chunks of points.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh

__all__ = ["SCHEMES", "assign", "interpolate", "window_k"]

SCHEMES = {"ngp": 1, "cic": 2, "tsc": 3}


def _scheme(scheme: str) -> str:
    s = str(scheme).lower()
    if s not in SCHEMES:
        raise ValueError(f"scheme must be one of {tuple(SCHEMES)}, got {scheme!r}")
    return s


def _axis_stencil(u: np.ndarray, scheme: str):
    """Node indices (unwrapped) and weights along one axis.

    u : positions in grid units, shape (N,).  Returns (idx, w) of shape
    (N, p)."""
    if scheme == "ngp":
        i = np.rint(u)
        return i[:, None].astype(np.int64), np.ones((len(u), 1))
    if scheme == "cic":
        i0 = np.floor(u)
        f = u - i0
        idx = np.stack([i0, i0 + 1.0], -1).astype(np.int64)
        return idx, np.stack([1.0 - f, f], -1)
    # tsc
    i = np.rint(u)
    d = u - i
    idx = np.stack([i - 1.0, i, i + 1.0], -1).astype(np.int64)
    w = np.stack([0.5 * (0.5 - d) ** 2, 0.75 - d * d, 0.5 * (0.5 + d) ** 2], -1)
    return idx, w


def _stencil(xs: np.ndarray, mesh: Mesh, scheme: str):
    """Flat node indices and weights of every point, shape (N, p^3)."""
    n = mesh.n_grids
    u = np.asarray(xs, dtype=np.float64) / mesh.l_grid
    (ix, wx), (iy, wy), (iz, wz) = (_axis_stencil(u[:, a], scheme)
                                    for a in range(3))
    ix, iy, iz = ix % n, iy % n, iz % n
    flat = ((ix[:, :, None, None] * n + iy[:, None, :, None]) * n
            + iz[:, None, None, :])
    w = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    p3 = flat.shape[1] * flat.shape[2] * flat.shape[3]
    return flat.reshape(len(u), p3), w.reshape(len(u), p3)


def assign(xs: np.ndarray, mesh: Mesh, weights: np.ndarray | None = None,
           scheme: str = "cic", chunk_size: int = 2_000_000) -> np.ndarray:
    """Sum of ``weights`` assigned to the nodes of ``mesh``.

    Parameters
    ----------
    xs : (N, 3) positions [same units as ``mesh.l_box``]; periodic.
    weights : (N,) per-point weights (default 1, i.e. number counts).
    scheme : 'ngp', 'cic' or 'tsc'.

    Returns
    -------
    grid : float64 array of shape ``mesh.shape`` whose total equals
    ``weights.sum()``.
    """
    scheme = _scheme(scheme)
    xs = np.asarray(xs, dtype=np.float64)
    if xs.ndim != 2 or xs.shape[1] != 3:
        raise ValueError(f"xs must be (N, 3), got {xs.shape}")
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(xs),):
            raise ValueError("weights must have one entry per point")
    out = np.zeros(mesh.n_cells, dtype=np.float64)
    for a in range(0, len(xs), chunk_size):
        b = min(a + chunk_size, len(xs))
        flat, w = _stencil(xs[a:b], mesh, scheme)
        if weights is not None:
            w = w * weights[a:b, None]
        out += np.bincount(flat.ravel(), weights=w.ravel(),
                           minlength=mesh.n_cells)
    return out.reshape(mesh.shape)


def interpolate(grid: np.ndarray, xs: np.ndarray, mesh: Mesh,
                scheme: str = "cic", chunk_size: int = 2_000_000) -> np.ndarray:
    """Values of a node field at arbitrary positions, using the same stencil
    as :func:`assign` (NGP = value of the nearest node, CIC = trilinear)."""
    scheme = _scheme(scheme)
    grid = np.asarray(grid)
    trailing = grid.shape[3:]
    mesh.check_shape(grid, "grid", trailing)
    flat_grid = grid.reshape((mesh.n_cells,) + trailing)
    xs = np.asarray(xs, dtype=np.float64)
    out = np.empty((len(xs),) + trailing, dtype=np.result_type(grid, np.float64)
                   if scheme != "ngp" else grid.dtype)
    for a in range(0, len(xs), chunk_size):
        b = min(a + chunk_size, len(xs))
        flat, w = _stencil(xs[a:b], mesh, scheme)
        vals = flat_grid[flat]                         # (n, p^3, ...)
        if scheme == "ngp":
            out[a:b] = vals[:, 0]
        else:
            w = w.reshape(w.shape + (1,) * len(trailing))
            out[a:b] = (vals * w).sum(1)
    return out


def window_k(mesh: Mesh, scheme: str = "cic", rfft: bool = True) -> np.ndarray:
    """Fourier transform of the assignment kernel,
    ``W(k) = prod_i sinc(pi k_i / (2 k_Ny))^p`` (p = 1, 2, 3 for NGP, CIC,
    TSC), on an ``rfftn`` (default) or ``fftn`` grid.  Divide by it to
    deconvolve the assignment."""
    p = SCHEMES[_scheme(scheme)]
    n = mesh.n_grids
    f = np.fft.fftfreq(n)                    # cycles per cell, in [-1/2, 1/2)
    fz = np.fft.rfftfreq(n) if rfft else f
    sx = np.sinc(f) ** p                     # np.sinc(x) = sin(pi x)/(pi x)
    sz = np.sinc(fz) ** p
    return sx[:, None, None] * sx[None, :, None] * sz[None, None, :]
