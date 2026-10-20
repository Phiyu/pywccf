"""Separation bins of the 1D and 3D WCCF.

Both bin types follow the convention of the paper: the edges ``e_0 < e_1 <
... < e_n`` define ``n + 1`` bins, the first one being the *underflow* bin
``[0, e_0)``; pairs beyond ``e_n`` are discarded.

* :class:`RadialBins` -- spherical shells (1D-WCCF), volume
  ``4 pi / 3 (e_k^3 - e_{k-1}^3)``.
* :class:`FrameBins` -- rectangular bins in the folded eigenframe
  coordinates ``r_i = |dx . e_i|`` (3D-WCCF), volume
  ``8 dr_1 dr_2 dr_3`` (8 = folding of the octants).
"""
from __future__ import annotations

import numpy as np

__all__ = ["PAPER_EDGES", "RadialBins", "FrameBins"]

#: edges of the paper (17 bins with the underflow bin [0, 0.5) h^-1 Mpc)
PAPER_EDGES = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
                        10.5, 11.6904, 13.0158, 14.4914, 16.1343, 17.9635,
                        20.0])


def _check_edges(e, name="edges") -> np.ndarray:
    e = np.asarray(e, dtype=np.float64)
    if e.ndim != 1 or len(e) < 1:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if e[0] <= 0 or np.any(np.diff(e) <= 0):
        raise ValueError(f"{name} must be positive and strictly increasing")
    return e


def _centers(e: np.ndarray) -> np.ndarray:
    inner = np.concatenate([[0.0], e[:-1]])
    return 0.5 * (inner + e)


class RadialBins:
    """Spherical shells ``[0, e_0), [e_0, e_1), ..., [e_{n-1}, e_n]``."""

    ndim = 1

    def __init__(self, edges=PAPER_EDGES):
        self.edges = _check_edges(edges)

    @property
    def n_bins(self) -> int:
        return len(self.edges)

    @property
    def shape(self) -> tuple[int]:
        return (self.n_bins,)

    @property
    def r_max(self) -> float:
        """Search radius."""
        return float(self.edges[-1])

    @property
    def centers(self) -> np.ndarray:
        return _centers(self.edges)

    @property
    def inner_edges(self) -> np.ndarray:
        return np.concatenate([[0.0], self.edges[:-1]])

    @property
    def volumes(self) -> np.ndarray:
        return 4.0 * np.pi / 3.0 * (self.edges ** 3 - self.inner_edges ** 3)

    def bin_of(self, r: np.ndarray) -> np.ndarray:
        """Bin index (``-1`` outside ``[0, e_n]``)."""
        r = np.asarray(r, dtype=np.float64)
        b = np.minimum(np.searchsorted(self.edges, r, side="right"),
                       self.n_bins - 1)
        return np.where((r >= 0) & (r <= self.edges[-1]), b, -1)

    def __repr__(self):
        return (f"RadialBins(n_bins={self.n_bins}, "
                f"r_max={self.r_max:g})")


class FrameBins:
    """Folded eigenframe bins in ``(r_1, r_2, r_3)``; one edge array per
    axis (a single array is used for all three)."""

    ndim = 3

    def __init__(self, e1=PAPER_EDGES, e2=None, e3=None):
        e1 = _check_edges(e1, "e1")
        self.e1 = e1
        self.e2 = e1.copy() if e2 is None else _check_edges(e2, "e2")
        self.e3 = e1.copy() if e3 is None else _check_edges(e3, "e3")

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.e1, self.e2, self.e3

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(len(e) for e in self.axes)

    @property
    def n_bins(self) -> int:
        n1, n2, n3 = self.shape
        return n1 * n2 * n3

    @property
    def r_axis_max(self) -> np.ndarray:
        return np.array([e[-1] for e in self.axes])

    @property
    def r_max(self) -> float:
        """Search radius enclosing the box ``|r_i| <= e_i[-1]``."""
        return float(np.sqrt((self.r_axis_max ** 2).sum()))

    @property
    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(_centers(e) for e in self.axes)

    @property
    def volumes(self) -> np.ndarray:
        d = [np.diff(np.concatenate([[0.0], e])) for e in self.axes]
        return 8.0 * d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :]

    def bin_of(self, r: np.ndarray) -> np.ndarray:
        """Flat bin index of folded coordinates ``r`` (N, 3) (``-1`` outside
        the box)."""
        r = np.asarray(r, dtype=np.float64)
        n1, n2, n3 = self.shape
        b = [np.minimum(np.searchsorted(e, r[:, a], side="right"), len(e) - 1)
             for a, e in enumerate(self.axes)]
        flat = (b[0] * n2 + b[1]) * n3 + b[2]
        inside = np.all((r >= 0) & (r <= self.r_axis_max[None, :]), axis=1)
        return np.where(inside, flat, -1)

    def __repr__(self):
        return f"FrameBins(shape={self.shape}, r_axis_max={self.r_axis_max})"
