"""Periodic cubic mesh (replacement of ``pyhipp.field.cubic_box.Mesh``).

Convention (same as pyhipp / the ELUCID field files): the value of grid
point ``(i, j, k)`` sits on the *node* ``x = (i, j, k) * l_grid``; a point
belongs (NGP) to the nearest node, ``i = round(x / l_grid) mod n_grids``.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

__all__ = ["Mesh"]


class Mesh:
    """A periodic cube of side ``l_box`` sampled by ``n_grids^3`` nodes."""

    def __init__(self, n_grids: int, l_box: float):
        n_grids, l_box = int(n_grids), float(l_box)
        if n_grids < 1 or l_box <= 0:
            raise ValueError(f"invalid mesh: n_grids={n_grids}, l_box={l_box}")
        self.n_grids = n_grids
        self.l_box = l_box

    # -- basic geometry ----------------------------------------------------
    @property
    def l_grid(self) -> float:
        return self.l_box / self.n_grids

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_grids,) * 3

    @property
    def n_cells(self) -> int:
        return self.n_grids ** 3

    @property
    def volume(self) -> float:
        return self.l_box ** 3

    @property
    def cell_volume(self) -> float:
        return self.l_grid ** 3

    def __eq__(self, other) -> bool:
        return (isinstance(other, Mesh) and other.n_grids == self.n_grids
                and np.isclose(other.l_box, self.l_box))

    def __hash__(self):
        return hash((self.n_grids, round(self.l_box, 9)))

    def __repr__(self) -> str:
        return f"Mesh(n_grids={self.n_grids}, l_box={self.l_box:g})"

    def check_shape(self, arr: np.ndarray, name: str = "array",
                    trailing: tuple = ()) -> np.ndarray:
        arr = np.asarray(arr)
        expected = self.shape + tuple(trailing)
        if arr.shape != expected:
            raise ValueError(f"{name} has shape {arr.shape}, expected "
                             f"{expected} for {self}")
        return arr

    # -- positions <-> indices --------------------------------------------
    def wrap(self, xs: np.ndarray) -> np.ndarray:
        """Positions folded into ``[0, l_box)``."""
        return np.mod(np.asarray(xs, dtype=np.float64), self.l_box)

    def min_image(self, dx: np.ndarray) -> np.ndarray:
        """Separation vectors mapped to ``[-l_box/2, l_box/2]``."""
        dx = np.asarray(dx, dtype=np.float64)
        return dx - self.l_box * np.round(dx / self.l_box)

    def x_to_index(self, xs: np.ndarray, mode: str = "nearest") -> np.ndarray:
        """Integer node index of every position (periodic).

        mode : 'nearest' (NGP, the node convention) or 'floor' (the node at
        the lower-left corner of the enclosing cell).
        """
        if mode not in ("nearest", "floor"):
            raise ValueError("mode must be 'nearest' or 'floor'")
        u = np.asarray(xs, dtype=np.float64) / self.l_grid
        i = np.rint(u) if mode == "nearest" else np.floor(u)
        return np.mod(i.astype(np.int64), self.n_grids)

    def index_to_x(self, idx: np.ndarray) -> np.ndarray:
        return np.asarray(idx, dtype=np.float64) * self.l_grid

    def flat_index(self, ijk: np.ndarray) -> np.ndarray:
        ijk = np.asarray(ijk, dtype=np.int64)
        n = self.n_grids
        return (ijk[..., 0] * n + ijk[..., 1]) * n + ijk[..., 2]

    def unravel(self, flat: np.ndarray) -> np.ndarray:
        return np.stack(np.unravel_index(np.asarray(flat), self.shape), -1)

    def node_positions(self, mask: np.ndarray | None = None) -> np.ndarray:
        """(N, 3) node positions, all nodes (C order) or those in ``mask``,
        without materialising the full ``(n^3, 3)`` grid when masked."""
        if mask is None:
            idx = np.indices(self.shape).reshape(3, -1).T
        else:
            idx = np.stack(np.nonzero(self.check_shape(mask, "mask")), -1)
        return idx.astype(np.float64) * self.l_grid

    # -- Fourier space -----------------------------------------------------
    def k_axes(self, rfft: bool = True):
        """Wavenumbers [h/Mpc] along the three axes, shaped for broadcasting
        on an ``rfftn`` (default) or ``fftn`` grid."""
        n, h = self.n_grids, self.l_grid
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
        kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=h) if rfft else k
        return k[:, None, None], k[None, :, None], kz[None, None, :]

    def k2(self, rfft: bool = True) -> np.ndarray:
        kx, ky, kz = self.k_axes(rfft)
        return kx * kx + ky * ky + kz * kz

    @cached_property
    def separation(self) -> np.ndarray:
        """``|r|`` of the minimum-image displacement of every node from node
        (0, 0, 0); the radial coordinate of an FFT correlation grid."""
        n, h = self.n_grids, self.l_grid
        c = np.arange(n, dtype=np.float64)
        c = np.minimum(c, n - c) * h
        return np.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2
                       + c[None, None, :] ** 2)

    # -- sub-volumes -------------------------------------------------------
    def region_of(self, xs: np.ndarray, n_side: int) -> np.ndarray:
        """Index of the sub-cube (``n_side^3`` in total) containing each
        position; used for jackknife resampling."""
        s = int(n_side)
        t = np.clip((self.wrap(xs) / self.l_box * s).astype(np.int64), 0, s - 1)
        return (t[:, 0] * s + t[:, 1]) * s + t[:, 2]

    def region_grid(self, n_side: int) -> np.ndarray:
        """Sub-cube index of every node, shape ``mesh.shape``."""
        s = int(n_side)
        t = np.minimum(np.arange(self.n_grids) * s // self.n_grids, s - 1)
        return (t[:, None, None] * s + t[None, :, None]) * s + t[None, None, :]
