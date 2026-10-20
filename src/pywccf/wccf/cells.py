"""Web cells used as the centres of the WCCF estimators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh
from ..samples.tidal import WEB_TYPES, TidalField, _n_lam

__all__ = ["WebCells"]


@dataclass
class WebCells:
    """Nodes of one web type: positions, weights and (for 3D) eigenframes.

    Attributes
    ----------
    web : web type name.
    flat_index : flat node indices (C order, ascending).
    positions : (N, 3) node positions [h^-1 Mpc].
    weights : (N,) non-negative cell weights ``w_c``.
    evecs : (N, 3, 3) eigenframes, ``evecs[c, :, m] = e_{m+1}`` (or None).
    mesh : the mesh.
    n_total : number of cells of this type before sub-sampling.
    """

    web: str
    flat_index: np.ndarray
    positions: np.ndarray
    weights: np.ndarray
    evecs: np.ndarray | None
    mesh: Mesh
    n_total: int

    @classmethod
    def from_tidal(cls, tf: TidalField, web, *, weight="density",
                   max_cells: int | None = None, seed: int = 0,
                   need_evecs: bool = True) -> "WebCells":
        """Select the cells of ``web`` in ``tf``.

        weight : 'density' (``w_c = (1 + delta)_+``, the paper), None
            (uniform), or a node array of weights.
        max_cells : random subset of that many cells (without replacement,
            ``numpy.random.default_rng(seed)``; identical to the reference
            implementation used for the paper).
        """
        name = WEB_TYPES[_n_lam(web)]
        mask = tf.web_mask(web)
        flat = np.flatnonzero(mask)
        n_total = len(flat)
        if n_total == 0:
            raise ValueError(f"no {name} cells")
        if max_cells is not None and n_total > int(max_cells):
            rng = np.random.default_rng(seed)
            flat = flat[np.sort(rng.choice(n_total, int(max_cells),
                                           replace=False))]
        mesh = tf.mesh
        positions = mesh.unravel(flat).astype(np.float64) * mesh.l_grid
        if weight is None:
            w = np.ones(len(flat))
        elif isinstance(weight, str):
            if weight != "density":
                raise ValueError("weight must be 'density', None or an array")
            w = (1.0 + tf.delta.ravel()[flat].astype(np.float64)).clip(0.0)
        else:
            w = np.asarray(mesh.check_shape(weight, "weight"),
                           dtype=np.float64).ravel()[flat].clip(0.0)
        if w.sum() <= 0:
            raise ValueError(f"all {name} cell weights are zero")
        evecs = None
        if need_evecs:
            sel = np.zeros(mesh.n_cells, dtype=bool)
            sel[flat] = True
            evecs = tf.evecs_in(sel.reshape(mesh.shape))
            norms = np.linalg.norm(evecs, axis=1, keepdims=True).clip(1e-30)
            evecs = evecs / norms
        return cls(name, flat, positions, w, evecs, mesh, n_total)

    @property
    def n(self) -> int:
        return len(self.flat_index)

    @property
    def probs(self) -> np.ndarray:
        return self.weights / self.weights.sum()

    def regions(self, n_side: int):
        """Jackknife sub-cube of every cell and the number of sub-cubes."""
        return self.mesh.region_of(self.positions, n_side), int(n_side) ** 3

    def weight_grid(self) -> np.ndarray:
        """Cell weights painted on the mesh (zero elsewhere)."""
        g = np.zeros(self.mesh.n_cells)
        g[self.flat_index] = self.weights
        return g.reshape(self.mesh.shape)
