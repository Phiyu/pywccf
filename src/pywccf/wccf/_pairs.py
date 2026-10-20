"""Cell--tracer pair counting shared by the 1D and 3D PH estimators.

For a set of centres ``g`` (web cells, weights ``w_g``) and tracers ``h``
with one or more weightings ``v_h``, the engine returns row sums

    S[k, b] = sum_g c_kg sum_h v_h 1[(g, h) in bin b]

for row coefficients ``c_kg`` that are either "regions" (``c_kg = w_g`` if
centre ``g`` lies in region ``k``; one region = no resampling) or a dense
matrix (bootstrap resamples).  The 1D binning uses ``|dx|``; the 3D binning
the folded projections ``|dx . e_i(g)|`` on the centre's eigenframe.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from ..core.log import Log
from ..core.mesh import Mesh
from .cells import WebCells

__all__ = ["PairEngine"]


class PairEngine:
    """Pair counts between web cells and a tracer set.

    Parameters
    ----------
    cells : centres (positions, weights, eigenframes for 3D).
    tracer_xs : (N, 3) tracer positions.
    mesh : periodic box.
    bins : :class:`RadialBins` (1D) or :class:`FrameBins` (3D).
    exclude_zero : drop pairs at zero separation (a tracer exactly on a
        node); True for discrete tracers, False for field nodes.
    backend : 'numpy', 'numba' or 'auto' (numba for 3D when available).
    chunk_size : centres per vectorised chunk (NumPy path).
    workers : threads of the KD-tree query and of the numba kernel.
    """

    def __init__(self, cells: WebCells, tracer_xs: np.ndarray, mesh: Mesh,
                 bins, *, exclude_zero: bool = True, backend: str = "auto",
                 chunk_size: int = 2000, workers: int = 1,
                 log: Log | None = None):
        self.cells = cells
        self.mesh = mesh
        self.l_box = mesh.l_box
        self.bins = bins
        self.ndim = bins.ndim
        if self.ndim == 3 and cells.evecs is None:
            raise ValueError("3D binning needs the cells' eigenvectors")
        self.xs = mesh.wrap(tracer_xs)
        if len(self.xs) == 0:
            raise ValueError("empty tracer set")
        self.exclude_zero = bool(exclude_zero)
        self.chunk_size = int(chunk_size)
        self.workers = max(1, int(workers))
        self.log = log or Log("PairEngine", False)
        if 2.0 * bins.r_max >= self.l_box:
            raise ValueError("box too small for the separation range")
        self.backend = self._init_backend(backend)
        self._tree = None

    # -- backend -----------------------------------------------------------
    def _init_backend(self, backend: str) -> str:
        backend = str(backend).lower()
        if backend not in ("auto", "numpy", "numba"):
            raise ValueError("backend must be 'auto', 'numpy' or 'numba'")
        if backend == "numpy" or (self.ndim != 3):
            if backend == "numba":
                raise ValueError("the numba kernel is 3D only")
            return "numpy"
        try:
            from . import _numba
            half = float(np.sqrt((self.bins.r_axis_max ** 2).sum()))
            self._cells_list = _numba.CellList(self.xs, self.l_box, half)
            self._nb = _numba
        except Exception as err:                       # no numba / tiny box
            if backend == "numba":
                raise
            self.log(f"numba backend unavailable ({err}); using numpy")
            return "numpy"
        return "numba"

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.xs, boxsize=self.l_box)
        return self._tree

    # -- geometry (NumPy) --------------------------------------------------
    def _chunk_pairs(self, cs: int, ce: int):
        """(local centre index, tracer index, flat bin) of centres cs..ce-1."""
        cx = self.cells.positions
        try:
            lists = self.tree.query_ball_point(cx[cs:ce], r=self.bins.r_max,
                                               workers=self.workers)
        except TypeError:                               # old scipy
            lists = self.tree.query_ball_point(cx[cs:ce], r=self.bins.r_max)
        lens = np.fromiter(map(len, lists), dtype=np.int64, count=ce - cs)
        jj = np.fromiter(itertools.chain.from_iterable(lists),
                         dtype=np.int64, count=int(lens.sum()))
        ii = np.repeat(np.arange(ce - cs), lens)
        if jj.size == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e, e
        dx = self.xs[jj] - cx[cs + ii]
        half = 0.5 * self.l_box
        dx[dx > half] -= self.l_box
        dx[dx < -half] += self.l_box
        d2 = np.einsum("pk,pk->p", dx, dx)
        if self.ndim == 1:
            r = np.sqrt(d2)
            keep = r <= self.bins.r_max
            if self.exclude_zero:
                keep &= r > 1e-12
            b = self.bins.bin_of(r[keep])
        else:
            # projection on each centre's own frame, block by block
            r = np.empty_like(dx)
            st = np.zeros(len(lens) + 1, dtype=np.int64)
            np.cumsum(lens, out=st[1:])
            ev = self.cells.evecs
            for i in np.flatnonzero(lens):
                np.matmul(dx[st[i]:st[i + 1]], ev[cs + i], out=r[st[i]:st[i + 1]])
            np.abs(r, out=r)
            keep = np.all(r <= self.bins.r_axis_max[None, :], axis=1)
            if self.exclude_zero:
                keep &= d2 > 1e-24
            b = self.bins.bin_of(r[keep])
        return ii[keep], jj[keep], b

    def _chunks(self):
        n = self.cells.n
        for cs in range(0, n, self.chunk_size):
            yield cs, min(cs + self.chunk_size, n)

    # -- accumulation ------------------------------------------------------
    def sums(self, V, *, regions: np.ndarray | None = None, n_regions: int = 1,
             A: np.ndarray | None = None) -> list[np.ndarray]:
        """Row sums for every tracer weighting in ``V``.

        regions : region index of every centre (default: all in region 0);
            the row coefficients are then the cell weights.
        A : dense (n_rows, n_centres) coefficients instead of regions.

        Returns one (n_rows, n_bins) array per weighting.
        """
        V = [np.asarray(v, dtype=np.float64) for v in V]
        for v in V:
            if v.shape != (len(self.xs),):
                raise ValueError("tracer weights must have one entry per tracer")
        n_c = self.cells.n
        if A is None and regions is None:
            regions = np.zeros(n_c, dtype=np.int64)
        t0 = time.perf_counter()
        if self.backend == "numba" and A is None:
            out = self._numba_sums(V, regions, n_regions)
        else:
            out = self._numpy_sums(V, regions, n_regions, A)
        self.log(f"{self.ndim}D pairs: {n_c:,} centres x {len(self.xs):,} "
                 f"tracers x {len(V)} weighting(s) [{self.backend}] in "
                 f"{time.perf_counter() - t0:.1f} s")
        return out

    def _numpy_sums(self, V, regions, n_regions, A):
        w = self.cells.probs
        nb = self.bins.n_bins
        n_rows = A.shape[0] if A is not None else int(n_regions)
        out = [np.zeros((n_rows, nb)) for _ in V]
        for cs, ce in self._chunks():
            ii, jj, bf = self._chunk_pairs(cs, ce)
            nc = ce - cs
            if A is not None:
                coef = sparse.csr_matrix(A[:, cs:ce])
            else:
                coef = sparse.csr_matrix(
                    (w[cs:ce], (regions[cs:ce], np.arange(nc))),
                    shape=(n_rows, nc))
            for v, o in zip(V, out):
                vv = v[jj]
                keep = vv != 0
                if not keep.any():
                    continue
                C = sparse.csr_matrix((vv[keep], (ii[keep], bf[keep])),
                                      shape=(nc, nb))
                o += (coef @ C).toarray()
        return out

    def _numba_sums(self, V, regions, n_regions):
        import numba
        nbk, cl = self._nb, self._cells_list
        n_thr = max(1, min(self.workers, numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(n_thr)
        order, lo, hi, t_reg = nbk.make_tasks(regions, int(n_regions), n_thr)
        h_ptr, h_v, h_w = cl.weight_csr(V)
        e1, e2, e3 = (np.asarray(e, np.float64) for e in self.bins.axes)
        nb = self.bins.n_bins
        buf = np.zeros((len(lo), len(V), nb))
        nbk.region_sums(lo, hi, order,
                        np.ascontiguousarray(self.cells.positions),
                        np.ascontiguousarray(self.cells.evecs),
                        np.ascontiguousarray(self.cells.probs),
                        cl.xs, cl.cell_ptr, cl.n_side, cl.cell_size,
                        self.l_box, True, self.exclude_zero, e1, e2, e3,
                        h_ptr, h_v, h_w,
                        len(V), buf)
        S = np.zeros((len(V), int(n_regions), nb))
        for t, r in enumerate(t_reg):
            S[:, r] += buf[t]
        return [S[i] for i in range(len(V))]
