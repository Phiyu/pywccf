# halo_web_fast.py
"""Faster drop-in replacement for ``halo_web.HaloWeb3DCross``.

Same estimator, same constructor / ``from_field_data`` / ``corrs`` /
``profiles`` interface, same bootstrap resamples (identical random draws),
so results agree with ``halo_web`` to floating-point rounding.

What changed (see ``benchmark`` notes in the docstring of the class):

1. No dense ``(N_centers, N_bins)`` arrays.  The original keeps ``cnts``,
   the accumulated counts and the per-centre xi as dense float64 arrays of
   N_centers x (n1+1)(n2+1)(n3+1) -- 39 GB each for 10^6 centres and
   17^3 bins -- and copies one of them for every bootstrap iteration.
   Here every centre's contribution is streamed straight into the weighted
   mean and the bootstrap means: a bootstrap mean is sum_g m_kg xi_g / N,
   with m_kg the multiplicity of centre g in resample k, so only the
   (n_boot + 1) x N_centers multiplicity matrix is kept.
2. The pair geometry (neighbour search, projection on the eigen-frame,
   binning) is vectorised over chunks of centres instead of a Python loop
   over centres; the KD-tree query runs on ``workers`` threads.
3. The pair cache stores 6 bytes per pair (int32 halo id + int16 bin)
   instead of 12, or is not kept at all (``cache_pairs=False``).
4. ``corrs_many`` evaluates many halo weightings (e.g. the subsamples of a
   catalogue as 0/1 weights) from ONE geometry pass.  Because the estimator
   normalises by the sum of the weights, the correlation of a subsample S
   computed with weights 1[h in S] on the parent catalogue equals the one
   computed on S alone, so the neighbour search is done once per web type
   instead of once per (sample, tail, mass bin).
"""
from __future__ import annotations

import itertools
import time
from multiprocessing import cpu_count

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .halo_web import Edges, _sample_positions        # noqa: F401  (re-export)

try:
    from pyhipp.core import DataDict
except ImportError:                                  # pragma: no cover
    DataDict = dict


_N_LAM_TABLE = {"knot": 3, "filament": 2, "sheet": 1, "void": 0}


class HaloWeb3DCross:
    """3-D anisotropic halo -- web-cell cross-correlation in the local tidal
    eigen-frame (same estimator as ``halo_web.HaloWeb3DCross``):

        xi(r1, r2, r3) = sum_g w_g [ DD_g / (rho_bar_h V(B)) - 1 ]

    Extra keyword arguments (all optional):
        cache_pairs : keep the binned pairs after construction (default
            True, as the original) so that repeated ``corrs`` calls skip the
            geometry; False = stream the geometry inside each call (minimal
            memory; best with ``corrs_many``).
        workers : threads for the KD-tree query (default ``n_processes``).
        chunk_size : centres per vectorised geometry chunk (default 2000).
        error : 'jackknife' (default), 'bootstrap' (as halo_web) or 'none'.
        backend : 'auto' (default: the numba kernel of halo_web_numba when
            numba imports; used for jackknife / no errors), 'numba', 'numpy'.
            Threads = ``workers`` (default n_processes).
        cell_size : halo cell size of the numba kernel [Mpc/h]
            (default: largest search half-width / 4).
        n_jk_side : jackknife sub-cubes per side (default 4 -> 64 regions).
    ``use_disk_cache`` / ``cache_dir`` are accepted for compatibility;
    ``use_disk_cache=True`` means ``cache_pairs=False``.
    """

    def __init__(
        self,
        center_xs,
        center_evecs,
        halo_sample,
        mesh,
        edges: Edges,
        *,
        tidal_field=None,
        center_weights=None,
        periodic: bool = True,
        chunk_size: int = 2000,
        use_disk_cache: bool = False,
        cache_dir: str | None = None,
        n_processes: int = 4,
        rng: int = 42,
        max_centers: int | None = None,
        test_mode: bool = False,
        verbose: bool = True,
        cache_pairs: bool | None = None,
        workers: int | None = None,
        error: str = "jackknife",
        n_jk_side: int = 4,
        backend: str = "auto",
        cell_size: float | None = None,
    ):
        center_xs = np.asarray(center_xs, dtype=np.float64)
        center_evecs = np.asarray(center_evecs, dtype=np.float64)
        if center_xs.ndim != 2 or center_xs.shape[1] != 3:
            raise ValueError(f"center_xs must be (N,3), got {center_xs.shape}")
        if center_evecs.ndim != 3 or center_evecs.shape[1:] != (3, 3):
            raise ValueError(
                f"center_evecs must be (N,3,3), got {center_evecs.shape}")
        if len(center_xs) != len(center_evecs):
            raise ValueError("center_xs / center_evecs length mismatch")
        if not isinstance(edges, Edges):
            # accept an Edges from another import of halo_web
            if not all(hasattr(edges, k) for k in ("e1", "e2", "e3")):
                raise TypeError("edges must be an instance of Edges")

        norms = np.linalg.norm(center_evecs, axis=1, keepdims=True).clip(1e-30)
        center_evecs = center_evecs / norms

        # identical centre sub-sampling to halo_web (seed 0)
        if max_centers is not None and len(center_xs) > max_centers:
            sub_rng = np.random.default_rng(0)
            sel = np.sort(sub_rng.choice(len(center_xs), max_centers,
                                         replace=False))
            center_xs = center_xs[sel]
            center_evecs = center_evecs[sel]
            if center_weights is not None:
                center_weights = np.asarray(center_weights,
                                            dtype=np.float64)[sel]

        self.verbose = bool(verbose)
        self.periodic = bool(periodic)
        self.mesh = mesh
        self.boxsize = float(mesh.l_box)
        self.center_xs = self._wrap_positions(center_xs)
        self.center_evecs = center_evecs
        self.halo_sample = halo_sample
        self.tidal_field = tidal_field
        self.field_xs = self._wrap_positions(_sample_positions(halo_sample))
        if len(self.field_xs) == 0:
            raise ValueError("Empty halo sample.")
        if len(self.center_xs) == 0:
            raise ValueError("No web centers (mask + subsample empty).")
        if self.verbose:
            print(f"[3DWeb-fast] N_centers = {len(self.center_xs):,}, "
                  f"N_halos = {len(self.field_xs):,}")

        t0 = time.time()
        self.field_kdt = (cKDTree(self.field_xs, boxsize=self.boxsize)
                          if self.periodic else cKDTree(self.field_xs))
        if self.verbose:
            print(f"[3DWeb-fast] KDTree built in {time.time() - t0:.2f} s")

        if center_weights is not None:
            ws = np.asarray(center_weights, dtype=np.float64).clip(0.0)
            if len(ws) != len(self.center_xs):
                raise ValueError("center_weights length mismatch")
            s = ws.sum()
            if s <= 0:
                raise ValueError("All center weights are zero")
            self.ws = ws / s
        else:
            n = len(self.center_xs)
            self.ws = np.full(n, 1.0 / n, dtype=np.float64)

        self.chunk_size = int(chunk_size)
        self.n_processes = min(int(n_processes), cpu_count())
        self.workers = int(workers) if workers is not None else max(
            1, self.n_processes)
        self.test_mode = bool(test_mode)
        self.rng_seed = int(rng)
        self.error = str(error).lower()
        if self.error not in ("jackknife", "bootstrap", "none"):
            raise ValueError("error must be 'jackknife', 'bootstrap' or 'none'")
        self.n_jk_side = int(n_jk_side)
        self._set_bins(edges)
        self.backend = self._init_backend(backend, cell_size)
        if cache_pairs is None:
            # the numba kernel recomputes the geometry, no cache needed
            cache_pairs = (not use_disk_cache) and self.backend == "numpy"
        self.cache_pairs = bool(cache_pairs)
        self.n_pairs = None
        self._cache = None
        if self.cache_pairs:
            self._build_cache()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_field_data(cls, evecs, halo_sample, tidal_field=None, *,
                        edges, web_mask=None, web_type="filament",
                        use_density_weight=True, mesh=None, **kwargs):
        """Same as ``halo_web.HaloWeb3DCross.from_field_data``."""
        if mesh is None:
            if tidal_field is None:
                raise ValueError("Either tidal_field or mesh must be provided.")
            mesh = tidal_field.mesh
        evecs = np.asarray(evecs)
        if evecs.ndim != 5 or evecs.shape[-2:] != (3, 3):
            raise ValueError(f"evecs must be (Ng,Ng,Ng,3,3), got {evecs.shape}")
        ng = int(mesh.n_grids)
        if web_type not in _N_LAM_TABLE:
            raise ValueError(f"Unknown web_type {web_type!r}")
        if web_mask is None:
            if tidal_field is None:
                raise ValueError("Need tidal_field if web_mask is not given.")
            web_mask = np.asarray(tidal_field.n_lams == _N_LAM_TABLE[web_type],
                                  dtype=bool)
            if getattr(tidal_field, "recon_mask", None) is not None:
                web_mask &= np.asarray(tidal_field.recon_mask, dtype=bool)
        else:
            web_mask = np.asarray(web_mask, dtype=bool)
        if web_mask.shape != (ng, ng, ng):
            raise ValueError(f"web_mask shape {web_mask.shape} != {(ng,)*3}")
        if not web_mask.any():
            raise ValueError(f"web_mask for {web_type!r} is empty.")

        # cell positions without materialising the full (Ng^3, 3) grid
        h = mesh.l_grid
        idx = np.nonzero(web_mask)
        center_xs = np.stack(idx, axis=-1).astype(np.float64) * h
        center_evecs = evecs[web_mask].reshape(-1, 3, 3).astype(
            np.float64, copy=False)
        center_weights = None
        if use_density_weight:
            if tidal_field is None:
                raise ValueError("use_density_weight=True needs tidal_field.")
            delta = np.asarray(tidal_field.delta)
            if delta.shape != (ng, ng, ng):
                raise ValueError(f"delta shape {delta.shape} != {(ng,)*3}")
            center_weights = (1.0 + delta[web_mask].astype(np.float64)
                              ).clip(0.0).ravel()
        return cls(center_xs, center_evecs, halo_sample, mesh, edges,
                   tidal_field=tidal_field, center_weights=center_weights,
                   **kwargs)

    # ------------------------------------------------------------------ #
    # Public API (same outputs as halo_web)
    # ------------------------------------------------------------------ #
    def corrs(self, field_vals=None, n_bootstrap: int = 100,
              bootstrap: bool = True, mass_weighted: bool = True,
              weight_key: str = "mass", error: str | None = None,
              n_jk_side: int | None = None):
        """xi(r1, r2, r3) with errors.

        error : 'jackknife' (default of the object; delete-one over
            n_jk_side^3 sub-cubes of the box, see below), 'bootstrap' (the
            resampling of halo_web, bit-compatible; uses n_bootstrap and
            bootstrap) or 'none'.
        Returns 'mean' (the full-sample estimate for jackknife, the
        bootstrap average for bootstrap), 'sd', the bin centres and edges,
        and 'n_jk' (regions used) for jackknife.
        """
        vals = self._resolve_field_vals(field_vals, mass_weighted=mass_weighted,
                                        weight_key=weight_key)
        return self.corrs_many([vals], n_bootstrap=n_bootstrap,
                               bootstrap=bootstrap, error=error,
                               n_jk_side=n_jk_side)[0]

    def corrs_many(self, field_vals_list, n_bootstrap: int = 100,
                   bootstrap: bool = True, error: str | None = None,
                   n_jk_side: int | None = None):
        """``corrs`` for several halo weightings with one geometry pass.

        Each entry is a length-N_halo weight vector (e.g. 0/1 membership of
        a subsample, times any halo weights).  Every result is what
        ``corrs(field_vals=entry)`` of a fresh object returns.
        """
        V = [np.asarray(v, dtype=np.float64) for v in field_vals_list]
        for v in V:
            if v.shape != (len(self.field_xs),):
                raise ValueError(f"field_vals length {len(v)} != "
                                 f"{len(self.field_xs)}")
        denoms = []
        for v in V:
            rho_mean = v.sum() / self.boxsize ** 3
            denoms.append((rho_mean * self.dVs_flat).clip(1.0e-12))
        mode = self._mode(error, n_bootstrap, bootstrap)
        if self.backend == "numba" and mode in ("jackknife", "none"):
            if mode == "jackknife":
                reg, n_reg = self._jk_regions(n_jk_side)
            else:
                reg, n_reg = np.zeros(len(self.center_xs), np.int64), 1
            sums = self._numba_sums(V, reg, n_reg)
            return [self._finish_jk(S, reg, n_reg, offset=1.0, denom=d,
                                    errors=mode == "jackknife")
                    for S, d in zip(sums, denoms)]
        if mode == "jackknife":
            reg, n_reg = self._jk_regions(n_jk_side)
            sums = self._stream(V, kind="corr", reg=reg, n_reg=n_reg)
            return [self._finish_jk(S, reg, n_reg, offset=1.0, denom=d)
                    for S, d in zip(sums, denoms)]
        A, n_res = self._resample_matrix(n_bootstrap, mode == "bootstrap")
        sums = self._stream(V, kind="corr", A=A)
        rowsum = A.sum(1)
        return [self._finish(S / d[None, :] - rowsum[:, None], n_res)
                for S, d in zip(sums, denoms)]

    def profiles(self, field_vals=None, n_bootstrap: int = 100,
                 bootstrap: bool = True, mass_weighted: bool = True,
                 weight_key: str = "mass", error: str | None = None,
                 n_jk_side: int | None = None):
        vals = self._resolve_field_vals(field_vals, mass_weighted=mass_weighted,
                                        weight_key=weight_key)
        mode = self._mode(error, n_bootstrap, bootstrap)
        if mode == "jackknife":
            reg, n_reg = self._jk_regions(n_jk_side)
            S = self._stream([vals], kind="profile", reg=reg, n_reg=n_reg)[0]
            return self._finish_jk(S, reg, n_reg, offset=0.0, denom=None)
        A, n_res = self._resample_matrix(n_bootstrap, mode == "bootstrap")
        means = self._stream([vals], kind="profile", A=A)[0]
        return self._finish(means, n_res)

    # ------------------------------------------------------------------ #
    # Backend
    # ------------------------------------------------------------------ #
    def _init_backend(self, backend, cell_size):
        """'numba' (compiled kernel, jackknife / no errors), 'numpy', or
        'auto' = numba when it can be imported and the box allows it."""
        backend = str(backend).lower()
        if backend not in ("auto", "numba", "numpy"):
            raise ValueError("backend must be 'auto', 'numba' or 'numpy'")
        if backend == "numpy":
            return "numpy"
        try:
            from . import halo_web_numba as hwn
            half = float(max(self.edges.e1[-1], self.edges.e2[-1],
                             self.edges.e3[-1]) * np.sqrt(3.0))
            self._cells = hwn.CellList(self.field_xs, self.boxsize, half,
                                       cell_size=cell_size,
                                       periodic=self.periodic)
            self._hwn = hwn
        except Exception as err:                     # no numba, small box
            if backend == "numba":
                raise
            if self.verbose:
                print(f"[3DWeb-fast] numba backend unavailable ({err}); "
                      "using numpy")
            return "numpy"
        if self.verbose:
            print(f"[3DWeb-fast] numba backend: {self._cells.n_side}^3 "
                  f"cells of {self._cells.cell_size:.2f} Mpc/h")
        return "numba"

    def _numba_sums(self, V, reg, n_reg):
        """Per-region sums  sum_{g in k} w_g DD_g(bin)  for every weighting,
        from the compiled kernel (one pass over centres and haloes)."""
        import numba
        hwn, cells = self._hwn, self._cells
        n_thr = max(1, min(self.workers, numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(n_thr)
        order, lo, hi, t_reg = hwn.make_tasks(reg, n_reg, n_thr)
        h_ptr, h_v, h_w = cells.weight_csr(V)
        nb = self.n_bins_flat
        out = np.zeros((len(lo), len(V), nb))
        t0 = time.time()
        hwn.region_sums(lo, hi, order, np.ascontiguousarray(self.center_xs),
                        np.ascontiguousarray(self.center_evecs),
                        np.ascontiguousarray(self.ws), cells.xs,
                        cells.cell_ptr, cells.n_side, cells.cell_size,
                        self.boxsize, self.periodic,
                        np.asarray(self.edges.e1, np.float64),
                        np.asarray(self.edges.e2, np.float64),
                        np.asarray(self.edges.e3, np.float64),
                        h_ptr, h_v, h_w, len(V), out)
        S = np.zeros((len(V), n_reg, nb))
        for t, r in enumerate(t_reg):
            S[:, r] += out[t]
        if self.verbose:
            print(f"[3DWeb-fast] numba: {len(self.center_xs):,} centres x "
                  f"{len(V)} weighting(s) on {n_thr} thread(s) in "
                  f"{time.time() - t0:.1f} s")
        return [S[i] for i in range(len(V))]

    # ------------------------------------------------------------------ #
    # Error modes
    # ------------------------------------------------------------------ #
    def _mode(self, error, n_bootstrap, bootstrap):
        mode = self.error if error is None else str(error).lower()
        if mode not in ("jackknife", "bootstrap", "none"):
            raise ValueError("error must be 'jackknife', 'bootstrap' or 'none'")
        if self.test_mode:
            return "none"
        if mode == "bootstrap" and ((not bootstrap) or n_bootstrap <= 0):
            return "none"
        return mode

    def _jk_regions(self, n_jk_side=None):
        """Sub-cube (of n_jk_side^3) of every web-cell centre."""
        s = self.n_jk_side if n_jk_side is None else int(n_jk_side)
        if s < 2:
            raise ValueError("n_jk_side must be >= 2")
        t = np.clip((self.center_xs / self.boxsize * s).astype(np.int64),
                    0, s - 1)
        return (t[:, 0] * s + t[:, 1]) * s + t[:, 2], s ** 3

    def _finish_jk(self, S_k, reg, n_reg, offset, denom, errors=True):
        """Full estimate and delete-one jackknife error from the per-region
        sums S_k = sum_{g in k} w_g X_g (X_g = pair counts or profiles).

        Removing region k removes its web cells (and with them every pair
        centred there); the halo field and its mean density stay those of
        the whole box, so no edge effect enters the analytic normalisation.
        """
        W_k = np.bincount(reg, weights=self.ws, minlength=n_reg)
        S, W = S_k.sum(0), W_k.sum()
        scale = (lambda x: x / denom[None, :] - offset) if denom is not None \
            else (lambda x: x)
        full = scale((S / W)[None, :])[0]
        use = (W_k > 0) & (W - W_k > 0)
        n_jk = int(use.sum()) if errors else 0
        if n_jk >= 2:
            th = scale((S[None, :] - S_k[use]) / (W - W_k[use])[:, None])
            sd = np.sqrt((n_jk - 1.0) / n_jk
                         * ((th - th.mean(0)) ** 2).sum(0))
        else:
            sd = np.zeros_like(full)
        res = self._finish(full[None, :], 0)
        res["sd"] = sd.reshape(res["mean"].shape)
        if errors:
            res["n_jk"] = n_jk
        return res

    # ------------------------------------------------------------------ #
    def _resolve_field_vals(self, field_vals, *, mass_weighted, weight_key):
        n_h = len(self.field_xs)
        if field_vals is not None:
            v = np.asarray(field_vals, dtype=np.float64)
            if len(v) != n_h:
                raise ValueError(f"field_vals length {len(v)} != {n_h}")
            return v
        if mass_weighted and weight_key in self.halo_sample:
            w = np.asarray(self.halo_sample[weight_key], dtype=np.float64)
            if len(w) != n_h:
                raise ValueError(f"weight_key length {len(w)} != {n_h}")
            return w
        return np.ones(n_h, dtype=np.float64)

    def _finish(self, means, n_res):
        if n_res == 0:
            mean, sd = means[0], np.zeros_like(means[0])
        else:
            mean, sd = means.mean(0), means.std(0)
        shape = (self.n1 + 1, self.n2 + 1, self.n3 + 1)
        return DataDict({
            "mean": mean.reshape(shape), "sd": sd.reshape(shape),
            "r1_centers": self.r1_centers, "r2_centers": self.r2_centers,
            "r3_centers": self.r3_centers, "r1_edges": self.edges.e1,
            "r2_edges": self.edges.e2, "r3_edges": self.edges.e3,
        })

    def _resample_matrix(self, n_bootstrap, bootstrap):
        """Rows of centre coefficients whose products with the per-centre
        estimates give the reported means: the weights w (no bootstrap) or
        the multiplicities / N of every bootstrap resample, drawn with the
        same generators, seeds and call order as halo_web (fresh object)."""
        n_c = len(self.center_xs)
        ws = self.ws
        if self.test_mode:
            bootstrap, n_bootstrap = False, 0
        if (not bootstrap) or n_bootstrap <= 0:
            return ws[None, :].copy(), 0
        draws = []
        if n_bootstrap >= 20 and self.n_processes > 1:
            seed_rng = np.random.default_rng(self.rng_seed)
            n_per = n_bootstrap // self.n_processes
            rem = n_bootstrap % self.n_processes
            plan = [(n_per, int(seed_rng.integers(0, 2**31)))
                    for _ in range(self.n_processes)]
            if rem > 0:
                plan.append((rem, int(seed_rng.integers(0, 2**31))))
            for n_iter, seed in plan:
                r = np.random.default_rng(seed)
                for _ in range(n_iter):
                    draws.append(r.choice(n_c, n_c, p=ws))
        else:
            r = np.random.default_rng(self.rng_seed)
            for _ in range(n_bootstrap):
                draws.append(r.choice(n_c, n_c, p=ws))
        A = np.empty((len(draws), n_c), dtype=np.float64)
        for k, ids in enumerate(draws):
            A[k] = np.bincount(ids, minlength=n_c) / n_c
        return A, len(draws)

    # ------------------------------------------------------------------ #
    # Bins
    # ------------------------------------------------------------------ #
    def _set_bins(self, edges):
        self.edges = edges
        n1, n2, n3 = (len(edges.e1) - 1, len(edges.e2) - 1, len(edges.e3) - 1)
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.n_bins_flat = (n1 + 1) * (n2 + 1) * (n3 + 1)
        self.r_max_search = float(np.sqrt(edges.e1[-1] ** 2 + edges.e2[-1] ** 2
                                          + edges.e3[-1] ** 2))
        dr1 = np.diff(np.concatenate([[0.0], edges.e1]))
        dr2 = np.diff(np.concatenate([[0.0], edges.e2]))
        dr3 = np.diff(np.concatenate([[0.0], edges.e3]))
        dVs = 8.0 * (dr1[:, None, None] * dr2[None, :, None]
                     * dr3[None, None, :])
        self.dVs_flat = dVs.ravel().astype(np.float64)
        m = lambda e: 0.5 * (e[:-1] + e[1:])
        self.r1_centers = np.concatenate([[0.5 * edges.e1[0]], m(edges.e1)])
        self.r2_centers = np.concatenate([[0.5 * edges.e2[0]], m(edges.e2)])
        self.r3_centers = np.concatenate([[0.5 * edges.e3[0]], m(edges.e3)])
        self._bin_dtype = np.int16 if self.n_bins_flat < 2**15 else np.int32
        if self.verbose:
            print(f"[3DWeb-fast] bins: ({n1+1}, {n2+1}, {n3+1}) -> "
                  f"{self.n_bins_flat} flat; r_max_search = "
                  f"{self.r_max_search:.3f} Mpc/h")

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def _iter_chunks(self):
        n_c = len(self.center_xs)
        for cs in range(0, n_c, self.chunk_size):
            yield cs, min(cs + self.chunk_size, n_c)

    def _query(self, xs):
        try:
            return self.field_kdt.query_ball_point(
                xs, r=self.r_max_search, workers=self.workers)
        except TypeError:                            # scipy < 1.6
            return self.field_kdt.query_ball_point(xs, r=self.r_max_search)

    def _chunk_pairs(self, cs, ce):
        """Binned pairs of centres cs..ce-1: (local centre index, halo id,
        flat bin), sorted by centre."""
        lists = self._query(self.center_xs[cs:ce])
        lens = np.fromiter(map(len, lists), dtype=np.int64, count=ce - cs)
        jj = np.fromiter(itertools.chain.from_iterable(lists),
                         dtype=np.int64, count=int(lens.sum()))
        ii = np.repeat(np.arange(ce - cs), lens)
        if jj.size == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e, e.astype(self._bin_dtype)
        dx = self.field_xs[jj]
        dx -= self.center_xs[cs + ii]
        if self.periodic:
            # same as dx -= L * round(dx / L) for |dx| < L, but cheaper
            half = 0.5 * self.boxsize
            m = dx > half
            dx[m] -= self.boxsize
            m = dx < -half
            dx[m] += self.boxsize
        d2 = np.einsum("pk,pk->p", dx, dx)
        # projection on each centre's own triad: one small matmul per centre
        # on its contiguous block of pairs (as in halo_web, bit for bit)
        r = np.empty_like(dx)
        st = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=st[1:])
        ev = self.center_evecs
        for i in np.flatnonzero(lens):
            a, b = st[i], st[i + 1]
            np.matmul(dx[a:b], ev[cs + i], out=r[a:b])
        np.abs(r, out=r)
        e1, e2, e3 = self.edges.e1, self.edges.e2, self.edges.e3
        valid = ((d2 > 1e-24) & (r[:, 0] <= e1[-1]) & (r[:, 1] <= e2[-1])
                 & (r[:, 2] <= e3[-1]))
        ii, jj, r = ii[valid], jj[valid], r[valid]
        b1 = np.minimum(np.searchsorted(e1, r[:, 0], side="right"), self.n1)
        b2 = np.minimum(np.searchsorted(e2, r[:, 1], side="right"), self.n2)
        b3 = np.minimum(np.searchsorted(e3, r[:, 2], side="right"), self.n3)
        bf = (b1 * (self.n2 + 1) + b2) * (self.n3 + 1) + b3
        return ii, jj, bf.astype(self._bin_dtype)

    def _build_cache(self):
        t0 = time.time()
        jj_l, bf_l, lens_l = [], [], []
        for cs, ce in self._iter_chunks():
            ii, jj, bf = self._chunk_pairs(cs, ce)
            jj_l.append(jj.astype(np.int32 if len(self.field_xs) < 2**31
                                  else np.int64))
            bf_l.append(bf)
            lens_l.append(np.bincount(ii, minlength=ce - cs))
        lens = np.concatenate(lens_l)
        starts = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=starts[1:])
        self._cache = (starts, np.concatenate(jj_l), np.concatenate(bf_l))
        self.n_pairs = int(starts[-1])
        if self.verbose:
            print(f"[3DWeb-fast] pair cache: {self.n_pairs:,} pairs "
                  f"({self.n_pairs * (jj_l[0].itemsize + bf_l[0].itemsize) / 2**20:.0f}"
                  f" MB) in {time.time() - t0:.1f} s")

    def _pairs(self):
        """Iterate over (cs, ce, local centre index, halo id, flat bin)."""
        if self._cache is not None:
            starts, jj_all, bf_all = self._cache
            for cs, ce in self._iter_chunks():
                a, b = starts[cs], starts[ce]
                lens = np.diff(starts[cs:ce + 1])
                ii = np.repeat(np.arange(ce - cs), lens)
                yield cs, ce, ii, jj_all[a:b], bf_all[a:b]
        else:
            for cs, ce in self._iter_chunks():
                ii, jj, bf = self._chunk_pairs(cs, ce)
                yield cs, ce, ii, jj, bf

    # ------------------------------------------------------------------ #
    # Streaming accumulation
    # ------------------------------------------------------------------ #
    def _stream(self, V, kind, A=None, reg=None, n_reg=None):
        """For every weighting v: sum_g c_kg X_g(bin), with X_g the weighted
        pair count of centre g (kind='corr') or its mean value per bin
        (kind='profile', the per-centre ratio of halo_web).  The rows c_k are
        the dense rows of A, or (reg given) c_kg = w_g [g in region k]."""
        nb = self.n_bins_flat
        n_rows = A.shape[0] if A is not None else n_reg
        out = [np.zeros((n_rows, nb)) for _ in V]
        t0 = time.time()
        for cs, ce, ii, jj, bf in self._pairs():
            nc = ce - cs
            if A is not None:
                rows, Ac = slice(None), A[:, cs:ce]
            else:                       # only the regions of this chunk
                rc = reg[cs:ce]
                rows, inv = np.unique(rc, return_inverse=True)
                Ac = np.zeros((len(rows), nc))
                Ac[inv, np.arange(nc)] = self.ws[cs:ce]
            if kind == "profile":
                cnt = np.zeros((nc, nb))
                np.add.at(cnt, (ii, bf.astype(np.int64)), 1.0)
            for v, o in zip(V, out):
                w = v[jj]
                keep = w != 0
                if not keep.any():
                    continue
                ik, bk, wk = ii[keep], bf[keep], w[keep]
                if kind == "corr":
                    indptr = np.zeros(nc + 1, dtype=np.int64)
                    np.cumsum(np.bincount(ik, minlength=nc), out=indptr[1:])
                    C = sparse.csr_matrix((wk, bk.astype(np.int32), indptr),
                                          shape=(nc, nb))
                    o[rows] += np.asarray(C.T @ Ac.T).T
                else:
                    tot = np.zeros((nc, nb))
                    np.add.at(tot, (ik, bk.astype(np.int64)), wk)
                    o[rows] += Ac @ (tot / cnt.clip(1e-2))
        if self.verbose:
            print(f"[3DWeb-fast] accumulated {len(V)} weighting(s) x "
                  f"{n_rows} row(s) in {time.time() - t0:.1f} s")
        return out

    def _wrap_positions(self, xs):
        xs = np.asarray(xs, dtype=np.float64)
        return np.mod(xs, self.boxsize) if self.periodic else xs
