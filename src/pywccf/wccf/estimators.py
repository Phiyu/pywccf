"""1D and 3D cosmic-web cross-correlation functions (WCCF).

Both estimators centre on the cells of one web type ``W`` of a
:class:`~pywccf.samples.TidalField` (cell weights ``w_c``, by default
``(1 + delta)_+``) and correlate them with a tracer or a field.

Peebles--Hauser (PH) estimator, tracer with weights ``v_h``::

    xi_W(B) = sum_c w_c N_v(c, B) / sum_c w_c / (n_v V_B) - 1,
    N_v(c, B) = sum_h v_h 1[(c, h) in B],   n_v = sum_h v_h / L^3,

with ``B`` a spherical shell (1D, :class:`RadialBins`) or a folded
eigenframe bin ``r_i = |(x_h - x_c) . e_i(x_c)|`` (3D, :class:`FrameBins`,
``V_B = 8 dr_1 dr_2 dr_3``).  A field enters the PH estimator through its
nodes, weighted by ``1 + delta``.

Convolution (1D only, field on the same mesh)::

    xi_W(r) = < sum_x W(x) delta_f(x + r) / sum_x W(x) >_{|r| in B},
    W(x) = w_c 1[x in W],

computed with FFTs and averaged over the grid displacements of each shell.

Central values are always the deterministic estimator; ``error`` selects
the uncertainty: 'none', 'jackknife' (delete-one over ``n_jk_side^3``
sub-cubes of web cells) or 'bootstrap' (weighted resampling of cells; PH
only).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..core.assignment import assign
from ..core.bins import PAPER_EDGES, FrameBins, RadialBins
from ..core.log import HasLog
from ..core.resampling import bootstrap_matrix, check_error_mode, jackknife
from ..samples.field import Field
from ..samples.tidal import WEB_TYPES, TidalField
from ..samples.tracer import Tracer
from ._pairs import PairEngine
from .cells import WebCells
from .result import WCCFResult

__all__ = ["WCCF1D", "WCCF3D"]


class _WCCFBase(HasLog):
    ndim = 0
    _bins_cls = None

    def __init__(self, tidal_field: TidalField, tracer: Tracer | None = None,
                 field: Field | None = None, *, edges=PAPER_EDGES, bins=None,
                 cell_weight="density", max_cells: int | None = 1_000_000,
                 cell_seed: int = 0, tracer_weight=None,
                 field_key: str | None = None, field_stride: int = 1,
                 backend: str = "auto", chunk_size: int = 2000,
                 workers: int = 1, verbose: bool = True):
        super().__init__(verbose=verbose)
        if not isinstance(tidal_field, TidalField):
            raise TypeError("tidal_field must be a TidalField")
        if (tracer is None) == (field is None):
            raise ValueError("give exactly one of tracer= or field=")
        if field is not None and not np.isclose(field.l_box,
                                                tidal_field.l_box):
            raise ValueError("field and tidal field have different boxes")
        self.tf = tidal_field
        self.mesh = tidal_field.mesh
        self.tracer = tracer
        self.field = field
        self.bins = bins if bins is not None else self._bins_cls(edges)
        if self.bins.ndim != self.ndim:
            raise ValueError(f"{type(self).__name__} needs {self.ndim}D bins")
        self.cell_weight = cell_weight
        self.max_cells = max_cells
        self.cell_seed = int(cell_seed)
        self.tracer_weight = tracer_weight
        self.field_key = field_key
        self.field_stride = int(field_stride)
        self.backend = backend
        self.chunk_size = int(chunk_size)
        self.workers = int(workers)
        self._cells: dict = {}
        self._points = None
        self._engines: dict = {}
        src = (f"tracer {type(tracer).__name__} (N={tracer.n_objs:,})"
               if tracer is not None else f"field {type(field).__name__}")
        self.log(f"{src}; {self.bins!r}; R_s={tidal_field.R_s}, "
                 f"lambda_th={tidal_field.lam_th:g}")

    # -- centres and points ------------------------------------------------
    def cells(self, web) -> WebCells:
        key = WEB_TYPES[WEB_TYPES.index(web)] if isinstance(web, str) else web
        if key not in self._cells:
            self._cells[key] = WebCells.from_tidal(
                self.tf, web, weight=self.cell_weight,
                max_cells=self.max_cells, seed=self.cell_seed,
                need_evecs=self.ndim == 3)
            c = self._cells[key]
            self.log(f"{c.web}: {c.n:,} of {c.n_total:,} cells")
        return self._cells[key]

    def _point_set(self):
        """(positions, base weights, exclude_zero) of the PH point set."""
        if self._points is None:
            if self.tracer is not None:
                self._points = (self.tracer.positions,
                                self.tracer.weights(self.tracer_weight), True)
            else:
                pts = self.field.to_tracer(self.field_key,
                                           stride=self.field_stride)
                self._points = (pts.positions, np.asarray(pts["weight"]),
                                False)
                self.log(f"field as {pts.n_objs:,} weighted nodes "
                         f"(stride {self.field_stride})")
        return self._points

    def _weightings(self, samples, n_points: int, base: np.ndarray):
        """Names and weight vectors of the requested sub-samples."""
        if samples is None:
            return ["all"], [base]
        if isinstance(samples, Mapping):
            items = list(samples.items())
        elif isinstance(samples, Sequence) and not isinstance(samples, str):
            items = [(f"s{i}", s) for i, s in enumerate(samples)]
        else:
            raise TypeError("samples must be None, a dict or a list")
        names, V = [], []
        for name, s in items:
            s = np.asarray(s)
            if s.shape != (n_points,):
                raise ValueError(f"sample {name!r}: shape {s.shape} != "
                                 f"({n_points},)")
            v = base * s.astype(np.float64)
            if v.sum() <= 0:
                raise ValueError(f"sample {name!r} has zero total weight")
            names.append(str(name))
            V.append(v)
        return names, V

    def _engine(self, web) -> PairEngine:
        cells = self.cells(web)
        if cells.web not in self._engines:
            xs, _, excl = self._point_set()
            self._engines[cells.web] = PairEngine(
                cells, xs, self.mesh, self.bins, exclude_zero=excl,
                backend=self.backend, chunk_size=self.chunk_size,
                workers=self.workers, log=self.log)
        return self._engines[cells.web]

    # -- PH estimator ------------------------------------------------------
    def _compute_ph(self, web, samples, error, n_jk_side, n_bootstrap, seed):
        error = check_error_mode(error)
        cells = self.cells(web)
        _, base, _ = self._point_set()
        if self.field is not None and samples is not None:
            raise ValueError("sub-samples are only defined for tracers")
        names, V = self._weightings(samples, len(base), base)
        eng = self._engine(web)
        vol = self.bins.volumes.ravel()
        L3 = self.mesh.volume
        probs = cells.probs
        regions, n_reg, A = None, 1, None
        if error == "jackknife":
            regions, n_reg = cells.regions(n_jk_side)
        elif error == "bootstrap":
            A = np.vstack([probs[None, :],
                           bootstrap_matrix(probs, n_bootstrap, seed)])
        sums = eng.sums(V, regions=regions, n_regions=n_reg, A=A)
        shape = self.bins.shape
        out = {}
        for name, v, S in zip(names, V, sums):
            nbar = v.sum() / L3
            norm = nbar * vol

            def transform(m, norm=norm):
                return m / norm - 1.0

            samples_ = None
            if error == "jackknife":
                W_k = np.bincount(regions, weights=probs, minlength=n_reg)
                xi, sd, samples_ = jackknife(S, W_k, transform)
            elif error == "bootstrap":
                xi = transform(S[0])
                samples_ = S[1:] / norm[None, :] - A[1:].sum(1)[:, None]
                sd = samples_.std(0)
            else:
                xi, sd = transform(S[0]), np.zeros(len(vol))
            meta = self._meta(cells, error, n_jk_side, n_bootstrap, seed,
                              method="ph", n_tracers=int((v != 0).sum()),
                              tracer_weight_sum=float(v.sum()),
                              backend=eng.backend)
            if samples_ is not None:
                samples_ = samples_.reshape((len(samples_),) + shape)
                meta["n_samples"] = len(samples_)
            out[name] = WCCFResult(xi=xi.reshape(shape), xi_sd=sd.reshape(shape),
                                   bins=self.bins, web=cells.web, name=name,
                                   method="ph", error=error,
                                   samples=samples_, meta=meta)
        return out

    def _meta(self, cells, error, n_jk_side, n_bootstrap, seed, **kw):
        return {"n_cells": cells.n, "n_cells_total": cells.n_total,
                "cell_weight": (self.cell_weight if isinstance(
                    self.cell_weight, str) or self.cell_weight is None
                    else "custom") or "uniform",
                "R_s": self.tf.R_s, "lambda_th": self.tf.lam_th,
                "l_box": self.mesh.l_box, "n_grids": self.mesh.n_grids,
                "n_jk_side": n_jk_side if error == "jackknife" else None,
                "n_bootstrap": n_bootstrap if error == "bootstrap" else None,
                "seed": seed if error == "bootstrap" else None,
                "max_cells": self.max_cells, "cell_seed": self.cell_seed,
                **kw}

    # -- public API --------------------------------------------------------
    def compute(self, web, samples=None, *, error: str = "jackknife",
                n_jk_side: int = 4, n_bootstrap: int = 100,
                seed: int = 0) -> dict[str, WCCFResult]:
        """WCCF of web type ``web`` for every sub-sample.

        samples : None (the whole tracer), or a dict ``name -> mask`` (or
            per-tracer weights, multiplied with ``tracer_weight``), or a
            list of masks.  All sub-samples share one pair-counting pass.
        error : 'none', 'jackknife' or 'bootstrap'.
        Returns ``{name: WCCFResult}``.
        """
        return self._compute_ph(web, samples, error, n_jk_side, n_bootstrap,
                                seed)

    def compute_all(self, webs=WEB_TYPES, samples=None, **kw):
        """``{web: {name: WCCFResult}}`` for several web types."""
        return {w: self.compute(w, samples, **kw) for w in webs}


class WCCF3D(_WCCFBase):
    """3D-WCCF in the local tidal eigenframe (PH estimator only).

    Parameters
    ----------
    tidal_field : :class:`TidalField` with eigenvectors.
    tracer : a :class:`Tracer` (halos, galaxies, ...), or
    field : a :class:`Field`, used through its nodes weighted by
        ``1 + value`` (``field_stride`` thins the nodes).
    edges / bins : separation bins (default: the 17 bins of the paper).
    cell_weight : 'density' (``(1 + delta)_+``), None or a node array.
    max_cells : random subset of cells per web type (paper: 10^6).
    tracer_weight : base tracer weight ('number', 'mass', column, array).
    backend : 'auto' (numba kernel when available), 'numpy' or 'numba'.
    workers : threads of the KD-tree query / numba kernel.
    """

    ndim = 3
    _bins_cls = FrameBins


class WCCF1D(_WCCFBase):
    """Isotropic (1D) WCCF.

    ``method='ph'`` (default for a tracer): pair counts in spherical shells.
    ``method='fft'`` (default for a field): convolution of the web-cell
    weight field with the field's overdensity on the tidal-field mesh; a
    tracer is first assigned to that mesh (``scheme``, ``tracer_weight``).
    Other parameters as :class:`WCCF3D`.
    """

    ndim = 1
    _bins_cls = RadialBins

    def __init__(self, tidal_field, tracer=None, field=None, *,
                 method: str = "auto", scheme: str = "cic",
                 field_is_overdensity: bool | None = None, **kw):
        super().__init__(tidal_field, tracer, field, **kw)
        method = str(method).lower()
        if method == "auto":
            method = "ph" if tracer is not None else "fft"
        if method not in ("ph", "fft"):
            raise ValueError("method must be 'auto', 'ph' or 'fft'")
        if method == "fft" and field is not None and field.mesh != self.mesh:
            raise ValueError("method='fft' needs the field on the tidal mesh")
        self.method = method
        self.scheme = scheme
        self.field_is_overdensity = field_is_overdensity

    def _field_delta(self) -> np.ndarray:
        f = self.field
        key = self.field_key or f.default_key
        is_od = self.field_is_overdensity
        if is_od is None:
            is_od = key == "delta"
        return (np.asarray(f[key], dtype=np.float64) if is_od
                else f.overdensity(key))

    def _tracer_deltas(self, samples):
        base = self.tracer.weights(self.tracer_weight)
        names, V = self._weightings(samples, len(base), base)
        xs = self.tracer.positions
        out = {}
        for name, v in zip(names, V):
            grid = assign(xs, self.mesh, weights=v, scheme=self.scheme)
            out[name] = grid / grid.mean() - 1.0
        return out

    def _compute_fft(self, web, samples, error, n_jk_side):
        error = check_error_mode(error)
        if error == "bootstrap":
            raise ValueError("method='fft' supports error='none' or "
                             "'jackknife'")
        if self.field is not None:
            if samples is not None:
                raise ValueError("sub-samples are only defined for tracers")
            deltas = {"all": self._field_delta()}
        else:
            deltas = self._tracer_deltas(samples)
        cells = WebCells.from_tidal(self.tf, web, weight=self.cell_weight,
                                    max_cells=None, need_evecs=False)
        mesh = self.mesh
        Wg = cells.weight_grid()
        b = self.bins.bin_of(mesh.separation).ravel()
        inb = b >= 0
        b_in = b[inb]
        nb = self.bins.n_bins
        counts = np.bincount(b_in, minlength=nb).astype(np.float64)
        if np.any(counts == 0):
            self.log("warning: some shells contain no grid displacement")
        Fk = {name: np.fft.rfftn(d) for name, d in deltas.items()}

        def shell_sums(Wgrid):
            Wk = np.conj(np.fft.rfftn(Wgrid))
            res = {}
            for name, fk in Fk.items():
                U = np.fft.irfftn(Wk * fk, s=mesh.shape, axes=(0, 1, 2)).ravel()
                res[name] = np.bincount(b_in, weights=U[inb], minlength=nb)
            return res

        def transform(m):
            with np.errstate(invalid="ignore", divide="ignore"):
                return m / counts

        out = {}
        if error == "none":
            S = shell_sums(Wg)
            W = Wg.sum()
            for name in Fk:
                out[name] = (transform(S[name] / W), np.zeros(nb), None)
        else:
            reg_grid = mesh.region_grid(n_jk_side)
            n_reg = int(n_jk_side) ** 3
            S_k = {name: np.zeros((n_reg, nb)) for name in Fk}
            W_k = np.zeros(n_reg)
            for k in range(n_reg):
                Wgk = np.where(reg_grid == k, Wg, 0.0)
                W_k[k] = Wgk.sum()
                if W_k[k] == 0:
                    continue
                for name, s in shell_sums(Wgk).items():
                    S_k[name][k] = s
            for name in Fk:
                out[name] = jackknife(S_k[name], W_k, transform)
        res = {}
        for name, (xi, sd, smp) in out.items():
            meta = self._meta(cells, error, n_jk_side, 0, 0, method="fft", 
            scheme=self.scheme if self.tracer is not None else None)
            meta["n_cells"] = cells.n
            res[name] = WCCFResult(xi=xi, xi_sd=sd, bins=self.bins, web=cells.web, name=name, method="fft", error=error, samples=smp, meta=meta)
        return res

    def compute(self, web, samples=None, *, error: str = "jackknife", n_jk_side: int = 4, n_bootstrap: int = 100, seed: int = 0) -> dict[str, WCCFResult]:
        if self.method == "fft":
            return self._compute_fft(web, samples, error, n_jk_side)
        return self._compute_ph(web, samples, error, n_jk_side, n_bootstrap,
                                seed)
