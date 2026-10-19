# halo_web_3d.py
from __future__ import annotations

import os
import time
import tempfile
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from pyhipp.core import DataDict
from pyhipp.field.cubic_box import Mesh
from scipy.spatial import cKDTree

from ..samples import HaloSample, TidalField

try:
    from tqdm import tqdm
except ImportError:                                # pragma: no cover
    def tqdm(it, **kw):
        return it


# --------------------------------------------------------------------------- #
# Edges adapter
# --------------------------------------------------------------------------- #
class Edges:
    """Anisotropic radial bin edges for the 3 tidal eigen-axes.

    Each axis carries strictly increasing positive edges. If only one
    array is given, it is broadcast to all three axes (isotropic case).
    """

    def __init__(self, e1, e2=None, e3=None):
        e1 = np.asarray(e1, dtype=np.float64)
        e2 = e1.copy() if e2 is None else np.asarray(e2, dtype=np.float64)
        e3 = e1.copy() if e3 is None else np.asarray(e3, dtype=np.float64)
        for name, e in zip(("e1", "e2", "e3"), (e1, e2, e3)):
            if e.ndim != 1 or len(e) < 2:
                raise ValueError(f"{name} must be 1-D with length >= 2")
            if e[0] <= 0:
                raise ValueError(f"{name}[0] must be > 0")
            if not np.all(np.diff(e) > 0):
                raise ValueError(f"{name} must be strictly increasing")
        self.e1, self.e2, self.e3 = e1, e2, e3

    @property
    def n_bins(self):
        return len(self.e1) - 1, len(self.e2) - 1, len(self.e3) - 1

    @property
    def r_axis_max(self):
        return self.e1[-1], self.e2[-1], self.e3[-1]

    @property
    def r_max(self):
        return float(np.sqrt(self.e1[-1]**2 + self.e2[-1]**2 + self.e3[-1]**2))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bootstrap_worker_3d(args):
    outs, ws, n_iter, seed = args
    rng = np.random.default_rng(seed)
    means = np.zeros((n_iter, outs.shape[1]), dtype=np.float64)
    for i in range(n_iter):
        ids = rng.choice(len(outs), len(outs), p=ws)
        means[i] = outs[ids].mean(0)
    return means


def _sample_positions(sample: HaloSample) -> np.ndarray:
    if "position" in sample:
        xs = np.asarray(sample["position"], dtype=np.float64)
        if xs.ndim != 2 or xs.shape[1] != 3:
            raise ValueError(f"sample['position'] must be (N,3), got {xs.shape}")
        return xs
    if all(k in sample for k in ("x", "y", "z")):
        return np.column_stack(
            [sample["x"], sample["y"], sample["z"]]
        ).astype(np.float64, copy=False)
    raise ValueError("HaloSample needs 'position' or 'x','y','z'.")


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #
class HaloWeb3DCross:
    """3-D anisotropic cross-correlation between halos and cosmic-web cells,
    decomposed in the local tidal eigen-frame.

    The estimator (per cell, then weighted-averaged over cells) is

        xi_hat(r1, r2, r3)
            = sum_g w_tilde_g * [ DD_g / (rho_bar_h * V(B)) - 1 ]

    with
        r_i  = |(x_h - x_g) . e_i(x_g)|              (folded to first octant)
        V(B) = 8 * dr1 * dr2 * dr3                   (8 = octant-folding)
        w_g  = (1 + delta_web)_+   (when use_density_weight=True)

    See the accompanying theory note for full derivation.
    """

    # ------------------------------------------------------------------ #
    # Constructor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        center_xs: np.ndarray,
        center_evecs: np.ndarray,           # shape (N, 3, 3); [:, :, i] = e_i
        halo_sample: HaloSample,
        mesh: Mesh,
        edges: Edges,
        *,
        tidal_field: TidalField | None = None,
        center_weights: np.ndarray | None = None,
        periodic: bool = True,
        chunk_size: int = 500,
        use_disk_cache: bool = False,
        cache_dir: str | None = None,
        n_processes: int = 4,
        rng: int = 42,
        max_centers: int | None = None,
        test_mode: bool = False,
        verbose: bool = True,
    ):
        center_xs = np.asarray(center_xs, dtype=np.float64)
        center_evecs = np.asarray(center_evecs, dtype=np.float64)

        if center_xs.ndim != 2 or center_xs.shape[1] != 3:
            raise ValueError(f"center_xs must be (N,3), got {center_xs.shape}")
        if center_evecs.ndim != 3 or center_evecs.shape[1:] != (3, 3):
            raise ValueError(f"center_evecs must be (N,3,3), got {center_evecs.shape}")
        if len(center_xs) != len(center_evecs):
            raise ValueError("center_xs / center_evecs length mismatch")
        if not isinstance(edges, Edges):
            raise TypeError("edges must be an instance of Edges")

        # Normalize each eigenvector column to unit length (defensive).
        norms = np.linalg.norm(center_evecs, axis=1, keepdims=True).clip(1e-30)
        center_evecs = center_evecs / norms

        # Optional sub-sampling of centers (preserve sync of weights)
        if max_centers is not None and len(center_xs) > max_centers:
            sub_rng = np.random.default_rng(0)
            sel = np.sort(sub_rng.choice(len(center_xs), max_centers, replace=False))
            center_xs = center_xs[sel]
            center_evecs = center_evecs[sel]
            if center_weights is not None:
                center_weights = np.asarray(center_weights, dtype=np.float64)[sel]

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
            print(f"[3DWeb] N_centers = {len(self.center_xs):,}, "
                  f"N_halos = {len(self.field_xs):,}")

        # KDTree on field side
        t0 = time.time()
        if self.periodic:
            self.field_kdt = cKDTree(self.field_xs, boxsize=self.boxsize)
        else:
            self.field_kdt = cKDTree(self.field_xs)
        if self.verbose:
            print(f"[3DWeb] KDTree built in {time.time() - t0:.2f} s")

        # Center weights (rho_web), normalized to a probability vector
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
        self.use_disk_cache = bool(use_disk_cache)
        self.cache_dir = cache_dir
        self.n_processes = min(int(n_processes), cpu_count())
        self.test_mode = bool(test_mode)

        self._seed_rng = np.random.default_rng(int(rng))
        self._rng = np.random.default_rng(int(rng))
        self.cache_file: str | None = None

        self._set_bins(edges)
        self._build_neighbor_cache()

    # ------------------------------------------------------------------ #
    # Builder from gridded field
    # ------------------------------------------------------------------ #
    @classmethod
    def from_field_data(
        cls,
        evecs: np.ndarray,                      # (Ng,Ng,Ng,3,3); [...,i] = e_i
        halo_sample: HaloSample,
        tidal_field: TidalField | None = None,
        *,
        edges: Edges,
        web_mask: np.ndarray | None = None,
        web_type: str = "filament",
        use_density_weight: bool = True,
        mesh: Mesh | None = None,
        **kwargs,
    ):
        if mesh is None:
            if tidal_field is None:
                raise ValueError("Either tidal_field or mesh must be provided.")
            mesh = tidal_field.mesh

        evecs = np.asarray(evecs)
        if evecs.ndim != 5 or evecs.shape[-2:] != (3, 3):
            raise ValueError(f"evecs must be (Ng,Ng,Ng,3,3), got {evecs.shape}")
        ng = int(mesh.n_grids)

        _N_LAM_TABLE = {"knot": 3, "filament": 2, "sheet": 1, "void": 0}
        if web_type not in _N_LAM_TABLE:
            raise ValueError(f"Unknown web_type {web_type!r}")
        default_n_lam = _N_LAM_TABLE[web_type]

        # Resolve mask
        if web_mask is None:
            if tidal_field is None:
                raise ValueError("Need tidal_field if web_mask is not given.")
            web_mask = np.asarray(tidal_field.n_lams == default_n_lam, dtype=bool)
            if getattr(tidal_field, "recon_mask", None) is not None:
                web_mask &= np.asarray(tidal_field.recon_mask, dtype=bool)
        else:
            web_mask = np.asarray(web_mask, dtype=bool)

        if web_mask.shape != (ng, ng, ng):
            raise ValueError(f"web_mask shape {web_mask.shape} != {(ng,ng,ng)}")
        if not web_mask.any():
            raise ValueError(f"web_mask for {web_type!r} is empty.")

        # Grid positions of masked cells
        h = mesh.l_grid
        axis = np.arange(ng, dtype=np.float64) * h
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
        center_xs = np.stack([gx, gy, gz], axis=-1)[web_mask].reshape(-1, 3)

        # Full triad (Nc, 3, 3) in the SAME column convention as input
        center_evecs = evecs[web_mask].reshape(-1, 3, 3).astype(np.float64, copy=False)

        # rho_web at each cell
        center_weights = None
        if use_density_weight:
            if tidal_field is None:
                raise ValueError("use_density_weight=True needs tidal_field.")
            delta = np.asarray(tidal_field.delta)
            if delta.shape != (ng, ng, ng):
                raise ValueError(f"delta shape {delta.shape} != {(ng,ng,ng)}")
            center_weights = (1.0 + delta[web_mask].astype(np.float64)).clip(0.0).ravel()

        return cls(
            center_xs, center_evecs, halo_sample, mesh, edges,
            tidal_field=tidal_field,
            center_weights=center_weights,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def corrs(
        self,
        field_vals: np.ndarray | None = None,
        n_bootstrap: int = 100,
        bootstrap: bool = True,
        mass_weighted: bool = True,
        weight_key: str = "mass",
    ):
        vals = self._resolve_field_vals(
            field_vals, mass_weighted=mass_weighted, weight_key=weight_key
        )
        c = self._w_corr(vals)
        mean, sd = self._bootstrap(c, n_bootstrap=n_bootstrap, bootstrap=bootstrap)
        shape = (self.n1 + 1, self.n2 + 1, self.n3 + 1)
        return DataDict({
            "mean":      mean.reshape(shape),
            "sd":        sd.reshape(shape),
            "r1_centers": self.r1_centers,
            "r2_centers": self.r2_centers,
            "r3_centers": self.r3_centers,
            "r1_edges":   self.edges.e1,
            "r2_edges":   self.edges.e2,
            "r3_edges":   self.edges.e3,
        })

    def profiles(
        self,
        field_vals: np.ndarray | None = None,
        n_bootstrap: int = 100,
        bootstrap: bool = True,
        mass_weighted: bool = True,
        weight_key: str = "mass",
    ):
        vals = self._resolve_field_vals(
            field_vals, mass_weighted=mass_weighted, weight_key=weight_key
        )
        p = self._v_profiles(vals)
        mean, sd = self._bootstrap(p, n_bootstrap=n_bootstrap, bootstrap=bootstrap)
        shape = (self.n1 + 1, self.n2 + 1, self.n3 + 1)
        return DataDict({
            "mean":      mean.reshape(shape),
            "sd":        sd.reshape(shape),
            "r1_centers": self.r1_centers,
            "r2_centers": self.r2_centers,
            "r3_centers": self.r3_centers,
            "r1_edges":   self.edges.e1,
            "r2_edges":   self.edges.e2,
            "r3_edges":   self.edges.e3,
        })

    # ------------------------------------------------------------------ #
    # Halo-side weights
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

    # ------------------------------------------------------------------ #
    # Binning setup
    # ------------------------------------------------------------------ #
    def _set_bins(self, edges: Edges):
        self.edges = edges
        n1, n2, n3 = edges.n_bins
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.n_bins_flat = (n1 + 1) * (n2 + 1) * (n3 + 1)
        self.r_max_search = edges.r_max

        # Volume element with leading underflow bin per axis
        dr1 = np.diff(np.concatenate([[0.0], edges.e1]))   # (n1+1,)
        dr2 = np.diff(np.concatenate([[0.0], edges.e2]))   # (n2+1,)
        dr3 = np.diff(np.concatenate([[0.0], edges.e3]))   # (n3+1,)
        # Folding factor 8 (first-octant)
        dVs = 8.0 * (dr1[:, None, None] * dr2[None, :, None] * dr3[None, None, :])
        self.dVs_flat = dVs.ravel().astype(np.float64)

        # Bin centers (linear midpoints; underflow center = e_i^(1) / 2)
        m1 = 0.5 * (edges.e1[:-1] + edges.e1[1:])
        m2 = 0.5 * (edges.e2[:-1] + edges.e2[1:])
        m3 = 0.5 * (edges.e3[:-1] + edges.e3[1:])
        self.r1_centers = np.concatenate([[0.5 * edges.e1[0]], m1])
        self.r2_centers = np.concatenate([[0.5 * edges.e2[0]], m2])
        self.r3_centers = np.concatenate([[0.5 * edges.e3[0]], m3])

        if self.verbose:
            print(f"[3DWeb] bins: ({n1+1}, {n2+1}, {n3+1}) "
                  f"-> {self.n_bins_flat} flat;  r_max_search = "
                  f"{self.r_max_search:.3f} Mpc/h")

    # ------------------------------------------------------------------ #
    # CSR neighbor cache
    # ------------------------------------------------------------------ #
    def _build_neighbor_cache(self):
        n_c = len(self.center_xs)
        self.cnts = np.zeros((n_c, self.n_bins_flat), dtype=np.float64)
        self.starts = np.zeros(n_c + 1, dtype=np.int64)

        if self.use_disk_cache:
            if self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                suffix=".hdf5", prefix="halo3d_cache_",
                dir=self.cache_dir, delete=False,
            ) as tmp:
                self.cache_file = tmp.name

            self.nn_ids_flat = None
            self.bin_flat_flat = None
            self._fill_cache_disk()
        else:
            self.cache_file = None
            self._fill_cache_memory()

        if self.verbose:
            total_pairs = int(self.starts[-1])
            print(f"[3DWeb] neighbor cache: {total_pairs:,} kept pairs "
                  f"({'disk' if self.use_disk_cache else 'memory'})")

    def _iter_chunks(self):
        n_c = len(self.center_xs)
        for cs in range(0, n_c, self.chunk_size):
            yield cs, min(cs + self.chunk_size, n_c)

    def _fill_cache_memory(self):
        n_c = len(self.center_xs)
        nn_chunks, bf_chunks = [], []
        running = 0

        chunk_iter = self._iter_chunks()
        if self.verbose:
            chunk_iter = tqdm(
                list(chunk_iter), desc="[3DWeb] cache (mem)", unit="chunk",
            )
        for cs, ce in chunk_iter:
            chunk_nn_ids = self.field_kdt.query_ball_point(
                self.center_xs[cs:ce], r=self.r_max_search,
            )
            for i, nn_id in enumerate(chunk_nn_ids):
                ci = cs + i
                nn_id = np.asarray(nn_id, dtype=np.int64)
                nn_keep, bf = self._neighbor_bins(ci, nn_id)
                nn_chunks.append(nn_keep)
                bf_chunks.append(bf)
                running += len(nn_keep)
                self.starts[ci + 1] = running
                if bf.size:
                    self.cnts[ci] = np.bincount(
                        bf, minlength=self.n_bins_flat
                    )[: self.n_bins_flat]

        self.nn_ids_flat = (np.concatenate(nn_chunks)
                            if nn_chunks else np.empty(0, dtype=np.int64))
        self.bin_flat_flat = (np.concatenate(bf_chunks)
                              if bf_chunks else np.empty(0, dtype=np.int32))

    def _fill_cache_disk(self):
        with h5py.File(self.cache_file, "w", locking=False) as f:
            nn_dset = f.create_dataset(
                "nn_ids", shape=(0,), maxshape=(None,),
                dtype=np.int64, chunks=True,
            )
            bf_dset = f.create_dataset(
                "bin_flat", shape=(0,), maxshape=(None,),
                dtype=np.int32, chunks=True,
            )
            running = 0

            chunk_iter = self._iter_chunks()
            if self.verbose:
                chunk_iter = tqdm(
                    list(chunk_iter), desc="[3DWeb] cache (disk)", unit="chunk",
                )
            for cs, ce in chunk_iter:
                chunk_nn_ids = self.field_kdt.query_ball_point(
                    self.center_xs[cs:ce], r=self.r_max_search,
                )
                local_nn, local_bf = [], []
                for i, nn_id in enumerate(chunk_nn_ids):
                    ci = cs + i
                    nn_id = np.asarray(nn_id, dtype=np.int64)
                    nn_keep, bf = self._neighbor_bins(ci, nn_id)
                    local_nn.append(nn_keep)
                    local_bf.append(bf)
                    running += len(nn_keep)
                    self.starts[ci + 1] = running
                    if bf.size:
                        self.cnts[ci] = np.bincount(
                            bf, minlength=self.n_bins_flat
                        )[: self.n_bins_flat]
                if local_nn:
                    nn_concat = np.concatenate(local_nn)
                    bf_concat = np.concatenate(local_bf)
                    new_total = nn_dset.shape[0] + len(nn_concat)
                    nn_dset.resize((new_total,))
                    bf_dset.resize((new_total,))
                    nn_dset[new_total - len(nn_concat):] = nn_concat
                    bf_dset[new_total - len(bf_concat):] = bf_concat

            f.create_dataset("starts", data=self.starts)

    def _neighbor_bins(self, ci, nn_id):
        if len(nn_id) == 0:
            return (np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int32))

        dx = self.field_xs[nn_id] - self.center_xs[ci]
        if self.periodic:
            dx -= self.boxsize * np.round(dx / self.boxsize)

        nn_d = np.linalg.norm(dx, axis=1)
        triad = self.center_evecs[ci]                  # (3,3) cols = e_i
        # |dx . e_i| for i=1,2,3, vectorized: shape (n_nn, 3)
        r_comp = np.abs(dx @ triad)

        r1_max, r2_max, r3_max = self.edges.r_axis_max
        valid = (
            (nn_d > 1e-12)
            & (r_comp[:, 0] <= r1_max)
            & (r_comp[:, 1] <= r2_max)
            & (r_comp[:, 2] <= r3_max)
        )
        if not np.any(valid):
            return (np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int32))

        nn_keep = nn_id[valid]
        rk = r_comp[valid]

        # searchsorted side='right': 0 -> underflow, 1..N -> normal,
        # N+1 -> overflow (already filtered out by `valid`)
        b1 = np.searchsorted(self.edges.e1, rk[:, 0], side="right").astype(np.int32)
        b2 = np.searchsorted(self.edges.e2, rk[:, 1], side="right").astype(np.int32)
        b3 = np.searchsorted(self.edges.e3, rk[:, 2], side="right").astype(np.int32)
        np.clip(b1, 0, self.n1, out=b1)
        np.clip(b2, 0, self.n2, out=b2)
        np.clip(b3, 0, self.n3, out=b3)

        s2 = self.n2 + 1
        s3 = self.n3 + 1
        bf = (b1 * s2 * s3 + b2 * s3 + b3).astype(np.int32)
        return nn_keep, bf

    # ------------------------------------------------------------------ #
    # Accumulators
    # ------------------------------------------------------------------ #
    def _process_chunk(self, vals, start, end):
        chunk_n = end - start
        out = np.zeros((chunk_n, self.n_bins_flat), dtype=np.float64)

        if self.use_disk_cache:
            s_start = int(self.starts[start])
            s_end = int(self.starts[end])
            if s_end == s_start:
                return out
            with h5py.File(self.cache_file, "r", locking=False) as f:
                nn_chunk = f["nn_ids"][s_start:s_end]
                bf_chunk = f["bin_flat"][s_start:s_end]
            for i in range(chunk_n):
                ci = start + i
                ls = int(self.starts[ci]) - s_start
                le = int(self.starts[ci + 1]) - s_start
                if le > ls:
                    out[i] = np.bincount(
                        bf_chunk[ls:le],
                        weights=vals[nn_chunk[ls:le]],
                        minlength=self.n_bins_flat,
                    )[: self.n_bins_flat]
        else:
            for i in range(chunk_n):
                ci = start + i
                ls = int(self.starts[ci])
                le = int(self.starts[ci + 1])
                if le > ls:
                    out[i] = np.bincount(
                        self.bin_flat_flat[ls:le],
                        weights=vals[self.nn_ids_flat[ls:le]],
                        minlength=self.n_bins_flat,
                    )[: self.n_bins_flat]
        return out

    def _accumulate(self, vals):
        vals = np.asarray(vals, dtype=np.float64)
        if len(vals) != len(self.field_xs):
            raise ValueError(
                f"field_vals length {len(vals)} != {len(self.field_xs)}"
            )
        n_c = len(self.center_xs)
        tot = np.zeros((n_c, self.n_bins_flat), dtype=np.float64)
        chunk_iter = self._iter_chunks()
        if self.verbose:
            chunk_iter = tqdm(
                list(chunk_iter), desc="[3DWeb] accumulate", unit="chunk",
            )
        for cs, ce in chunk_iter:
            tot[cs:ce] = self._process_chunk(vals, cs, ce)
        return tot

    def _v_profiles(self, vals):
        return self._accumulate(vals) / self.cnts.clip(1e-2)

    def _w_corr(self, wgts):
        tot = self._accumulate(wgts)
        rho_mean = wgts.sum() / (self.boxsize ** 3)
        denom = (rho_mean * self.dVs_flat).clip(1.0e-12)
        return tot / denom - 1.0

    # ------------------------------------------------------------------ #
    # Bootstrap
    # ------------------------------------------------------------------ #
    def _bootstrap(self, outs, n_bootstrap, bootstrap=True):
        if self.test_mode:
            bootstrap = False
            n_bootstrap = 0
        ws = self.ws
        if (not bootstrap) or n_bootstrap <= 0:
            mean = (outs * ws[:, None]).sum(0)
            return mean, np.zeros_like(mean)

        if n_bootstrap >= 20 and self.n_processes > 1:
            return self._parallel_bootstrap(outs, n_bootstrap, ws)
        return self._serial_bootstrap(outs, n_bootstrap, ws)

    def _serial_bootstrap(self, outs, n_bootstrap, ws):
        means = np.zeros((n_bootstrap, outs.shape[1]), dtype=np.float64)
        it = range(n_bootstrap)
        if self.verbose:
            it = tqdm(it, desc="[3DWeb] bootstrap", unit="it")
        for i in it:
            ids = self._rng.choice(len(outs), len(outs), p=ws)
            means[i] = outs[ids].mean(0)
        return means.mean(0), means.std(0)

    def _parallel_bootstrap(self, outs, n_bootstrap, ws):
        n_per = n_bootstrap // self.n_processes
        rem = n_bootstrap % self.n_processes
        args = [
            (outs, ws, n_per, int(self._seed_rng.integers(0, 2**31)))
            for _ in range(self.n_processes)
        ]
        if rem > 0:
            args.append(
                (outs, ws, rem, int(self._seed_rng.integers(0, 2**31)))
            )
        if self.verbose:
            print(f"[3DWeb] bootstrap: {n_bootstrap} iters / "
                  f"{self.n_processes} procs")
        with Pool(self.n_processes) as pool:
            parts = pool.map(_bootstrap_worker_3d, args)
        all_means = np.vstack(parts)
        return all_means.mean(0), all_means.std(0)

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def _wrap_positions(self, xs):
        xs = np.asarray(xs, dtype=np.float64)
        if not self.periodic:
            return xs
        return np.mod(xs, self.boxsize)

    def __del__(self):
        cf = getattr(self, "cache_file", None)
        if cf is not None and os.path.exists(cf):
            try:
                os.remove(cf)
            except Exception:
                pass