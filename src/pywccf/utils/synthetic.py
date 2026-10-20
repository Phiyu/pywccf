"""Synthetic boxes for tests and examples (no simulation needed).

A Gaussian random field with a power-law spectrum is turned into a
log-normal density; tracers are Poisson-sampled from ``(1 + delta)^b`` and
given a mass and a formation redshift that correlates with the local
density, so that all statistics of the package have a non-trivial signal.
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import Mesh
from ..samples.field import MatterField
from ..samples.halo import HaloSample

__all__ = ["gaussian_random_field", "lognormal_delta", "poisson_points",
           "synthetic_box"]


def gaussian_random_field(mesh: Mesh, *, slope: float = -2.0,
                          k_min: float | None = None, sigma: float = 1.0,
                          seed: int = 0) -> np.ndarray:
    """Zero-mean Gaussian field with ``P(k) ~ k^slope``, rescaled to
    standard deviation ``sigma``."""
    rng = np.random.default_rng(seed)
    k = np.sqrt(mesh.k2())
    kf = 2.0 * np.pi / mesh.l_box
    k_min = kf if k_min is None else k_min
    amp = np.where(k > 0, np.maximum(k, k_min) ** (0.5 * slope), 0.0)
    noise = np.fft.rfftn(rng.standard_normal(mesh.shape))
    g = np.fft.irfftn(noise * amp, s=mesh.shape, axes=(0, 1, 2))
    return sigma * (g - g.mean()) / g.std()


def lognormal_delta(mesh: Mesh, *, sigma: float = 1.0, seed: int = 0,
                    **kw) -> np.ndarray:
    g = gaussian_random_field(mesh, sigma=sigma, seed=seed, **kw)
    rho = np.exp(g - 0.5 * sigma ** 2)
    return rho / rho.mean() - 1.0


def poisson_points(delta: np.ndarray, mesh: Mesh, n_mean: float, *,
                   bias_power: float = 1.0, seed: int = 0) -> np.ndarray:
    """Poisson points with rate ``proportional to (1 + delta)^bias_power``,
    uniformly placed inside the cell around each node."""
    rng = np.random.default_rng(seed)
    rate = np.clip(1.0 + delta, 0.0, None) ** bias_power
    lam = n_mean * rate / rate.sum()
    counts = rng.poisson(lam.ravel())
    idx = np.repeat(np.arange(mesh.n_cells), counts)
    xs = mesh.unravel(idx).astype(np.float64) * mesh.l_grid
    xs += (rng.random((len(idx), 3)) - 0.5) * mesh.l_grid
    return mesh.wrap(xs)


def synthetic_box(n_grids: int = 64, l_box: float = 128.0, *,
                  n_halos: int = 20_000, R_s: float = 2.0,
                  lam_th: float = 0.0, sigma: float = 1.0, seed: int = 0,
                  verbose: bool = False):
    """``(matter, tidal, halos)`` of a synthetic periodic box.

    halos: ``position``, ``mass`` (log10, 11.5--14), ``zf`` (higher in
    denser regions at fixed mass), ``is_backsplash`` (0), ``halo_id``.
    """
    mesh = Mesh(n_grids, l_box)
    delta = lognormal_delta(mesh, sigma=sigma, seed=seed)
    matter = MatterField({"delta": delta}, mesh, verbose=verbose)
    tidal = matter.tidal_field(R_s, lam_th=lam_th)
    tidal.verbose = verbose
    rng = np.random.default_rng(seed + 1)
    xs = poisson_points(delta, mesh, n_halos, bias_power=1.5, seed=seed + 2)
    n = len(xs)
    mass = 11.5 + rng.exponential(0.6, n).clip(0, 2.5)
    d_s = tidal.values_at(xs, "delta", scheme="cic")
    zf = np.clip(1.6 - 0.25 * (mass - 11.5) + 0.3 * np.tanh(d_s)
                 + 0.4 * rng.standard_normal(n), 0.02, 6.0)
    halos = HaloSample({"position": xs, "mass": mass, "zf": zf,
                        "is_backsplash": np.zeros(n, dtype=np.int8),
                        "halo_id": np.arange(n)},
                       l_box=l_box, verbose=verbose,
                       meta={"synthetic": True, "seed": seed})
    return matter, tidal, halos
