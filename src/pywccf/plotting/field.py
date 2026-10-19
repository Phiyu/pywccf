"""Field- and catalogue-level figures of Sections 2 and Appendix B.

* :func:`plot_tidal_field_slice` -- Figure 1, density slice with the
  projected least-compressing eigenvector e_3 (former ``figure.ipynb`` cell).
* :func:`plot_zf_mass` -- the (M_h, z_f) plane of the halo sample.
* :func:`volume_filling_fractions`, :func:`plot_volume_fractions` -- volume
  filling fraction of each web type vs. lambda_th (former
  ``pyccb.utils.check.plot_vff``).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .xi import _paper_figure, style_axis

__all__ = ["plot_tidal_field_slice", "plot_zf_mass",
           "volume_filling_fractions", "plot_volume_fractions"]

MASS_LABEL = r"$\log\,(M_{\rm h}/h^{-1}{\rm M_\odot})$"


def _save(fig, save_path, dpi=300):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {save_path}")


# ---------------------------------------------------------------------------
# Figure 1: density slice + e_3
# ---------------------------------------------------------------------------
@_paper_figure
def plot_tidal_field_slice(field_file, *, box_size=500.0, z_window=6,
                           z_index=None, eigvec=3, n_arrows=32,
                           vmin=-1.0, vmax=10.0, cmap="turbo",
                           arrow_scale=0.15, figsize=(14.2, 7.8),
                           save_path=None, show=True):
    """Smoothed density averaged over ``z_window`` cells around ``z_index``
    (default: the box centre) with the projected eigenvector ``e_{eigvec}``
    of the central slice.

    ``field_file`` must contain ``delta_sm_x`` (Ng^3) and ``evec``
    (Ng^3 x 3 x 3, ``evec[..., :, m]`` = e_{m+1}).  With the defaults and
    Ng = 500 this reproduces Figure 1 of the paper (slice 247 <= z < 253).
    """
    with h5py.File(field_file, "r", locking=False) as f:
        delta = f["delta_sm_x"][:]
        evecs = f["evec"][:]
    if evecs.ndim != 5 or evecs.shape[-2:] != (3, 3):
        raise ValueError(f"evec must have shape (Ng, Ng, Ng, 3, 3), "
                         f"got {evecs.shape}")
    ng = delta.shape[0]
    if delta.shape != (ng, ng, ng) or evecs.shape[:3] != (ng, ng, ng):
        raise ValueError(f"inconsistent shapes {delta.shape}, {evecs.shape}")

    z_idx = ng // 2 if z_index is None else int(z_index)
    z_start = max(0, z_idx - z_window // 2)
    z_stop = min(ng, z_start + z_window)
    z_start = max(0, z_stop - z_window)

    density_slice = delta[:, :, z_start:z_stop].mean(axis=2).T
    e_slice = evecs[:, :, z_idx, :, eigvec - 1].T
    step = max(1, ng // n_arrows)
    grid = np.arange(0, ng, step)
    gx, gy = np.meshgrid(grid, grid)
    u = e_slice[0][np.ix_(grid, grid)]
    v = e_slice[1][np.ix_(grid, grid)]

    extent = [0.0, box_size, 0.0, box_size]
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(density_slice, origin="lower", cmap=cmap, extent=extent,
                   interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.quiver(gx / (ng - 1) * box_size, gy / (ng - 1) * box_size, u, v,
              color="white", angles="xy", scale_units="xy",
              scale=arrow_scale, width=0.0050, headwidth=3.5,
              headlength=5.5, headaxislength=3.5, alpha=0.92)
    ax.set_xlim(0.0, box_size)
    ax.set_ylim(0.0, box_size)
    ax.set_xlabel(r"$x\,[h^{-1}{\rm Mpc}]$")
    ax.set_ylabel(r"$y\,[h^{-1}{\rm Mpc}]$")
    style_axis(ax)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\delta_{\rm s}$")
    _save(fig, save_path, dpi=240)
    print(f"Used z slices {z_start}:{z_stop - 1}")
    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# z_f -- M_h plane
# ---------------------------------------------------------------------------
def _median_in_bins(x, y, edges, min_count=5):
    out = np.full(len(edges) - 1, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (x >= lo) & (x < hi)
        if m.sum() > min_count:
            out[i] = np.nanmedian(y[m])
    return out


@_paper_figure
def plot_zf_mass(halo, *, mass_range=(11.5, 14.5), n_mass=20, bins=60,
                 zf_quantile=0.995, cmap="viridis", figsize=(7, 5),
                 save_path=None, show=True):
    """2D histogram of (log M_h, z_f) with the running median.

    ``halo`` is a halo HDF5 file (datasets ``mass``, ``zf``,
    ``is_backsplash``; backsplash halos are removed) or a
    :class:`~pywccf.samples.HaloSample` that is already cut.
    Returns ``(fig, ax, stats)`` with the median z_f per mass bin.
    """
    if isinstance(halo, (str, Path)):
        with h5py.File(halo, "r") as f:
            lgM = f["mass"][:]
            zf = f["zf"][:]
            bs = f["is_backsplash"][:]
        sel = np.isfinite(lgM) & np.isfinite(zf) & (bs != 1)
    else:
        lgM = np.asarray(halo["mass"])
        zf = np.asarray(halo["zf"])
        sel = np.isfinite(lgM) & np.isfinite(zf)
    print(f"N usable: {sel.sum():,} / {len(lgM):,}")
    lgM, zf = lgM[sel], zf[sel]

    m_lo, m_hi = mass_range
    m_edges = np.linspace(m_lo, m_hi, n_mass + 1)
    m_cen = 0.5 * (m_edges[:-1] + m_edges[1:])
    zf_hi = np.quantile(zf, zf_quantile)

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.hist2d(lgM, zf, bins=[bins, bins], range=[[m_lo, m_hi], [0., zf_hi]],
                  cmap=cmap, cmin=1, norm="log")
    fig.colorbar(h[3], ax=ax, label=r"$N_{\rm halo}$")
    med = _median_in_bins(lgM, zf, m_edges)
    ax.plot(m_cen, med, "r-", lw=2.0, label="median")
    ax.set_xlabel(MASS_LABEL)
    ax.set_ylabel(r"$z_{\rm f}$")
    ax.legend()
    style_axis(ax)
    fig.tight_layout()
    _save(fig, save_path)
    print(f"median z_f = {np.nanmedian(zf):.3f}")
    if show:
        plt.show()
    return fig, ax, {"mass_centers": m_cen, "zf_median": med}


# ---------------------------------------------------------------------------
# Appendix B: volume filling fractions
# ---------------------------------------------------------------------------
def volume_filling_fractions(lams, thresholds):
    """Fraction of cells with 3/2/1/0 eigenvalues ``> lambda_th``.

    ``lams`` has shape (..., 3).  Returns a dict web -> array over
    ``thresholds``.
    """
    lams = np.asarray(lams)
    total = np.prod(lams.shape[:-1])
    out = {"knot": [], "filament": [], "sheet": [], "void": []}
    for l_th in np.atleast_1d(thresholds):
        n_above = (lams > l_th).sum(axis=-1)
        for web, n in (("knot", 3), ("filament", 2), ("sheet", 1),
                       ("void", 0)):
            out[web].append(np.sum(n_above == n) / total)
    return {k: np.array(v) for k, v in out.items()}


@_paper_figure
def plot_volume_fractions(field, *, thresholds=None, fiducial=0.0,
                          figsize=(8, 6), save_path=None, show=True):
    """Volume filling fraction of each web type vs. lambda_th.

    ``field`` is a :class:`~pywccf.samples.TidalField` (``.lams``) or an
    eigenvalue array of shape (..., 3).
    """
    if thresholds is None:
        thresholds = np.linspace(-1.0, 1.0, 200)
    lams = getattr(field, "lams", field)
    vff = volume_filling_fractions(lams, thresholds)

    fig, ax = plt.subplots(figsize=figsize)
    for web, ls in (("void", "-"), ("sheet", "--"), ("filament", ":"),
                    ("knot", "-.")):
        ax.plot(thresholds, vff[web], color="black", linestyle=ls, lw=2,
                label=web)
    if fiducial is not None:
        ax.axvline(fiducial, color="0.5", lw=1.0, ls="--")
    ax.set_xlabel(r"$\lambda_{\rm th}$")
    ax.set_ylabel("Volume filling fraction")
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(0.0, 1.0)
    style_axis(ax)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig, ax, vff
