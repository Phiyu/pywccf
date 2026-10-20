"""Host halos (FoF groups / distinct halos)."""
from __future__ import annotations

import numpy as np

from .tracer import Tracer

__all__ = ["HaloSample"]


class HaloSample(Tracer):
    """Distinct (host) halos.

    Default columns (the ELUCID catalogue of the paper, see
    ``docs/data_format.md``): ``position`` (N, 3) [h^-1 Mpc], ``mass``
    = log10(M_200m / h^-1 Msun), ``R200m``, ``V200``, ``Vmax``,
    ``concentration``, ``lambda``, ``zf``, ``is_backsplash``.
    Any other columns are kept.
    """

    id_key = "halo_id"
    radius_key = "R200m"
    zf_key = "zf"
    backsplash_key = "is_backsplash"

    @property
    def radii(self) -> np.ndarray:
        return np.asarray(self.data[self.radius_key], dtype=np.float64)

    @property
    def formation_redshift(self) -> np.ndarray:
        return np.asarray(self.data[self.zf_key], dtype=np.float64)

    def select_paper(self, *, zf_range=(0.01, 20.0),
                     exclude_backsplash: bool = True) -> "HaloSample":
        """Cuts of the paper: ``zf_lo <= z_f < zf_hi`` and, optionally,
        ``is_backsplash == 0`` (strict flag; unmatched halos, flag -1, are
        removed as well)."""
        out = self
        if zf_range is not None:
            out = out.subset_by_val(self.zf_key, lo=zf_range[0], hi=zf_range[1])
        if exclude_backsplash and self.backsplash_key in self:
            out = out.subset_by_val(self.backsplash_key, eq=False)
        return out

    def mass_bins(self, edges):
        """List of sub-samples with ``lo <= log M < hi``."""
        edges = np.asarray(edges, dtype=np.float64)
        return [self.subset_by_val(self.mass_key, lo=lo, hi=hi)
                for lo, hi in zip(edges[:-1], edges[1:])]

    def count_mass_bins(self, edges) -> np.ndarray:
        m = self.log_masses
        edges = np.asarray(edges, dtype=np.float64)
        return np.array([int(((m >= lo) & (m < hi)).sum())
                         for lo, hi in zip(edges[:-1], edges[1:])])
