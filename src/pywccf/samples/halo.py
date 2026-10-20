"""Host halos (FoF groups / distinct halos)."""
from __future__ import annotations

import numpy as np

from .tracer import Tracer

__all__ = ["HaloSample"]


class HaloSample(Tracer):
    """Distinct (host) halos.

    Required keys: ``position``, ``mass`` = log10(M_h / h^-1 Msun).
    Optional keys used by the methods below: ``id`` (matched by the
    ``host_id`` of subhalos / galaxies), ``zf`` (formation redshift),
    ``is_backsplash`` (strict backsplash flag: 1 yes, 0 no, -1 unknown).
    """

    required_keys = ("position", "mass")

    @property
    def formation_redshift(self) -> np.ndarray:
        return self._need("zf").astype(np.float64)

    def select_paper(self, *, zf_range=(0.01, 20.0),
                     exclude_backsplash: bool = True) -> "HaloSample":
        """Cuts of the paper: ``zf_lo <= z_f < zf_hi`` and, optionally,
        ``is_backsplash == 0`` (halos with flag -1 are removed as well)."""
        out = self
        if zf_range is not None:
            out = out.subset_by_val("zf", lo=zf_range[0], hi=zf_range[1])
        if exclude_backsplash and "is_backsplash" in self:
            out = out.subset_by_val("is_backsplash", eq=False)
        return out
