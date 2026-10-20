"""Galaxies (simulated or observed)."""
from __future__ import annotations

from .tracer import Tracer

__all__ = ["GalaxySample"]


class GalaxySample(Tracer):
    """Galaxies.

    Required keys: ``position``, ``mass`` = log10(M_* / h^-1 Msun),
    ``central`` (bool) and ``host_id`` (int, the ``id`` of the host halo or
    group).  Same host methods as :class:`SubhaloSample`.
    """

    required_keys = ("position", "mass", "central", "host_id")
