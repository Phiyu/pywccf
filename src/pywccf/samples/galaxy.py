"""Galaxies (simulated or observed)."""
from __future__ import annotations

from .tracer import HostedTracer

__all__ = ["GalaxySample"]


class GalaxySample(HostedTracer):
    """Galaxies with their host.

    ``host_id`` is the id of the host halo / group (e.g. the group id of a
    group catalogue, or the id of the host subhalo for a semi-analytic
    model); ``is_c`` flags centrals when available, otherwise a galaxy is
    central iff ``galaxy_id == host_id``.  ``mass`` is the stellar mass,
    log10(M_* / h^-1 Msun) by default.
    """

    id_key = "galaxy_id"
    host_key = "host_id"
    central_key = "is_c"
