"""Subhalos inside host halos."""
from __future__ import annotations

import numpy as np

from .tracer import HostedTracer

__all__ = ["SubhaloSample"]


class SubhaloSample(HostedTracer):
    """SUBFIND subhalos.

    Follows the tree conventions used in CosmicCoreBox / pyhipp_sims:
    ``subhalo_id`` is the id of the subhalo and ``f_in_grp`` the id of the
    first (most massive, central) subhalo of its FoF group, which serves as
    the host id: a subhalo is central iff ``f_in_grp == subhalo_id``.
    ``mass`` is log10(m_sub / h^-1 Msun) by default.
    """

    id_key = "subhalo_id"
    host_key = "f_in_grp"
    central_key = "is_c"

    def host_halo_index(self, halos, host_id_key: str = "subhalo_id") -> np.ndarray:
        """Row of each subhalo's host in a halo table whose ``host_id_key``
        column stores the id of the group's central subhalo."""
        return self.host_index(halos, host_id_key)
