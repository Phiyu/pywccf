"""Subhalos inside host halos."""
from __future__ import annotations

from .tracer import Tracer

__all__ = ["SubhaloSample"]


class SubhaloSample(Tracer):
    """Subhalos (e.g. SUBFIND).

    Required keys: ``position``, ``mass`` = log10(m_sub / h^-1 Msun),
    ``central`` (bool, the main subhalo of its host) and ``host_id`` (int,
    the ``id`` of the host halo).  Host methods: :meth:`centrals`,
    :meth:`satellites`, :meth:`n_per_host`, :meth:`host_index`,
    :meth:`attach_host`.
    """

    required_keys = ("position", "mass", "central", "host_id")
