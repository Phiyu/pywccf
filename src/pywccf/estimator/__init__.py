"""3D-WCCF estimators.

``halo_web``       reference implementation (engine ``legacy``; bootstrap
                   errors).  Used for the fiducial measurements of the paper.
``halo_web_fast``  drop-in replacement (engine ``fast``): one neighbour search
                   per web type shared by all samples; jackknife or bootstrap.
``halo_web_numba`` compiled pair-counting kernel used by ``halo_web_fast``.

The isotropic 1D-WCCF estimator lives in :mod:`pywccf.pipeline.runner`
(``HaloWeb1DCross``) and is re-exported here.
"""
from .halo_web import Edges, HaloWeb3DCross
from .halo_web_fast import HaloWeb3DCross as HaloWeb3DCrossFast
from ..pipeline.runner import HaloWeb1DCross

__all__ = ["Edges", "HaloWeb3DCross", "HaloWeb3DCrossFast", "HaloWeb1DCross"]
