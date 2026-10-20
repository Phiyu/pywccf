"""Samples: discrete tracers and gridded fields.

Tracers (``Tracer`` base)            Fields (``Field`` base)
-------------------------            -----------------------
HaloSample                            MatterField   (delta)
SubhaloSample   (host: f_in_grp)      TidalField    (lam, evec, delta; T-web)
GalaxySample    (host: host_id)       GasField      (rho, temperature, ...)

``Tracer.to_field(mesh, scheme='ngp'|'cic'|'tsc', weight='number'|'mass')``
turns a tracer into a :class:`MatterField`; ``Field.to_tracer`` does the
opposite (weighted nodes) for pair-count estimators.
"""
from .tracer import Tracer, HostedTracer
from .halo import HaloSample
from .subhalo import SubhaloSample
from .galaxy import GalaxySample
from .field import Field, MatterField, GasField, gaussian_smooth
from .tidal import WEB_TYPES, WEB_N_LAM, TidalField, tidal_eigensystem

__all__ = ["Tracer", "HostedTracer", "HaloSample", "SubhaloSample",
           "GalaxySample", "Field", "MatterField", "GasField",
           "gaussian_smooth", "WEB_TYPES", "WEB_N_LAM", "TidalField",
           "tidal_eigensystem"]
