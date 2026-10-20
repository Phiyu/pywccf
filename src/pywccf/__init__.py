"""pywccf -- cosmic-web cross-correlation functions in the tidal eigenframe.

Companion code of Yu, Wang et al., "Halo formation and distribution around
cosmic web" (in prep.).

>>> from pywccf import TidalField, HaloSample, WCCF3D
>>> tf = TidalField.from_file('field_099.2.hdf5')
>>> halos = HaloSample.from_file('halo_0701.hdf5', l_box=tf.l_box)
>>> res = WCCF3D(tf, halos).compute('filament', error='jackknife')
>>> r, xi, sd = res['all'].profile(3)          # xi_3D along e_3

Sub-packages: ``core`` (Log, Mesh, assignment, bins, resampling),
``samples`` (tracers and fields), ``wccf`` (estimators), ``io``, ``utils``.
"""

__version__ = "0.2.0"

from . import core, samples, wccf, io, utils
from .core import Mesh, RadialBins, FrameBins, PAPER_EDGES
from .samples import (Tracer, HaloSample, SubhaloSample, GalaxySample, Field,
                      MatterField, TidalField, GasField, WEB_TYPES)
from .wccf import WCCF1D, WCCF3D, WCCFResult
from .io import save_results, load_results

__all__ = ["core", "samples", "wccf", "io", "utils", "Mesh", "RadialBins",
           "FrameBins", "PAPER_EDGES", "Tracer", "HaloSample",
           "SubhaloSample", "GalaxySample", "Field", "MatterField",
           "TidalField", "GasField", "WEB_TYPES", "WCCF1D", "WCCF3D",
           "WCCFResult", "save_results", "load_results", "__version__"]
