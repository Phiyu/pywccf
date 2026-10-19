"""pywccf -- the three-dimensional cosmic-web cross-correlation function.

Companion code of Yu & Wang, "Anisotropic Cosmic Web Cross-Correlation in
Tidal Eigenspace I: Cosmic Web-Halo Correlation and Assembly Bias".

Sub-packages
------------
samples      halo catalogue and tidal-field containers (from CosmicCoreBox)
estimator    3D-WCCF estimators (reference / fast / numba engines)
pipeline     YAML-driven runner that writes one combined HDF5 file
io           readers for the combined HDF5 files
ratio        scale-averaged amplitudes and the amplitude ratio (AR)
geometry     directional walk: frame coherence and web-type transitions
environment  halo mass function / formation time vs density and web type
plotting     the figures of the paper

Heavy modules (matplotlib, numba) are only imported when the corresponding
sub-package is imported.
"""

__version__ = "0.1.0"

from . import samples
from .samples import FieldSample, HaloSample, TidalField

__all__ = ["samples", "TidalField", "FieldSample", "HaloSample",
           "__version__"]
