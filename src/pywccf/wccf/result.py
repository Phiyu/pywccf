"""Container of one measured WCCF."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.bins import FrameBins, RadialBins

__all__ = ["WCCFResult"]


@dataclass
class WCCFResult:
    """One correlation function (one web type, one tracer sample).

    Attributes
    ----------
    xi : (n,) for 1D or (n1, n2, n3) for 3D; the deterministic estimator.
    xi_sd : uncertainty of the chosen error mode (zeros for 'none').
    samples : jackknife delete-one or bootstrap estimates (or None).
    bins : the :class:`RadialBins` / :class:`FrameBins`.
    web, name : web type and sample name.
    method : 'ph' (pair counts) or 'fft' (convolution).
    error : 'none', 'jackknife' or 'bootstrap'.
    meta : numbers of cells / tracers, resampling settings, ...
    """

    xi: np.ndarray
    xi_sd: np.ndarray
    bins: RadialBins | FrameBins
    web: str
    name: str = "all"
    method: str = "ph"
    error: str = "none"
    samples: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ndim(self) -> int:
        return self.bins.ndim

    @property
    def r(self) -> np.ndarray:
        """Bin centres (1D)."""
        if self.ndim != 1:
            raise AttributeError("r is defined for 1D results; use r_axes")
        return self.bins.centers

    @property
    def r_axes(self):
        """Bin centres along e_1, e_2, e_3 (3D)."""
        if self.ndim != 3:
            raise AttributeError("r_axes is defined for 3D results; use r")
        return self.bins.centers

    def profile(self, axis: int, fixed_index: int = 0):
        """Directional slice ``xi_3D(r_axis)`` along ``e_{axis}`` (1, 2, 3)
        with the other two coordinates in bin ``fixed_index`` (the
        innermost bin by default, as in the paper).

        Returns ``(r, xi, xi_sd)``.
        """
        if self.ndim != 3:
            raise ValueError("profile() needs a 3D result")
        a = int(axis) - 1
        if a not in (0, 1, 2):
            raise ValueError("axis must be 1, 2 or 3")
        sl = [fixed_index] * 3
        sl[a] = slice(None)
        sl = tuple(sl)
        return self.bins.centers[a], self.xi[sl], self.xi_sd[sl]

    def plane(self, axes=(1, 2), fixed_index: int = 0):
        """2D section in the plane of ``axes`` (third coordinate fixed)."""
        a, b = (int(x) - 1 for x in axes)
        c = ({0, 1, 2} - {a, b}).pop()
        sl = [slice(None)] * 3
        sl[c] = fixed_index
        xi = self.xi[tuple(sl)]
        return (self.bins.centers[a], self.bins.centers[b],
                xi if a < b else xi.T)

    # -- dict round trip (used by pywccf.io.results) ------------------------
    def to_dict(self) -> dict:
        d = {"xi": self.xi, "xi_sd": self.xi_sd, "samples": self.samples,
             "web": self.web, "name": self.name, "method": self.method,
             "error": self.error, "ndim": self.ndim}
        if self.ndim == 1:
            d["edges"] = self.bins.edges
        else:
            d["e1"], d["e2"], d["e3"] = self.bins.axes
        d["meta"] = {k: v for k, v in self.meta.items() if v is not None}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WCCFResult":
        bins = (RadialBins(d["edges"]) if int(d["ndim"]) == 1
                else FrameBins(d["e1"], d["e2"], d["e3"]))
        return cls(xi=np.asarray(d["xi"]), xi_sd=np.asarray(d["xi_sd"]),
                   bins=bins, web=str(d["web"]), name=str(d["name"]),
                   method=str(d["method"]), error=str(d["error"]),
                   samples=(None if d.get("samples") is None
                            else np.asarray(d["samples"])),
                   meta=dict(d.get("meta", {})))

    def __repr__(self) -> str:
        return (f"WCCFResult({self.ndim}D, web={self.web!r}, "
                f"name={self.name!r}, method={self.method!r}, "
                f"error={self.error!r}, shape={self.xi.shape})")
