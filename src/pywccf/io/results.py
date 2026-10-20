"""Save / load :class:`~pywccf.wccf.WCCFResult` collections.

Layout: ``/<web>/<name>/{xi, xi_sd, samples, edges | e1 e2 e3, meta/...}``
plus string datasets ``web``, ``name``, ``method``, ``error`` and ``ndim``;
an optional ``/run`` group stores free-form run information.
"""
from __future__ import annotations

from typing import Mapping

import h5py

from .hdf5 import dump_dict, load_dict

__all__ = ["save_results", "load_results"]


def _flatten(results):
    """Accept a WCCFResult, {name: res} or {web: {name: res}}."""
    from ..wccf.result import WCCFResult
    if isinstance(results, WCCFResult):
        return {results.web: {results.name: results}}
    out = {}
    for k, v in results.items():
        if isinstance(v, WCCFResult):
            out.setdefault(v.web, {})[v.name] = v
        else:
            for name, r in v.items():
                out.setdefault(r.web, {})[name] = r
    return out


def save_results(path, results, *, run_info: Mapping | None = None,
                 mode: str = "w", compression: str | None = "gzip"):
    """Write results to HDF5 (``mode='a'`` adds to an existing file)."""
    with h5py.File(path, mode) as f:
        for web, by_name in _flatten(results).items():
            g = f.require_group(web)
            for name, r in by_name.items():
                if name in g:
                    del g[name]
                dump_dict(g.create_group(name), r.to_dict(),
                          compression=compression)
        if run_info:
            if "run" in f:
                del f["run"]
            dump_dict(f.create_group("run"), run_info)
    return path


def load_results(path, webs=None) -> dict:
    """``{web: {name: WCCFResult}}`` (and ``'run'`` if stored)."""
    from ..wccf.result import WCCFResult
    out = {}
    with h5py.File(path, "r", locking=False) as f:
        for web in (webs or [k for k in f.keys() if k != "run"]):
            out[web] = {name: WCCFResult.from_dict(load_dict(f[web][name]))
                        for name in f[web].keys()}
        if "run" in f and webs is None:
            out["run"] = load_dict(f["run"])
    return out
