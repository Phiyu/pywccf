"""Sample selection helpers for assembly-bias measurements."""
from __future__ import annotations

import numpy as np

__all__ = ["tail_masks", "mass_match_weights"]


def tail_masks(tracer, key: str = "zf", p: float = 0.2, *,
               bin_key: str | None = "mass", bin_edges=None):
    """Lower / upper ``p`` tails of ``key`` inside each bin of ``bin_key``
    (wrapper of :meth:`Tracer.quantile_masks`); with the paper defaults the
    upper tail in ``zf`` is the early-forming (old) sample."""
    if bin_edges is None:
        bin_key = None
    return tracer.quantile_masks(key, p, bin_key=bin_key, bin_edges=bin_edges)


def mass_match_weights(target_mass, tail_masses, *, sub_bins: int = 10,
                       max_weight: float | None = 10.0):
    """Weights giving every tail the mass distribution of ``target_mass``.

    The target range is cut into ``sub_bins`` equal-count sub-bins; only
    sub-bins populated by every tail can be matched.  A member of a tail in
    sub-bin j gets ``w = P_target(j) / P_tail(j)``, normalised to mean 1 and
    optionally capped at ``max_weight`` (members of unmatched sub-bins get
    0).  Returns one ``(weights, diagnostics)`` pair per tail.
    """
    target_mass = np.asarray(target_mass, dtype=np.float64)
    tails = [np.asarray(m, dtype=np.float64) for m in tail_masses]
    if len(target_mass) == 0 or any(len(m) == 0 for m in tails):
        return [(np.ones(len(m)), {"n_eff": float(len(m))}) for m in tails]
    inner = np.unique(np.quantile(target_mass,
                                  np.linspace(0.0, 1.0, sub_bins + 1)[1:-1]))
    n_sub = len(inner) + 1
    t_count = np.bincount(np.searchsorted(inner, target_mass, side="right"),
                          minlength=n_sub).astype(np.float64)
    subs = [np.searchsorted(inner, m, side="right") for m in tails]
    fracs = [np.bincount(s, minlength=n_sub) / len(s) for s in subs]
    covered = np.logical_and.reduce([f > 0 for f in fracs])
    coverage = float(t_count[covered].sum() / t_count.sum())
    t_frac = np.where(covered, t_count, 0.0)
    t_frac /= t_frac.sum()
    out = []
    for m, s, f in zip(tails, subs, fracs):
        ratio = np.zeros(n_sub)
        ratio[covered] = t_frac[covered] / f[covered]
        w = ratio[s]
        w /= w.mean()
        n_capped = 0
        if max_weight is not None:
            for _ in range(20):
                over = w > max_weight * (1.0 + 1e-12)
                if not over.any():
                    break
                n_capped = max(n_capped, int(over.sum()))
                w = np.minimum(w, max_weight)
                w /= w.mean()
        n_eff = float(w.sum() ** 2 / (w ** 2).sum())
        out.append((w, {"n_eff": n_eff, "n_eff_frac": n_eff / len(m),
                        "coverage": coverage, "n_capped": n_capped,
                        "n_zero": int((w == 0).sum()),
                        "mean_mass_raw": float(m.mean()),
                        "mean_mass_weighted": float(np.average(m, weights=w))}))
    return out
