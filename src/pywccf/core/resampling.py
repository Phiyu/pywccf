"""Error estimation for estimators that are weighted means over centres.

All WCCF estimators here have the form

    xi = F( sum_g w_g X_g / sum_g w_g ),

with ``X_g`` the per-centre pair counts and ``F`` a fixed affine map.  The
helpers below turn per-region or per-resample sums of ``w_g X_g`` into the
central value and its uncertainty.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ERROR_MODES", "check_error_mode", "jackknife",
           "bootstrap_matrix", "summarize_samples"]

ERROR_MODES = ("none", "jackknife", "bootstrap")


def check_error_mode(error: str) -> str:
    e = "none" if error is None else str(error).lower()
    if e not in ERROR_MODES:
        raise ValueError(f"error must be one of {ERROR_MODES}, got {error!r}")
    return e


def jackknife(S_k: np.ndarray, W_k: np.ndarray, transform):
    """Delete-one jackknife from region sums.

    S_k : (n_reg, ...) sums of ``w_g X_g`` over the centres of region k.
    W_k : (n_reg,) sums of ``w_g`` over region k.
    transform : maps the weighted mean ``S / W`` to the statistic.

    Returns ``(full, sd, samples)`` where ``full`` uses all centres,
    ``samples`` the delete-one estimates of the regions that contain
    centres (regions without centres, or holding all of them, are skipped).
    """
    S_k = np.asarray(S_k, dtype=np.float64)
    W_k = np.asarray(W_k, dtype=np.float64)
    S, W = S_k.sum(0), W_k.sum()
    full = transform(S / W)
    use = (W_k > 0) & (W - W_k > 0)
    n = int(use.sum())
    if n < 2:
        return full, np.zeros_like(full), np.empty((0,) + full.shape)
    shape = (n,) + (1,) * (S_k.ndim - 1)
    samples = transform((S[None] - S_k[use]) / (W - W_k[use]).reshape(shape))
    sd = np.sqrt((n - 1.0) / n * ((samples - samples.mean(0)) ** 2).sum(0))
    return full, sd, samples


def bootstrap_matrix(probs: np.ndarray, n_bootstrap: int,
                     seed: int | None = 0) -> np.ndarray:
    """Weighted bootstrap of centres as a dense coefficient matrix.

    Each resample draws ``N`` centres with replacement with probabilities
    ``probs`` (normalised weights); row ``b`` holds the multiplicity of
    every centre divided by ``N``, so that ``A @ X`` is the (unweighted)
    mean of ``X`` over the resample, as in the weighted cell bootstrap of
    the paper (Section 4.2).
    """
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    n = len(p)
    rng = np.random.default_rng(seed)
    A = np.empty((int(n_bootstrap), n), dtype=np.float64)
    for b in range(int(n_bootstrap)):
        A[b] = np.bincount(rng.choice(n, n, p=p), minlength=n) / n
    return A


def summarize_samples(samples: np.ndarray):
    """Standard deviation and 16/84 percentiles of resampled statistics."""
    samples = np.asarray(samples)
    if len(samples) == 0:
        z = np.full(samples.shape[1:], np.nan)
        return {"sd": z, "p16": z, "p84": z}
    return {"sd": samples.std(0),
            "p16": np.percentile(samples, 16, axis=0),
            "p84": np.percentile(samples, 84, axis=0)}
