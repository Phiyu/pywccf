# halo_web_numba.py
"""Numba kernel for ``halo_web_fast.HaloWeb3DCross`` (backend='numba').

One compiled pass per web type: for every web-cell centre, loop over the
halo cells that cover the centre's rotated search box (the cube
|r_i| <= e_i[-1] in its own tidal frame), project each halo on the centre's
eigenvectors, bin it, and add  w_centre * w_halo  to the accumulator of the
centre's jackknife region for every halo weighting the halo belongs to.
No pair is stored; centres are processed in parallel (one task = a block of
centres of a single region, so tasks never write to the same row).

The search, the projection (dx @ triad), the periodic wrap and the binning
follow halo_web.HaloWeb3DCross pair for pair.
"""
from __future__ import annotations

import numpy as np
import numba as nb


@nb.njit(cache=True)
def _bin_right(edges, r, n):
    """np.searchsorted(edges, r, side='right'), clipped to n."""
    lo, hi = 0, edges.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if r < edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo if lo < n else n


@nb.njit(parallel=True, cache=True)
def region_sums(task_lo, task_hi, order, cx, cev, cw,
                hx, cell_ptr, n_side, cell_size, box, periodic,
                e1, e2, e3, h_ptr, h_v, h_w, n_v, out):
    """out[t, v, bin] += sum over the pairs of task t of w_c * w_h(v)."""
    n1 = e1.shape[0] - 1
    n2 = e2.shape[0] - 1
    n3 = e3.shape[0] - 1
    s3 = n3 + 1
    s23 = (n2 + 1) * s3
    a1 = e1[n1]
    a2 = e2[n2]
    a3 = e3[n3]
    half = 0.5 * box
    inv = 1.0 / cell_size
    for t in nb.prange(task_lo.shape[0]):
        acc = out[t]
        for p in range(task_lo[t], task_hi[t]):
            g = order[p]
            c0 = cx[g, 0]
            c1 = cx[g, 1]
            c2 = cx[g, 2]
            E = cev[g]
            wg = cw[g]
            # axis-aligned half extents of the rotated search box
            h0 = a1 * abs(E[0, 0]) + a2 * abs(E[0, 1]) + a3 * abs(E[0, 2])
            h1 = a1 * abs(E[1, 0]) + a2 * abs(E[1, 1]) + a3 * abs(E[1, 2])
            h2 = a1 * abs(E[2, 0]) + a2 * abs(E[2, 1]) + a3 * abs(E[2, 2])
            lo0 = int(np.floor((c0 - h0) * inv))
            hi0 = int(np.floor((c0 + h0) * inv))
            lo1 = int(np.floor((c1 - h1) * inv))
            hi1 = int(np.floor((c1 + h1) * inv))
            lo2 = int(np.floor((c2 - h2) * inv))
            hi2 = int(np.floor((c2 + h2) * inv))
            for i0 in range(lo0, hi0 + 1):
                if periodic:
                    k0 = i0 % n_side
                elif i0 < 0 or i0 >= n_side:
                    continue
                else:
                    k0 = i0
                for i1 in range(lo1, hi1 + 1):
                    if periodic:
                        k1 = i1 % n_side
                    elif i1 < 0 or i1 >= n_side:
                        continue
                    else:
                        k1 = i1
                    base = (k0 * n_side + k1) * n_side
                    for i2 in range(lo2, hi2 + 1):
                        if periodic:
                            k2 = i2 % n_side
                        elif i2 < 0 or i2 >= n_side:
                            continue
                        else:
                            k2 = i2
                        cell = base + k2
                        for q in range(cell_ptr[cell], cell_ptr[cell + 1]):
                            d0 = hx[q, 0] - c0
                            d1 = hx[q, 1] - c1
                            d2 = hx[q, 2] - c2
                            if periodic:
                                if d0 > half:
                                    d0 -= box
                                elif d0 < -half:
                                    d0 += box
                                if d1 > half:
                                    d1 -= box
                                elif d1 < -half:
                                    d1 += box
                                if d2 > half:
                                    d2 -= box
                                elif d2 < -half:
                                    d2 += box
                            r1 = abs(d0 * E[0, 0] + d1 * E[1, 0] + d2 * E[2, 0])
                            if r1 > a1:
                                continue
                            r2 = abs(d0 * E[0, 1] + d1 * E[1, 1] + d2 * E[2, 1])
                            if r2 > a2:
                                continue
                            r3 = abs(d0 * E[0, 2] + d1 * E[1, 2] + d2 * E[2, 2])
                            if r3 > a3:
                                continue
                            if d0 * d0 + d1 * d1 + d2 * d2 <= 1e-24:
                                continue
                            bf = (_bin_right(e1, r1, n1) * s23
                                  + _bin_right(e2, r2, n2) * s3
                                  + _bin_right(e3, r3, n3))
                            for e in range(h_ptr[q], h_ptr[q + 1]):
                                acc[h_v[e], bf] += wg * h_w[e]
    return out


class CellList:
    """Haloes sorted into a periodic grid of cubic cells."""

    def __init__(self, xs, box, search_half_max, cell_size=None,
                 periodic=True):
        xs = np.asarray(xs, dtype=np.float64)
        if cell_size is None:
            cell_size = max(search_half_max / 4.0, box / 512.0)
        n_side = max(1, int(np.floor(box / cell_size)))
        cell_size = box / n_side
        # a search box must never wrap onto itself
        if periodic and 2.0 * search_half_max + 2.0 * cell_size >= box:
            raise ValueError("box too small for the cell list")
        idx = np.clip((xs / cell_size).astype(np.int64), 0, n_side - 1)
        cell = (idx[:, 0] * n_side + idx[:, 1]) * n_side + idx[:, 2]
        self.order = np.argsort(cell, kind="stable")
        self.xs = np.ascontiguousarray(xs[self.order])
        counts = np.bincount(cell, minlength=n_side ** 3)
        self.cell_ptr = np.zeros(n_side ** 3 + 1, dtype=np.int64)
        np.cumsum(counts, out=self.cell_ptr[1:])
        self.n_side, self.cell_size = n_side, float(cell_size)

    def weight_csr(self, V):
        """Per sorted halo: the weightings it belongs to (index, weight)."""
        W = np.asarray(V, dtype=np.float64)[:, self.order]      # (n_v, n_h)
        v_idx, h_idx = np.nonzero(W)
        # group by halo
        o = np.argsort(h_idx, kind="stable")
        h_idx, v_idx = h_idx[o], v_idx[o]
        h_ptr = np.zeros(W.shape[1] + 1, dtype=np.int64)
        np.cumsum(np.bincount(h_idx, minlength=W.shape[1]), out=h_ptr[1:])
        return h_ptr, v_idx.astype(np.int32), W[v_idx, h_idx]


def make_tasks(reg, n_reg, n_threads, min_block=64):
    """Blocks of centres, each inside one region (no write conflicts)."""
    order = np.argsort(reg, kind="stable")
    counts = np.bincount(reg, minlength=n_reg)
    block = max(min_block, int(np.ceil(len(reg) / max(1, 4 * n_threads))))
    lo, hi, rr = [], [], []
    start = 0
    for r in range(n_reg):
        n = counts[r]
        for b in range(start, start + n, block):
            lo.append(b)
            hi.append(min(b + block, start + n))
            rr.append(r)
        start += n
    return (order, np.asarray(lo, np.int64), np.asarray(hi, np.int64),
            np.asarray(rr, np.int64))
