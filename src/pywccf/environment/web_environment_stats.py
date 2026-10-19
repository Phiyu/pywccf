"""
Environment dependence of the density field and of the halo mass function.

Part of ``pywccf`` (split out of the analysis scripts of the 3D-WCCF paper): it only
uses the public data model of ``pywccf.samples.field.TidalField`` and
``pywccf.samples.halo.HaloSample``.

The two headline functions reproduce, for the T-web classification used in the
3D-WCCF paper, the two control plots of Hahn, Porciani, Carollo & Dekel
(2007, MNRAS 375, 489):

* :func:`density_pdf_by_web` -- their Fig. 3.  Volume-weighted probability
  distribution of the local density :math:`1+\\delta` separately for voids,
  sheets, filaments and knots.  Optionally the same distribution *weighted by
  haloes* instead of by volume, which is what one needs to build a
  density-matched (re-weighted) halo sample.

* :func:`hmf_by_web` -- their Fig. 4.  Halo mass function in each web type,
  normalised by the volume actually occupied by that web type (Hahn et al.
  phrase this as "abundances in the whole box rescaled by the corresponding
  volume fractions"), together with the total mass function and with the
  fraction of haloes of a given mass residing in each environment.

Conventions
-----------
* Web type index == ``tf.n_lams`` == number of tidal eigenvalues above
  ``tf.lam_off``:  0 = void, 1 = sheet, 2 = filament, 3 = knot.  This matches
  ``pyccb.process.halo_web`` and ``pywccf.geometry``.
* Only cells with ``tf.recon_mask`` are used, and only haloes that fall in a
  reconstructed cell.  All volumes are therefore *effective* volumes,
  ``N_cell * l_grid**3``, not ``l_box**3``.
* The halo mass field is assumed to be **already logarithmic**,
  :math:`\\log_{10}(M_{\\rm h}\\,/\\,h^{-1}M_\\odot)`.  Pass ``log_mass=False``
  for a linear mass in :math:`h^{-1}M_\\odot`, and ``lm_offset`` /
  ``mass_factor`` for any other unit (e.g. ``lm_offset=10`` if the catalogue
  stores :math:`\\log_{10}(M/10^{10}h^{-1}M_\\odot)`).
* Lengths in :math:`h^{-1}\\mathrm{Mpc}`, number densities in
  :math:`h^{3}\\mathrm{Mpc}^{-3}\\,\\mathrm{dex}^{-1}`.

Example
-------
>>> from pywccf.samples import TidalField, HaloSample
>>> from pywccf.environment.web_environment_stats import (density_pdf_by_web, hmf_by_web,
...                                    plot_density_pdf_by_web, plot_hmf_by_web)
>>> tf = TidalField.from_file('elucid_z0_Rs3.hdf5')
>>> hs = HaloSample.from_file('elucid_z0_halos.hdf5')
>>> d_res = density_pdf_by_web(tf, hs)
>>> m_res = hmf_by_web(hs, tf, n_jk_side=4)
>>> plot_density_pdf_by_web(d_res)
>>> plot_hmf_by_web(m_res)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'density_pdf_by_web', 'hmf_by_web',
    'plot_density_pdf_by_web', 'plot_hmf_by_web',
    'WEB_NAMES', 'WEB_COLORS', 'WEB_LINESTYLES',
]

# index = tf.n_lams = number of eigenvalues above the threshold
WEB_NAMES = ('Void', 'Sheet', 'Filament', 'Knot')
WEB_COLORS = ('#5e76ff', '#00ac38', '#f9a825', '#f33b1f')
WEB_LINESTYLES = ('-', '--', '-.', ':')
N_WEB = 4


# ---------------------------------------------------------------------------
# grid / sampling helpers
# ---------------------------------------------------------------------------

def _grid_spec(tf):
    """(n_grids, l_grid) of the field mesh."""
    return int(tf.mesh.n_grids), float(tf.mesh.l_grid)


def _recon_mask(tf):
    m = getattr(tf, 'recon_mask', None)
    if m is None:
        n, _ = _grid_spec(tf)
        return np.ones((n, n, n), dtype=bool)
    return np.asarray(m)


def _cell_index(tf, xs):
    """Nearest-grid-point cell indices (N, 3) of positions `xs` (N, 3).

    Uses the same mesh call as ``TidalField.wet_t_at``; falls back to a plain
    periodic floor division if the private ``mesh._impl`` is not available.
    """
    xs = np.asarray(xs, dtype=np.float64)
    assert xs.ndim == 2 and xs.shape[1] == 3, 'positions must be (N, 3)'
    try:
        xis = tf.mesh._impl.x_to_xi_nd(xs)
    except AttributeError:                                   # pragma: no cover
        n, h = _grid_spec(tf)
        xis = np.floor(xs / h + 0.5).astype(np.int64) % n   # pyhipp: nodes at i*h
    return np.asarray(xis, dtype=np.int64)


def _gather(arr, ijk):
    return np.asarray(arr)[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def _cic_at(arr, xs, n, h):
    """Cloud-in-cell interpolation of a periodic grid `arr` at `xs`.

    pyhipp / pyccb store field values on *nodes* at ``i * l_grid`` (see
    ``Mesh.x_to_xi_nd = floor(x/h + 0.5)`` and ``TidalField.xs_grids``), so
    the left node of `x` is ``floor(x / h)`` with no half-cell shift.
    """
    t = np.asarray(xs, dtype=np.float64) / h
    i0 = np.floor(t).astype(np.int64)
    f = t - i0
    out = np.zeros(len(xs), dtype=np.float64)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1.0 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                out += wx * wy * wz * arr[(i0[:, 0] + dx) % n,
                                          (i0[:, 1] + dy) % n,
                                          (i0[:, 2] + dz) % n]
    return out


def _delta_at(tf, xs, interp='ngp'):
    """delta at halo positions.  'ngp' is consistent with the cell-based web
    classification; 'cic' is smoother but mixes neighbouring web types."""
    n, h = _grid_spec(tf)
    if interp == 'ngp':
        return _gather(tf.delta, _cell_index(tf, xs)).astype(np.float64)
    if interp == 'cic':
        return _cic_at(np.asarray(tf.delta), xs, n, h)
    raise ValueError(f"interp must be 'ngp' or 'cic', got {interp!r}")


def _halo_arrays(h_samp, tf, *, mass_key=None, position_key='position',
                 use_weight=True):
    """Positions / web types / weights (/ raw mass field) of the haloes that
    sit in a reconstructed cell.  Returns a dict; weights are *not*
    renormalised and the mass field is returned exactly as stored."""
    xs = np.asarray(h_samp[position_key], dtype=np.float64)
    ijk = _cell_index(tf, xs)
    inside = _gather(_recon_mask(tf), ijk).astype(bool)

    out = {
        'xs': xs[inside],
        'ijk': ijk[inside],
        'web': _gather(tf.n_lams, ijk[inside]).astype(np.int64),
        'inside': inside,
        'n_in': int(inside.sum()),
        'n_tot': int(len(xs)),
    }
    if use_weight and ('weight' in h_samp):
        out['w'] = np.asarray(h_samp['weight'], dtype=np.float64)[inside]
    else:
        out['w'] = np.ones(out['n_in'], dtype=np.float64)
    if mass_key is not None:
        out['m'] = np.asarray(h_samp[mass_key], dtype=np.float64)[inside]
    return out


def _bin_ids(x, lo, dx, n_bins):
    """Bin index with 0 = underflow and n_bins+1 = overflow."""
    ids = np.floor((np.asarray(x) - lo) / dx).astype(np.int64) + 1
    return np.clip(ids, 0, n_bins + 1)


def _quantile_from_hist(cnt, edges, q):
    """Quantiles of a histogram that carries under/overflow in cnt[0] and
    cnt[-1].  Returns NaN when the quantile lies in a tail bin."""
    tot = cnt.sum()
    if tot <= 0:
        return np.full(np.shape(q), np.nan, dtype=float)
    cdf = np.cumsum(cnt) / tot
    q = np.atleast_1d(np.asarray(q, dtype=float))
    out = np.full(q.shape, np.nan)
    # cdf[i] is the cumulative fraction at the *upper* edge of bin i;
    # finite bins are 1 .. len(edges)-1, with upper edges edges[1:].
    lo_frac, hi_frac = cdf[0], cdf[-2]
    inner = np.concatenate([[cdf[0]], cdf[1:-1]])
    for i, qi in enumerate(q):
        if qi <= lo_frac or qi >= hi_frac:
            continue
        out[i] = np.interp(qi, inner, edges)
    return out if out.size > 1 else float(out[0])


# ---------------------------------------------------------------------------
# 1. density distribution per web type  (Hahn et al. 2007, Fig. 3)
# ---------------------------------------------------------------------------

def density_pdf_by_web(tf, h_samp=None, *,
                       x_range=(-2.0, 2.0), n_bins=80, log=True,
                       position_key='position', use_weight=True,
                       interp='ngp', chunk=32, verbose=True):
    """Distribution of the local density in the four web types.

    Grid ("volume-weighted") part reproduces Hahn et al. (2007) Fig. 3: for
    every reconstructed cell the density is histogrammed separately for
    ``tf.n_lams == 0, 1, 2, 3``.  If a halo sample is given, the same
    histogram weighted by haloes is returned as well -- the pair
    ``pdf_vol`` / ``pdf_halo`` is what a density-matched re-weighting of the
    halo sample is built from.

    Parameters
    ----------
    tf : pywccf.samples.field.TidalField
    h_samp : pywccf.samples.halo.HaloSample, optional
        If given, the halo-weighted density PDF is computed too.
    x_range, n_bins : range and number of bins of the abscissa.
    log : bool
        If True the abscissa is ``log10(1 + delta)``, else ``delta``.
    use_weight : bool
        Use ``h_samp['weight']`` if present.
    interp : {'ngp', 'cic'}
        Interpolation of ``delta`` at halo positions.
    chunk : int
        Number of grid slabs processed at a time (memory control).

    Returns
    -------
    dict with
        ``x``, ``x_edges``        -- bin centres / edges of the abscissa
        ``log``                   -- whether the abscissa is log10(1+delta)
        ``pdf_vol``      (4, nb)  -- volume PDF *within* each web type, so that
                                     ``sum(pdf_vol[i] * dx) = 1`` up to tails
        ``pdf_vol_joint``(4, nb)  -- ``pdf_vol * f_vol[:, None]``; the four
                                     curves add up to ``pdf_vol_all``
        ``pdf_vol_all``   (nb,)   -- PDF over all reconstructed cells
        ``f_vol``         (4,)    -- volume filling fraction of each web type
        ``q_vol``        (4, 3)   -- 16/50/84 percentiles of ``delta``
        ``n_cell``        (4,)    -- number of reconstructed cells per web type
        ``V_eff``, ``V_web``      -- effective volumes [ (Mpc/h)^3 ]
    and, when ``h_samp`` is given, the same with ``_halo`` instead of ``_vol``
    (``f_halo`` = fraction of haloes per web type, ``n_halo`` = weighted
    counts, ``n_halo_raw`` = raw counts).
    """
    n, h = _grid_spec(tf)
    lo, hi = float(x_range[0]), float(x_range[1])
    dx = (hi - lo) / n_bins
    edges = np.linspace(lo, hi, n_bins + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    nb2 = n_bins + 2                                   # + under/overflow

    def _x_of_delta(d):
        return np.log10(np.maximum(1.0 + d, 1e-10)) if log else d

    mask = _recon_mask(tf)
    cnt = np.zeros((N_WEB, nb2), dtype=np.float64)
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(i0 + chunk, n))
        m = np.asarray(mask[sl], dtype=bool)
        if not m.any():
            continue
        d = np.asarray(tf.delta[sl])[m]
        w = np.asarray(tf.n_lams[sl])[m].astype(np.int64)
        ids = _bin_ids(_x_of_delta(d), lo, dx, n_bins)
        cnt += np.bincount(w * nb2 + ids,
                           minlength=N_WEB * nb2).reshape(N_WEB, nb2)

    n_cell = cnt.sum(1)
    n_cell_tot = n_cell.sum()
    f_vol = n_cell / max(n_cell_tot, 1.0)
    V_eff = n_cell_tot * h**3
    V_web = n_cell * h**3

    pdf_vol = cnt[:, 1:-1] / np.maximum(n_cell, 1.0)[:, None] / dx
    pdf_vol_joint = pdf_vol * f_vol[:, None]
    pdf_vol_all = cnt.sum(0)[1:-1] / max(n_cell_tot, 1.0) / dx

    q_vol = np.array([_quantile_from_hist(cnt[i], edges, [0.16, 0.50, 0.84])
                      for i in range(N_WEB)])
    if log:                                            # back to delta
        q_vol_delta = 10.0**q_vol - 1.0
    else:
        q_vol_delta = q_vol

    res = {
        'x': cen, 'x_edges': edges, 'dx': dx, 'log': bool(log),
        'pdf_vol': pdf_vol, 'pdf_vol_joint': pdf_vol_joint,
        'pdf_vol_all': pdf_vol_all,
        'f_vol': f_vol, 'n_cell': n_cell,
        'q_vol': q_vol, 'q_vol_delta': q_vol_delta,
        'V_eff': V_eff, 'V_web': V_web,
        'lam_off': float(getattr(tf, 'lam_off', 0.0)),
        'web_names': WEB_NAMES,
    }

    if h_samp is not None:
        hd = _halo_arrays(h_samp, tf, position_key=position_key,
                          use_weight=use_weight)
        d_h = _delta_at(tf, hd['xs'], interp=interp)
        ids = _bin_ids(_x_of_delta(d_h), lo, dx, n_bins)
        key = hd['web'] * nb2 + ids
        cnt_h = np.bincount(key, weights=hd['w'],
                            minlength=N_WEB * nb2).reshape(N_WEB, nb2)
        cnt_h_raw = np.bincount(key, minlength=N_WEB * nb2).reshape(N_WEB, nb2)

        n_h = cnt_h.sum(1)
        f_halo = n_h / max(n_h.sum(), 1e-30)
        pdf_halo = cnt_h[:, 1:-1] / np.maximum(n_h, 1e-30)[:, None] / dx
        q_halo = np.array([_quantile_from_hist(cnt_h[i], edges,
                                               [0.16, 0.50, 0.84])
                           for i in range(N_WEB)])
        res.update({
            'pdf_halo': pdf_halo,
            'pdf_halo_joint': pdf_halo * f_halo[:, None],
            'pdf_halo_all': cnt_h.sum(0)[1:-1] / max(n_h.sum(), 1e-30) / dx,
            'f_halo': f_halo,
            'n_halo': n_h, 'n_halo_raw': cnt_h_raw.sum(1),
            'q_halo': q_halo,
            'q_halo_delta': 10.0**q_halo - 1.0 if log else q_halo,
            'delta_halo': d_h, 'web_halo': hd['web'],
            'n_halo_in': hd['n_in'], 'n_halo_tot': hd['n_tot'],
        })

    if verbose:
        print(f'density_pdf_by_web: lam_th={res["lam_off"]:g}, '
              f'V_eff={V_eff:.4g} (Mpc/h)^3')
        for i in range(N_WEB):
            line = (f'  {WEB_NAMES[i]:<9s} f_V={f_vol[i]:6.3f}  '
                    f'median delta={q_vol_delta[i, 1]:8.3f}')
            if h_samp is not None:
                line += (f'  f_halo={res["f_halo"][i]:6.3f}'
                         f'  median delta_h={res["q_halo_delta"][i, 1]:8.3f}')
            print(line)
    return res


# ---------------------------------------------------------------------------
# 2. halo mass function per web type  (Hahn et al. 2007, Fig. 4)
# ---------------------------------------------------------------------------

def _web_cell_counts(tf, n_jk_side=0, chunk=32):
    """Cell counts per (jackknife sub-cube, web type).

    Returns an array of shape ``(n_jk, 4)`` with ``n_jk = max(1, n_jk_side**3)``.
    """
    n, _ = _grid_spec(tf)
    n_jk = max(1, int(n_jk_side)**3)
    mask = _recon_mask(tf)
    out = np.zeros((n_jk, N_WEB), dtype=np.float64)
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(i0 + chunk, n))
        m = np.asarray(mask[sl], dtype=bool)
        if not m.any():
            continue
        w = np.asarray(tf.n_lams[sl])[m].astype(np.int64)
        if n_jk == 1:
            out[0] += np.bincount(w, minlength=N_WEB)
            continue
        di, jj, kk = np.nonzero(m)
        s = int(n_jk_side)
        jk = ((di + i0) * s // n) * s * s + (jj * s // n) * s + (kk * s // n)
        out += np.bincount(jk * N_WEB + w,
                           minlength=n_jk * N_WEB).reshape(n_jk, N_WEB)
    return out


def _jk_index_of_pos(xs, l_box, n_jk_side):
    s = int(n_jk_side)
    t = np.clip((np.asarray(xs, dtype=np.float64) / l_box * s).astype(np.int64),
                0, s - 1)
    return t[:, 0] * s * s + t[:, 1] * s + t[:, 2]


def hmf_by_web(h_samp, tf, *, lm_range=None, n_bins=16,
               mass_key='mass', position_key='position',
               log_mass=True, lm_offset=0.0, mass_factor=1.0,
               use_weight=True, n_jk_side=0, chunk=32, verbose=True):
    """Halo mass function split by cosmic-web type.

    Reproduces Hahn et al. (2007) Fig. 4: haloes are counted in the whole
    (reconstructed) volume and the abundance of each web type is divided by
    that web type's own volume, so the curves are number densities *inside*
    voids / sheets / filaments / knots and the grey total is the usual mass
    function.  Also returns ``f_web`` -- the fraction of haloes of a given mass
    that live in each environment -- which is the quantity to control for when
    asking whether a signal is driven by environment rather than by mass.

    Parameters
    ----------
    h_samp : pywccf.samples.halo.HaloSample
        Needs ``position`` (cMpc/h) and ``mass``; ``weight`` is used if present.
    tf : pywccf.samples.field.TidalField
    lm_range : (lo, hi), optional
        log10(M / h^-1 Msun) range.  Default: from the 0.1 and 99.9 percentiles
        of the sample, rounded outwards to 0.1 dex.
    log_mass : bool
        True (default) if ``h_samp[mass_key]`` already is
        log10(M_h / h^-1 Msun); False if it is a linear mass.
    lm_offset : float
        Added to the log mass (``log_mass=True``), e.g. 10.0 for a catalogue in
        log10(M / 10^10 h^-1 Msun).
    mass_factor : float
        Multiplies the linear mass (``log_mass=False``) to bring it to
        h^-1 Msun.
    n_jk_side : int
        If > 0, errors from a ``n_jk_side**3`` sub-cube jackknife (haloes *and*
        volumes are dropped together).  If 0, Poisson errors are used.

    Returns
    -------
    dict with
        ``lm``, ``lm_edges``, ``dlm``   -- log10(M) bin centres / edges / width
        ``dndlgm``      (4, nb)  -- dn/dlog10M inside each web type
                                    [ h^3 Mpc^-3 dex^-1 ]
        ``dndlgm_err``  (4, nb)
        ``dndlgm_all``   (nb,)   -- total, over the whole effective volume
        ``dndlgm_all_err`` (nb,)
        ``f_web``       (4, nb)  -- fraction of haloes per web type at fixed M
        ``f_web_err``   (4, nb)
        ``n_halo``      (4, nb)  -- weighted counts; ``n_halo_raw`` raw counts
        ``f_vol``, ``V_web``, ``V_eff``, ``err_kind``
    """
    n, h = _grid_spec(tf)
    l_box = n * h

    hd = _halo_arrays(h_samp, tf, mass_key=mass_key, position_key=position_key,
                      use_weight=use_weight)
    m = hd['m']
    if log_mass:                       # already log10(M_h / h^-1 Msun)
        good = np.isfinite(m)
        lm_h = m[good] + float(lm_offset)
    else:                              # linear mass
        good = np.isfinite(m) & (m > 0)
        lm_h = np.log10(m[good] * float(mass_factor))
    web_h = hd['web'][good]
    w_h = hd['w'][good]
    xs_h = hd['xs'][good]

    if lm_range is None:
        lo = np.floor(np.percentile(lm_h, 0.1) * 10.0) / 10.0
        hi = np.ceil(np.percentile(lm_h, 99.9) * 10.0) / 10.0
    else:
        lo, hi = float(lm_range[0]), float(lm_range[1])
    dlm = (hi - lo) / n_bins
    edges = np.linspace(lo, hi, n_bins + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    nb2 = n_bins + 2

    n_jk = max(1, int(n_jk_side)**3)
    jk_h = (_jk_index_of_pos(xs_h, l_box, n_jk_side) if n_jk > 1
            else np.zeros(len(lm_h), dtype=np.int64))

    ids = _bin_ids(lm_h, lo, dlm, n_bins)
    key = (jk_h * N_WEB + web_h) * nb2 + ids
    size = n_jk * N_WEB * nb2
    cnt_jk = np.bincount(key, weights=w_h, minlength=size).reshape(
        n_jk, N_WEB, nb2)
    cnt_jk_raw = np.bincount(key, minlength=size).reshape(n_jk, N_WEB, nb2)
    cnt_jk_w2 = np.bincount(key, weights=w_h**2, minlength=size).reshape(
        n_jk, N_WEB, nb2)

    cells_jk = _web_cell_counts(tf, n_jk_side=n_jk_side, chunk=chunk)  # (n_jk,4)
    cells = cells_jk.sum(0)
    V_web = cells * h**3
    V_eff = V_web.sum()
    f_vol = cells / max(cells.sum(), 1.0)

    cnt = cnt_jk.sum(0)[:, 1:-1]                       # (4, nb)
    cnt_raw = cnt_jk_raw.sum(0)[:, 1:-1]
    cnt_w2 = cnt_jk_w2.sum(0)[:, 1:-1]

    dndlgm = cnt / np.maximum(V_web, 1e-30)[:, None] / dlm
    dndlgm_all = cnt.sum(0) / V_eff / dlm

    # sub-cubes that hold no reconstructed volume would be null resamplings
    # and would bias the jackknife variance low -- drop them.
    occupied = np.flatnonzero(cells_jk.sum(1) > 0)
    if n_jk > 1 and len(occupied) > 1:
        n_use = len(occupied)
        err_kind = f'jackknife ({n_use} of {n_jk_side}^3 sub-cubes)'
        vals = np.empty((n_use, N_WEB, n_bins))
        vals_all = np.empty((n_use, n_bins))
        for j, k in enumerate(occupied):
            c_k = cnt - cnt_jk[k][:, 1:-1]
            V_k = np.maximum(V_web - cells_jk[k] * h**3, 1e-30)
            vals[j] = c_k / V_k[:, None] / dlm
            vals_all[j] = c_k.sum(0) / V_k.sum() / dlm
        fac = (n_use - 1.0) / n_use
        dndlgm_err = np.sqrt(fac * ((vals - vals.mean(0))**2).sum(0))
        dndlgm_all_err = np.sqrt(fac * ((vals_all - vals_all.mean(0))**2).sum(0))
        n_jk = n_use
    else:
        err_kind = 'Poisson'
        dndlgm_err = np.sqrt(cnt_w2) / np.maximum(V_web, 1e-30)[:, None] / dlm
        dndlgm_all_err = np.sqrt(cnt_w2.sum(0)) / V_eff / dlm

    tot = np.maximum(cnt.sum(0), 1e-30)
    f_web = cnt / tot
    n_eff = np.maximum(cnt.sum(0)**2 / np.maximum(cnt_w2.sum(0), 1e-30), 1.0)
    f_web_err = np.sqrt(np.clip(f_web * (1.0 - f_web), 0.0, None) / n_eff)

    res = {
        'lm': cen, 'lm_edges': edges, 'dlm': dlm,
        'dndlgm': dndlgm, 'dndlgm_err': dndlgm_err,
        'dndlgm_all': dndlgm_all, 'dndlgm_all_err': dndlgm_all_err,
        'f_web': f_web, 'f_web_err': f_web_err,
        'n_halo': cnt, 'n_halo_raw': cnt_raw,
        'f_vol': f_vol, 'V_web': V_web, 'V_eff': V_eff,
        'err_kind': err_kind, 'n_jk': n_jk,
        'lam_off': float(getattr(tf, 'lam_off', 0.0)),
        'n_halo_in': hd['n_in'], 'n_halo_tot': hd['n_tot'],
        'web_names': WEB_NAMES,
    }

    if verbose:
        print(f'hmf_by_web: {hd["n_in"]}/{hd["n_tot"]} haloes in the '
              f'reconstructed volume, lam_th={res["lam_off"]:g}, '
              f'errors: {err_kind}')
        print(f'  log10 M range = [{lo:.2f}, {hi:.2f}], '
              f'{n_bins} bins of {dlm:.3f} dex, V_eff={V_eff:.4g} (Mpc/h)^3')
        for i in range(N_WEB):
            print(f'  {WEB_NAMES[i]:<9s} f_V={f_vol[i]:6.3f}  '
                  f'N_halo={cnt_raw[i].sum():10.0f}  '
                  f'f_halo={cnt[i].sum() / tot.sum():6.3f}')
    return res


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_density_pdf_by_web(res, ax=None, *, which='vol', joint=False,
                            show_all=True, show_median=False,
                            colors=WEB_COLORS, linestyles=WEB_LINESTYLES,
                            **line_kw):
    """Hahn et al. (2007) Fig. 3 style plot of :func:`density_pdf_by_web`.

    @which: 'vol' (volume weighted) or 'halo' (halo weighted).
    @joint: plot ``pdf * f`` so the four curves add up to the global PDF.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
    suffix = '_joint' if joint else ''
    pdf = res[f'pdf_{which}{suffix}']
    frac = res[f'f_{which}']
    q = res.get(f'q_{which}')
    x = res['x']

    kw = dict(lw=1.8)
    kw.update(line_kw)
    for i in range(N_WEB):
        ax.plot(x, pdf[i], color=colors[i], ls=linestyles[i],
                label=f'{WEB_NAMES[i]} ({frac[i] * 100:.1f}%)', **kw)
        if show_median and q is not None and np.isfinite(q[i, 1]):
            ax.axvline(q[i, 1], color=colors[i], ls=linestyles[i],
                       lw=0.8, alpha=0.5)
    if show_all and joint:
        ax.plot(x, res[f'pdf_{which}_all'], color='0.35', lw=2.2, zorder=0,
                label='All')

    ax.set_xlabel(r'$\log_{10}(1+\delta)$' if res['log'] else r'$\delta$')
    ax.set_ylabel(r'$p(\log_{10}(1+\delta))$' if res['log'] else r'$p(\delta)$')
    ax.set_xlim(res['x_edges'][0], res['x_edges'][-1])
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_hmf_by_web(res, axs=None, *, with_fraction=True, show_total=True,
                    colors=WEB_COLORS, linestyles=WEB_LINESTYLES, **line_kw):
    """Hahn et al. (2007) Fig. 4 style plot of :func:`hmf_by_web`.

    Top panel: dn/dlog10M inside each web type plus the total.
    Bottom panel (optional): fraction of haloes per web type at fixed mass,
    with the volume filling fractions as horizontal reference lines.
    """
    if axs is None:
        n_row = 2 if with_fraction else 1
        _, axs = plt.subplots(n_row, 1, figsize=(4.2, 3.0 + 1.8 * (n_row - 1)),
                              sharex=True, constrained_layout=True,
                              gridspec_kw={'height_ratios': [2, 1][:n_row]})
    axs = np.atleast_1d(axs)
    ax = axs[0]
    lm, kw = res['lm'], dict(lw=1.8)
    kw.update(line_kw)

    if show_total:
        ax.plot(lm, res['dndlgm_all'], color='0.55', lw=3.0, zorder=0,
                label='All')
    for i in range(N_WEB):
        y, e = res['dndlgm'][i], res['dndlgm_err'][i]
        ok = y > 0
        ax.plot(lm[ok], y[ok], color=colors[i], ls=linestyles[i],
                label=WEB_NAMES[i], **kw)
        ax.fill_between(lm[ok], np.maximum(y[ok] - e[ok], 1e-30), y[ok] + e[ok],
                        color=colors[i], alpha=0.20, lw=0)
    ax.set_yscale('log')
    ax.set_ylabel(r'${\rm d}n/{\rm d}\log_{10}M\ '
                  r'[h^{3}\,{\rm Mpc}^{-3}\,{\rm dex}^{-1}]$')
    ax.legend(frameon=False, fontsize=8, ncol=2)

    if with_fraction and len(axs) > 1:
        ax2 = axs[1]
        for i in range(N_WEB):
            f, e = res['f_web'][i], res['f_web_err'][i]
            ax2.plot(lm, f, color=colors[i], ls=linestyles[i], **kw)
            ax2.fill_between(lm, f - e, f + e, color=colors[i], alpha=0.20,
                             lw=0)
            ax2.axhline(res['f_vol'][i], color=colors[i], lw=0.8, alpha=0.4)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel(r'$f_{\rm web}(M)$')
        ax2.set_xlabel(r'$\log_{10}(M\,/\,h^{-1}M_\odot)$')
    else:
        ax.set_xlabel(r'$\log_{10}(M\,/\,h^{-1}M_\odot)$')
    return axs
