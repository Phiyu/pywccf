"""
Separating the local-density effect from the tidal (cosmic-web) effect in the
halo-age dependence of the 3D-WCCF.

Part of ``pywccf`` (split out of the analysis scripts of the 3D-WCCF paper) and to ``web_environment_stats`` (neither is
modified).  The workflow it supports:

1. :func:`quantile_split` -- split haloes into "old" and "young" by formation
   redshift **at fixed mass** (tails of ``zf`` inside每个 mass bin).  Splitting
   on ``zf`` globally would also split on mass, because the two correlate.

2. :func:`density_pdf_by_sample_web` / :func:`plot_density_pdf_by_sample_web`
   -- the diagnostic figure: p(log10(1+delta)) for every (sub-sample, web type)
   pair, i.e. "inside one web type, do old and young haloes sit at different
   densities?".  Returns the median offsets and two-sample KS statistics, since
   a 0.05 dex offset is invisible by eye but not by the WCCF.

3. :func:`density_matched_weights` / :func:`matched_subsets` -- the main
   density control.  Per mass bin, haloes are re-weighted on a
   (log M, log10(1+delta)) grid so that every sub-sample shares one reference
   (M, delta) distribution.  The result is a ``weight`` column that
   ``pyccb.process.halo_web_optimized.HaloWebCross`` already consumes, so the
   WCCF pipeline is unchanged.  ``N_eff = (sum w)^2 / sum w^2`` quantifies what
   the matching costs in signal-to-noise.

4. :func:`density_bin_labels` / :func:`density_bin_subsets` -- the transparent
   cross-check: hard delta bins inside each mass bin, cut at the *same*
   thresholds for old and young.  Noisier than (3); intended for an appendix.

5. :func:`smooth_field` -- re-smooth ``tf.delta`` to another scale, so the
   matching scale can be scanned (R = 3, 5, 8 h^-1 Mpc).  Matching delta at the
   classification scale alone does not remove the density on the scales the
   3D-WCCF actually probes; a residual signal that survives R = 8 is a much
   stronger claim.

Caveat to carry into the text: matching one-point density at one smoothing
scale removes *that* density, not "all density".  The honest phrasing is
"residual anisotropy after controlling for the local density", not
"pure tidal effect".

Example
-------
>>> from pywccf.environment import web_environment_stats as wes
>>> from pywccf.environment.age_density_control import *
>>> sp = quantile_split(hs)                       # zf tails at fixed mass
>>> res = density_pdf_by_sample_web(tf, hs, sp)
>>> plot_density_pdf_by_sample_web(res)
>>> mw = density_matched_weights(hs, tf, sp)      # ref = pooled old+young
>>> subs = matched_subsets(hs, mw)                # {'Young': ..., 'Old': ...}
>>> HaloWebCross(subs['Old'], tf, rng=42).corrs()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import web_environment_stats as wes
from .web_environment_stats import WEB_NAMES, WEB_COLORS, N_WEB

__all__ = [
    'quantile_split', 'delta_at_halos', 'smooth_field',
    'density_pdf_by_sample_web', 'plot_density_pdf_by_sample_web',
    'density_matched_weights', 'matched_subsets',
    'density_bin_labels', 'density_bin_subsets',
    'halo_number_field', 'summarize_density_bins',
    'mass_distribution_by_sample', 'plot_mass_distribution_by_sample',
    'LM_EDGES_DEFAULT',
]

LM_EDGES_DEFAULT = np.arange(11.5, 15.001, 0.5)
SAMPLE_LINESTYLES = ('--', '-', ':')          # Young, Old, All


# ---------------------------------------------------------------------------
# 1. age split at fixed mass
# ---------------------------------------------------------------------------

def quantile_split(h_samp, *, key='zf', mass_key='mass', log_mass=True,
                   lm_edges=None, p=0.2, high_is_old=True,
                   names=('Young', 'Old'), verbose=True):
    """Tails of `key` taken separately inside each mass bin.

    Parameters
    ----------
    h_samp : HaloSample
    key : str
        Age proxy; ``zf`` by default.  ``high_is_old=True`` means a larger
        value is older (true for a formation redshift).
    mass_key, log_mass :
        ``h_samp[mass_key]`` is log10(M / h^-1 Msun) when ``log_mass`` is True.
    lm_edges : array, optional
        Mass bin edges in log10(M / h^-1 Msun).  Default ``LM_EDGES_DEFAULT``
        (0.5 dex from 11.5 to 15).
    p : float
        Tail fraction on each side, per mass bin (0.2 -> oldest 20% and
        youngest 20% of every bin).

    Returns
    -------
    dict with
        ``label``   (n,)  -- 0 = young, 1 = old, -1 = neither / outside range
        ``in_range``(n,)  -- halo falls inside ``lm_edges``
        ``m_bin``   (n,)  -- mass bin index, -1 outside
        ``lm``      (n,)  -- log10 mass actually used
        ``names``, ``lm_edges``, ``p``, ``key``
        ``thresholds`` (nm, 2) -- (young cut, old cut) of `key` per mass bin
        ``counts``     (nm, 2) -- sample size per mass bin and sub-sample
    """
    lm = np.asarray(h_samp[mass_key], dtype=np.float64)
    if not log_mass:
        lm = np.log10(lm)
    v = np.asarray(h_samp[key], dtype=np.float64)
    n = len(lm)

    edges = np.asarray(LM_EDGES_DEFAULT if lm_edges is None else lm_edges,
                       dtype=np.float64)
    nm = len(edges) - 1
    m_bin = np.digitize(lm, edges) - 1
    m_bin[(m_bin < 0) | (m_bin >= nm) | ~np.isfinite(lm) | ~np.isfinite(v)] = -1

    label = np.full(n, -1, dtype=np.int64)
    thr = np.full((nm, 2), np.nan)
    cnt = np.zeros((nm, 2), dtype=np.int64)
    q_lo, q_hi = (p, 1.0 - p) if high_is_old else (1.0 - p, p)

    for b in range(nm):
        sel = m_bin == b
        if sel.sum() < 10:
            continue
        vb = v[sel]
        c_young, c_old = np.quantile(vb, q_lo), np.quantile(vb, q_hi)
        if high_is_old:
            young = sel & (v <= c_young)
            old = sel & (v >= c_old)
        else:
            young = sel & (v >= c_young)
            old = sel & (v <= c_old)
        label[young] = 0
        label[old] = 1
        thr[b] = (c_young, c_old)
        cnt[b] = (young.sum(), old.sum())

    out = {
        'label': label, 'in_range': m_bin >= 0, 'm_bin': m_bin, 'lm': lm,
        'names': tuple(names), 'lm_edges': edges, 'p': float(p),
        'key': key, 'high_is_old': bool(high_is_old),
        'thresholds': thr, 'counts': cnt,
    }
    if verbose:
        print(f'quantile_split: {key} tails at p={p:g}, {nm} mass bins')
        for b in range(nm):
            if cnt[b].sum() == 0:
                continue
            print(f'  logM [{edges[b]:5.2f}, {edges[b+1]:5.2f}): '
                  f'N={int((m_bin == b).sum()):8d}  '
                  f'{names[0]}={cnt[b, 0]:7d} ({key}<={thr[b, 0]:.3f})  '
                  f'{names[1]}={cnt[b, 1]:7d} ({key}>={thr[b, 1]:.3f})')
    return out


# ---------------------------------------------------------------------------
# 2. density at halo positions, at an arbitrary smoothing scale
# ---------------------------------------------------------------------------

def smooth_field(delta, l_box, R, kind='gaussian', dtype=np.float32):
    """Re-smooth a periodic cubic field to scale `R` [same units as `l_box`].

    ``kind='gaussian'`` applies exp(-k^2 R^2 / 2); ``'tophat'`` applies the
    real-space top-hat window.  Note that ``tf.delta`` is *already* smoothed at
    R_s, so the result is smoothed at sqrt(R_s^2 + R^2) for a Gaussian --
    pass ``R = sqrt(R_target**2 - R_s**2)`` if you want a target total scale.

    Memory: a 500^3 grid needs about 2-3 GB during the transform.
    """
    a = np.asarray(delta, dtype=dtype)
    n = a.shape[0]
    assert a.shape == (n, n, n), 'field must be a cubic grid'
    kf = 2.0 * np.pi / float(l_box)
    k1 = np.fft.fftfreq(n, d=1.0 / n) * kf
    kz = np.fft.rfftfreq(n, d=1.0 / n) * kf
    k2 = (k1[:, None, None]**2 + k1[None, :, None]**2 + kz[None, None, :]**2)

    ak = np.fft.rfftn(a)
    if kind == 'gaussian':
        ak *= np.exp(-0.5 * k2 * R**2)
    elif kind == 'tophat':
        kr = np.sqrt(k2) * R
        w = np.ones_like(kr)
        nz = kr > 1e-6
        w[nz] = 3.0 * (np.sin(kr[nz]) - kr[nz] * np.cos(kr[nz])) / kr[nz]**3
        ak *= w
    else:
        raise ValueError("kind must be 'gaussian' or 'tophat'")
    return np.fft.irfftn(ak, s=(n, n, n)).astype(dtype)


def delta_at_halos(tf, h_samp, *, field=None, interp='ngp',
                   position_key='position', prefix='env_'):
    """delta at halo positions, plus the in-reconstruction mask.

    @tf: the TidalField, or None to read the per-halo columns written by
         ``halo_env.attach_halo_environment`` (``env_delta``, ``env_web``,
         ``env_in_recon``) -- identical values, no field needed.  With
         tf=None only the NGP values on the original field are available.
    @field: a grid to sample instead of ``tf.delta`` (e.g. from
            :func:`smooth_field`).  Must share ``tf``'s mesh.
    Returns ``(delta, web_type, inside)``, all aligned to the full sample.
    """
    if tf is None:
        if field is not None or interp != 'ngp':
            raise ValueError("tf=None reads the stored NGP environment; "
                             "field/interp need the TidalField")
        need = [prefix + k for k in ('delta', 'web', 'in_recon')]
        miss = [k for k in need if k not in h_samp]
        if miss:
            raise KeyError(f'tf=None but the sample lacks {miss}; build it '
                           'with halo_env.attach_halo_environment')
        return (np.asarray(h_samp[need[0]], dtype=np.float64),
                np.asarray(h_samp[need[1]], dtype=np.int64),
                np.asarray(h_samp[need[2]], dtype=bool))
    xs = np.asarray(h_samp[position_key], dtype=np.float64)
    ijk = wes._cell_index(tf, xs)
    inside = wes._gather(wes._recon_mask(tf), ijk).astype(bool)
    web = wes._gather(tf.n_lams, ijk).astype(np.int64)

    grid = tf.delta if field is None else field
    n, h = wes._grid_spec(tf)
    if interp == 'ngp':
        d = wes._gather(grid, ijk).astype(np.float64)
    elif interp == 'cic':
        d = wes._cic_at(np.asarray(grid), xs, n, h)
    else:
        raise ValueError("interp must be 'ngp' or 'cic'")
    return d, web, inside


def _x_of_delta(d, log=True):
    return np.log10(np.maximum(1.0 + np.asarray(d), 1e-10)) if log else \
        np.asarray(d)


# ---------------------------------------------------------------------------
# 3. the diagnostic figure: p(log(1+delta)) per (sub-sample, web type)
# ---------------------------------------------------------------------------

def density_pdf_by_sample_web(tf, h_samp, split, *, field=None, interp='ngp',
                              x_range=(-2.0, 2.0), n_bins=60, log=True,
                              include_all=True, use_weight=True, weights=None,
                              position_key='position', verbose=True):
    """Density distribution of each sub-sample inside each web type.

    `split` is the dict returned by :func:`quantile_split`.  The "All" row, if
    requested, is every halo inside the mass range, regardless of age.

    Returns
    -------
    dict with
        ``x``, ``x_edges``, ``log``
        ``pdf``    (ns, 4, nb) -- normalised inside each (sample, web) cell
        ``n``      (ns, 4)     -- weighted counts
        ``n_raw``  (ns, 4)     -- raw counts
        ``f_web``  (ns, 4)     -- web-type fractions of each sub-sample
        ``median`` (ns, 4)     -- median of the abscissa
        ``d_median``   (4,)    -- median(Old) - median(Young), per web type
        ``ks``, ``ks_p`` (4,)  -- weighted two-sample KS between Old and
                                  Young, per web (binned; responds to
                                  ``weights``)
        ``n_eff`` (ns, 4)      -- effective sample size per (sample, web)
        ``names``  (ns,)
    """
    names = list(split['names']) + (['All'] if include_all else [])
    ns = len(names)
    lo, hi = float(x_range[0]), float(x_range[1])
    dx = (hi - lo) / n_bins
    edges = np.linspace(lo, hi, n_bins + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    nb2 = n_bins + 2

    d, web, inside = delta_at_halos(tf, h_samp, field=field, interp=interp,
                                    position_key=position_key)
    x = _x_of_delta(d, log)
    ids = wes._bin_ids(x, lo, dx, n_bins)
    if weights is not None:            # e.g. density_matched_weights output
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(x):
            raise ValueError('weights must be aligned to the full sample')
    elif use_weight and ('weight' in h_samp):
        w = np.asarray(h_samp['weight'], dtype=np.float64)
    else:
        w = np.ones(len(x), dtype=np.float64)

    lab, in_range = split['label'], split['in_range']
    rows = [(lab == i) & inside for i in range(len(split['names']))]
    if include_all:
        rows.append(in_range & inside)

    cnt = np.zeros((ns, N_WEB, nb2))
    cnt_raw = np.zeros((ns, N_WEB, nb2))
    cnt_w2 = np.zeros((ns, N_WEB, nb2))
    for s, sel in enumerate(rows):
        key = web[sel] * nb2 + ids[sel]
        cnt[s] = np.bincount(key, weights=w[sel],
                             minlength=N_WEB * nb2).reshape(N_WEB, nb2)
        cnt_raw[s] = np.bincount(key,
                                 minlength=N_WEB * nb2).reshape(N_WEB, nb2)
        cnt_w2[s] = np.bincount(key, weights=w[sel]**2,
                                minlength=N_WEB * nb2).reshape(N_WEB, nb2)

    n_sw = cnt.sum(2)
    pdf = cnt[:, :, 1:-1] / np.maximum(n_sw, 1e-30)[:, :, None] / dx
    f_web = n_sw / np.maximum(n_sw.sum(1, keepdims=True), 1e-30)
    quant = np.array([[wes._quantile_from_hist(cnt[s, i], edges,
                                               [0.16, 0.5, 0.84])
                       for i in range(N_WEB)] for s in range(ns)])
    med = quant[:, :, 1]
    sig = 0.5 * (quant[:, :, 2] - quant[:, :, 0])
    # standardised effect size: the raw dex offset means little on its own,
    # because conditioning on web type already pins delta to a narrow range.
    sig_pool = np.sqrt(0.5 * (sig[0]**2 + sig[1]**2))
    d_cohen = (med[1] - med[0]) / np.maximum(sig_pool, 1e-30)

    # weighted two-sample KS between the first two sub-samples, computed from
    # the weighted histograms so that it responds to `weights` (an unweighted
    # scipy KS would return the same number before and after matching).
    n_eff = cnt.sum(2)**2 / np.maximum(cnt_w2.sum(2), 1e-30)
    ks = np.full(N_WEB, np.nan)
    ks_p = np.full(N_WEB, np.nan)
    for i in range(N_WEB):
        t0, t1 = cnt[0, i].sum(), cnt[1, i].sum()
        if t0 <= 0 or t1 <= 0:
            continue
        F0, F1 = np.cumsum(cnt[0, i]) / t0, np.cumsum(cnt[1, i]) / t1
        d_ks = float(np.abs(F0 - F1).max())
        ne = (n_eff[0, i] * n_eff[1, i]) / max(n_eff[0, i] + n_eff[1, i], 1e-30)
        lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(max(ne, 1e-30))) * d_ks
        k = np.arange(1, 101)
        ks[i] = d_ks
        ks_p[i] = float(np.clip(
            2.0 * np.sum((-1.0)**(k - 1) * np.exp(-2.0 * k**2 * lam**2)),
            0.0, 1.0))

    res = {
        'x': cen, 'x_edges': edges, 'dx': dx, 'log': bool(log),
        'pdf': pdf, 'n': n_sw, 'n_raw': cnt_raw.sum(2), 'f_web': f_web,
        'quantiles': quant, 'median': med, 'sigma': sig,
        'd_median': med[1] - med[0], 'd_cohen': d_cohen,
        'ks': ks, 'ks_p': ks_p, 'n_eff': n_eff, 'names': names,
        'web_names': WEB_NAMES, 'key': split['key'],
        'lm_edges': split['lm_edges'], 'p': split['p'],
    }

    if verbose:
        print(f'density_pdf_by_sample_web: split on {split["key"]}, '
              f'p={split["p"]:g}')
        head = '  ' + ' '.join(f'{nm:>10s}' for nm in names)
        print(f'  {"web":<9s}' + head.replace('  ', '', 1) +
              f'{"d_median":>11s}{"d_cohen":>9s}{"KS":>8s}{"p_KS":>10s}')
        for i in range(N_WEB):
            row = f'  {WEB_NAMES[i]:<9s}'
            for s in range(ns):
                row += f'{med[s, i]:10.3f} '
            row += (f'{res["d_median"][i]:10.3f} {d_cohen[i]:8.3f} '
                    f'{ks[i]:7.4f} {ks_p[i]:9.2e}')
            print(row)
        print('  d_median [dex], d_cohen = d_median / sigma_pooled, '
              'KS weighted (binned)')
        print('  f_web  ' + '  '.join(
            f'{names[s]}: ' + np.array2string(f_web[s], precision=3)
            for s in range(ns)))
    return res


def plot_density_pdf_by_sample_web(res, axs=None, *, panels=True, xlim='auto',
                                   colors=WEB_COLORS,
                                   linestyles=SAMPLE_LINESTYLES,
                                   annotate=True, **line_kw):
    """Plot :func:`density_pdf_by_sample_web`.

    @panels: True -> one panel per web type (recommended; the four web types
             are offset by a decade in delta, so a single panel squashes the
             old/young difference that is the point of the figure).
             False -> all curves on one axes, colour = web, style = sub-sample.
    """
    names, pdf, x = res['names'], res['pdf'], res['x']
    ns = len(names)
    kw = dict(lw=1.6)
    kw.update(line_kw)

    if panels:
        if axs is None:
            _, axs = plt.subplots(1, N_WEB, figsize=(11.0, 2.9), sharey=True,
                                  constrained_layout=True)
        axs = np.atleast_1d(axs).ravel()
        for i in range(N_WEB):
            ax = axs[i]
            for s in range(ns):
                ax.plot(x, pdf[s, i], color=colors[i],
                        ls=linestyles[s % len(linestyles)],
                        alpha=1.0 if s < 2 else 0.45,
                        label=names[s], **kw)
            ax.set_title(WEB_NAMES[i], fontsize=10)
            ax.set_xlabel(r'$\log_{10}(1+\delta)$' if res['log']
                          else r'$\delta$')
            ax.set_ylim(bottom=0.0)
            if xlim == 'auto':           # zoom to where this web type lives
                c = np.cumsum(pdf[:, i].sum(0))
                if c[-1] > 0:
                    c = c / c[-1]
                    a, b = np.interp([0.002, 0.998], c, x)
                    pad = 0.1 * max(b - a, 1e-3)
                    ax.set_xlim(a - pad, b + pad)
            elif xlim is not None:
                ax.set_xlim(*xlim)
            if annotate:
                txt = rf'$\Delta\tilde{{x}}={res["d_median"][i]:+.3f}$'
                if 'd_cohen' in res:
                    txt += '\n' + rf'$d={res["d_cohen"][i]:+.2f}$'
                if np.isfinite(res['ks'][i]):
                    txt += '\n' + rf'$D_{{\rm KS}}={res["ks"][i]:.3f}$'
                ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top',
                        fontsize=8)
            if i == 0:
                ax.set_ylabel(r'$p(\log_{10}(1+\delta))$')
                ax.legend(frameon=False, fontsize=8)
        return axs

    if axs is None:
        _, axs = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
    ax = np.atleast_1d(axs).ravel()[0]
    for i in range(N_WEB):
        for s in range(ns):
            ax.plot(x, pdf[s, i], color=colors[i],
                    ls=linestyles[s % len(linestyles)],
                    alpha=1.0 if s < 2 else 0.45, **kw)
    h_web = [plt.Line2D([], [], color=colors[i], lw=1.6, label=WEB_NAMES[i])
             for i in range(N_WEB)]
    h_smp = [plt.Line2D([], [], color='0.3',
                        ls=linestyles[s % len(linestyles)], lw=1.6,
                        label=names[s]) for s in range(ns)]
    ax.add_artist(ax.legend(handles=h_web, frameon=False, fontsize=8,
                            loc='upper left'))
    ax.legend(handles=h_smp, frameon=False, fontsize=8, loc='upper right')
    ax.set_xlabel(r'$\log_{10}(1+\delta)$' if res['log'] else r'$\delta$')
    ax.set_ylabel(r'$p(\log_{10}(1+\delta))$')
    ax.set_ylim(bottom=0.0)
    return ax


# ---------------------------------------------------------------------------
# 4. density matching by re-weighting  (main control)
# ---------------------------------------------------------------------------

def density_matched_weights(h_samp, tf, split, *, ref='pooled', n_dbins=6,
                            n_msub=3, by_web=False, field=None, interp='ngp',
                            log=True, w_clip=10.0, min_ref=20,
                            position_key='position', verbose=True):
    """Weights that give every sub-sample the same (log M, delta) distribution.

    Inside each analysis mass bin the reference sample is split into
    ``n_msub`` equal-count mass sub-bins times ``n_dbins`` equal-count delta
    bins, and every sub-sample is re-weighted cell by cell to the reference
    counts.  Matching on the cells therefore matches the mass distribution
    *inside* the bin as well -- which matters, because ``zf`` correlates with
    mass, so old and young are not identically distributed even within a
    0.5 dex bin.  ``n_msub * n_dbins`` should stay well below the smallest
    sub-sample size (aim for >= 20 haloes per cell).

    @ref: 'pooled' (all labelled haloes, the symmetric choice), or one of
          ``split['names']`` to leave that sample unweighted.
    @by_web: False (default) matches the *marginal* (M, delta) distribution.
             Because web type is not conditioned on, the sub-samples keep
             whatever web preference they have *at fixed density* -- which is
             a purely tidal statement and usually the quantity of interest.
             True adds the halo's own web type to the matching cells, which
             also equalises the web-type fractions and therefore removes part
             of the signal; use it only as a deliberately conservative check.
    @w_clip: weights are clipped at this multiple of the sub-sample's median
             weight; ``None`` disables.  Clipping breaks the match slightly
             but stops a handful of haloes from dominating the WCCF.
    @min_ref: mass bins with fewer reference haloes than this are dropped
              (weight 0).

    Returns
    -------
    dict with
        ``weight``   (n,) -- per halo, 0 outside the labelled samples;
                             normalised so each sub-sample sums to 1
        ``label``, ``names``, ``ref``
        ``n_eff``    (ns,) -- (sum w)^2 / sum w^2
        ``n_raw``    (ns,) -- number of haloes carrying a non-zero weight
        ``eff_frac`` (ns,) -- n_eff / n_raw, the S/N cost of the matching
        ``coverage`` (ns,) -- fraction of the reference distribution that the
                              sub-sample could actually cover
        ``w_max_ratio`` (ns,) -- max weight / median weight after clipping
    """
    names = list(split['names'])
    ns = len(names)
    lab, m_bin = split['label'], split['m_bin']
    nm = len(split['lm_edges']) - 1

    d, web, inside = delta_at_halos(tf, h_samp, field=field, interp=interp,
                                    position_key=position_key)
    x = _x_of_delta(d, log)
    ok = inside & np.isfinite(x) & (m_bin >= 0)
    lm = split['lm']

    if ref == 'pooled':
        ref_sel_all = ok & (lab >= 0)
    else:
        if ref not in names:
            raise ValueError(f'ref must be "pooled" or one of {names}')
        ref_sel_all = ok & (lab == names.index(ref))

    w = np.zeros(len(x), dtype=np.float64)
    cover_num = np.zeros(ns)
    cover_den = np.zeros(ns)

    for b in range(nm):
        in_b = ok & (m_bin == b)
        ref_b = in_b & ref_sel_all
        n_ref_b = int(ref_b.sum())
        if n_ref_b < min_ref:
            continue
        def _qbin(vals, ref_vals, k):
            """Equal-count bin index defined by the reference sample."""
            e = np.quantile(ref_vals, np.linspace(0.0, 1.0, k + 1))
            e[0], e[-1] = -np.inf, np.inf
            e = np.unique(e)
            nk = max(len(e) - 1, 1)
            return np.clip(np.searchsorted(e, vals, side='right') - 1,
                           0, nk - 1), nk

        db, nd = _qbin(x, x[ref_b], n_dbins)
        mb, nms = _qbin(lm, lm[ref_b], max(int(n_msub), 1))
        cell = mb * nd + db
        n_cell = nms * nd
        if by_web:
            cell = web * n_cell + cell
            n_cell *= N_WEB

        n_ref = np.bincount(cell[ref_b], minlength=n_cell).astype(float)
        for s in range(ns):
            sel = in_b & (lab == s)
            if not sel.any():
                continue
            n_s = np.bincount(cell[sel], minlength=n_cell).astype(float)
            ratio = np.where(n_s > 0, n_ref / np.maximum(n_s, 1.0), 0.0)
            w_s = ratio[cell[sel]]
            tot = w_s.sum()
            if tot <= 0:
                continue
            w[sel] = w_s * (n_ref_b / tot)   # restore the mass-bin total
            cover_num[s] += n_ref[n_s > 0].sum()
            cover_den[s] += n_ref.sum()

    n_eff = np.zeros(ns)
    n_raw = np.zeros(ns, dtype=np.int64)
    w_ratio = np.zeros(ns)
    for s in range(ns):
        sel = (lab == s) & (w > 0)
        n_raw[s] = int(sel.sum())
        if n_raw[s] == 0:
            continue
        ws = w[sel]
        if w_clip is not None:
            ws = np.minimum(ws, w_clip * np.median(ws))
        ws = ws / ws.sum()
        w[sel] = ws
        n_eff[s] = 1.0 / np.sum(ws**2)
        w_ratio[s] = ws.max() / np.median(ws)

    cov = np.divide(cover_num, np.maximum(cover_den, 1e-30))
    res = {
        'weight': w, 'label': lab, 'names': names, 'ref': ref,
        'n_eff': n_eff, 'n_raw': n_raw,
        'eff_frac': n_eff / np.maximum(n_raw, 1),
        'coverage': cov, 'w_max_ratio': w_ratio,
        'n_dbins': n_dbins, 'w_clip': w_clip,
    }
    if verbose:
        print(f'density_matched_weights: ref={ref}, '
              f'{n_msub} x {n_dbins} (mass, delta) cells per mass bin, '
              f'w_clip={w_clip}')
        for s in range(ns):
            print(f'  {names[s]:<8s} N={n_raw[s]:8d}  N_eff={n_eff[s]:10.1f}'
                  f'  N_eff/N={res["eff_frac"][s]:5.3f}'
                  f'  coverage={cov[s]:5.3f}'
                  f'  w_max/w_med={w_ratio[s]:6.2f}')
    return res


def matched_subsets(h_samp, matched, *, weight_key='weight'):
    """Sub-samples carrying the matched weights, ready for ``HaloWebCross``.

    Returns ``{name: HaloSample}``; ``sample['weight']`` is the matched weight
    (``HaloWebCross`` renormalises it internally).
    """
    w, lab, names = matched['weight'], matched['label'], matched['names']
    out = {}
    for s, nm in enumerate(names):
        sel = (lab == s) & (w > 0)
        sub = h_samp.subset(sel)
        sub.data[weight_key] = w[sel]
        out[nm] = sub
    return out


# ---------------------------------------------------------------------------
# 5. hard density bins  (cross-check)
# ---------------------------------------------------------------------------

def density_bin_labels(h_samp, tf, split, *, n_dbins=3, field=None,
                       interp='ngp', log=True, position_key='position',
                       verbose=True):
    """Equal-count delta bins inside each mass bin.

    The thresholds come from the *combined* old+young sample of that mass bin,
    so the two sub-samples are cut at identical delta values -- otherwise the
    comparison between them reintroduces the very offset being controlled for.

    Returns dict with ``d_bin`` (n,), -1 where undefined, plus ``edges``
    (nm, n_dbins+1) and ``counts`` (nm, ns, n_dbins).
    """
    names = list(split['names'])
    ns = len(names)
    lab, m_bin = split['label'], split['m_bin']
    nm = len(split['lm_edges']) - 1

    d, _web, inside = delta_at_halos(tf, h_samp, field=field, interp=interp,
                                     position_key=position_key)
    x = _x_of_delta(d, log)
    ok = inside & np.isfinite(x) & (m_bin >= 0) & (lab >= 0)

    d_bin = np.full(len(x), -1, dtype=np.int64)
    edges = np.full((nm, n_dbins + 1), np.nan)
    counts = np.zeros((nm, ns, n_dbins), dtype=np.int64)
    for b in range(nm):
        sel = ok & (m_bin == b)
        if sel.sum() < 10 * n_dbins:
            continue
        e = np.quantile(x[sel], np.linspace(0.0, 1.0, n_dbins + 1))
        edges[b] = e
        e_open = e.copy()
        e_open[0], e_open[-1] = -np.inf, np.inf
        db = np.clip(np.searchsorted(e_open, x, side='right') - 1,
                     0, n_dbins - 1)
        d_bin[sel] = db[sel]
        for s in range(ns):
            ss = sel & (lab == s)
            counts[b, s] = np.bincount(d_bin[ss], minlength=n_dbins)

    if verbose:
        print(f'density_bin_labels: {n_dbins} equal-count delta bins per mass '
              f'bin (thresholds from the combined sample)')
        for b in range(nm):
            if counts[b].sum() == 0:
                continue
            lm_e = split['lm_edges']
            print(f'  logM [{lm_e[b]:5.2f}, {lm_e[b+1]:5.2f}): '
                  f'edges={np.array2string(edges[b], precision=2)}  '
                  + '  '.join(f'{names[s]}={counts[b, s]}'
                              for s in range(ns)))
    return {'d_bin': d_bin, 'edges': edges, 'counts': counts,
            'names': names, 'n_dbins': n_dbins, 'log': bool(log)}


def density_bin_subsets(h_samp, split, dbins):
    """``{(name, i_dbin): HaloSample}`` for the hard-bin cross-check."""
    lab, db = split['label'], dbins['d_bin']
    out = {}
    for s, nm in enumerate(dbins['names']):
        for i in range(dbins['n_dbins']):
            sel = (lab == s) & (db == i)
            if sel.sum() == 0:
                continue
            out[(nm, i)] = h_samp.subset(sel)
    return out


# ---------------------------------------------------------------------------
# 6. glue to the WCCF pipeline and to the appendix table
# ---------------------------------------------------------------------------

def halo_number_field(h_samp, tf, *, sel=None, weights=None,
                      weight_key='weight', position_key='position',
                      scheme='cic'):
    """Deposit a (weighted) halo sample on ``tf``'s mesh.

    ``pyccb.process.web_web.WebWeb2DCross`` takes the web cells as centres and
    a *scalar field on the grid* as ``field_vals``, so a halo sub-sample has to
    be gridded before it can be cross-correlated.  The field points of that
    class are ``xs_all[field_mask]`` in C order, hence

    >>> g = halo_number_field(hs_old, tf)
    >>> xi = cross.corrs(g[field_mask])          # same field_mask as the class

    ``corrs`` divides by the mean of ``field_vals``, so the raw weighted counts
    per cell can be passed as they are -- no need to build a contrast.

    @sel: optional boolean mask selecting a sub-sample of `h_samp`.
    @weights: explicit per-halo weights, **aligned to the full sample** (not
              to `sel`); defaults to ``h_samp[weight_key]`` if present, else
              unity.  So pass ``weights=mw['weight'], sel=(label == 1)``, not
              ``weights=mw['weight'][sel]``.
    @scheme: 'cic' (default) or 'ngp'.  NGP is self-consistent with the
             cell-based web classification; CIC is smoother.

    Memory note: a 500^3 grid needs about 2 GB while depositing.
    """
    n, h = wes._grid_spec(tf)
    xs = np.asarray(h_samp[position_key], dtype=np.float64)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
    elif weight_key in h_samp:
        w = np.asarray(h_samp[weight_key], dtype=np.float64)
    else:
        w = np.ones(len(xs), dtype=np.float64)
    if len(w) != len(xs):
        raise ValueError(
            f'weights has length {len(w)} but the sample has {len(xs)} haloes '
            '-- pass full-length weights and use `sel` to subset')
    if sel is not None:
        sel = np.asarray(sel)
        xs, w = xs[sel], w[sel]

    flat = np.zeros(n**3, dtype=np.float64)
    if scheme == 'ngp':
        ijk = wes._cell_index(tf, xs)
        idx = (ijk[:, 0] * n + ijk[:, 1]) * n + ijk[:, 2]
        flat += np.bincount(idx, weights=w, minlength=n**3)
    elif scheme == 'cic':
        t = xs / h                 # nodes at i*h, as in pyhipp's CIC
        i0 = np.floor(t).astype(np.int64)
        f = t - i0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1.0 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1.0 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1.0 - f[:, 2]
                    idx = (((i0[:, 0] + dx) % n) * n
                           + ((i0[:, 1] + dy) % n)) * n \
                        + ((i0[:, 2] + dz) % n)
                    flat += np.bincount(idx, weights=w * wx * wy * wz,
                                        minlength=n**3)
    else:
        raise ValueError("scheme must be 'cic' or 'ngp'")
    return flat.reshape(n, n, n)


def summarize_density_bins(h_samp, tf, split, dbins, *, field=None,
                           interp='ngp', log=True, age_key=None,
                           position_key='position', verbose=True):
    """Tidy per-(mass bin, delta bin, sub-sample) table for the appendix.

    ``density_bin_labels`` returns the labelling, not a result; what belongs in
    a paper is this table: how many haloes each cell holds, what delta range it
    covers, and -- the point of the exercise -- that the old and young cells of
    the same (mass, delta) cell really do have matched densities while still
    differing in age.

    Returns a dict of 1-D arrays (one entry per row) plus ``rows``, a list of
    dicts, so it can go straight into pandas or an AASTeX deluxetable.
    """
    names = list(dbins['names'])
    lab, m_bin = split['label'], split['m_bin']
    lm_e, nd = split['lm_edges'], dbins['n_dbins']
    age_key = split['key'] if age_key is None else age_key
    age = np.asarray(h_samp[age_key], dtype=np.float64)

    d, _web, _in = delta_at_halos(tf, h_samp, field=field, interp=interp,
                                  position_key=position_key)
    x = _x_of_delta(d, log)
    db = dbins['d_bin']

    rows = []
    for b in range(len(lm_e) - 1):
        for i in range(nd):
            for s, nm in enumerate(names):
                sel = (m_bin == b) & (db == i) & (lab == s)
                k = int(sel.sum())
                if k == 0:
                    continue
                rows.append({
                    'm_bin': b, 'lm_lo': lm_e[b], 'lm_hi': lm_e[b + 1],
                    'd_bin': i,
                    'x_lo': dbins['edges'][b, i],
                    'x_hi': dbins['edges'][b, i + 1],
                    'sample': nm, 'n': k,
                    'lm_med': float(np.median(split['lm'][sel])),
                    'x_med': float(np.median(x[sel])),
                    'age_med': float(np.median(age[sel])),
                })
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}
    out['rows'] = rows

    if verbose and rows:
        print(f'summarize_density_bins: {len(rows)} cells '
              f'({age_key} medians shown)')
        print(f'  {"logM":>13s} {"dbin":>5s} {"sample":>8s} {"N":>8s} '
              f'{"med x":>8s} {"med logM":>9s} {"med " + age_key:>10s}')
        for r in rows:
            print(f'  [{r["lm_lo"]:5.2f},{r["lm_hi"]:5.2f}) {r["d_bin"]:5d} '
                  f'{r["sample"]:>8s} {r["n"]:8d} {r["x_med"]:8.3f} '
                  f'{r["lm_med"]:9.3f} {r["age_med"]:10.3f}')
    return out


# ---------------------------------------------------------------------------
# 7. residual mass difference inside each mass bin
# ---------------------------------------------------------------------------

def mass_distribution_by_sample(h_samp, split, *, n_bins=20, weights=None,
                                include_all=True, verbose=True):
    """Mass distribution of each sub-sample *inside* each analysis mass bin.

    Binning in mass does not make old and young identical in mass: ``zf``
    correlates with ``M``, so inside a 0.5 dex bin the young sample sits
    systematically at the high-mass end.  This is the figure that shows how
    large that residual is -- and, if ``weights`` from
    :func:`density_matched_weights` are passed, that the matching removes it.

    Returns a dict with
        ``x``, ``x_edges``   -- offset from the bin's lower edge [dex]
        ``pdf``  (nm, ns, nb)
        ``median``(nm, ns)   -- median log10 M in that bin
        ``d_median`` (nm,)   -- median(Old) - median(Young) [dex]
        ``d_cohen``  (nm,)   -- the same in units of the pooled scatter
        ``n``    (nm, ns)
    """
    names = list(split['names']) + (['All'] if include_all else [])
    ns = len(names)
    lm, lab, m_bin = split['lm'], split['label'], split['m_bin']
    edges_m = split['lm_edges']
    nm = len(edges_m) - 1
    w = (np.ones(len(lm)) if weights is None
         else np.asarray(weights, dtype=np.float64))

    dlm = float(np.median(np.diff(edges_m)))
    xe = np.linspace(0.0, dlm, n_bins + 1)
    xc = 0.5 * (xe[:-1] + xe[1:])

    pdf = np.zeros((nm, ns, n_bins))
    med = np.full((nm, ns), np.nan)
    sig = np.full((nm, ns), np.nan)
    cnt = np.zeros((nm, ns))
    for b in range(nm):
        base = edges_m[b]
        rows = [(lab == s) & (m_bin == b) for s in range(len(split['names']))]
        if include_all:
            rows.append(m_bin == b)
        for s, sel in enumerate(rows):
            if not sel.any() or w[sel].sum() <= 0:
                continue
            v, ws = lm[sel] - base, w[sel]
            h, _ = np.histogram(v, bins=xe, weights=ws)
            tot = h.sum()
            cnt[b, s] = ws.sum()
            if tot > 0:
                pdf[b, s] = h / tot / (xe[1] - xe[0])
            o = np.argsort(v)
            c = np.cumsum(ws[o]) / ws.sum()
            q16, q50, q84 = np.interp([0.16, 0.5, 0.84], c, v[o])
            med[b, s], sig[b, s] = base + q50, 0.5 * (q84 - q16)

    sig_pool = np.sqrt(0.5 * (sig[:, 0]**2 + sig[:, 1]**2))
    res = {
        'x': xc, 'x_edges': xe, 'pdf': pdf, 'median': med, 'sigma': sig,
        'n': cnt, 'names': names, 'lm_edges': edges_m,
        'd_median': med[:, 1] - med[:, 0],
        'd_cohen': (med[:, 1] - med[:, 0]) / np.maximum(sig_pool, 1e-30),
    }
    if verbose:
        print('mass_distribution_by_sample: residual mass offset per bin'
              + ('' if weights is None else ' (weighted)'))
        print(f'  {"logM bin":>16s} ' + ' '.join(f'{nm_:>10s}' for nm_ in names)
              + f'{"d_median":>11s}{"d_cohen":>9s}')
        for b in range(nm):
            if cnt[b].sum() == 0:
                continue
            print(f'  [{edges_m[b]:5.2f}, {edges_m[b+1]:5.2f}) '
                  + ' '.join(f'{med[b, s]:10.3f}' for s in range(ns))
                  + f'{res["d_median"][b]:11.4f}{res["d_cohen"][b]:9.3f}')
    return res


def plot_mass_distribution_by_sample(res, axs=None, *, colors=None,
                                     linestyles=SAMPLE_LINESTYLES,
                                     annotate=True, **line_kw):
    """One panel per analysis mass bin; x is the offset from the bin edge."""
    pdf, names, edges_m = res['pdf'], res['names'], res['lm_edges']
    nm, ns = pdf.shape[0], pdf.shape[1]
    show = [b for b in range(nm) if res['n'][b].sum() > 0]
    if axs is None:
        _, axs = plt.subplots(1, len(show), figsize=(2.6 * len(show), 2.8),
                              sharey=True, constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    cols = colors or ['#5e76ff', '#f33b1f', '0.5']
    kw = dict(lw=1.6)
    kw.update(line_kw)
    for j, b in enumerate(show):
        ax = axs[j]
        for s in range(ns):
            ax.plot(res['x'] + edges_m[b], pdf[b, s],
                    color=cols[s % len(cols)],
                    ls=linestyles[s % len(linestyles)],
                    alpha=1.0 if s < 2 else 0.5, label=names[s], **kw)
        ax.set_title(rf'$\log_{{10}}M \in [{edges_m[b]:.1f}, '
                     rf'{edges_m[b+1]:.1f})$', fontsize=9)
        ax.set_xlabel(r'$\log_{10}(M\,/\,h^{-1}M_\odot)$')
        ax.set_ylim(bottom=0.0)
        if annotate:
            ax.text(0.04, 0.96,
                    rf'$\Delta\tilde{{M}}={res["d_median"][b]:+.3f}$'
                    + '\n' + rf'$d={res["d_cohen"][b]:+.2f}$',
                    transform=ax.transAxes, va='top', fontsize=8)
        if j == 0:
            ax.set_ylabel(r'$p(\log_{10}M)$')
            ax.legend(frameon=False, fontsize=8)
    return axs
