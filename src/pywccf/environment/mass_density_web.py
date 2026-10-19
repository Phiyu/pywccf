"""
Joint one-point statistics of halo mass, local density and local web type --
"fix one, calibrate the other two".

Part of ``pywccf`` (split out of the analysis scripts of the 3D-WCCF paper).

Notation (used consistently in this module and its docstrings)
---------------------------------------------------------------
Per-halo variables
    m = log10(M / h^-1 Msun)       halo mass                       code: lm
    x = log10(1 + delta)           delta at the halo's node,       code: x
                                   Gaussian-smoothed on R_s
    W in {0,1,2,3}                 web type of the halo's node =   code: web
                                   number of eigenvalues >= lambda_th
                                   (void, sheet, filament, knot)
    S                              sub-sample label (old/young/all)

Counts and volumes (un-normalised)
    N(m, x, W)    haloes in the (m, x, W) cell (sum of weights if weighted);
                  dropping an argument sums over it, e.g. N(m, W).
                                                              code: res['T']
    V(x, W)       comoving volume of reconstructed grid nodes with density x
                  and type W  [(h^-1 Mpc)^3];  V_tot = total.

Probabilities (dimensionless; "|" is only ever used for these)
    P( . | . )    halo-weighted: fraction of HALOES
    P_V( . | . )  volume-weighted: fraction of VOLUME (grid nodes)
    p(x | . )     probability density in x [dex^-1] = P / dx
    e.g. P(W | m, x)  fraction of haloes of mass m at density x that are in W
         P(W | S)     web composition of sub-sample S
         P_V(W)       volume filling fraction of W
         P_V(W | x)   fraction of the volume at density x that is W
    p(m, x, W) = N(m, x, W) / (N_tot dm dx) is the halo joint distribution;
    the three figures are its conditional slices p(x, W | m), p(m, W | x),
    p(m, x | W).

Abundances (dimensional; subscripts = "inside the volume of")
    n(m)          = N(m) / (V_tot dm)             total HMF, dn/dlog10 M
    n_W(m)        = N(m, W) / (V(W) dm)           HMF inside web W (Hahn+07 F4)
    n_x(m)        = N(m, x) / (V(x) dm)           HMF inside density x
    n_{W,x}(m)    = N(m, x, W) / (V(x, W) dm)     HMF inside (W, x)
    units h^3 Mpc^-3 dex^-1.  Identity:
        n_{W,x}(m) = n_x(m) * P(W | m, x) / P_V(W | x)

Dependence measures
    rho_ab, rho_ab.c     Spearman / partial Spearman (c held fixed)
    H(W | m)             halo-weighted conditional entropy [bits]
    I(x; W | m)          conditional mutual information [bits]
    U(W; x | m)          = I(x; W | m) / H(W | m) in [0, 1]: fraction of the
                         web-type entropy at fixed m removed by knowing x

Scales
    R_s, lambda_th       smoothing scale and eigenvalue threshold of the T-web
    M_s = (2 pi)^{3/2} R_s^3 rho_m    mass inside the smoothing kernel; a
                         halo raises delta at its own node by M / M_s

The three slices
----------------
* fix m -> x vs W      :func:`plot_fix_mass`
  heat maps of p(m, x | W, m-bin): rows = mass bins, columns = W
  (``norm='column'`` for p(x | m, W)).  Its projection onto x, one panel
  per mass bin with four curves p(x | W, m), is :func:`plot_fix_mass_pdf`.
  The earlier stacked P(W | x, m) version is :func:`plot_fix_mass_fractions`.
* fix x -> m vs W      :func:`plot_fix_density`
  P(W | m, x): flat in m  <=>  W independent of m at fixed x.
  Optional abundance form: :func:`hmf_by_web_fixed_density` /
  :func:`plot_hmf_fixed_density` give n_{W,x}(m) and n_{W,x}(m) / n_x(m);
  :func:`plot_hmf_fixed_web` shows the same table the other way round
  (one panel per W, one curve per density bin, ratio to n_W(m)).
* fix W -> m vs x      :func:`plot_fix_web`
  median x(m) inside W, with the self-contribution reference
  log10(1 + delta_bg + M / M_s).

Everything is computed from one histogram N(m, x, W) on a fine grid, so all
derived quantities get a spatial-jackknife error for free (``n_jk_side > 0``).
Spearman coefficients use mid-ranks from the histogram (W is ordinal); the
mutual informations carry a Miller-Madow correction.  U(W; x | m) and
U(W; m | x) are directly comparable; I(m; x | W), between two binned
continuous variables, depends on the binning.

Example
-------
>>> from pywccf.environment.mass_density_web import *
>>> mdw = mass_density_web(hs, tf, n_jk_side=4)
>>> plot_fix_mass(mdw); plot_fix_density(mdw); plot_fix_web(mdw)
>>> hmf = hmf_by_web_fixed_density(mdw, tf)          # optional
>>> plot_hmf_fixed_density(hmf, kind='ratio')
"""

from __future__ import annotations

import functools
import math
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, FixedLocator, LogLocator,
                               NullFormatter, ScalarFormatter)

from . import web_environment_stats as wes
from .web_environment_stats import WEB_NAMES, WEB_COLORS, N_WEB
from .age_density_control import delta_at_halos

__all__ = ['mass_density_web', 'plot_fix_mass', 'plot_fix_density',
           'plot_fix_web', 'self_mass_scale', 'print_mdw_summary',
           'hmf_by_web_fixed_density', 'plot_hmf_fixed_density',
           'plot_fix_mass_fractions', 'plot_fix_mass_pdf',
           'plot_hmf_fixed_web', 'zf_by_web_fixed_density',
           'plot_zf_fixed_density', 'plot_zf_fixed_web',
           'hmf_sample_ratio', 'plot_zf_fixed_web_single',
           'plot_hmf_fixed_web_single', 'spin_by_web_fixed_density',
           'concentration_by_web_fixed_density', 'plot_spin_fixed_density',
           'plot_spin_fixed_web', 'plot_spin_fixed_web_single',
           'plot_c_fixed_density', 'plot_c_fixed_web',
           'plot_c_fixed_web_single', 'plot_zf_by_density',
           'plot_spin_by_density', 'plot_c_by_density',
           'smoothing_mass_cut', 'FIDUCIAL_R_S',
           'MASS_CUT_OFFSET', 'plot_hmf_by_density']

# ---------------------------------------------------------------------------
# paper figure style (skill "paper-figure-style"): thick frames, large text,
# figure legend on top in <= 2 rows, inward major + minor ticks on all four
# sides, grid at major ticks only
# ---------------------------------------------------------------------------
PAPER_RC = {
    'font.size': 13, 'axes.labelsize': 15, 'axes.titlesize': 14,
    'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 12,
    'axes.linewidth': 1.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.major.size': 6.0, 'ytick.major.size': 6.0,
    'xtick.minor.size': 3.2, 'ytick.minor.size': 3.2,
    'xtick.major.width': 1.3, 'ytick.major.width': 1.3,
    'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
    'grid.alpha': 0.25, 'grid.linewidth': 0.7,
    'legend.frameon': False,
}


def style_axis(ax):
    """Paper tick/grid/frame rules on one axis, whatever its scale."""
    for axis, scale in ((ax.xaxis, ax.get_xscale()),
                        (ax.yaxis, ax.get_yscale())):
        if scale == 'linear' and not isinstance(axis.get_minor_locator(),
                                                FixedLocator):
            axis.set_minor_locator(AutoMinorLocator())
        elif scale == 'log':
            axis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
            axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=PAPER_RC['xtick.labelsize'])
    ax.tick_params(which='major', length=6.0, width=1.3)
    ax.tick_params(which='minor', length=3.2, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(PAPER_RC['axes.linewidth'])
    ax.grid(True, which='major', alpha=0.25, linewidth=0.7)
    ax.grid(False, which='minor')


def _is_range(v):
    """True for one (lo, hi) pair of numbers."""
    return (isinstance(v, (tuple, list, np.ndarray)) and len(v) == 2
            and all(isinstance(x, (int, float, np.integer, np.floating))
                    for x in v))


def _apply_xlim(out, xlim, cells, kwargs):
    """Per-panel x ranges.  ``out`` are the axes a plot returned; a panel
    ("cell") is one column of a (main, ratio) pair, or one axis.

    xlim : (lo, hi) for every panel; a sequence with one (lo, hi) or None
           per panel, in panel order; or a dict {panel index: (lo, hi)} --
           for the fixed-web plots the keys may also be web names
           ('void', 'sheet', 'filament', 'knot').  None = automatic.
    """
    arr = np.asarray(out, dtype=object)
    if cells == 'single':
        groups = [list(np.ravel(arr))]
    elif arr.ndim == 2:                       # (main row, ratio row)
        groups = [list(arr[:, c]) for c in range(arr.shape[1])]
    else:
        groups = [[a] for a in np.ravel(arr)]
    n = len(groups)
    names = ([WEB_NAMES[int(w)].lower()
              for w in kwargs.get('webs', range(N_WEB))]
             if cells == 'webs' else None)
    if _is_range(xlim):
        per = [xlim] * n
    elif isinstance(xlim, dict):
        per = [None] * n
        for key, lim in xlim.items():
            if isinstance(key, str):
                if names is None or key.lower() not in names:
                    raise ValueError(
                        f'xlim key {key!r}: use panel indices 0..{n - 1}'
                        + (f' or web names {names}' if names else ''))
                idx = names.index(key.lower())
            else:
                idx = int(key)
                if not -n <= idx < n:
                    raise ValueError(f'xlim key {key}: only {n} panels')
            per[idx] = lim
    else:
        per = list(xlim)
        if len(per) != n:
            raise ValueError(f'xlim gives {len(per)} ranges for {n} panels '
                             '(pass one (lo, hi), one per panel, or a dict)')
    for group, lim in zip(groups, per):
        if lim is None:
            continue
        if not _is_range(lim):
            raise ValueError(f'xlim entry {lim!r} is not a (lo, hi) pair')
        for a in group:
            if a is not None and hasattr(a, 'set_xlim'):
                a.set_xlim(float(lim[0]), float(lim[1]))


# gaps between panels of the figures these plots create (constrained
# layout): pads in inch around each panel, spaces as a fraction of the
# panel size (matplotlib defaults: 3/72 inch and 0.02)
PANEL_PAD = 2.0 / 72.0
PANEL_GAP = 0.01


def _tighten(out, panel_gap):
    """Smaller gaps between panels: panel_gap None = (PANEL_GAP, PANEL_GAP),
    a number = the same fraction for width and height, (wspace, hspace)."""
    if panel_gap is None:
        wspace = hspace = PANEL_GAP
    elif np.isscalar(panel_gap):
        wspace = hspace = float(panel_gap)
    else:
        wspace, hspace = (float(v) for v in panel_gap)
    figs = {getattr(a, 'figure', None)
            for a in np.ravel(np.asarray(out, dtype=object))} - {None}
    for fig in figs:
        engine = fig.get_layout_engine()
        if engine is not None and hasattr(engine, 'set'):
            try:
                engine.set(w_pad=PANEL_PAD, h_pad=PANEL_PAD,
                           wspace=wspace, hspace=hspace)
            except TypeError:                 # not a constrained layout
                pass


def _paper_figure(cells='columns'):
    """Decorator of the plot engines: draw inside the paper rc context,
    apply the optional per-panel ``xlim`` (see :func:`_apply_xlim`; cells =
    'columns', 'webs' or 'single'), narrow the gaps between panels of a
    figure the plot created (``panel_gap``, see :func:`_tighten`; figures
    passed in through ``axs`` are left alone), then style_axis on every
    axis of the figure(s) the plot used."""
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, xlim=None, panel_gap=None, **kwargs):
            axs_in = kwargs.get('axs', args[2] if len(args) > 2 else None)
            with mpl.rc_context(PAPER_RC):
                out = func(*args, **kwargs)
                if xlim is not None:
                    _apply_xlim(out, xlim, cells, kwargs)
                if axs_in is None:
                    _tighten(out, panel_gap)
                _trim_edge_labels(out)
                _style_figures(out)
            return out
        return wrapper
    return decorate


class _EdgeTrimFormatter(ScalarFormatter):
    """x tick labels that would stick out of the panel on a side facing a
    neighbouring panel are left blank, so that the layout does not have to
    open a gap for them (the outer sides of the figure keep theirs)."""

    def __init__(self, trim_left, trim_right):
        super().__init__()
        self.trim_left, self.trim_right = trim_left, trim_right

    def __call__(self, x, pos=None):
        text = super().__call__(x, pos)
        if not text or not (self.trim_left or self.trim_right):
            return text
        lo, hi = sorted(self.axis.get_view_interval())
        ax = self.axis.axes
        width = ax.bbox.width * 72.0 / ax.figure.dpi          # pt
        if hi <= lo or width <= 0:
            return text
        size = PAPER_RC['xtick.labelsize']                    # set by style_axis
        half = 0.5 * 0.62 * size * len(text)                  # pt
        at = (x - lo) / (hi - lo) * width
        if (self.trim_left and at < half) or (
                self.trim_right and width - at < half):
            return ''
        return text


def _trim_edge_labels(out):
    """Install _EdgeTrimFormatter on the linear x axes of the panels a plot
    returned, trimming only the sides that face another panel."""
    for a in np.ravel(np.asarray(out, dtype=object)):
        if not hasattr(a, 'get_subplotspec') or a.get_xscale() != 'linear':
            continue
        spec = a.get_subplotspec()
        if spec is None:
            continue
        top = spec.get_topmost_subplotspec()
        a.xaxis.set_major_formatter(_EdgeTrimFormatter(
            trim_left=not top.is_first_col(),
            trim_right=not top.is_last_col()))


def _style_figures(out):
    """style_axis on every axis of the figure(s) holding ``out``."""
    figs = []
    for a in np.ravel(np.asarray(out, dtype=object)):
        fig = getattr(a, 'figure', None)
        if fig is not None and fig not in figs:
            figs.append(fig)
    for fig in figs:
        for a in fig.axes:
            style_axis(a)


RHO_CRIT = 2.775e11          # h^2 Msun Mpc^-3  ->  rho_m = RHO_CRIT * Omega_m
                             # in h^-1 Msun / (h^-1 Mpc)^3
FIDUCIAL_R_S = 2.0           # h^-1 Mpc, fiducial T-web smoothing scale
                             # (log10 M_s = 12.95 for Omega_m = 0.258)
MASS_CUT_OFFSET = -0.5       # dex: plot-time cut log10 M_h < log10 M_s
                             # + MASS_CUT_OFFSET + log10(1 + delta)


def self_mass_scale(R_s=FIDUCIAL_R_S, omega_m=0.258, kernel='gaussian'):
    """Mass enclosed by the smoothing kernel, M_s [h^-1 Msun].

    A point mass M at the kernel centre raises the smoothed delta there by
    M / M_s.  Gaussian: (2 pi)^{3/2} R^3 rho_m ; top-hat: 4 pi/3 R^3 rho_m.
    Default Omega_m = 0.258 (ELUCID / WMAP5).
    """
    rho = RHO_CRIT * float(omega_m)
    if kernel == 'gaussian':
        return (2.0 * np.pi)**1.5 * R_s**3 * rho
    if kernel == 'tophat':
        return 4.0 * np.pi / 3.0 * R_s**3 * rho
    raise ValueError("kernel must be 'gaussian' or 'tophat'")


# ---------------------------------------------------------------------------
# histogram statistics
# ---------------------------------------------------------------------------

def _entropy_bits(c):
    """Plug-in entropy [bits], number of occupied cells, total weight."""
    c = np.asarray(c, dtype=np.float64).ravel()
    c = c[c > 0]
    tot = c.sum()
    if tot <= 0:
        return 0.0, 0, 0.0
    p = c / tot
    return float(-(p * np.log2(p)).sum()), len(c), float(tot)


def _cond_mi(T, a, b, c):
    """I(A;B|C) [bits] of a 3-D table with axes (a, b, c) given as indices,
    Miller-Madow corrected.  Also returns H(B|C) for the uncertainty
    coefficient U(B; A | C) = I / H(B|C)."""
    T = np.moveaxis(np.asarray(T, dtype=np.float64), (a, b, c), (0, 1, 2))
    h_abc, k_abc, n = _entropy_bits(T)
    if n <= 0:
        return np.nan, np.nan
    h_ac, k_ac, _ = _entropy_bits(T.sum(1))
    h_bc, k_bc, _ = _entropy_bits(T.sum(0))
    h_c, k_c, _ = _entropy_bits(T.sum((0, 1)))
    mi = h_ac + h_bc - h_abc - h_c
    # Miller-Madow: H_MM = H + (K-1) / (2 N ln 2)
    mi += ((k_ac - 1) + (k_bc - 1) - (k_abc - 1) - (k_c - 1)) / (
        2.0 * n * np.log(2.0))
    h_b_given_c = (h_bc + (k_bc - 1) / (2.0 * n * np.log(2.0))) - (
        h_c + (k_c - 1) / (2.0 * n * np.log(2.0)))
    return float(mi), float(h_b_given_c)


def _spearman_2d(H):
    """Spearman rho of a weighted 2-D contingency table (mid-ranks)."""
    H = np.asarray(H, dtype=np.float64)
    N = H.sum()
    if N <= 0:
        return np.nan
    na, nb = H.sum(1), H.sum(0)
    ra, rb = np.cumsum(na) - 0.5 * na, np.cumsum(nb) - 0.5 * nb
    da = ra - (na * ra).sum() / N
    db = rb - (nb * rb).sum() / N
    va, vb = (na * da**2).sum() / N, (nb * db**2).sum() / N
    if va <= 0 or vb <= 0:
        return np.nan
    return float((H * da[:, None] * db[None, :]).sum() / N / np.sqrt(va * vb))


def _partial(r_ab, r_ac, r_bc):
    den = np.sqrt(max((1 - r_ac**2) * (1 - r_bc**2), 1e-30))
    return float((r_ab - r_ac * r_bc) / den)


def _spearman_3(T):
    """Pairwise and partial Spearman of a 3-D table with axes (M, x, W)."""
    r_mx = _spearman_2d(T.sum(2))
    r_mw = _spearman_2d(T.sum(1))
    r_xw = _spearman_2d(T.sum(0))
    return {
        'Mx': r_mx, 'MW': r_mw, 'xW': r_xw,
        'Mx.W': _partial(r_mx, r_mw, r_xw),
        'MW.x': _partial(r_mw, r_mx, r_xw),
        'xW.M': _partial(r_xw, r_mx, r_mw),
    }


def _hist_quantiles(h, edges, q):
    tot = h.sum()
    if tot <= 0:
        return np.full(len(q), np.nan)
    cdf = np.concatenate([[0.0], np.cumsum(h) / tot])
    return np.interp(q, cdf, edges)


def _standardise_fixx(T, x_pan, n_xp, ratio_to=None, ref_rows=None):
    r"""Direct standardisation of the fix-x slice inside each x panel.

    A panel is a range of x, not a value: inside it, heavier haloes sit at
    higher x, so P(W | m, panel) inherits an m-dependence from the x-mixture
    even when W is independent of m at fixed x.  Every mass bin is therefore
    re-mixed to the panel's pooled x-distribution w_q(x) (halo-weighted, all
    masses):

        P_std (W | m, q) = sum_x w_q(x) P(W | m, x)        / c(m)
        P_null(W | m, q) = sum_x w_q(x) P(W | x; ref)      / c(m)

    with sums over the fine x bins populated by both mass bin m and the
    reference, and c(m) = the w_q-weight of those bins (coverage).
    P(W | x; ref) is measured on the reference mass rows ``ref_rows``
    (default: all masses).  P_null is therefore what a halo of mass m would
    show if it sat in the same web as a reference halo at the same x; the
    solid-vs-dashed difference is the genuine m-W dependence at fixed x.
    Using a low-mass reference keeps the null from being pulled by the very
    haloes being tested when a web type is rare at that density.

    With ``ratio_to`` = P_V(W | x) on the fine grid (n_x, 4), both are
    computed for P(W | m, x) / P_V(W | x) instead, i.e. for
    n_{W,x}(m) / n_x(m).
    """
    n_mf = T.shape[0]
    std = np.full((n_xp, n_mf, N_WEB), np.nan)
    null = np.full((n_xp, n_mf, N_WEB), np.nan)
    cov = np.full((n_xp, n_mf), np.nan)
    for q in range(n_xp):
        sel = x_pan == q
        Tq = T[:, sel, :]                                  # (m, x_q, W)
        Nx = Tq.sum((0, 2))
        if Nx.sum() <= 0:
            continue
        wq = Nx / Nx.sum()                                 # w_q(x)
        Tref = Tq if ref_rows is None else Tq[ref_rows]
        Nx_ref = Tref.sum((0, 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            Pw_x = Tref.sum(0) / Nx_ref[:, None]           # P(W | x; ref)
            Nmx = Tq.sum(2)
            Pw_mx = Tq / Nmx[..., None]                    # P(W | m, x)
            if ratio_to is not None:
                pv = ratio_to[sel]                         # P_V(W | x)
                Pw_x = Pw_x / pv
                Pw_mx = Pw_mx / pv[None]
        covd = (Nmx > 0) & np.isfinite(Pw_x).all(1)[None, :]
        wc = wq[None, :] * covd                            # (m, x_q)
        c = wc.sum(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            std[q] = np.nansum(wc[..., None] * np.nan_to_num(Pw_mx), 1) \
                / c[:, None]
            null[q] = (wc[..., None] * np.nan_to_num(Pw_x)[None]).sum(1) \
                / c[:, None]
        std[q][c <= 0] = np.nan
        null[q][c <= 0] = np.nan
        cov[q] = c
    return std, null, cov


def _stats(T, ctx):
    """Every derived quantity of the 3-D table T[M, x, W]; returns a dict of
    float arrays so that the jackknife can treat all of them uniformly."""
    lm_e, x_e = ctx['lm_edges'], ctx['x_edges']
    mp, xp = ctx['m_panel_of_bin'], ctx['x_panel_of_bin']
    n_mp, n_xp = ctx['n_mp'], ctx['n_xp']
    n_mf, n_xf = T.shape[0], T.shape[1]
    dx = np.diff(x_e)

    out = {}
    # ---- fix M: P(W | x, M-panel), p(x | W, M-panel) ----------------------
    Tm = np.zeros((n_mp, n_xf, N_WEB))
    for p in range(n_mp):
        Tm[p] = T[mp == p].sum(0)
    nx = Tm.sum(2, keepdims=True)
    out['fixM_pw_x'] = np.where(nx > 0, Tm / np.maximum(nx, 1e-30), np.nan)
    nw = Tm.sum(1, keepdims=True)
    out['fixM_px_w'] = np.where(nw > 0, Tm / np.maximum(nw, 1e-30)
                                / dx[None, :, None], np.nan)
    out['fixM_n_x'] = nx[..., 0]
    u, r = np.full(n_mp, np.nan), np.full(n_mp, np.nan)
    for p in range(n_mp):
        sub = T[mp == p]                       # (m_in_panel, x, W)
        if sub.sum() > 0:
            mi, hw = _cond_mi(sub, 1, 2, 0)    # I(x;W|M)
            u[p] = mi / hw if hw > 0 else np.nan
            r[p] = _spearman_3(sub)['xW.M']
    out['fixM_U'], out['fixM_rho'] = u, r

    # ---- fix x: f_W(M | x-panel) ------------------------------------------
    Tx = np.zeros((n_xp, n_mf, N_WEB))
    for q in range(n_xp):
        Tx[q] = T[:, xp == q].sum(1)
    nm = Tx.sum(2, keepdims=True)
    out['fixX_fw_m'] = np.where(nm > 0, Tx / np.maximum(nm, 1e-30), np.nan)
    out['fixX_n_m'] = nm[..., 0]
    u, r = np.full(n_xp, np.nan), np.full(n_xp, np.nan)
    for q in range(n_xp):
        sub = T[:, xp == q]                    # (M, x_in_panel, W)
        if sub.sum() > 0:
            mi, hw = _cond_mi(sub, 0, 2, 1)    # I(M;W|x)
            u[q] = mi / hw if hw > 0 else np.nan
            r[q] = _spearman_3(sub)['MW.x']
    out['fixX_U'], out['fixX_rho'] = u, r
    std, null, cov = _standardise_fixx(T, xp, n_xp,
                                       ref_rows=ctx.get('ref_rows'))
    out['fixX_fw_m_std'], out['fixX_fw_m_null'], out['fixX_cov'] = std, null, cov

    # ---- fix W: quantiles of x vs M ---------------------------------------
    qx = np.full((N_WEB, n_mf, 3), np.nan)
    for w in range(N_WEB):
        for m in range(n_mf):
            qx[w, m] = _hist_quantiles(T[m, :, w], x_e, [0.16, 0.5, 0.84])
    out['fixW_qx'] = qx
    out['fixW_n_m'] = T.sum(1).T                # (W, M)
    r = np.full(N_WEB, np.nan)
    mi_w = np.full(N_WEB, np.nan)
    for w in range(N_WEB):
        r[w] = _spearman_2d(T[:, :, w])
        h_mx, k_mx, n = _entropy_bits(T[:, :, w])
        if n > 0:
            h_m, k_m, _ = _entropy_bits(T[:, :, w].sum(1))
            h_x, k_x, _ = _entropy_bits(T[:, :, w].sum(0))
            mi_w[w] = (h_m + h_x - h_mx
                       + ((k_m - 1) + (k_x - 1) - (k_mx - 1))
                       / (2.0 * n * np.log(2.0)))
    out['fixW_rho'], out['fixW_I'] = r, mi_w

    # ---- global -----------------------------------------------------------
    sp = _spearman_3(T)
    for k, v in sp.items():
        out['rho_' + k] = np.array(v)
    mi_xw_m, hw_m = _cond_mi(T, 1, 2, 0)
    mi_mw_x, hw_x = _cond_mi(T, 0, 2, 1)
    mi_mx_w, _ = _cond_mi(T, 0, 1, 2)
    out['I_xW.M'] = np.array(mi_xw_m)
    out['I_MW.x'] = np.array(mi_mw_x)
    out['I_Mx.W'] = np.array(mi_mx_w)
    out['U_W_x.M'] = np.array(mi_xw_m / hw_m if hw_m > 0 else np.nan)
    out['U_W_M.x'] = np.array(mi_mw_x / hw_x if hw_x > 0 else np.nan)
    out['f_web'] = T.sum((0, 1)) / max(T.sum(), 1e-30)
    return out


def _snap(edges_coarse, edges_fine):
    """Snap coarse panel edges onto the fine grid and return, for each fine
    bin, the panel it belongs to (-1 if none)."""
    e = np.array([edges_fine[np.argmin(np.abs(edges_fine - v))]
                  for v in edges_coarse])
    e = np.unique(e)
    c = 0.5 * (edges_fine[:-1] + edges_fine[1:])
    pan = np.digitize(c, e) - 1
    pan[(pan < 0) | (pan >= len(e) - 1)] = -1
    return e, pan


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def _resolve_R_s(R_s, tf, h_samp):
    """Smoothing scale of the field: explicit > tf attribute > halo_env meta
    > fiducial.  Returns (R_s, where it came from)."""
    if R_s is not None:
        return float(R_s), 'given'
    for a in ('R_s', 'r_s', 'smoothing_scale', 'R_smooth'):
        v = getattr(tf, a, None) if tf is not None else None
        if v is not None and np.ndim(v) == 0:
            return float(v), f'tf.{a}'
    v = (getattr(h_samp, 'meta', None) or {}).get('env_R_s')
    if v is not None:
        return float(v), "meta['env_R_s']"
    warnings.warn(f'mass_density_web: smoothing scale not given and not found '
                  f'in tf or h_samp.meta (HaloSample.subset drops .meta); '
                  f'assuming the fiducial R_s = {FIDUCIAL_R_S:g} h^-1 Mpc '
                  f'(log10 M_s = {np.log10(self_mass_scale(FIDUCIAL_R_S)):.3f}'
                  f').  Pass R_s= if the field is smoothed on another scale: '
                  f'it sets the plot-time mass cut.', stacklevel=3)
    return FIDUCIAL_R_S, 'fiducial default'


def mass_density_web(h_samp, tf, *, sel=None, weights=None,
                     mass_key='mass', log_mass=True,
                     lm_range=(11.5, 15.0), dlm=0.1,
                     x_range=(-1.5, 2.0), dx=0.05,
                     lm_panels=None, x_panels=None, n_x_panels=5,
                     field=None, interp='ngp', position_key='position',
                     n_jk_side=0, R_s=None, omega_m=0.258, kernel='gaussian',
                     ref_lm=(11.5, 12.0), l_box=None, verbose=True):
    """Joint (M, delta, W) statistics of a halo sample.

    Parameters
    ----------
    h_samp : HaloSample
    tf : TidalField, or None to use the ``env_*`` columns stored by
          ``halo_env`` (NGP on the original field only).
    sel : bool array, optional  -- restrict to a sub-sample (e.g. old haloes,
          so that the same machinery compares the triple structure of two
          sub-samples).
    weights : per-halo weights aligned to `h_samp`, optional.  Not read from
          ``h_samp['weight']`` implicitly, so stale matching weights stored in
          the sample cannot leak into an age-free test.
    lm_range, dlm : fine log-mass grid [dex] -- the mass bins of every
          curve of the HMF / z_f functions (they inherit it; re-run this
          function to change it).  Haloes outside are dropped.
    x_range, dx : fine log10(1+delta) grid.  Values outside are clipped into
          the end bins, so no halo is lost.
    lm_panels : mass-panel edges for "fix M" (default 0.5 dex from 11.5 to
          14.5).  Snapped to the fine grid.
    x_panels : density-panel edges for "fix x".  Default: ``n_x_panels``
          equal-count panels of the sample, snapped to the fine grid.
    field, interp : as in ``age_density_control.delta_at_halos``.
    n_jk_side : spatial jackknife with ``n_jk_side**3`` sub-cubes (only the
          occupied ones are used).  0 = no errors.
    R_s, omega_m, kernel : smoothing scale [h^-1 Mpc] of the density/T-web
          field in ``tf`` and the cosmology; they set M_s = (2 pi)^{3/2}
          R_s^3 rho_m, used by the self-contribution reference curve and by
          the plot-time mass cut log10 M_h < log10 M_s - 0.5 + log10(1 +
          delta) of every HMF / z_f / spin / c plot.  R_s=None: ``tf.R_s`` if the field carries it, else
          ``h_samp.meta['env_R_s']`` (halo_env), else FIDUCIAL_R_S = 2.
    ref_lm : log-mass range whose P(W | x) defines the null curve of the
          standardised fix-x slice (None = all masses).
    l_box : box size for the jackknife sub-cubes when ``tf`` is None
          (default: ``h_samp.meta['env_l_box']`` if present).

    Returns
    -------
    dict with the fine grids, panel edges, the table ``T`` (M, x, W), every
    quantity listed in the module docstring as ``res[key]`` and, with
    jackknife, ``res[key + '_err']``; plus ``M_s`` and the per-halo arrays
    ``lm``, ``x``, ``web`` actually used.
    """
    R_s, R_s_from = _resolve_R_s(R_s, tf, h_samp)
    lm = np.asarray(h_samp[mass_key], dtype=np.float64)
    if not log_mass:
        lm = np.log10(lm)
    d, web, inside = delta_at_halos(tf, h_samp, field=field, interp=interp,
                                    position_key=position_key)
    x = np.log10(np.maximum(1.0 + d, 1e-10))
    w = (np.ones(len(lm)) if weights is None
         else np.asarray(weights, dtype=np.float64))
    if len(w) != len(lm):
        raise ValueError('weights must be aligned to the full sample')

    ok = inside & np.isfinite(lm) & np.isfinite(x) & (w > 0)
    ok &= (lm >= lm_range[0]) & (lm < lm_range[1])
    if sel is not None:
        ok &= np.asarray(sel, dtype=bool)

    lm_e = np.arange(lm_range[0], lm_range[1] + 0.5 * dlm, dlm)
    x_e = np.arange(x_range[0], x_range[1] + 0.5 * dx, dx)
    # if dlm does not divide lm_range the last edge falls short of
    # lm_range[1]: drop the haloes above it instead of clipping them into
    # the last bin
    ok &= lm < lm_e[-1]
    n_mf, n_xf = len(lm_e) - 1, len(x_e) - 1
    im = np.clip(np.floor((lm[ok] - lm_e[0]) / dlm).astype(np.int64),
                 0, n_mf - 1)
    ix = np.clip(np.floor((x[ok] - x_e[0]) / dx).astype(np.int64),
                 0, n_xf - 1)
    iw = web[ok].astype(np.int64)
    ww = w[ok]

    # jackknife regions
    n_jk = max(1, int(n_jk_side)**3)
    if n_jk > 1:
        if tf is not None:
            n, h = wes._grid_spec(tf)
            lb = n * h
        else:
            lb = l_box if l_box is not None else (
                getattr(h_samp, 'meta', None) or {}).get('env_l_box')
            if lb is None:
                raise ValueError('jackknife with tf=None needs the box size: '
                                 'pass l_box= (HaloSample.subset drops .meta)')
        xs = np.asarray(h_samp[position_key], dtype=np.float64)[ok]
        jk = wes._jk_index_of_pos(xs, float(lb), n_jk_side)
        jk_lb = float(lb)
    else:
        jk = np.zeros(len(im), dtype=np.int64)
        jk_lb = None

    size = n_mf * n_xf * N_WEB
    key = ((im * n_xf) + ix) * N_WEB + iw
    Tjk = np.zeros((n_jk, n_mf, n_xf, N_WEB))
    for k in range(n_jk):
        s = jk == k
        if s.any():
            Tjk[k] = np.bincount(key[s], weights=ww[s],
                                 minlength=size).reshape(n_mf, n_xf, N_WEB)
    T = Tjk.sum(0)

    # panels
    if lm_panels is None:
        lm_panels = np.arange(11.5, 14.5 + 1e-6, 0.5)
    lm_p, m_pan = _snap(np.asarray(lm_panels, float), lm_e)
    if x_panels is None:
        xq = np.quantile(x[ok], np.linspace(0.0, 1.0, n_x_panels + 1))
        x_panels = xq
    x_p, x_pan = _snap(np.asarray(x_panels, float), x_e)

    lm_c = 0.5 * (lm_e[:-1] + lm_e[1:])
    ref_rows = (None if ref_lm is None
                else (lm_c >= ref_lm[0]) & (lm_c < ref_lm[1]))
    ctx = {'lm_edges': lm_e, 'x_edges': x_e,
           'm_panel_of_bin': m_pan, 'x_panel_of_bin': x_pan,
           'n_mp': len(lm_p) - 1, 'n_xp': len(x_p) - 1,
           'ref_rows': ref_rows}
    res = _stats(T, ctx)

    if n_jk > 1:
        occ = [k for k in range(n_jk) if Tjk[k].sum() > 0]
        samples = [_stats(T - Tjk[k], ctx) for k in occ]
        nj = len(samples)
        for kk in list(res):
            arr = np.array([s_[kk] for s_ in samples], dtype=np.float64)
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                mean = np.nanmean(arr, axis=0)
                res[kk + '_err'] = np.sqrt((nj - 1.0) / nj * np.nansum(
                    (arr - mean)**2, axis=0))
        res['n_jk'] = nj
    else:
        res['n_jk'] = 0

    res.update({
        'T': T, 'lm_edges': lm_e, 'x_edges': x_e,
        'lm_c': 0.5 * (lm_e[:-1] + lm_e[1:]),
        'x_c': 0.5 * (x_e[:-1] + x_e[1:]),
        'lm_panels': lm_p, 'x_panels': x_p,
        'M_s': self_mass_scale(R_s, omega_m, kernel),
        'R_s': R_s, 'R_s_from': R_s_from, 'omega_m': omega_m,
        'kernel': kernel,
        'lm': lm[ok], 'x': x[ok], 'web': iw, 'w': ww,
        'n_halo': int(ok.sum()),
        # kept for hmf_by_web_fixed_density (volume normalisation + jackknife)
        'Tjk': Tjk, 'jk_side': int(n_jk_side), 'jk_l_box': jk_lb,
        # per-halo bookkeeping so that other halo properties (e.g. z_f) can be
        # binned into exactly the same (m, x, W, jackknife) cells later
        'idx': np.flatnonzero(ok), 'n_all': int(len(ok)),
        'im': im.astype(np.int32),
        'ix': ix.astype(np.int32), 'jk': np.asarray(jk, dtype=np.int32),
        'mass_key': mass_key, 'log_mass': bool(log_mass),
        'field_default': field is None, 'interp': interp,
        'ref_lm': ref_lm, 'ref_rows': ref_rows,
    })
    if verbose:
        print_mdw_summary(res)
    return res


def print_mdw_summary(res):
    """Compact text summary of :func:`mass_density_web`."""
    e = (lambda k: f' +- {res[k + "_err"]:.3f}' if k + '_err' in res
         and np.ndim(res[k + '_err']) == 0 else '')
    print(f'mass_density_web: N = {res["n_halo"]}, '
          f'jackknife regions = {res["n_jk"]}, '
          f'log10 M_s = {np.log10(res["M_s"]):.3f} '
          f'({res["kernel"]}, R_s = {res["R_s"]:g} h^-1 Mpc'
          + (f', {res["R_s_from"]}' if res.get('R_s_from') else '') + ')')
    print('  f_web = ' + np.array2string(res['f_web'], precision=3))
    print('  Spearman   unconditional     partial')
    for a, b in (('Mx', 'Mx.W'), ('MW', 'MW.x'), ('xW', 'xW.M')):
        print(f'    {a:>3s}     {float(res["rho_" + a]):+.3f}{e("rho_" + a):<10s}'
              f'  {b:>5s} {float(res["rho_" + b]):+.3f}{e("rho_" + b)}')
    print(f'  I(x;W|M) = {float(res["I_xW.M"]):.4f} bits   '
          f'U(W;x|M) = {float(res["U_W_x.M"]):.3f}{e("U_W_x.M")}')
    print(f'  I(M;W|x) = {float(res["I_MW.x"]):.4f} bits   '
          f'U(W;M|x) = {float(res["U_W_M.x"]):.3f}{e("U_W_M.x")}')
    print(f'  I(M;x|W) = {float(res["I_Mx.W"]):.4f} bits  (binning-dependent)')

    lp, xp = res['lm_panels'], res['x_panels']
    print('  fix M:   ' + '  '.join(
        f'[{lp[p]:.1f},{lp[p+1]:.1f}) U={res["fixM_U"][p]:.3f} '
        f'rho={res["fixM_rho"][p]:+.3f}' for p in range(len(lp) - 1)))
    print('  fix x:   ' + '  '.join(
        f'[{xp[q]:+.2f},{xp[q+1]:+.2f}) U={res["fixX_U"][q]:.3f} '
        f'rho={res["fixX_rho"][q]:+.3f}' for q in range(len(xp) - 1)))
    print('  fix W:   ' + '  '.join(
        f'{WEB_NAMES[w]} rho={res["fixW_rho"][w]:+.3f}'
        for w in range(N_WEB)))


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _panel_axes(axs, n, width=2.7, height=2.8, sharey=True):
    if axs is None:
        _, axs = plt.subplots(1, n, figsize=(width * n, height),
                              sharey=sharey, constrained_layout=True)
    return np.atleast_1d(axs).ravel()


def _fig_legend(axs, handles=None, labels=None, ncol=4):
    """One legend for the whole row, above the panels."""
    fig = np.atleast_1d(axs).ravel()[0].figure
    if handles is None:                  # collect over ALL panels, de-duplicated,
        seen = {}                        # web types first in their fixed order
        for a in np.atleast_1d(axs).ravel():
            for hnd, lab in zip(*a.get_legend_handles_labels()):
                seen.setdefault(lab, hnd)
        order = [n for n in WEB_NAMES if n in seen] + \
                [n for n in seen if n not in WEB_NAMES]
        handles, labels = [seen[n] for n in order], order
    ncol = max(int(ncol), math.ceil(len(labels) / 2))   # <= 2 rows
    try:                                   # matplotlib >= 3.7
        fig.legend(handles, labels, loc='outside upper center', ncol=ncol,
                   frameon=False, fontsize=12, columnspacing=1.0,
                   handlelength=1.8)
    except (ValueError, TypeError):
        fig.legend(handles, labels, loc='upper center', ncol=ncol,
                   frameon=False, fontsize=12, columnspacing=1.0,
                   handlelength=1.8, bbox_to_anchor=(0.5, 1.08))


def _fmt_err(res, key, idx, fmt='{:.3f}'):
    v = res[key][idx]
    s = fmt.format(v)
    if key + '_err' in res:
        s += r'\pm' + fmt.format(res[key + '_err'][idx]).lstrip('+')
    return s


def plot_fix_mass(res, axs=None, *, mass_panels=None, norm='joint',
                  sharey='row', n_m=25, n_x=40, cmap='Blues',
                  log_color=False, n_min=20, n_scatter=300, vmax_pct=99.5,
                  median=True, annotate=True):
    r"""Fix m: the (m, x) distribution of the haloes of each web type, inside
    each mass bin -- a grid of heat maps, rows = mass bins, columns = W.

    Reading the figure
    ------------------
    * across a row (fixed mass bin): how the density distribution differs
      between web types at the same mass -- the x-W relation at fixed m;
    * inside a panel: whether x still depends on m within the bin for that
      web type (the running median, and rho_mx annotated);
    * down a column: how that relation moves with mass.

    Parameters
    ----------
    res : output of :func:`mass_density_web` (uses its per-halo m, x, W and
          weights, so the heat-map resolution is independent of the fine grid).
    mass_panels : indices of ``res['lm_panels']`` bins to show (default all
          populated ones).
    norm : 'joint'  -> p(m, x | W, bin)  [dex^-2]; every panel integrates to 1.
                       Keeps the HMF slope inside the bin.
           'column' -> p(x | m, W)        [dex^-1]; every mass column
                       integrates to 1, which removes the HMF slope and shows
                       the m-x relation directly.
           'count'  -> raw (weighted) counts; only comparable between panels
                       with identical bin edges.
    sharey : 'row' (default; webs compared at fixed mass on one y-axis),
             'col' (one y-range per web type), or 'all'.
    n_m, n_x : heat-map bins per panel along m and x.
    cmap : one-hue sequential colour map; the colour scale is shared by the
           four panels of a row.
    log_color : logarithmic colour scale.
    n_min : minimum (weighted) haloes in a mass column for the running median.
    n_scatter : panels with fewer haloes than this show the individual haloes
          as points instead of a normalised density (a handful of haloes
          normalised to unit integral would look like a strong peak) and are
          left out of the row's colour scale.
    vmax_pct : the row's colour scale tops out at this percentile of the
          positive cell values of its well-populated panels.
    median : overlay the running median of x (solid) and its 16-84 range
             (dashed).
    """
    from matplotlib.colors import Normalize, LogNorm

    lm, x, web = res['lm'], res['x'], res['web']
    w = res.get('w', np.ones(len(lm)))
    lp = res['lm_panels']
    cand = range(len(lp) - 1) if mass_panels is None else mass_panels
    rows = [p for p in cand if np.any((lm >= lp[p]) & (lm < lp[p + 1]))]
    n_r = len(rows)
    if n_r == 0:
        raise ValueError('no populated mass panel to show')

    # y-ranges according to `sharey`
    def _rng(sel):
        if sel.sum() < 2:
            return (-1.0, 1.0)
        lo, hi = np.quantile(x[sel], [0.005, 0.995])
        pad = 0.05 * max(hi - lo, 0.1)
        return (lo - pad, hi + pad)
    in_rows = np.zeros(len(lm), dtype=bool)
    for p in rows:
        in_rows |= (lm >= lp[p]) & (lm < lp[p + 1])
    yr = {}
    for i, p in enumerate(rows):
        in_p = (lm >= lp[p]) & (lm < lp[p + 1])
        for W in range(N_WEB):
            if sharey == 'row':
                yr[i, W] = _rng(in_p)
            elif sharey == 'col':
                yr[i, W] = _rng(in_rows & (web == W))
            elif sharey == 'all':
                yr[i, W] = _rng(in_rows)
            else:
                raise ValueError("sharey must be 'row', 'col' or 'all'")

    if axs is None:
        _, axs = plt.subplots(
            n_r, N_WEB, figsize=(2.55 * N_WEB + 0.9, 2.25 * n_r),
            sharex='row', sharey={'row': 'row', 'col': 'col',
                                  'all': True}[sharey],
            constrained_layout=True, squeeze=False)
    axs = np.asarray(axs).reshape(n_r, N_WEB)

    cb_label = {'joint': r'$p(m,x\,|\,W,\ {\rm bin})\ [{\rm dex}^{-2}]$',
                'column': r'$p(x\,|\,m,W)\ [{\rm dex}^{-1}]$',
                'count': r'$N$'}[norm]
    ink = '0.15'

    for i, p in enumerate(rows):
        in_p = (lm >= lp[p]) & (lm < lp[p + 1])
        m_e = np.linspace(lp[p], lp[p + 1], n_m + 1)
        dm = m_e[1] - m_e[0]
        panels = []
        for W in range(N_WEB):
            x_e = np.linspace(*yr[i, W], n_x + 1)
            dx = x_e[1] - x_e[0]
            sel = in_p & (web == W)
            H, _, _ = np.histogram2d(lm[sel], x[sel], bins=[m_e, x_e],
                                     weights=w[sel])
            if norm == 'joint':
                Z = H / max(H.sum(), 1e-30) / (dm * dx)
            elif norm == 'column':
                col = H.sum(1, keepdims=True)
                with np.errstate(invalid='ignore', divide='ignore'):
                    Z = np.where(col > 0, H / col / dx, 0.0)
            elif norm == 'count':
                Z = H
            else:
                raise ValueError("norm must be 'joint', 'column' or 'count'")
            panels.append((x_e, H, Z, sel))

        dense = [(Z, sel) for _, _, Z, sel in panels
                 if sel.sum() >= n_scatter and np.any(Z > 0)]
        src = dense if dense else [(Z, sel) for _, _, Z, sel in panels
                                   if np.any(Z > 0)]
        vmax = max((np.percentile(Z[Z > 0], vmax_pct) for Z, _ in src),
                   default=1.0)
        pos = [Z[Z > 0].min() for Z, _ in src]
        cnorm = (LogNorm(vmin=max(min(pos) if pos else 1e-3, vmax * 1e-3),
                         vmax=vmax) if log_color
                 else Normalize(vmin=0.0, vmax=vmax))
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad('white')
        mappable = None

        for W, (x_e, H, Z, sel) in enumerate(panels):
            ax = axs[i, W]
            if sel.sum() >= n_scatter or not dense:
                Zm = np.ma.masked_where(Z <= 0, Z)
                mp = ax.pcolormesh(m_e, x_e, Zm.T, cmap=cm, norm=cnorm,
                                   shading='flat', rasterized=True)
                if sel.sum() >= n_scatter or mappable is None:
                    mappable = mp
            else:                          # too few haloes for a density
                ax.scatter(lm[sel], x[sel], s=4, color=ink, alpha=0.6,
                           lw=0, rasterized=True)
            if median and H.sum() > 0:
                mc = 0.5 * (m_e[:-1] + m_e[1:])
                ok = H.sum(1) >= n_min
                q = np.full((len(mc), 3), np.nan)
                for j in np.flatnonzero(ok):
                    q[j] = _hist_quantiles(H[j], x_e, [0.16, 0.5, 0.84])
                ax.plot(mc, q[:, 1], color=ink, lw=1.3)
                ax.plot(mc, q[:, 0], color=ink, lw=0.8, ls='--')
                ax.plot(mc, q[:, 2], color=ink, lw=0.8, ls='--')
            ax.set_xlim(m_e[0], m_e[-1])
            ax.set_ylim(*yr[i, W])
            if annotate:
                n_w = int(sel.sum())
                if n_w > 1:
                    rho = _spearman_2d(H)
                    xmed = _hist_quantiles(H.sum(0), x_e, [0.5])[0]
                    txt = (rf'$N={n_w}$' + '\n' + rf'$\tilde x={xmed:+.2f}$'
                           + '\n' + rf'$\rho_{{mx}}={rho:+.2f}$')
                else:
                    txt = rf'$N={n_w}$'
                ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top',
                        fontsize=7, color=ink,
                        bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.2))
            if i == 0:
                ax.set_title(WEB_NAMES[W], fontsize=10)
            if i == n_r - 1:
                ax.set_xlabel(r'$\log_{10}(M\,/\,h^{-1}M_\odot)$')
            if W == 0:
                ax.set_ylabel(rf'$m\in[{lp[p]:.1f},{lp[p+1]:.1f})$' + '\n'
                              + r'$\log_{10}(1+\delta)$')
        cb = axs[i, 0].figure.colorbar(mappable, ax=list(axs[i, :]),
                                       location='right', shrink=0.95,
                                       pad=0.01)
        cb.set_label(cb_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return axs


def plot_fix_mass_fractions(res, axs=None, *, kind='frac', n_min=30,
                            colors=WEB_COLORS, annotate=True, **line_kw):
    """Fix m (previous version of the fix-m figure): relation between x and
    W, one panel per mass bin.

    @kind: 'frac' -> stacked P(W | x, M)   (how sharply delta decides W)
           'pdf'  -> p(x | W, M) curves    (Hahn+07 Fig. 3 style)
    """
    lp, xc = res['lm_panels'], res['x_c']
    n_p = len(lp) - 1
    axs = _panel_axes(axs, n_p)
    kw = dict(lw=1.6)
    kw.update(line_kw)
    for p in range(n_p):
        ax = axs[p]
        nx = res['fixM_n_x'][p]
        good = nx >= n_min
        if not good.any():
            ax.set_visible(False)
            continue
        if kind == 'frac':
            f = np.where(good[:, None], res['fixM_pw_x'][p], np.nan)
            base = np.zeros(len(xc))
            for w in range(N_WEB):
                top = base + np.nan_to_num(f[:, w])
                ax.fill_between(xc, base, top, where=good, step='mid',
                                color=colors[w], lw=0, label=WEB_NAMES[w])
                base = top
            ax.set_ylim(0.0, 1.0)
            if p == 0:
                ax.set_ylabel(r'$P(W\,|\,\delta, M)$')
        elif kind == 'pdf':
            for w in range(N_WEB):
                y = res['fixM_px_w'][p][:, w]
                ax.plot(xc, y, color=colors[w], label=WEB_NAMES[w], **kw)
            ax.set_ylim(bottom=0.0)
            if p == 0:
                ax.set_ylabel(r'$p(\log_{10}(1+\delta)\,|\,W, M)$')
        else:
            raise ValueError("kind must be 'frac' or 'pdf'")
        xs_good = xc[good]
        ax.set_xlim(xs_good[0] - 0.05, xs_good[-1] + 0.05)
        ax.set_title(rf'$\log_{{10}}M\in[{lp[p]:.1f},{lp[p+1]:.1f})$',
                     fontsize=9)
        ax.set_xlabel(r'$\log_{10}(1+\delta)$')
        if annotate:
            ax.text(0.04, 0.96,
                    rf'$U={_fmt_err(res, "fixM_U", p)}$' + '\n'
                    + rf'$\rho_{{xW\cdot M}}={_fmt_err(res, "fixM_rho", p, "{:+.3f}")}$',
                    transform=ax.transAxes, va='top', fontsize=7.5,
                    bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))
    if kind == 'frac':
        from matplotlib.patches import Patch
        _fig_legend(axs, [Patch(color=c) for c in colors[:N_WEB]],
                    list(WEB_NAMES))
    else:
        _fig_legend(axs)
    return axs


def plot_fix_density(res, axs=None, *, standardise=True, n_min=30,
                     min_cov=0.3, colors=WEB_COLORS, logy=False,
                     annotate=True, **line_kw):
    """Fix x: P(W | m, x) as a function of mass, one panel per density bin.

    @standardise: True (default) plots P_std(W | m, x) -- every mass bin
        re-mixed to the panel's pooled x-distribution -- as solid lines and
        the null expectation P_null (W independent of m at fixed x, same
        coverage) dashed.  Solid on dashed <=> no mass information beyond x.
        False plots the raw panel fractions, which inherit an m-trend from the
        x-mixture inside the panel and can show structure that the fine-x
        statistics U and rho do not (see ``_standardise_fixx``).
    @min_cov: hide mass bins whose populated x bins carry less than this
        fraction of the panel's x-distribution.
    """
    xp, lc = res['x_panels'], res['lm_c']
    n_p = len(xp) - 1
    axs = _panel_axes(axs, n_p)
    kw = dict(lw=1.6)
    kw.update(line_kw)
    key = 'fixX_fw_m_std' if standardise else 'fixX_fw_m'
    has_err = key + '_err' in res
    for q in range(n_p):
        ax = axs[q]
        good = res['fixX_n_m'][q] >= n_min
        if standardise:
            good &= res['fixX_cov'][q] >= min_cov
        if not good.any():
            ax.set_visible(False)
            continue
        for w in range(N_WEB):
            f = np.where(good, res[key][q][:, w], np.nan)
            ax.plot(lc, f, color=colors[w], label=WEB_NAMES[w], **kw)
            if has_err:
                e = res[key + '_err'][q][:, w]
                ax.fill_between(lc, f - e, f + e, where=good,
                                color=colors[w], alpha=0.2, lw=0)
            if standardise:
                ax.plot(lc, np.where(good, res['fixX_fw_m_null'][q][:, w],
                                     np.nan),
                        color=colors[w], ls='--', lw=0.9, alpha=0.9)
        if logy:
            ax.set_yscale('log')
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_title(rf'$\log_{{10}}(1+\delta)\in[{xp[q]:+.2f},{xp[q+1]:+.2f})$',
                     fontsize=9)
        ax.set_xlabel(r'$\log_{10}(M\,/\,h^{-1}M_\odot)$')
        if q == 0:
            ax.set_ylabel(r'$P(W\,|\,m, x)$'
                          + (' (standardised)' if standardise else ''))
        if annotate:
            ax.text(0.04, 0.96,
                    rf'$U={_fmt_err(res, "fixX_U", q)}$' + '\n'
                    + rf'$\rho_{{mW\cdot x}}={_fmt_err(res, "fixX_rho", q, "{:+.3f}")}$'
                    + ('\ndashed: null' if standardise else ''),
                    transform=ax.transAxes, va='top', fontsize=7.5,
                    bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))
    _fig_legend(axs)
    return axs


def plot_fix_web(res, axs=None, *, n_min=30, colors=WEB_COLORS,
                 self_term=True, n_bg=3, annotate=True, **line_kw):
    """Fix W: median log10(1+delta) vs mass inside each web type.

    Band = 16-84 percentile range of the distribution (not an error).
    Error bars on the median = jackknife, if computed.
    Dashed = log10(1 + delta_bg + M / M_s): the delta a halo of mass M would
    show from its own mass alone, on top of the background delta_bg taken
    from the ``n_bg`` lowest populated mass bins of that web type.
    """
    lc = res['lm_c']
    axs = _panel_axes(axs, N_WEB, sharey=False)
    kw = dict(lw=1.8)
    kw.update(line_kw)
    has_err = 'fixW_qx_err' in res
    M_s = res['M_s']
    for w in range(N_WEB):
        ax = axs[w]
        n = res['fixW_n_m'][w]
        good = n >= n_min
        if not good.any():
            ax.set_visible(False)
            continue
        q = np.where(good[:, None], res['fixW_qx'][w], np.nan)
        ax.fill_between(lc, q[:, 0], q[:, 2], where=good, color=colors[w],
                        alpha=0.25, lw=0)
        ax.plot(lc, q[:, 1], color=colors[w], **kw)
        if has_err:
            ax.errorbar(lc, q[:, 1], yerr=res['fixW_qx_err'][w][:, 1],
                        fmt='none', ecolor=colors[w], elinewidth=0.9,
                        capsize=0)
        if self_term:
            x_bg = np.nanmedian(q[good, 1][:n_bg])     # q is NaN-masked
            d_bg = 10.0**x_bg - 1.0
            lm_ref = np.linspace(lc[good][0], lc[good][-1], 100)
            ax.plot(lm_ref, np.log10(np.maximum(
                1.0 + d_bg + 10.0**lm_ref / M_s, 1e-10)),
                ls='--', color='0.3', lw=1.0,
                label=r'$1+\delta_{\rm bg}+M/M_s$')
            ax.axvline(np.log10(M_s), color='0.6', lw=0.8, ls=':')
        ax.set_title(WEB_NAMES[w], fontsize=10)
        ax.set_xlabel(r'$\log_{10}(M\,/\,h^{-1}M_\odot)$')
        if w == 0:
            ax.set_ylabel(r'$\log_{10}(1+\delta)$')
            if self_term:
                ax.legend(frameon=False, fontsize=7.5, loc='lower right')
        if annotate:
            ax.text(0.04, 0.96,
                    rf'$\rho_{{Mx}}={_fmt_err(res, "fixW_rho", w, "{:+.3f}")}$'
                    + '\n' + rf'$N={int(n.sum())}$',
                    transform=ax.transAxes, va='top', fontsize=7.5)
    return axs


# ---------------------------------------------------------------------------
# optional: the fix-x slice as abundances n_{W,x}(m) instead of P(W | m, x)
# ---------------------------------------------------------------------------

def _volume_table(tf, x_edges, *, field=None, n_jk_side=0, chunk=32):
    """V(x, W) on the fine x grid, per jackknife region: (n_jk, n_x, 4).

    Grid nodes sit at i * l_grid (pyhipp convention); their jackknife region
    uses the same floor(x / l_box * s) rule as the haloes, and their x is
    clipped into the end bins exactly like the haloes' x.
    """
    n, h = wes._grid_spec(tf)
    grid = tf.delta if field is None else field
    mask = wes._recon_mask(tf)
    x0 = float(x_edges[0])
    dx = float(x_edges[1] - x_edges[0])
    n_xf = len(x_edges) - 1
    s = int(n_jk_side)
    n_jk = max(1, s**3)
    V = np.zeros((n_jk, n_xf, N_WEB))
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(i0 + chunk, n))
        msk = np.asarray(mask[sl], dtype=bool)
        if not msk.any():
            continue
        d = np.asarray(grid[sl])[msk]
        w = np.asarray(tf.n_lams[sl])[msk].astype(np.int64)
        xv = np.log10(np.maximum(1.0 + d, 1e-10))
        ix = np.clip(np.floor((xv - x0) / dx).astype(np.int64), 0, n_xf - 1)
        if n_jk == 1:
            jk = 0
        else:
            di, jj, kk = np.nonzero(msk)       # C order == msk-indexing order
            jk = ((((di + i0) * s) // n) * s * s
                  + ((jj * s) // n) * s + (kk * s) // n)
        key = (jk * n_xf + ix) * N_WEB + w
        V += np.bincount(key, minlength=n_jk * n_xf * N_WEB).reshape(
            n_jk, n_xf, N_WEB)
    return V * h**3


def _hmf_fixx(T, V, x_pan, n_xp, dlm, ref_rows=None):
    """n_{W,x}(m), n_x(m) and derived ratios for every x panel."""
    n_mf = T.shape[0]
    out = {k: np.full(shp, np.nan) for k, shp in (
        ('n_Wx', (n_xp, n_mf, N_WEB)), ('n_x', (n_xp, n_mf)),
        ('ratio', (n_xp, n_mf, N_WEB)), ('level', (n_xp, N_WEB)),
        ('P_W', (n_xp, N_WEB)), ('P_V_W', (n_xp, N_WEB)),
        ('V_Wx', (n_xp, N_WEB)), ('N_mW', (n_xp, n_mf, N_WEB)),
        ('ratio_null_raw', (n_xp, n_mf, N_WEB)), ('cov_raw', (n_xp, n_mf)))}
    for q in range(n_xp):
        sel = x_pan == q
        # numerator and denominator are the SAME set of cells: nodes whose
        # type is W and whose x lies in panel q.  A halo belongs to the cell
        # of its own NGP node, so N counts the haloes living in those nodes
        # and V is those nodes' volume.
        Nq = T[:, sel, :].sum(1)       # haloes in nodes with (W, x in q)
        Vq = V[sel, :].sum(0)          # volume of nodes with (W, x in q)
        out['N_mW'][q], out['V_Wx'][q] = Nq, Vq
        if Vq.sum() <= 0 or Nq.sum() <= 0:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            out['n_Wx'][q] = np.where(Vq[None, :] > 0,
                                      Nq / (Vq[None, :] * dlm), np.nan)
            out['n_x'][q] = Nq.sum(1) / (Vq.sum() * dlm)
            out['ratio'][q] = out['n_Wx'][q] / out['n_x'][q][:, None]
            out['P_W'][q] = Nq.sum(0) / Nq.sum()
            out['P_V_W'][q] = Vq / Vq.sum()
            out['level'][q] = out['P_W'][q] / out['P_V_W'][q]
        # Null for the RAW panel ratio n_{W,q}(m) / n_q(m): if W depended on
        # x only (fine x), a mass-m halo would be in W with probability
        # <P(W|x; ref)> averaged over the x-distribution of mass-m haloes in
        # this panel.  This keeps the within-panel x-mixture that the raw
        # ratio contains, so solid vs dashed isolates the m-W dependence.
        Tq = T[:, sel, :]
        Nmx = Tq.sum(2)                                   # (m, x_q)
        Tref = Tq if ref_rows is None else Tq[ref_rows]
        nref = Tref.sum((0, 2))
        valid = nref > 0
        if valid.any() and Vq.sum() > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                Pw_x = Tref.sum(0)[valid] / nref[valid, None]   # (x_v, W)
                den = Nmx[:, valid].sum(1)
                Pbar = (Nmx[:, valid, None] * Pw_x[None]).sum(1) / den[:, None]
                out['ratio_null_raw'][q] = Pbar / (Vq / Vq.sum())[None, :]
                out['cov_raw'][q] = den / Nmx.sum(1)
    # n_W(m) over the union of all density panels ("All" at fixed W), and
    # each panel's HMF relative to it: n_{W,q}(m) / n_{W,all}(m)
    N_all = np.nansum(out['N_mW'], axis=0)                 # (m, W)
    V_all = np.nansum(out['V_Wx'], axis=0)                 # (W,)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['n_W_all'] = np.where(V_all[None, :] > 0,
                                  N_all / (V_all[None, :] * dlm), np.nan)
        out['ratio_W'] = out['n_Wx'] / out['n_W_all'][None]
        # n(m) over the union of all density panels (all webs), and each
        # panel's HMF relative to it: n_q(m) / n(m)
        V_tot = V_all.sum()
        out['n_all'] = (N_all.sum(1) / (V_tot * dlm) if V_tot > 0
                        else np.full(n_mf, np.nan))
        out['ratio_x_all'] = out['n_x'] / out['n_all'][None]
        pv_fine = V / V.sum(1, keepdims=True)              # P_V(W | x), fine
    out['ratio_std'], out['ratio_null'], out['cov'] = _standardise_fixx(
        T, x_pan, n_xp, ratio_to=pv_fine, ref_rows=ref_rows)
    return out


def _make_x_panels(x_e, Nx, Vx, spec, *, n_panels=5, edges_in='x',
                   panel_range=None, default=None):
    """Density panels on the fine x grid.

    spec : None            -> `default` edges (those of mass_density_web)
           'quantile'      -> n_panels with equal (weighted) halo counts
           'volume'        -> n_panels with equal grid volume
           'linear'        -> n_panels equally spaced in x (or in delta if
                              edges_in='delta') over `panel_range`, default
                              the 0.1-99.9 % range of the haloes
           1-D edges       -> contiguous panels [e0,e1), [e1,e2), ...
           [(lo, hi), ...] -> separate panels, gaps allowed, no overlap
    Values are in x = log10(1+delta), or in delta with edges_in='delta'.
    Every bound is snapped to the nearest fine-grid edge (the tables are
    binned there); panels that collapse to zero width are dropped.

    Returns (lo, hi, x_pan, contiguous, label): snapped bounds in x, the panel
    index of every fine bin (-1 = none), and a description.
    """
    x_e = np.asarray(x_e, float)
    xc = 0.5 * (x_e[:-1] + x_e[1:])

    def to_x(v):
        v = np.asarray(v, float)
        if edges_in == 'delta':
            if np.any(v <= -1):
                raise ValueError('delta edges must be > -1')
            return np.log10(1.0 + v)
        if edges_in != 'x':
            raise ValueError("edges_in must be 'x' or 'delta'")
        return v

    def snap(v):
        return np.round(x_e[np.argmin(np.abs(x_e[None, :]
                                             - np.atleast_1d(v)[:, None]),
                                      axis=1)], 10)

    def cum_edges(weights):
        tot = weights.sum()
        if tot <= 0:
            raise ValueError('no weight to split into panels')
        return _hist_quantiles(weights, x_e, np.linspace(0, 1, n_panels + 1))

    pairs = None
    if spec is None:
        edges, label = np.asarray(default, float), 'from mass_density_web'
    elif isinstance(spec, str):
        if spec == 'quantile':
            edges, label = cum_edges(Nx), f'{n_panels} equal-halo-count'
        elif spec == 'volume':
            edges, label = cum_edges(Vx), f'{n_panels} equal-volume'
        elif spec == 'linear':
            if panel_range is None:
                lo_, hi_ = _hist_quantiles(Nx, x_e, [0.001, 0.999])
                rng_ = (lo_, hi_) if edges_in == 'x' else (10**lo_ - 1,
                                                          10**hi_ - 1)
            else:
                rng_ = panel_range
            edges = to_x(np.linspace(rng_[0], rng_[1], n_panels + 1))
            label = f'{n_panels} equal-width in {edges_in}'
        else:
            raise ValueError("x_panels must be None, 'quantile', 'volume', "
                             "'linear', an edge array or (lo, hi) pairs")
    else:
        arr = np.asarray(spec, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            pairs = to_x(arr)
            label = f'{len(arr)} user panels ({edges_in})'
        elif arr.ndim == 1:
            edges, label = to_x(arr), f'user edges ({edges_in})'
        else:
            raise ValueError('explicit x_panels: 1-D edges or (lo, hi) pairs')

    if pairs is None:
        e = snap(np.sort(edges))
        lo, hi = e[:-1], e[1:]
        contiguous = True
    else:
        lo, hi = snap(pairs[:, 0]), snap(pairs[:, 1])
        order = np.argsort(lo)
        lo, hi = lo[order], hi[order]
        if np.any(lo[1:] < hi[:-1] - 1e-9):
            raise ValueError('user panels overlap after snapping to the grid')
        contiguous = bool(np.allclose(lo[1:], hi[:-1]))
    keep = hi > lo + 1e-9
    if not keep.all():
        warnings.warn(f'{int((~keep).sum())} panel(s) narrower than one fine '
                      'bin were dropped; rerun mass_density_web with smaller '
                      'dx for finer panels')
    lo, hi = lo[keep], hi[keep]
    x_pan = np.full(len(xc), -1, dtype=np.int64)
    for q, (a, b) in enumerate(zip(lo, hi)):
        x_pan[(xc >= a) & (xc < b)] = q
    return lo, hi, x_pan, contiguous, label


def hmf_by_web_fixed_density(mdw, tf, *, x_panels=None, n_panels=5,
                             edges_in='x', panel_range=None, field=None,
                             chunk=32, verbose=True):
    r"""Abundance form of the fix-x slice: n_{W,x}(m) for each density panel.

    Definition: the number of haloes whose own (NGP) node is of type W and has
    x inside the panel, divided by the volume of all nodes of type W with x
    inside the panel (and by the mass-bin width).  Numerator and denominator
    are the same set of cells; this requires ``interp='ngp'`` in
    :func:`mass_density_web` (the default) and is checked.

    .. math::
        n_{W,x}(m) = \frac{N(m, x, W)}{V(x, W)\,\Delta m}
                   = n_x(m)\,\frac{P(W\,|\,m, x)}{P_V(W\,|\,x)}

    Same information as :func:`plot_fix_density` as far as the *m-dependence*
    goes (shape differences between the n_{W,x} are exactly the slopes of
    P(W | m, x)); the only extra ingredient is the volume normalisation
    P_V(W | x), which comes from the grid, not from the haloes.

    Caveat: for M >~ M_s the halo creates the delta and the web type of its
    own node, so V(x, W) there is partly volume the halo made; the abundance
    "per unit environment volume" is then circular.

    Parameters
    ----------
    mdw : output of :func:`mass_density_web`; its x grid, panels, halo table
          and jackknife regions are reused.  Or a dict of them, e.g.
          ``{'young': mdw_y, 'old': mdw_o}`` (sub-samples run with the same
          mass_density_web settings): then every sample gets the same density
          panels (default: equal halo counts of all samples together, or the
          panels they were all given) and the same volume table, and a dict
          ``{label: result}`` is returned.  Plot it with the same functions
          (labels = keys); :func:`hmf_sample_ratio` gives the ratios between
          samples with their jackknife errors.
    tf : the same TidalField.
    x_panels : how to split the density axis (default: the panels of
            ``mdw``).  'quantile' (equal halo counts), 'volume' (equal grid
            volume), 'linear' (equal width), explicit edges
            ``[e0, e1, ...]``, or separate panels ``[(lo, hi), ...]`` with
            gaps allowed.  See ``_make_x_panels``.
    n_panels : number of panels for the string modes.
    edges_in : 'x' (log10(1+delta), default) or 'delta' -- the unit of
            explicit edges, of `panel_range`, and of the 'linear' spacing.
    panel_range : (lo, hi) for 'linear'.
            All bounds snap to the fine grid of ``mdw`` (``dx``, 0.05 dex by
            default); for finer panels rerun ``mass_density_web`` with a
            smaller ``dx``.
    field : must be the same density grid that ``mass_density_web`` used
            (None = ``tf.delta``).

    Returns
    -------
    dict with, per x panel q (first axis):
        n_Wx   (n_xp, n_m, 4)  n_{W,x}(m)            [h^3 Mpc^-3 dex^-1]
        n_x    (n_xp, n_m)     n_x(m), all web types together
        ratio  (n_xp, n_m, 4)  n_{W,x}(m) / n_x(m) = P(W|m,x) / P_V(W|x)
        level  (n_xp, 4)       P(W|x) / P_V(W|x): the value every ratio curve
                               would take if W were independent of m at
                               fixed x
        P_W, P_V_W (n_xp, 4)   halo- and volume-weighted web fractions
        V_Wx   (n_xp, 4)       V(W) inside the panel [(h^-1 Mpc)^3]
        N_mW   (n_xp, n_m, 4)  halo counts
        n_W_all (n_m, 4)       n_W(m) over the union of all panels (equals
                               Hahn+07's n_W(m) when the panels cover every
                               density); used by :func:`plot_hmf_fixed_web`
        ratio_W (n_xp, n_m, 4) n_{W,q}(m) / n_W_all(m)
    plus ``<key>_err`` from the jackknife if ``mdw`` was computed with one,
    and ``lm_c``, ``lm_edges``, ``x_lo``/``x_hi`` (snapped panel bounds in x),
    ``x_panels``, ``panel_mode``, ``edges_in``, ``M_s``.
    """
    if tf is None:
        raise ValueError('n_{W,x}(m) is normalised by grid volume V(x, W); '
                         'it needs the TidalField')
    if _is_mdw_dict(mdw):
        return _hmf_multi(mdw, tf, x_panels=x_panels, n_panels=n_panels,
                          edges_in=edges_in, panel_range=panel_range,
                          field=field, chunk=chunk, verbose=verbose)
    if bool(mdw.get('field_default', True)) != (field is None):
        warnings.warn('`field` differs from the one mass_density_web used; '
                      'halo x and volume x would not be on the same field.')
    if mdw.get('interp', 'ngp') != 'ngp':
        warnings.warn(
            "mass_density_web was run with interp=%r: a halo's x is then "
            "interpolated and need not equal the x of its own node, so the "
            "haloes in the numerator and the nodes in the denominator can "
            "fall in different density panels.  Re-run with interp='ngp' "
            "for n_{W,x}(m)." % mdw.get('interp'))
    T, x_e = mdw['T'], mdw['x_edges']
    s_jk = int(mdw.get('jk_side', 0))
    Vjk = _volume_table(tf, x_e, field=field, n_jk_side=s_jk, chunk=chunk)
    V = Vjk.sum(0)
    x_lo, x_hi, x_pan, contiguous, label = _make_x_panels(
        x_e, T.sum((0, 2)), V.sum(1), x_panels, n_panels=n_panels,
        edges_in=edges_in, panel_range=panel_range, default=mdw['x_panels'])
    res = _hmf_tables(mdw, V, Vjk, x_lo, x_hi, x_pan, contiguous, label,
                      edges_in)
    if verbose:
        _hmf_report(res, T, x_pan, label)
    return res


def _hmf_tables(mdw, V, Vjk, x_lo, x_hi, x_pan, contiguous, label, edges_in,
                occ=None):
    """n_{W,x}(m) & co. of one sample on given panels and volume tables,
    with jackknife errors and the jackknife replicates of the abundances
    (``res['jk']``, needed for ratios between samples)."""
    T, lm_e = mdw['T'], mdw['lm_edges']
    dlm = float(lm_e[1] - lm_e[0])
    n_xp = len(x_lo)
    ref_rows = mdw.get('ref_rows')
    res = _hmf_fixx(T, V, x_pan, n_xp, dlm, ref_rows)

    Tjk = mdw.get('Tjk')
    s_jk = int(mdw.get('jk_side', 0))
    if s_jk > 0 and Tjk is not None and Tjk.shape[0] == Vjk.shape[0]:
        if occ is None:
            occ = [k for k in range(Vjk.shape[0])
                   if Tjk[k].sum() > 0 or Vjk[k].sum() > 0]
        samples = [_hmf_fixx(T - Tjk[k], V - Vjk[k], x_pan, n_xp, dlm,
                             ref_rows) for k in occ]
        nj = len(samples)
        for kk in ('n_Wx', 'n_x', 'ratio', 'level', 'P_W', 'P_V_W',
                   'ratio_std', 'ratio_null', 'ratio_null_raw',
                   'n_W_all', 'ratio_W', 'n_all', 'ratio_x_all'):
            res[kk + '_err'] = _jk_std(np.array([sm[kk] for sm in samples]))
        res['jk'] = {kk: np.array([sm[kk] for sm in samples])
                     for kk in ('n_Wx', 'n_x', 'n_W_all', 'n_all')}
        res['jk_regions'] = np.asarray(occ)
        res['n_jk'] = nj
    else:
        res['n_jk'] = 0

    res.update({'lm_c': mdw['lm_c'], 'lm_edges': lm_e,
                'x_lo': x_lo, 'x_hi': x_hi,
                'x_panels': (np.append(x_lo, x_hi[-1]) if contiguous
                             else np.stack([x_lo, x_hi], 1)),
                'panel_mode': label, 'edges_in': edges_in,
                'M_s': mdw['M_s'], 'R_s': mdw.get('R_s'),
                'omega_m': mdw.get('omega_m'),
                'kernel': mdw.get('kernel', 'gaussian')})
    return res


def _jk_std(arr):
    """Delete-one jackknife standard deviation over the first axis."""
    nj = arr.shape[0]
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(arr, axis=0)
        return np.sqrt((nj - 1.0) / nj * np.nansum((arr - mean)**2, axis=0))


def _hmf_report(res, T, x_pan, label, name=None):
    x_lo, x_hi = res['x_lo'], res['x_hi']
    n_xp = len(x_lo)
    # the identity must hold to rounding: ratio == P(W|m,x) / P_V(W|x)
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = 0.0
        for q in range(n_xp):
            Nq = T[:, x_pan == q, :].sum(1)
            pw = Nq / Nq.sum(1, keepdims=True)         # P(W | m, q)
            pred = pw / res['P_V_W'][q][None, :]
            d = np.abs(res['ratio'][q] / pred - 1.0)
            if np.isfinite(d).any():
                dev = max(dev, np.nanmax(d))
    head = 'hmf_by_web_fixed_density' + (f' [{name}]' if name is not None
                                         else '')
    print(f'{head}: {n_xp} density panels ({label}), '
          f'jackknife regions = {res["n_jk"]}')
    print(f'  identity n_Wx/n_x = P(W|m,x)/P_V(W|x): max |dev| = {dev:.2e}')
    print(f'  {"panel log10(1+delta)":>24s}  '
          + '  '.join(f'{n:>15s}' for n in WEB_NAMES))
    print(f'  {"":>24s}  ' + '  '.join(f'{"P  P_V  P/P_V":>15s}'
                                       for _ in WEB_NAMES))
    for q in range(n_xp):
        print(f'  [{x_lo[q]:+.2f}, {x_hi[q]:+.2f})'.ljust(26) + '  '
              + '  '.join(f'{res["P_W"][q, w]:.2f} {res["P_V_W"][q, w]:.2f}'
                          f' {res["level"][q, w]:5.2f}'
                          for w in range(N_WEB)))


def _is_mdw_dict(m):
    return (isinstance(m, dict) and 'T' not in m and len(m) > 0
            and all(isinstance(v, dict) and 'T' in v for v in m.values()))


def _hmf_multi(mdws, tf, *, x_panels, n_panels, edges_in, panel_range, field,
               chunk, verbose):
    """hmf_by_web_fixed_density for a dict {label: mdw} (see there)."""
    names = list(mdws)
    M = [mdws[n] for n in names]
    b = M[0]

    def same(a, c):
        a, c = np.asarray(a, float), np.asarray(c, float)
        return a.shape == c.shape and np.allclose(a, c)

    for n, m in zip(names[1:], M[1:]):
        bad = [k for k, ok in (
            ('lm_range/dlm', same(m['lm_edges'], b['lm_edges'])),
            ('x_range/dx', same(m['x_edges'], b['x_edges'])),
            ('n_jk_side', int(m.get('jk_side', 0)) == int(b.get('jk_side', 0))),
            ('l_box', m.get('jk_l_box') == b.get('jk_l_box')),
            ('field', m.get('field_default') == b.get('field_default')),
            ('interp', m.get('interp') == b.get('interp')),
            ('ref_lm', m.get('ref_lm') == b.get('ref_lm'))) if not ok]
        if bad:
            raise ValueError(f'samples {names[0]!r} and {n!r}: '
                             'mass_density_web was run with different '
                             + ', '.join(bad) + '; rerun with the same '
                             'settings so that the cells coincide')
    for n, m in zip(names, M):
        if bool(m.get('field_default', True)) != (field is None):
            warnings.warn(f'{n!r}: `field` differs from the one '
                          'mass_density_web used')
        if m.get('interp', 'ngp') != 'ngp':
            warnings.warn(f"{n!r}: mass_density_web was run with "
                          f"interp={m.get('interp')!r}; use 'ngp' for "
                          "n_{W,x}(m)")

    x_e = b['x_edges']
    s_jk = int(b.get('jk_side', 0))
    Vjk = _volume_table(tf, x_e, field=field, n_jk_side=s_jk, chunk=chunk)
    V = Vjk.sum(0)
    # one set of density panels for every sample; by default the panels all
    # samples were given in mass_density_web if they agree, otherwise equal
    # halo counts of the samples together
    spec, default = x_panels, b['x_panels']
    if spec is None and not all(same(m['x_panels'], default) for m in M[1:]):
        spec = 'quantile'
    Nx = np.sum([m['T'].sum((0, 2)) for m in M], axis=0)
    x_lo, x_hi, x_pan, contiguous, label = _make_x_panels(
        x_e, Nx, V.sum(1), spec, n_panels=n_panels, edges_in=edges_in,
        panel_range=panel_range, default=default)
    if spec == 'quantile' and x_panels is None:
        label += ' of all samples together'
    occ = None
    if s_jk > 0:
        occ = [k for k in range(Vjk.shape[0])
               if Vjk[k].sum() > 0 or any(m['Tjk'][k].sum() > 0 for m in M)]
    out = {}
    for n, m in zip(names, M):
        out[n] = _hmf_tables(m, V, Vjk, x_lo, x_hi, x_pan, contiguous, label,
                             edges_in, occ=occ)
        if verbose:
            _hmf_report(out[n], m['T'], x_pan, label, name=n)
    return out


def _cross_ratio(S, R, key):
    """S[key] / R[key] for two samples on the same cells, with the jackknife
    error of the ratio itself (the same region removed from both)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.asarray(S[key], float) / np.asarray(R[key], float)
        js, jr = S.get('jk'), R.get('jk')
        if (js is None or jr is None or key not in js or key not in jr
                or not np.array_equal(S.get('jk_regions'),
                                      R.get('jk_regions'))):
            return r, None
        return r, _jk_std(js[key] / jr[key])


def hmf_sample_ratio(hres, ref=None):
    r"""Ratios between the samples of a multi-sample HMF result.

    hres : dict {label: result} from ``hmf_by_web_fixed_density`` called with
           a dict of ``mass_density_web`` outputs (same cells for all).
    ref  : label of the reference sample (default: the first one).

    For every other sample S returns a dict with

        ratio_Wx  (n_xp, n_m, 4)  n^S_{W,q}(m) / n^ref_{W,q}(m)
        ratio_x   (n_xp, n_m)     n^S_q(m) / n^ref_q(m)      (all web types)
        ratio_W   (n_m, 4)        n^S_W(m) / n^ref_W(m)      (all panels)
        rel_Wx    (n_xp, n_m, 4)  ratio_Wx / ratio_x
                                  = P(W | S, m, q) / P(W | ref, m, q)

    each with ``<key>_err`` (jackknife of the ratio itself, the same region
    removed from both samples; None without jackknife).  The volumes cancel,
    so these are ratios of halo counts in the same cells: ratio_x is how the
    two samples differ in density at fixed mass, and rel_Wx is how they
    differ in web type at fixed mass AND density (1 = no web dependence
    beyond density).
    """
    names = list(hres)
    ref = names[0] if ref is None else ref
    if ref not in hres:
        raise KeyError(f'ref {ref!r} is not one of {names}')
    R = hres[ref]
    out = {}
    for n in names:
        if n == ref:
            continue
        S = hres[n]
        o = {}
        for k_out, k in (('ratio_Wx', 'n_Wx'), ('ratio_x', 'n_x'),
                         ('ratio_W', 'n_W_all')):
            o[k_out], o[k_out + '_err'] = _cross_ratio(S, R, k)
        with np.errstate(divide='ignore', invalid='ignore'):
            o['rel_Wx'] = o['ratio_Wx'] / o['ratio_x'][..., None]
            if o['ratio_Wx_err'] is not None:
                js, jr = S['jk'], R['jk']
                o['rel_Wx_err'] = _jk_std((js['n_Wx'] / jr['n_Wx'])
                                          / (js['n_x'] / jr['n_x'])[..., None])
            else:
                o['rel_Wx_err'] = None
        out[n] = o
    return out


def _group_wquantile(g, z, w, n_groups, q=0.5):
    """Weighted quantile of z inside each group g (0..n_groups-1; g < 0 is
    ignored): the first value at which the cumulative weight reaches q of the
    group total (for unit weights and even counts: the lower median)."""
    out = np.full(n_groups, np.nan)
    keep = (g >= 0) & (w > 0) & np.isfinite(z)
    g, z, w = g[keep], z[keep], w[keep]
    if g.size == 0:
        return out
    order = np.lexsort((z, g))
    g, z, w = g[order], z[order], w[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    sizes = np.diff(np.r_[starts, len(g)])
    gi = np.repeat(np.arange(len(starts)), sizes)
    cw = np.cumsum(w)
    base = np.r_[0.0, cw[starts[1:] - 1]]
    tot = np.add.reduceat(w, starts)
    cw_in = cw - base[gi]
    cond = cw_in >= q * tot[gi] * (1 - 1e-12)
    first = np.minimum.reduceat(np.where(cond, np.arange(len(g)), len(g)),
                                starts)
    out[g[starts]] = z[np.minimum(first, len(g) - 1)]
    return out


_ZF_SUMS = ('n', 'z', 'x', 'm', 'xx', 'mm', 'xm', 'zx', 'zm')


def _zf_mean_tables(S, x_pan, n_q, null_ref='all'):
    """Mean-z_f statistics from per-cell sums on the fine (m, x, W) grid.

    S : dict of (n_m, n_x, 4) tables of weighted sums over the haloes of each
        cell: n = sum w, z = sum w z_f, x = sum w x, m = sum w log M, and the
        second moments xx, mm, xm, zx, zm.
    null_ref : reference of the null --
        'all'    all web types in the cell (default; matches the all-web
                 denominator of `ratio`),
        'others' the other three web types (leave-one-out),
        int w0   one baseline web type.

    Null = the curve minus the web offset, where the offset is measured cell
    by cell (web W vs reference in the same fine (m, x) cell) and corrected
    for the two groups' mean offsets in x and log M *inside* the cell, with
    slopes pooled from the within-(cell, web) scatter of the panel.  Without
    that correction, web types that sit at opposite edges of the same fine
    bin (knots high, voids low) would differ in z_f through the x- and
    M-gradients alone.
    """
    n_m = S['n'].shape[0]
    o = {k: np.full(shp, np.nan) for k, shp in (
        ('N_mW', (n_q, n_m, N_WEB)), ('z_Wx', (n_q, n_m, N_WEB)),
        ('z_x', (n_q, n_m)), ('ratio', (n_q, n_m, N_WEB)),
        ('z_null', (n_q, n_m, N_WEB)), ('ratio_null', (n_q, n_m, N_WEB)),
        ('ratio_W', (n_q, n_m, N_WEB)), ('cov_null', (n_q, n_m, N_WEB)),
        ('n_ref_eff', (n_q, n_m, N_WEB)),
        ('slope_x', (n_q, n_m)), ('slope_m', (n_q, n_m)))}
    in_any = x_pan >= 0
    with np.errstate(divide='ignore', invalid='ignore'):
        for q in range(n_q):
            sel = x_pan == q
            c = {k: S[k][:, sel, :] for k in _ZF_SUMS}          # (m, x_q, W)
            n = c['n']
            n_mw = n.sum(1)
            o['N_mW'][q] = n_mw
            o['z_Wx'][q] = c['z'].sum(1) / n_mw
            o['z_x'][q] = c['z'].sum((1, 2)) / n.sum((1, 2))
            o['ratio'][q] = o['z_Wx'][q] / o['z_x'][q][:, None]

            # cell means of the web type itself
            zc, xc, mc = c['z'] / n, c['x'] / n, c['m'] / n
            # cell means of the reference, shape (m, x_q, W)
            if null_ref == 'all':
                nr = n.sum(2, keepdims=True)
                ref = {k: c[k].sum(2, keepdims=True) / nr for k in 'zxm'}
                ref = {k: np.repeat(v, N_WEB, 2) for k, v in ref.items()}
                nr = np.repeat(nr, N_WEB, 2)
            elif null_ref == 'others':
                nr = n.sum(2, keepdims=True) - n
                ref = {k: (c[k].sum(2, keepdims=True) - c[k]) / nr
                       for k in 'zxm'}
            else:
                w0 = int(null_ref)
                ref = {k: np.repeat((c[k][..., w0] / n[..., w0])[..., None],
                                    N_WEB, 2) for k in 'zxm'}
                nr = np.repeat(n[..., w0][..., None], N_WEB, 2)
            # pooled within-(cell, web) slopes of z_f on x and log M
            def _cv(a, b, ab):
                return np.nansum(np.where(n > 0, c[ab] - c[a] * c[b] / n,
                                          0.0), axis=(1, 2))
            Cxx, Cmm, Cxm = _cv('x', 'x', 'xx'), _cv('m', 'm', 'mm'), \
                _cv('x', 'm', 'xm')
            Czx, Czm = _cv('z', 'x', 'zx'), _cv('z', 'm', 'zm')
            det = Cxx * Cmm - Cxm**2
            good2 = det > 1e-12 * np.maximum(Cxx * Cmm, 1e-300)
            bx = np.where(good2, (Czx * Cmm - Czm * Cxm) / det,
                          np.where(Cxx > 0, Czx / Cxx, 0.0))
            bm = np.where(good2, (Czm * Cxx - Czx * Cxm) / det, 0.0)
            o['slope_x'][q], o['slope_m'][q] = bx, bm

            d_cell = ((zc - ref['z']) - bx[:, None, None] * (xc - ref['x'])
                      - bm[:, None, None] * (mc - ref['m']))
            p_x = n / n_mw[:, None, :]                          # P(x|m,W,q)
            ok = np.isfinite(d_cell)
            cov = np.where(ok, p_x, 0.0).sum(1)
            o['cov_null'][q] = np.where(n_mw > 0, cov, np.nan)
            delta = np.where(ok, p_x * d_cell, 0.0).sum(1) / cov
            # effective number of reference haloes behind the null:
            # var(delta) from the reference ~ sigma_z^2 / n_ref_eff
            s2 = np.where(ok, p_x**2 / np.where(nr > 0, nr, np.inf),
                          0.0).sum(1)
            o['n_ref_eff'][q] = np.where(s2 > 0, cov**2 / s2, 0.0)
            o['z_null'][q] = np.where((n_mw > 0) & (cov > 0),
                                      o['z_Wx'][q] - delta, np.nan)
            o['ratio_null'][q] = o['z_null'][q] / o['z_x'][q][:, None]
        # carried explicitly so that the jackknife gives the error of the
        # comparison itself (the null is noisy when its reference is sparse)
        o['d_null'] = o['ratio'] - o['ratio_null']
        o['over_null'] = o['ratio'] / o['ratio_null']
        o['z_W_all'] = (S['z'][:, in_any, :].sum(1)
                        / S['n'][:, in_any, :].sum(1))
        o['ratio_W'] = o['z_Wx'] / o['z_W_all'][None]
        # all webs and all density panels together, and each panel's
        # all-web curve relative to it
        o['z_all'] = (S['z'][:, in_any, :].sum((1, 2))
                      / S['n'][:, in_any, :].sum((1, 2)))
        o['ratio_x_all'] = o['z_x'] / o['z_all'][None]
    return o


def _zf_median_tables(im, pan, web, z, w, n_m, n_q):
    """Median-z_f statistics from per-halo values (no null available)."""
    o = {}
    inq = pan >= 0
    g = np.where(inq, (pan * n_m + im) * N_WEB + web, -1)
    o['z_Wx'] = _group_wquantile(g, z, w, n_q * n_m * N_WEB).reshape(
        n_q, n_m, N_WEB)
    g = np.where(inq, pan * n_m + im, -1)
    o['z_x'] = _group_wquantile(g, z, w, n_q * n_m).reshape(n_q, n_m)
    g = np.where(inq, im * N_WEB + web, -1)
    o['z_W_all'] = _group_wquantile(g, z, w, n_m * N_WEB).reshape(n_m, N_WEB)
    o['z_all'] = _group_wquantile(np.where(inq, im, -1), z, w, n_m)
    key = np.where(inq, (pan * n_m + im) * N_WEB + web, 0)
    o['N_mW'] = np.bincount(key[inq], weights=w[inq],
                            minlength=n_q * n_m * N_WEB).reshape(
        n_q, n_m, N_WEB)
    with np.errstate(divide='ignore', invalid='ignore'):
        o['ratio'] = o['z_Wx'] / o['z_x'][..., None]
        o['ratio_W'] = o['z_Wx'] / o['z_W_all'][None]
        o['ratio_x_all'] = o['z_x'] / o['z_all'][None]
    o['z_null'] = np.full_like(o['z_Wx'], np.nan)
    o['ratio_null'] = np.full_like(o['z_Wx'], np.nan)
    return o


def _scalar(v):
    """A per-halo property as a 1-D array; vectors become their magnitude."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim >= 2:
        v = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1)
    return v


def _zf_input(mdw, h_samp, key):
    """z_f (or another per-halo property) of the haloes kept by
    ``mass_density_web`` (aligned with mdw['idx']), from the halo sample or
    from an array; a vector property becomes its magnitude."""
    if 'idx' not in mdw or 'jk' not in mdw:
        raise ValueError('re-run mass_density_web: this result has no '
                         'per-halo bookkeeping (idx, im, ix, jk)')
    if type(h_samp).__name__ == 'TidalField' or hasattr(h_samp, 'n_lams'):
        raise TypeError(
            "2nd argument is a TidalField, but z_f comes from the haloes: "
            "call zf_by_web_fixed_density(mdw, hs, ...) with hs the sample "
            "given to mass_density_web (or a z_f array aligned with it).  "
            "Unlike the HMF no volume is needed; tf= is only used for "
            "x_panels='volume'.")
    idx = mdw['idx']
    n_all = mdw.get('n_all')
    if isinstance(h_samp, (np.ndarray, list, tuple)):     # z_f values
        z = _scalar(h_samp)
        if z.ndim != 1 or (n_all is not None and len(z) != n_all) \
                or len(z) <= idx.max():
            raise ValueError(
                f'z_f array must have one value per halo of the sample given '
                f'to mass_density_web ({n_all}), got shape {z.shape}')
        return z[idx]
    try:
        lm_chk = np.asarray(h_samp[mdw.get('mass_key', 'mass')], float)
    except KeyError:
        raise
    except Exception as err:
        raise TypeError('2nd argument must be the halo sample given to '
                        'mass_density_web (or a z_f array aligned with '
                        f'it); got {type(h_samp).__name__}') from err
    if (n_all is not None and len(lm_chk) != n_all) or len(lm_chk) <= idx.max():
        raise ValueError('h_samp is not the sample mass_density_web used '
                         '(different length)')
    lm_chk = lm_chk[idx]
    if not mdw.get('log_mass', True):
        lm_chk = np.log10(lm_chk)
    if not np.allclose(lm_chk, mdw['lm']):
        raise ValueError('h_samp is not the sample mass_density_web used')
    return _scalar(h_samp[key])[idx]


def zf_by_web_fixed_density(mdw, h_samp, *, key='zf', stat='mean',
                            x_panels=None, n_panels=5, edges_in='x',
                            panel_range=None, panels_from=None, tf=None,
                            null_ref='all', valid_range=None, label=None,
                            verbose=True):
    r"""Formation time as a function of mass inside each (web type W, density
    panel q) -- the z_f counterpart of :func:`hmf_by_web_fixed_density`, on
    the same cells.

    For every mass bin m:

        z_Wx[q]    <z_f>_{W,q}(m)    haloes of web W in density panel q
        z_x[q]     <z_f>_q(m)        all web types in panel q
        ratio[q]   <z_f>_{W,q} / <z_f>_q          (fixed density, vary W)
        z_W_all    <z_f>_W(m)        web W over all density panels
        ratio_W[q] <z_f>_{W,q} / <z_f>_W          (fixed web, vary density)

    Null (stat='mean' only).  Inside a panel, web-W haloes may sit at other
    fine densities than the panel as a whole, and z_f depends on density, so
    part of ratio[q] is density mixing.  The null removes exactly that:

        z_null[q]     = sum_{x in q} P(x | m, W, q) <z_f>(m, x)
        ratio_null[q] = z_null / <z_f>_q

    with <z_f>(m, x) the all-web mean in the fine (m, x) cell: what web-W
    haloes would have if z_f depended on (m, x) only.  ratio vs ratio_null is
    the web effect on formation time at fixed mass and fine density.

    The comparison is relative: with the default reference (all web types),
    a genuine offset in one web type also moves the others the opposite way,
    in proportion to how much they share density cells with it (the same is
    true of web fractions, which sum to one).  To attribute an effect,
    ``null_ref='others'`` compares each web type with the other three, and
    ``null_ref=w0`` with one baseline web type (e.g. 1 = sheets): then only
    web types that really differ from the baseline leave their null.
    ``d_null`` = ratio - ratio_null and ``over_null`` = ratio / ratio_null
    carry their own jackknife errors: judge a departure from the null with
    those, not with the error of the curve alone (a single-web reference can
    be noisy where that web is rare).  ``n_ref_eff`` is the effective number
    of reference haloes behind each null point; below ~30 the jackknife
    itself fails (a handful of voids or knots in one or two sub-cubes), so
    the plots hide the null there (``n_ref_min``).  Mock tests: with 'all'
    and 'others' the null is always usable; with a baseline web, voids in the
    densest panel and knots in the emptiest one drop out.
    ``cov_null`` is the fraction of each curve's x-distribution where the
    reference exists (always 1 for 'all'); the web offset is measured on
    that part only and the null is the curve minus that offset, so a partial
    coverage never turns into a change of density mixture.

    Parameters
    ----------
    mdw : output of :func:`mass_density_web` (recent version: it must carry
          the per-halo bookkeeping 'idx', 'im', 'ix', 'jk').
    h_samp : the same halo sample that went into ``mass_density_web``, or a
          1-D array of z_f with one value per halo of that sample.  NOT the
          TidalField: unlike :func:`hmf_by_web_fixed_density` no volumes are
          needed, z_f comes from the haloes (tf= only for x_panels='volume').
    key : formation-time column (default 'zf').
    stat : 'mean' (default; exact null) or 'median' (weighted lower median;
          no null).
    x_panels, n_panels, edges_in, panel_range : as in
          :func:`hmf_by_web_fixed_density` ('volume' needs ``tf``).
    panels_from : a result of :func:`hmf_by_web_fixed_density` or of this
          function -- reuse its density panels exactly (overrides x_panels).
    tf : only for x_panels='volume'.
    valid_range : (lo, hi) -- values outside lo < v < hi are dropped like
          non-finite ones (None on either side = open); e.g. flagged
          concentrations.  ``res['n_dropped']`` counts them.
    label : TeX symbol for the plots (default from ``key``).
    Any per-halo property works (see :func:`spin_by_web_fixed_density`,
    :func:`concentration_by_web_fixed_density`); a vector property is
    replaced by its magnitude.

    Jackknife errors use the sub-cubes of ``mdw`` (``n_jk_side`` there).
    Weights are those of ``mdw``; haloes with non-finite z_f are dropped.
    """
    for k_ in ('idx', 'im', 'ix', 'jk'):
        if k_ not in mdw:
            raise KeyError("mdw lacks per-halo bookkeeping; rerun "
                           "mass_density_web with the current version")
    if stat not in ('mean', 'median'):
        raise ValueError("stat must be 'mean' or 'median'")
    idx = mdw['idx']
    label_tex = label        # `label` is reused below for the panel mode
    z = _zf_input(mdw, h_samp, key)
    ok_z = np.isfinite(z)
    if valid_range is not None:             # e.g. flagged values (c <= 0)
        lo_v, hi_v = valid_range
        if lo_v is not None:
            ok_z &= z > lo_v
        if hi_v is not None:
            ok_z &= z < hi_v
    n_bad = int((~ok_z).sum())
    w = np.asarray(mdw.get('w', np.ones(len(idx))), dtype=np.float64)
    w = np.where(ok_z, w, 0.0)
    z = np.where(ok_z, z, 0.0)
    im, ix, web, jk = mdw['im'], mdw['ix'], mdw['web'], mdw['jk']
    lm_e, x_e = mdw['lm_edges'], mdw['x_edges']
    n_m, n_x = len(lm_e) - 1, len(x_e) - 1

    # density panels
    if panels_from is not None:
        spec = np.stack([np.asarray(panels_from['x_lo']),
                         np.asarray(panels_from['x_hi'])], 1)
        e_in = 'x'
    else:
        spec, e_in = x_panels, edges_in
    Vx = None
    if isinstance(spec, str) and spec == 'volume':
        if tf is None:
            raise ValueError("x_panels='volume' needs the TidalField (tf=)")
        Vx = _volume_table(tf, x_e).sum(0).sum(1)
    Nx = np.bincount(ix, weights=w, minlength=n_x).astype(float)
    x_lo, x_hi, x_pan, contiguous, label = _make_x_panels(
        x_e, Nx, Vx if Vx is not None else Nx, spec, n_panels=n_panels,
        edges_in=e_in, panel_range=panel_range, default=mdw['x_panels'])
    n_q = len(x_lo)
    pan = x_pan[ix]                                   # panel of every halo

    n_jk = int(max(1, int(mdw.get('jk_side', 0))**3))
    if stat == 'mean':
        if not (null_ref in ('all', 'others')
                or (np.isscalar(null_ref) and int(null_ref) in range(N_WEB))):
            raise ValueError("null_ref must be 'all', 'others' or 0..3")
        size = n_m * n_x * N_WEB
        cell = (im * n_x + ix) * N_WEB + web
        xv = np.asarray(mdw['x'], dtype=np.float64)
        mv = np.asarray(mdw['lm'], dtype=np.float64)
        vals = {'n': np.ones_like(z), 'z': z, 'x': xv, 'm': mv,
                'xx': xv * xv, 'mm': mv * mv, 'xm': xv * mv,
                'zx': z * xv, 'zm': z * mv}
        Sjk = {k: np.zeros((n_jk, n_m, n_x, N_WEB)) for k in _ZF_SUMS}
        for k in range(n_jk):
            sk = jk == k
            if not sk.any():
                continue
            for kk in _ZF_SUMS:
                Sjk[kk][k] = np.bincount(
                    cell[sk], weights=w[sk] * vals[kk][sk],
                    minlength=size).reshape(n_m, n_x, N_WEB)
        St = {kk: v.sum(0) for kk, v in Sjk.items()}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = _zf_mean_tables(St, x_pan, n_q, null_ref)
            occ = ([k for k in range(n_jk) if Sjk['n'][k].sum() > 0]
                   if n_jk > 1 else [])
            samples = [_zf_mean_tables({kk: St[kk] - Sjk[kk][k]
                                        for kk in _ZF_SUMS},
                                       x_pan, n_q, null_ref) for k in occ]
    else:
        res = _zf_median_tables(im, pan, web, z, w, n_m, n_q)
        occ = [k for k in range(n_jk) if np.any(jk == k)] if n_jk > 1 else []
        samples = []
        for k in occ:
            wk = np.where(jk == k, 0.0, w)
            samples.append(_zf_median_tables(im, pan, web, z, wk, n_m, n_q))

    if len(samples) > 1:
        nj = len(samples)
        for kk in ('z_Wx', 'z_x', 'ratio', 'z_null', 'ratio_null',
                   'z_W_all', 'ratio_W', 'd_null', 'over_null', 'z_all',
                   'ratio_x_all'):
            if kk not in res:
                continue
            arr = np.array([sm[kk] for sm in samples])
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                mean = np.nanmean(arr, axis=0)
                res[kk + '_err'] = np.sqrt((nj - 1.0) / nj * np.nansum(
                    (arr - mean)**2, axis=0))
        res['n_jk'] = nj
    else:
        res['n_jk'] = 0

    res.update({'lm_c': mdw['lm_c'], 'lm_edges': lm_e,
                'x_lo': x_lo, 'x_hi': x_hi,
                'x_panels': (np.append(x_lo, x_hi[-1]) if contiguous
                             else np.stack([x_lo, x_hi], 1)),
                'panel_mode': label, 'edges_in': e_in if panels_from is None
                else panels_from.get('edges_in', 'x'),
                'M_s': mdw['M_s'], 'R_s': mdw.get('R_s'),
                'omega_m': mdw.get('omega_m'),
                'kernel': mdw.get('kernel', 'gaussian'), 'stat': stat, 'key': key,
                'label_tex': label_tex, 'n_dropped': n_bad,
                'null_ref': null_ref if stat == 'mean' else None})
    if verbose:
        print(f'<{key}> by web type at fixed density ({stat}): {n_q} density '
              f'panels ({res["panel_mode"]}), jackknife regions = '
              f'{res["n_jk"]}; {n_bad} haloes without a valid {key}')
        for q in range(n_q):
            n_q_w = np.nansum(res['N_mW'][q], axis=0)
            print(f'  [{x_lo[q]:+.2f}, {x_hi[q]:+.2f})  N per web = '
                  + ' '.join(f'{WEB_NAMES[w_][:4]}:{int(n_q_w[w_])}'
                             for w_ in range(N_WEB)))
    return res


def spin_by_web_fixed_density(mdw, h_samp, *, key='lambda',
                              valid_range=(0.0, None), label=r'\lambda', **kw):
    r"""Spin-mass relation inside each (web type, density panel):
    <lambda>_{W,x}(M), on the same cells as the HMF and z_f.

    Identical to :func:`zf_by_web_fixed_density` (same options: stat,
    x_panels / panels_from, null_ref, jackknife ...) with the spin column
    ``key`` (default 'lambda'; a spin vector is replaced by its magnitude)
    and non-positive values dropped (``valid_range``).  The mean is linear
    in lambda; stat='median' gives the median instead.
    Plot with :func:`plot_spin_fixed_density`, :func:`plot_spin_fixed_web`,
    :func:`plot_spin_fixed_web_single`.
    """
    res = zf_by_web_fixed_density(mdw, h_samp, key=key,
                                  valid_range=valid_range, label=label, **kw)
    res['quantity'] = 'spin'
    return res


def concentration_by_web_fixed_density(mdw, h_samp, *, key='concentration',
                                       valid_range=(0.0, None), label='c',
                                       **kw):
    r"""Concentration-mass relation inside each (web type, density panel):
    <c>_{W,x}(M), on the same cells as the HMF and z_f.

    Identical to :func:`zf_by_web_fixed_density` with the concentration
    column ``key`` (default 'concentration') and non-positive values dropped
    (``valid_range``; set e.g. (0, 100) to also drop failed fits).
    Plot with :func:`plot_c_fixed_density`, :func:`plot_c_fixed_web`,
    :func:`plot_c_fixed_web_single`.
    """
    res = zf_by_web_fixed_density(mdw, h_samp, key=key,
                                  valid_range=valid_range, label=label, **kw)
    res['quantity'] = 'concentration'
    return res


def _cell_grid(n, layout=None, two_rows=True, height_ratios=(2.2, 1.0),
               panel_size=(2.7, 2.8), share_top=True, share_bot=False):
    """n plotting cells on an (nrows, ncols) grid; each cell is one axis, or
    an upper (main) and a lower (ratio) axis sharing the x-axis.

    layout : None -> one row; int -> that many columns; (nrows, ncols).
    panel_size : (width, height) of a cell's main axis [inch].
    Returns (top, bot, first_col, last_row): arrays of length n (bot is all
    None for a single-axis cell) and per-cell booleans for axis labelling.
    """
    if layout is None:
        nrows, ncols = 1, n
    elif np.isscalar(layout):
        ncols = int(layout)
        nrows = int(np.ceil(n / ncols))
    else:
        nrows, ncols = (int(v) for v in layout)
        if nrows * ncols < n:
            raise ValueError(f'layout {layout} has {nrows * ncols} cells for '
                             f'{n} panels')
    hr = list(height_ratios)
    cell_h = panel_size[1] * (sum(hr) / hr[0] if two_rows else 1.0)
    fig = plt.figure(figsize=(panel_size[0] * ncols, cell_h * nrows),
                     constrained_layout=True)
    outer = fig.add_gridspec(nrows, ncols)
    top, bot = [], []
    for c in range(n):
        r, col = divmod(c, ncols)
        if two_rows:
            inner = outer[r, col].subgridspec(2, 1, height_ratios=hr,
                                              hspace=0.0)
            a = fig.add_subplot(inner[0])
            b = fig.add_subplot(inner[1], sharex=a)
            a.tick_params(labelbottom=False)
        else:
            a, b = fig.add_subplot(outer[r, col]), None
        top.append(a)
        bot.append(b)
    first_col = np.array([c % ncols == 0 for c in range(n)])
    last_row = np.array([c + ncols >= n for c in range(n)])
    for c in range(1, n):
        if share_top:
            top[c].sharey(top[0])
            if not first_col[c]:
                top[c].tick_params(labelleft=False)
        if two_rows and share_bot:
            bot[c].sharey(bot[0])
            if not first_col[c]:
                bot[c].tick_params(labelleft=False)
    return (np.array(top, dtype=object), np.array(bot, dtype=object),
            first_col, last_row)


def _style_ratio_axis(b, vals, *, log=True, ylim=None, auto=True):
    """y-range and ticks of a ratio axis: robust 2-98 % limits that always
    include 1, and plain tick labels (0.5, 1, 2) on a log axis."""
    from matplotlib.ticker import (LogLocator, MaxNLocator, FuncFormatter,
                                   NullFormatter, NullLocator)
    if log:
        b.set_yscale('log')
    if ylim is not None:
        b.set_ylim(*ylim)
    elif auto and vals:
        v = np.concatenate([np.ravel(a) for a in vals])
        v = v[np.isfinite(v) & (v > 0)] if log else v[np.isfinite(v)]
        if v.size:
            lo_, hi_ = np.percentile(v, [2, 98])
            if log:
                lo_, hi_ = min(lo_, 0.9), max(hi_, 1.1)
                b.set_ylim(lo_ / 1.25, hi_ * 1.25)
            else:
                lo_, hi_ = min(lo_, 0.95), max(hi_, 1.05)
                pad = 0.08 * (hi_ - lo_)
                b.set_ylim(lo_ - pad, hi_ + pad)
    if log:
        y0, y1 = b.get_ylim()
        b.yaxis.set_major_locator(
            MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
            if y1 / max(y0, 1e-30) < 8 else LogLocator(base=10, subs=(1, 2, 5)))
        b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        b.yaxis.set_minor_locator(NullLocator())
        b.yaxis.set_minor_formatter(NullFormatter())


def _panel_label(lo, hi, unit):
    if unit == 'delta':
        return rf'$\delta\in[{10**lo - 1:.3g},{10**hi - 1:.3g})$'
    return rf'$\log(1+\delta)\in[{lo:+.2f},{hi:+.2f})$'


_HMF_SPEC = dict(
    main='n_Wx', main_all='n_x', ratio='ratio', null='ratio_null_raw',
    cov='cov_raw', std='ratio_std', std_null='ratio_null', std_cov='cov',
    level='level', main_W_all='n_W_all', ratio_W='ratio_W',
    main_log=True, ratio_log=True, sym=r'n_{W,\delta}',
    main_label=r'$n_{W,\delta}(M_{\rm h})\ [h^3\,{\rm Mpc}^{-3}\,{\rm dex}^{-1}]$',
    ratio_label=r'$n_{W,\delta}/n_\delta$',
    over_label=r'$(n_{W,\delta}/n_\delta)\,/\,{\rm null}$',
    std_label=r'$n_{W,\delta}(M_{\rm h})\,/\,n_\delta(M_{\rm h})$',
    ratio_W_label=r'$n_{W,\delta}\,/\,n_W$',
    xlabel=r'$\log(M_{\rm h}/h^{-1}{\rm M}_\odot)$',
    by_x='n_x', by_all='n_all', by_ratio='ratio_x_all',
    all_label=r'$n_\delta(M_{\rm h})\ [h^3\,{\rm Mpc}^{-3}\,{\rm dex}^{-1}]$',
    ratio_x_all_label=r'$n_\delta\,/\,n$',
    null_n=None, cov_masks_curve=True,
    null_note=r'dashed: null (W depends on $\delta$ only)',
    over_note=r'1 = W depends on $\delta$ only',
    std_note=r'dashed: null (W indep. of $M_{\rm h}$ at fixed $\delta$)',
    level_note=r'dashed: $P(W|\delta)/P_V(W|\delta)$')


def _zf_spec(res):
    k = res.get('key', 'zf')
    kl = res.get('label_tex') or (
        r'z_{\rm f}' if k in ('zf', 'z_f') else k.replace('_', r'\_'))
    if res.get('stat', 'mean') == 'mean':
        a, b = rf'\langle {kl}\rangle', ''
    else:
        a, b = rf'\tilde{{{kl}}}', ''
    has_null = res.get('stat', 'mean') == 'mean'
    return dict(
        main='z_Wx', main_all='z_x', ratio='ratio',
        by_x='z_x', by_all='z_all', by_ratio='ratio_x_all',
        null='ratio_null' if has_null else None,
        cov='cov_null' if has_null else None,
        null_n='n_ref_eff' if has_null else None, cov_masks_curve=False,
        std=None, std_null=None, std_cov=None, level=None,
        main_W_all='z_W_all', ratio_W='ratio_W',
        main_log=False, ratio_log=False, sym=rf'{a}_{{W,\delta}}',
        main_label=rf'${a}_{{W,\delta}}(M_{{\rm h}})$',
        ratio_label=rf'${a}_{{W,\delta}}\,/\,{a}_\delta$',
        over_label=rf'$({a}_{{W,\delta}}/{a}_\delta)\,/\,{{\rm null}}$',
        std_label=None,
        ratio_W_label=rf'${a}_{{W,\delta}}\,/\,{a}_W$',
        all_label=rf'${a}(M_{{\rm h}})$',
        ratio_x_all_label=rf'${a}_\delta\,/\,{a}$',
        xlabel=r'$\log(M_{\rm h}/h^{-1}{\rm M}_\odot)$',
        null_note=('dashed: null (' + {
            'all': rf'${kl}$ depends on $M_{{\rm h}}$, $\delta$ only',
            'others': r'W like the other webs at same $M_{\rm h}$, $\delta$'
            }.get(
            res.get('null_ref'), f"W like {WEB_NAMES[int(res['null_ref'])]}"
            r" at same $M_{\rm h}$, $\delta$" if isinstance(res.get('null_ref'),
                                          (int, np.integer)) else '') + ')'),
        over_note='1 = null',
        std_note=None, level_note=None)


def smoothing_mass_cut(res=None, R_s=None, omega_m=None, kernel=None,
                       offset=None):
    r"""a = log10 M_s + offset of the plot-time cut
    log10 M_h < a + log10(1 + delta), with
    M_s = (2 pi)^{3/2} R_s^3 rho_m the mass inside the Gaussian smoothing
    kernel (log10 M_s = 12.95 at R_s = 2 h^-1 Mpc, Omega_m = 0.258) and
    offset = MASS_CUT_OFFSET = -0.5 dex by default, i.e.
    M_h < 10^{-0.5} M_s (1 + delta)  (a = 12.45 at R_s = 2).

    offset : dex added to log10 M_s (None = MASS_CUT_OFFSET; 0 = M_s itself).
    R_s, omega_m, kernel : explicit values; each one left None is taken from
        ``res`` (a result of mass_density_web / hmf_by_web_fixed_density /
        zf_by_web_fixed_density, which carry the R_s of mass_density_web),
        then from the fiducial set-up (R_s = FIDUCIAL_R_S, Omega_m = 0.258,
        Gaussian).
    """
    res = res or {}
    off = MASS_CUT_OFFSET if offset is None else float(offset)
    R = R_s if R_s is not None else res.get('R_s')
    if R is None and 'M_s' in res:          # result older than this version
        return float(np.log10(res['M_s'])) + off
    R = FIDUCIAL_R_S if R is None else float(R)
    om = omega_m if omega_m is not None else (res.get('omega_m') or 0.258)
    ke = kernel or res.get('kernel') or 'gaussian'
    return float(np.log10(self_mass_scale(R, om, ke))) + off


def _mass_cut_masks(res, mass_cut, mass_cut_at='lower', R_s=None,
                    omega_m=None, mass_cut_offset=None):
    """Mass bins kept under the cut log10 M_h < a + log10(1 + delta).

    mass_cut : None (no cut), 'smoothing' (alias 'self'; a = log10 M_s +
               mass_cut_offset, M_s from the smoothing scale, see
               :func:`smoothing_mass_cut`: M_h < 10^{-0.5} (2 pi)^{3/2}
               R_s^3 rho_m (1 + delta) by default), or a number a (manual
               override; mass_cut_offset is then ignored).
    R_s, omega_m, mass_cut_offset : override those of ``res`` / the default
               offset (MASS_CUT_OFFSET = -0.5) for mass_cut='smoothing'.
    mass_cut_at : the delta of a density panel at which the cut is taken --
               'lower' (default: the panel's lower edge, so EVERY halo of a
               shown mass bin satisfies the cut), 'center' or 'upper'.
    A mass bin is shown when its upper edge lies below the cut.
    Returns keep (n_panel, n_m), keep_all (n_m,) for curves over all
    panels (loosest panel), and the cut per panel.
    """
    lo, hi = np.asarray(res['x_lo'], float), np.asarray(res['x_hi'], float)
    n_m = len(res['lm_c'])
    if mass_cut is None:
        return (np.ones((len(lo), n_m), bool), np.ones(n_m, bool),
                np.full(len(lo), np.inf))
    if isinstance(mass_cut, str):
        if mass_cut not in ('smoothing', 'self'):
            raise ValueError("mass_cut must be None, 'smoothing' or a number")
        a = smoothing_mass_cut(res, R_s=R_s, omega_m=omega_m,
                               offset=mass_cut_offset)
    else:
        a = float(mass_cut)
    x_ref = {'lower': lo, 'center': 0.5 * (lo + hi), 'upper': hi}.get(
        mass_cut_at)
    if x_ref is None:
        raise ValueError("mass_cut_at must be 'lower', 'center' or 'upper'")
    m_max = a + x_ref
    m_hi = np.asarray(res['lm_edges'], float)[1:]
    keep = m_hi[None, :] <= m_max[:, None] + 1e-9
    return keep, keep.any(0), m_max


def _share_ratio_ylim(rat_ax, vals, log, ylim):
    """One robust y-range for all ratio axes (ratio_sharey=True)."""
    bs = [a for a in np.ravel(rat_ax) if a is not None and a.get_visible()]
    if not bs:
        return
    _style_ratio_axis(bs[0], vals, log=log, ylim=ylim)
    lim = bs[0].get_ylim()
    for a in bs[1:]:
        _style_ratio_axis(a, [], log=log, ylim=lim, auto=False)


def _band(ax, x, y, e, good, c, log=True, alpha=0.2, style='band'):
    """Jackknife error of one curve: shaded band, error bars or nothing."""
    if style in (None, False, 'none'):
        return
    if style == 'band':
        lo = np.maximum(y - e, 1e-30) if log else y - e
        ax.fill_between(x, lo, y + e, where=good, color=c, alpha=alpha, lw=0)
        return
    if style != 'bar':
        raise ValueError("err_style must be 'band', 'bar' or 'none'")
    y, e = np.asarray(y, float), np.broadcast_to(np.asarray(e, float),
                                                  np.shape(y))
    g = np.asarray(good, bool) & np.isfinite(y) & np.isfinite(e)
    if not g.any():
        return
    lo = np.minimum(e[g], 0.999 * y[g]) if log else e[g]   # stay > 0 on log
    ax.errorbar(np.asarray(x)[g], y[g], yerr=[lo, e[g]], fmt='none',
                ecolor=c, elinewidth=0.8, capsize=1.6, capthick=0.8,
                alpha=0.9, label='_nolegend_')


def _layout_cells(n, axs, *, main_and_ratio, layout, height_ratios,
                  panel_size, ratio_size, only_ratio, ratio_sharey):
    """Create or unpack the axes of a panel grid.  Returns (main, ratio,
    first_col, last_row, axs_out); main or ratio may be arrays of None."""
    none = np.array([None] * n, dtype=object)
    if axs is None:
        if only_ratio:
            size = ratio_size or (panel_size[0],
                                  max(1.5, panel_size[1] * 0.62))
            cells, _, first_col, last_row = _cell_grid(
                n, layout, False, height_ratios, size,
                share_top=ratio_sharey, share_bot=False)
            return none, cells, first_col, last_row, cells
        top, bot, first_col, last_row = _cell_grid(
            n, layout, main_and_ratio, height_ratios, panel_size,
            share_top=True, share_bot=ratio_sharey)
        if main_and_ratio:
            return top, bot, first_col, last_row, np.array([top, bot])
        return top, none, first_col, last_row, top
    axs = np.asarray(axs)
    first_col = np.arange(n) == 0
    last_row = np.ones(n, dtype=bool)
    if only_ratio:
        r = np.atleast_1d(axs).ravel()
        return none, r, first_col, last_row, r
    if main_and_ratio:
        return axs[0], axs[1], first_col, last_row, axs
    t = np.atleast_1d(axs).ravel()
    return t, none, first_col, last_row, t


_SAMPLE_LS = ('-', '--', ':', '-.')


def _as_samples(res, key):
    """(multi, names, results): a single result, or a dict {label: result}
    whose results share mass bins and density panels."""
    if isinstance(res, dict) and key in res:
        return False, [None], [res]
    if not (isinstance(res, dict) and len(res) > 0 and all(
            isinstance(v, dict) and key in v for v in res.values())):
        raise TypeError('expected a result of hmf/zf_by_web_fixed_density '
                        'or a dict {label: result} of them')
    names = list(res)
    R = [res[n] for n in names]
    b = R[0]
    for n, r in zip(names[1:], R[1:]):
        ok = all(np.shape(r[k]) == np.shape(b[k])
                 and np.allclose(r[k], b[k]) for k in ('lm_c', 'x_lo', 'x_hi'))
        if not ok:
            raise ValueError(
                f'samples {names[0]!r} and {n!r} have different mass bins or '
                'density panels: compute them together (a dict of '
                'mass_density_web outputs in hmf_by_web_fixed_density) or '
                'with the same x_panels / panels_from')
    return True, names, R


def _sample_setup(names, compare, ref, sample_ls):
    if compare not in ('ref', 'self'):
        raise ValueError("compare must be 'ref' or 'self'")
    if ref is None:
        ref_i = 0
    elif ref in names:
        ref_i = names.index(ref)
    else:
        raise KeyError(f'ref {ref!r} is not one of {names}')
    if len(sample_ls) < len(names):
        raise ValueError(f'{len(names)} samples but only {len(sample_ls)} '
                         'line styles in sample_ls')
    return ref_i


def _cross_label(sym, names, ref_i):
    others = [n for i, n in enumerate(names) if i != ref_i]
    if len(others) == 1:
        return rf'${sym}$: {others[0]} / {names[ref_i]}'
    return rf'${sym}\,/\,{sym}^{{\rm ref}}$ (ref: {names[ref_i]})'


def _sample_handles(names, sample_ls, lw, compare, ref_i):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color='0.15', ls=sample_ls[i], lw=lw)
         for i in range(len(names))]
    lab = [str(n) + (' (ref)' if compare == 'ref' and i == ref_i else '')
           for i, n in enumerate(names)]
    return h, lab


@_paper_figure('columns')
def _plot_fixed_density(res, spec, axs=None, *, kind='main',
                        ratio_panel=True, only_ratio=False, ratio_mode='raw',
                        standardise=True, n_min=30, min_cov=0.3,
                        n_ref_min=30, mass_cut=None, mass_cut_at='lower',
                        R_s=None, omega_m=None, mass_cut_offset=None,
                        all_mass_cut=True,
                        colors=WEB_COLORS, show_total=True, shade_self=None,
                        annotate=True, height_ratios=(2.2, 1.0),
                        ratio_sharey=False, ratio_ylim=None, title_in=None,
                        show_null=True, layout=None, panel_size=(3.4, 3.2),
                        ratio_size=None, err_style='band', compare='ref',
                        ref=None, sample_ls=_SAMPLE_LS, **line_kw):
    """Generic 'fixed density' figure: one cell per density panel, one curve
    per web type (and per sample, by line style, for a dict of results).
    See plot_hmf_fixed_density / plot_zf_fixed_density."""
    import functools
    _band_ = functools.partial(_band, style=err_style)
    from matplotlib.lines import Line2D

    multi, names, R = _as_samples(res, spec['main'])
    if kind == 'std' and spec['std'] is None:
        raise ValueError("kind='ratio' (standardised) exists for the HMF only")
    if ratio_mode == 'over_null' and spec['null'] is None:
        raise ValueError("ratio_mode='over_null' needs a null (stat='mean')")
    cross = False
    if multi:
        if kind != 'main':
            raise ValueError("kind='ratio' takes a single sample")
        if ratio_mode != 'raw':
            raise ValueError("ratio_mode='over_null' takes a single sample")
        ref_i = _sample_setup(names, compare, ref, sample_ls)
        cross = compare == 'ref'
        show_null = False                  # the null is a one-sample notion
        if cross:
            cr = {i: _cross_ratio(R[i], R[ref_i], spec['main'])
                  for i in range(len(R)) if i != ref_i}
            cra = {i: _cross_ratio(R[i], R[ref_i], spec['main_all'])
                   for i in range(len(R)) if i != ref_i}
    base = R[0]
    lc = base['lm_c']
    pan_lo, pan_hi = np.asarray(base['x_lo']), np.asarray(base['x_hi'])
    n_p = len(pan_lo)
    keep_m, _, m_cut = _mass_cut_masks(base, mass_cut, mass_cut_at,
                                    R_s=R_s, omega_m=omega_m,
                                    mass_cut_offset=mass_cut_offset)
    if shade_self is None:
        shade_self = mass_cut is None
    all_rvals = []
    t_in = title_in or base.get('edges_in', 'x')
    kw = dict(lw=1.6)
    kw.update(line_kw)
    kws = []
    for i in range(len(R)):
        k_ = dict(kw)
        if multi:
            k_.pop('linestyle', None)
            k_['ls'] = sample_ls[i]
        kws.append(k_)
    lMs = np.log10(base['M_s'])
    draw_main = not only_ratio
    draw_ratio = kind == 'main' and (ratio_panel or only_ratio)
    main_ax, rat_ax, first_col, last_row, axs_out = _layout_cells(
        n_p, axs, main_and_ratio=draw_main and draw_ratio, layout=layout,
        height_ratios=height_ratios, panel_size=panel_size,
        ratio_size=ratio_size, only_ratio=only_ratio,
        ratio_sharey=ratio_sharey)
    xl = spec.get('xlabel', r'$\log(M_{\rm h}/h^{-1}{\rm M}_\odot)$')
    drew_total = False

    for q in range(n_p):
        ax, b = main_ax[q], rat_ax[q]
        head = ax if ax is not None else b
        Nms = [r['N_mW'][q] for r in R]
        if all(np.nansum(N) <= 0 for N in Nms):
            for a in (ax, b):
                if a is not None:
                    a.set_visible(False)
            continue
        good_tots = [(np.nansum(N, 1) >= n_min)
                     & (keep_m[q] if all_mass_cut else True) for N in Nms]
        good_tot = np.logical_or.reduce(good_tots)
        use_std = kind == 'std' and standardise
        rvals = []
        for w in range(N_WEB):
            for i, r in enumerate(R):
                kw_i = kws[i]
                good = (Nms[i][:, w] >= n_min) & keep_m[q]
                if use_std:
                    good &= r[spec['std_cov']][q] >= min_cov
                if not good.any():
                    continue
                if ax is not None:                      # ---- main axis ----
                    mk = (spec['main'] if kind == 'main'
                          else spec['std'] if use_std else spec['ratio'])
                    y = np.where(good, r[mk][q][:, w], np.nan)
                    ax.plot(lc, y, color=colors[w], **kw_i)
                    if mk + '_err' in r:
                        _band_(ax, lc, y, r[mk + '_err'][q][:, w], good,
                               colors[w], log=spec['main_log'] or kind != 'main')
                    if use_std and show_null:
                        ax.plot(lc, np.where(good, r[spec['std_null']][q][:, w],
                                             np.nan),
                                color=colors[w], ls='--', lw=0.9, alpha=0.9)
                    elif kind == 'std' and show_null:
                        ax.axhline(r[spec['level']][q, w], color=colors[w],
                                   ls='--', lw=0.9, alpha=0.8)
                if not draw_ratio:
                    continue
                if cross:                               # ---- S / ref ----
                    if i == ref_i:
                        continue
                    g2 = good & (Nms[ref_i][:, w] >= n_min)
                    rr = np.where(g2, cr[i][0][q][:, w], np.nan)
                    b.plot(lc, rr, color=colors[w], **kw_i)
                    rvals.append(rr)
                    if cr[i][1] is not None:
                        _band_(b, lc, rr, cr[i][1][q][:, w], g2, colors[w],
                               log=spec['ratio_log'])
                    continue
                gn = good.copy()                 # where the null is usable
                if spec['cov'] is not None:
                    cv = r[spec['cov']][q]
                    gn &= (cv[:, w] if cv.ndim == 2 else cv) >= min_cov
                if spec.get('null_n') is not None:
                    gn &= r[spec['null_n']][q][:, w] >= n_ref_min
                # the coverage of the null masks the curve only where the
                # null is part of the comparison
                gr = (gn if spec.get('cov_masks_curve', True)
                      and (show_null or ratio_mode != 'raw') else good)
                rv = np.where(gr, r[spec['ratio']][q][:, w], np.nan)
                nl = (np.where(gn, r[spec['null']][q][:, w], np.nan)
                      if spec['null'] is not None else None)
                e = r.get(spec['ratio'] + '_err')
                e = None if e is None else e[q][:, w]
                if ratio_mode == 'raw':
                    b.plot(lc, rv, color=colors[w], **kw_i)
                    rvals.append(rv)
                    if show_null and nl is not None:
                        b.plot(lc, nl, color=colors[w], ls='--', lw=0.9,
                               alpha=0.9)
                        rvals.append(nl)
                    if e is not None:
                        _band_(b, lc, rv, e, gr, colors[w],
                               log=spec['ratio_log'])
                elif ratio_mode == 'over_null':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        rr = rv / nl
                    b.plot(lc, rr, color=colors[w], **kw_i)
                    rvals.append(rr)
                    if 'over_null_err' in r:         # error of the comparison
                        _band_(b, lc, rr, r['over_null_err'][q][:, w], gn,
                               colors[w], log=False)
                    elif e is not None:
                        _band_(b, lc, rr, e / nl, gn, colors[w], log=False)
                else:
                    raise ValueError("ratio_mode must be 'raw' or 'over_null'")

        for i, r in enumerate(R):                       # all web types
            gt = good_tots[i]
            if not (kind == 'main' and show_total and gt.any()):
                continue
            if ax is not None:
                ya = np.where(gt, r[spec['main_all']][q], np.nan)
                ax.plot(lc, ya, color='0.55', lw=2.6, zorder=0,
                        ls=kws[i].get('ls', '-'))
                if spec['main_all'] + '_err' in r:
                    _band_(ax, lc, ya, r[spec['main_all'] + '_err'][q],
                           gt, '0.55', log=spec['main_log'])
            if cross and draw_ratio and i != ref_i:
                g2 = gt & good_tots[ref_i]
                ra = np.where(g2, cra[i][0][q], np.nan)
                b.plot(lc, ra, color='0.55', lw=2.6, zorder=0,
                       ls=kws[i]['ls'])
                rvals.append(ra)
                if cra[i][1] is not None:
                    _band_(b, lc, ra, cra[i][1][q], g2, '0.55',
                           log=spec['ratio_log'])
                drew_total = True
            if ax is not None:
                drew_total = True
        if ax is not None:
            if kind == 'std':
                ax.axhline(1.0, color='0.75', lw=0.8, zorder=0)
            if spec['main_log'] or kind == 'std':
                ax.set_yscale('log')

        x0 = lc[good_tot][0] - 0.05 if good_tot.any() else lc[0]
        x1 = lc[good_tot][-1] + 0.05 if good_tot.any() else lc[-1]
        for a in (ax, b):
            if a is None:
                continue
            if shade_self:
                a.axvspan(lMs, lc[-1] + 0.5, color='0.9', zorder=-1, lw=0)
            a.set_xlim(x0, x1)
        head.set_title(_panel_label(pan_lo[q], pan_hi[q], t_in),
                       fontsize=13)
        if not good_tot.any() and np.isfinite(m_cut[q]):
            head.text(0.5, 0.5, 'no mass bin below\n'
                      rf'$\log M_{{\rm h}} < {m_cut[q]:.2f}$',
                      transform=head.transAxes, ha='center', va='center',
                      fontsize=11, color='0.4')

        if draw_ratio:
            b.axhline(1.0, color='0.6', lw=0.8, zorder=0)
            _style_ratio_axis(b, rvals,
                              log=(ratio_mode == 'raw' and spec['ratio_log']),
                              ylim=ratio_ylim, auto=not ratio_sharey)
            all_rvals += rvals
            if last_row[q]:
                b.set_xlabel(xl)
            if first_col[q]:
                b.set_ylabel(_cross_label(spec['sym'], names, ref_i) if cross
                             else spec['ratio_label'] if ratio_mode == 'raw'
                             else spec['over_label'])
            if annotate and q == 0 and spec['null'] is not None and (
                    show_null or ratio_mode != 'raw'):
                b.text(0.03, 0.04, spec['null_note'] if ratio_mode == 'raw'
                       else spec['over_note'],
                       transform=b.transAxes, fontsize=10, color='0.35',
                       bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.0))
        if ax is not None:
            if not draw_ratio and last_row[q]:
                ax.set_xlabel(xl)
            if first_col[q]:
                ax.set_ylabel(spec['main_label'] if kind == 'main'
                              else spec['std_label'])
            if annotate and kind == 'std' and show_null:
                ax.text(0.04, 0.04, spec['std_note'] if standardise
                        else spec['level_note'],
                        transform=ax.transAxes, fontsize=10, color='0.35')

    if draw_ratio and ratio_sharey:
        _share_ratio_ylim(rat_ax, all_rvals,
                          ratio_mode == 'raw' and spec['ratio_log'],
                          ratio_ylim)
    heads = [a for a in (main_ax if draw_main else rat_ax) if a is not None]
    handles = [Line2D([], [], color=colors[w], lw=kw['lw'])
               for w in range(N_WEB)]
    labs = list(WEB_NAMES)
    if drew_total:
        handles.append(Line2D([], [], color='0.55', lw=2.6))
        labs.append('All')
    if multi:
        h_, l_ = _sample_handles(names, sample_ls, kw['lw'], compare, ref_i)
        handles += h_
        labs += l_
    _fig_legend(np.array(heads, dtype=object), handles, labs,
                ncol=len(labs))
    return axs_out


@_paper_figure('webs')
def _plot_fixed_web(res, spec, axs=None, *, webs=(0, 1, 2, 3), layout=(2, 2),
                    ratio_panel=True, only_ratio=False, n_min=30,
                    cmap='viridis_r', cmap_range=(0.15, 1.0), show_all=True,
                    shade_self=None, title_in=None, height_ratios=(2.2, 1.0),
                    panel_size=(3.9, 3.2), ratio_size=None,
                    mass_cut=None, mass_cut_at='lower', R_s=None,
                    omega_m=None, mass_cut_offset=None, all_mass_cut=True,
                    ratio_sharey=False, ratio_ylim=None, err_style='band',
                    compare='ref', ref=None, sample_ls=_SAMPLE_LS, **line_kw):
    """Generic 'fixed web' figure: one cell per web type, one curve per
    density panel (and per sample, by line style, for a dict of results).
    See plot_hmf_fixed_web / plot_zf_fixed_web."""
    import functools
    _band_ = functools.partial(_band, style=err_style)
    from matplotlib.lines import Line2D

    multi, names, R = _as_samples(res, spec['main'])
    cross, ref_i = False, 0
    if multi:
        ref_i = _sample_setup(names, compare, ref, sample_ls)
        cross = compare == 'ref'
        if cross:
            cr = {i: _cross_ratio(R[i], R[ref_i], spec['main'])
                  for i in range(len(R)) if i != ref_i}
            cra = {i: _cross_ratio(R[i], R[ref_i], spec['main_W_all'])
                   for i in range(len(R)) if i != ref_i}
    base = R[0]
    lc = base['lm_c']
    pan_lo, pan_hi = np.asarray(base['x_lo']), np.asarray(base['x_hi'])
    n_q = len(pan_lo)
    keep_m, keep_all, _ = _mass_cut_masks(base, mass_cut, mass_cut_at,
                                    R_s=R_s, omega_m=omega_m,
                                    mass_cut_offset=mass_cut_offset)
    if shade_self is None:
        shade_self = mass_cut is None
    all_rvals = []
    t_in = title_in or base.get('edges_in', 'x')
    webs = list(webs)
    kw = dict(lw=1.6)
    kw.update(line_kw)
    kws = []
    for i in range(len(R)):
        k_ = dict(kw)
        if multi:
            k_.pop('linestyle', None)
            k_['ls'] = sample_ls[i]
        kws.append(k_)
    lMs = np.log10(base['M_s'])
    cm = plt.get_cmap(cmap)
    cols = [cm(v) for v in np.linspace(cmap_range[0], cmap_range[1],
                                       max(n_q, 2))][:n_q]
    labels = [_panel_label(pan_lo[q], pan_hi[q], t_in) for q in range(n_q)]
    draw_main = not only_ratio
    draw_ratio = ratio_panel or only_ratio
    main_ax, rat_ax, first_col, last_row, axs_out = _layout_cells(
        len(webs), axs, main_and_ratio=draw_main and draw_ratio,
        layout=layout, height_ratios=height_ratios, panel_size=panel_size,
        ratio_size=ratio_size, only_ratio=only_ratio,
        ratio_sharey=ratio_sharey)
    xl = spec.get('xlabel', r'$\log(M_{\rm h}/h^{-1}{\rm M}_\odot)$')
    drew_all = False

    for c, w in enumerate(webs):
        ax, b = main_ax[c], rat_ax[c]
        head = ax if ax is not None else b
        good_alls = [(np.nansum(r['N_mW'][:, :, w], axis=0) >= n_min)
                     & (keep_all if all_mass_cut else True) for r in R]
        good_all = np.logical_or.reduce(good_alls)
        if not good_all.any():
            for a in (ax, b):
                if a is not None:
                    a.set_visible(False)
            continue
        rvals = []
        for q in range(n_q):
            for i, r in enumerate(R):
                kw_i = kws[i]
                good = (r['N_mW'][q][:, w] >= n_min) & keep_m[q]
                if not good.any():
                    continue
                if ax is not None:
                    y = np.where(good, r[spec['main']][q][:, w], np.nan)
                    ax.plot(lc, y, color=cols[q], **kw_i)
                    if spec['main'] + '_err' in r:
                        _band_(ax, lc, y, r[spec['main'] + '_err'][q][:, w],
                               good, cols[q], log=spec['main_log'], alpha=0.18)
                if not draw_ratio:
                    continue
                if cross:
                    if i == ref_i:
                        continue
                    g2 = good & (R[ref_i]['N_mW'][q][:, w] >= n_min)
                    rr = np.where(g2, cr[i][0][q][:, w], np.nan)
                    b.plot(lc, rr, color=cols[q], **kw_i)
                    if cr[i][1] is not None:
                        _band_(b, lc, rr, cr[i][1][q][:, w], g2, cols[q],
                               log=spec['ratio_log'], alpha=0.18)
                    rvals.append(rr)
                    continue
                rv = np.where(good, r[spec['ratio_W']][q][:, w], np.nan)
                b.plot(lc, rv, color=cols[q], **kw_i)
                if spec['ratio_W'] + '_err' in r:
                    _band_(b, lc, rv, r[spec['ratio_W'] + '_err'][q][:, w],
                           good, cols[q], log=spec['ratio_log'], alpha=0.18)
                rvals.append(rv)
        if show_all:
            for i, r in enumerate(R):
                ga = good_alls[i]
                if not ga.any():
                    continue
                if ax is not None:
                    ya = np.where(ga, r[spec['main_W_all']][:, w], np.nan)
                    ax.plot(lc, ya, color='0.55', lw=2.6, zorder=0,
                            ls=kws[i].get('ls', '-'))
                    if spec['main_W_all'] + '_err' in r:
                        _band_(ax, lc, ya, r[spec['main_W_all'] + '_err'][:, w],
                               ga, '0.55', log=spec['main_log'], alpha=0.18)
                    drew_all = True
                if cross and draw_ratio and i != ref_i:
                    g2 = ga & good_alls[ref_i]
                    ra = np.where(g2, cra[i][0][:, w], np.nan)
                    b.plot(lc, ra, color='0.55', lw=2.6, zorder=0,
                           ls=kws[i]['ls'])
                    if cra[i][1] is not None:
                        _band_(b, lc, ra, cra[i][1][:, w], g2, '0.55',
                               log=spec['ratio_log'], alpha=0.18)
                    rvals.append(ra)
                    drew_all = True
        if ax is not None and spec['main_log']:
            ax.set_yscale('log')
        head.set_title(WEB_NAMES[w], fontsize=14)
        x0, x1 = lc[good_all][0] - 0.05, lc[good_all][-1] + 0.05
        for a in (ax, b):
            if a is None:
                continue
            if shade_self:
                a.axvspan(lMs, lc[-1] + 0.5, color='0.9', zorder=-1, lw=0)
            a.set_xlim(x0, x1)
        if draw_ratio:
            b.axhline(1.0, color='0.55', lw=1.0, zorder=0)
            _style_ratio_axis(b, rvals, log=spec['ratio_log'], ylim=ratio_ylim,
                              auto=not ratio_sharey)
            all_rvals += rvals
            if last_row[c]:
                b.set_xlabel(xl)
            if first_col[c]:
                b.set_ylabel(_cross_label(spec['sym'], names, ref_i) if cross
                             else spec['ratio_W_label'])
        if ax is not None:
            if not draw_ratio and last_row[c]:
                ax.set_xlabel(xl)
            if first_col[c]:
                ax.set_ylabel(spec['main_label'])

    if draw_ratio and ratio_sharey:
        _share_ratio_ylim(rat_ax, all_rvals, spec['ratio_log'], ratio_ylim)
    heads = [a for a in (main_ax if draw_main else rat_ax) if a is not None]
    handles = [Line2D([], [], color=cols[q], lw=kw['lw']) for q in range(n_q)]
    labs = list(labels)
    if show_all and (draw_main or (cross and drew_all)):
        handles.append(Line2D([], [], color='0.55', lw=2.6))
        labs.append('All densities')
    if multi:
        h_, l_ = _sample_handles(names, sample_ls, kw['lw'], compare, ref_i)
        handles += h_
        labs += l_
    _fig_legend(np.array(heads, dtype=object), handles, labs,
                ncol=min(len(labs), 3 if len(labs) > 4 else len(labs)))
    return axs_out


MASS_CUT = 'smoothing'     # default cut of every plot: log10 M_h < log10 M_s
                           # - 0.5 + log10(1 + delta), M_s from R_s
HMF_MASS_CUT = MASS_CUT    # (old name)

_COMMON_DOC = """
    Layout and size
        layout      None (one row), int (number of columns) or (nrows, ncols).
        panel_size  (width, height) in inch of one cell's upper panel.
        only_ratio  draw only the ratio panels (smaller cells, ratio_size
                    = (width, height) of one of them; default
                    (panel_size[0], ~0.62 panel_size[1])).
        ratio_panel draw the ratio panel under the main panel (default).
    x range
        xlim        in log(M_h / h^-1 M_sun): (lo, hi) for every panel; a
                    list with one (lo, hi) or None per panel, in panel
                    order; or a dict {panel index: (lo, hi)} (fixed-web
                    plots also accept web names, e.g. {'void': (11.5,
                    12.5)}).  None / missing panels: automatic, from the
                    mass bins shown.
        panel_gap   space between panels, as a fraction of the panel size:
                    None (default PANEL_GAP = 0.01, about half the matplotlib
                    default gap), a number, or (wspace, hspace).
    Mass cut
        mass_cut    'smoothing' (default), None or a number a: show only
                    mass bins with log10 M_h < log10 M_s + mass_cut_offset
                    + log10(1 + delta), M_s = (2 pi)^{3/2} R_s^3 rho_m
                    computed from the smoothing scale (log10 M_s = 12.95 at
                    R_s = 2 h^-1 Mpc, so the default cut is 12.45 +
                    log10(1 + delta)).  A number a replaces log10 M_s +
                    offset; None = no cut.
        all_mass_cut  False: the grey curve (all densities; n_delta in
                    the fixed-density plots) ignores the cut and is drawn
                    over every mass bin with >= n_min haloes (default True).
        mass_cut_offset  dex added to log10 M_s (default MASS_CUT_OFFSET
                    = -0.5; 0 = M_h < M_s (1 + delta)).
        R_s, omega_m  smoothing scale [h^-1 Mpc] and Omega_m of the cut;
                    default: those stored by mass_density_web (R_s= there).
        mass_cut_at delta of a panel at which the cut is taken: 'lower'
                    (default; every halo of a shown bin then obeys the cut),
                    'center' or 'upper'.  Curves over all panels use the
                    loosest panel.  shade_self defaults to False with a cut.
    Errors (jackknife; need n_jk_side > 0 in mass_density_web)
        err_style   'band' (shaded, default), 'bar' (error bars) or 'none'.
    Several samples: pass a dict {label: result} (same cells, e.g. from
    hmf_by_web_fixed_density with a dict of mass_density_web outputs).
        sample_ls   line style of each sample, in dict order
                    (default '-', '--', ':', '-.'); colours stay as before.
        compare     'ref' (default): the ratio panel shows sample / ref for
                    the same curve (volumes cancel: ratio of halo counts in
                    the same cells), grey = the same for all webs (fixed
                    density) or all densities (fixed web); jackknife errors
                    of the ratio itself.  'self': each sample's own ratio,
                    as for one sample.  The null is not drawn.
        ref         label of the reference sample (default: the first).
    Returns the axes: (2, n) [main, ratio] with both rows, else n axes."""


def plot_hmf_fixed_density(res, axs=None, *, kind='hmf', **kw):
    r"""Fixed density: one cell per density panel, four web types.

    Upper: n_{W,delta}(M_h) and, grey, n_delta(M_h).  Lower (ratio):
    n_{W,delta}/n_delta with the mixing-aware null dashed ('W depends on
    delta only'); ratio_mode='over_null' divides the null out (linear axis
    around 1).  kind='ratio': single row, standardised ratio.

    Defaults specific to this function: mass_cut='smoothing' (log10 M_h <
    log10 M_s - 0.5 + log10(1 + delta)), ratio panels as tall as the HMF
    panels (height_ratios=(1, 1), panel_size=(3.4, 3.0)) and one robust
    y-range for all ratio panels (ratio_sharey=True).  Pass other values to
    override.

    Options: ratio_panel, only_ratio, ratio_mode, standardise, show_null,
    n_min, min_cov, show_total, shade_self, annotate, ratio_sharey,
    ratio_ylim, title_in, layout, panel_size, ratio_size, height_ratios,
    colors, line keywords.
    """
    k = {'hmf': 'main', 'ratio': 'std'}.get(kind)
    if k is None:
        raise ValueError("kind must be 'hmf' or 'ratio'")
    kw.setdefault('mass_cut', MASS_CUT)
    # ratio panels as tall as the HMF panels, one y-range for all of them
    kw.setdefault('height_ratios', (1.0, 1.0))
    kw.setdefault('panel_size', (3.4, 3.0))
    kw.setdefault('ratio_sharey', True)
    return _plot_fixed_density(res, _HMF_SPEC, axs, kind=k, **kw)


def plot_hmf_fixed_web(res, axs=None, **kw):
    r"""Fixed web: one cell per web type (default 2 x 2), one curve per
    density panel, grey = all densities.  Lower: n_{W,delta}/n_W.
    Default mass_cut='smoothing' (see below).

    Options: webs, layout, ratio_panel, only_ratio, n_min, cmap, cmap_range,
    show_all, shade_self, title_in, ratio_sharey, ratio_ylim, panel_size,
    ratio_size, height_ratios, line keywords.
    """
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_web(res, _HMF_SPEC, axs, **kw)


def plot_zf_fixed_density(res, axs=None, **kw):
    r"""Fixed density, formation time: one cell per density panel q, four
    web types.  Upper: <z_f>_{W,x}(m) and, grey, <z_f>_x(m).  Lower:
    <z_f>_{W,x}/<z_f>_x with the null dashed (stat='mean'): what web-W
    haloes would have if z_f depended on (m, fine x) only -- solid vs dashed
    is the web effect on formation time at fixed mass and density.
    Linear axes.  Same options as :func:`plot_hmf_fixed_density` (except
    kind='ratio', which is HMF-specific), plus

        n_ref_min   hide the null (and over_null) where fewer than this many
                    effective reference haloes stand behind it (default 30).

    Unlike the HMF, the solid curve is kept where the null is hidden
    (min_cov / n_ref_min act on the null only).  Default
    mass_cut='smoothing' (log10 M_h < log10 M_s - 0.5 + log10(1 + delta),
    M_s from R_s), as for the HMF plots.
    """
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_density(res, _zf_spec(res), axs, kind='main', **kw)


def plot_zf_fixed_web(res, axs=None, **kw):
    r"""Fixed web, formation time: one cell per web type, one curve per
    density panel, grey = all densities.  Lower: <z_f>_{W,q}/<z_f>_W -- how
    formation time inside one web type changes with local density.  Same
    options as :func:`plot_hmf_fixed_web`; default mass_cut='smoothing'.
    """
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_web(res, _zf_spec(res), axs, **kw)


for _f in (plot_hmf_fixed_density, plot_hmf_fixed_web, plot_zf_fixed_density,
           plot_zf_fixed_web):
    _f.__doc__ = (_f.__doc__ or '') + _COMMON_DOC


WEB_LS = (':', '-.', '--', '-')          # void, sheet, filament, knot


@_paper_figure('single')
def _plot_fixed_web_single(res, spec, axs=None, *, webs=(0, 1, 2, 3),
                           ratio_panel=True, only_ratio=False, n_min=30,
                           cmap='viridis_r', cmap_range=(0.15, 1.0),
                           show_all=True, shade_self=None, title_in=None,
                           mass_cut=None, mass_cut_at='lower', R_s=None,
                           omega_m=None, mass_cut_offset=None,
                           all_mass_cut=True,
                           height_ratios=(2.2, 1.0), panel_size=(6.4, 4.0),
                           ratio_size=None, ratio_ylim=None, err_style='band',
                           err_alpha=0.12, web_ls=WEB_LS, title=None,
                           web_legend='upper right', legend_ncol=2,
                           layout=None, ratio_sharey=None, **line_kw):
    """Fixed web in ONE cell: colour = density panel, line style = web type.
    Same data and ratio as _plot_fixed_web (ratio to the web's own all-density
    curve, so every web's reference is the line y = 1)."""
    import functools
    _band_ = functools.partial(_band, style=err_style)
    from matplotlib.lines import Line2D

    multi, _, _ = _as_samples(res, spec['main'])
    if multi:
        raise ValueError('the single-panel view takes one sample (line '
                         'style already encodes the web type); use '
                         'plot_*_fixed_web for a dict of samples')
    if len(web_ls) < N_WEB:
        raise ValueError('web_ls needs one line style per web type (4)')
    lc = res['lm_c']
    pan_lo, pan_hi = np.asarray(res['x_lo']), np.asarray(res['x_hi'])
    n_q = len(pan_lo)
    keep_m, keep_all, _ = _mass_cut_masks(res, mass_cut, mass_cut_at,
                                    R_s=R_s, omega_m=omega_m,
                                    mass_cut_offset=mass_cut_offset)
    if shade_self is None:
        shade_self = mass_cut is None
    t_in = title_in or res.get('edges_in', 'x')
    webs = list(webs)
    kw = dict(lw=1.6)
    kw.update(line_kw)
    kw.pop('ls', None)
    kw.pop('linestyle', None)
    lMs = np.log10(res['M_s'])
    cm = plt.get_cmap(cmap)
    cols = [cm(v) for v in np.linspace(cmap_range[0], cmap_range[1],
                                       max(n_q, 2))][:n_q]
    labels = [_panel_label(pan_lo[q], pan_hi[q], t_in) for q in range(n_q)]

    draw_main = not only_ratio
    draw_ratio = ratio_panel or only_ratio
    if axs is None:
        if draw_main and draw_ratio:
            top, bot, _, _ = _cell_grid(1, None, True, height_ratios,
                                        panel_size)
            ax, b = top[0], bot[0]
            axs_out = np.array([ax, b], dtype=object)
        else:
            size = (ratio_size or (panel_size[0],
                                   max(1.8, panel_size[1] * 0.62))
                    if only_ratio else panel_size)
            top, _, _, _ = _cell_grid(1, None, False, height_ratios, size)
            ax, b = (None, top[0]) if only_ratio else (top[0], None)
            axs_out = top[0]
    else:
        a_ = np.atleast_1d(np.asarray(axs, dtype=object)).ravel()
        if draw_main and draw_ratio:
            if len(a_) != 2:
                raise ValueError('axs must be (main, ratio) axes')
            ax, b = a_[0], a_[1]
        elif only_ratio:
            ax, b = None, a_[0]
        else:
            ax, b = a_[0], None
        axs_out = axs

    good_any = np.zeros(len(lc), dtype=bool)
    used_q, used_w = set(), []
    rvals = []
    for w in webs:
        ls = web_ls[w]
        n_all = np.nansum(res['N_mW'][:, :, w], axis=0)
        good_all = (n_all >= n_min) & (keep_all if all_mass_cut else True)
        if not good_all.any():
            continue
        used_w.append(w)
        for q in range(n_q):
            good = (res['N_mW'][q][:, w] >= n_min) & keep_m[q]
            if not good.any():
                continue
            used_q.add(q)
            good_any |= good
            if ax is not None:
                y = np.where(good, res[spec['main']][q][:, w], np.nan)
                ax.plot(lc, y, color=cols[q], ls=ls, **kw)
                if spec['main'] + '_err' in res:
                    _band_(ax, lc, y, res[spec['main'] + '_err'][q][:, w],
                           good, cols[q], log=spec['main_log'],
                           alpha=err_alpha)
            if b is not None:
                r = np.where(good, res[spec['ratio_W']][q][:, w], np.nan)
                b.plot(lc, r, color=cols[q], ls=ls, **kw)
                rvals.append(r)
                if spec['ratio_W'] + '_err' in res:
                    _band_(b, lc, r, res[spec['ratio_W'] + '_err'][q][:, w],
                           good, cols[q], log=spec['ratio_log'],
                           alpha=err_alpha)
        if show_all and ax is not None:
            ya = np.where(good_all, res[spec['main_W_all']][:, w], np.nan)
            ax.plot(lc, ya, color='0.55', lw=2.4, ls=ls, zorder=0)
            if spec['main_W_all'] + '_err' in res:
                _band_(ax, lc, ya, res[spec['main_W_all'] + '_err'][:, w],
                       good_all, '0.55', log=spec['main_log'],
                       alpha=err_alpha)
            good_any |= good_all
    if not good_any.any():
        raise ValueError(f'no mass bin with >= {n_min} haloes')

    xl = spec.get('xlabel', r'$\log(M_{\rm h}/h^{-1}{\rm M}_\odot)$')
    x0, x1 = lc[good_any][0] - 0.05, lc[good_any][-1] + 0.05
    for a in (ax, b):
        if a is None:
            continue
        if shade_self:
            a.axvspan(lMs, lc[-1] + 0.5, color='0.9', zorder=-1, lw=0)
        a.set_xlim(x0, x1)
    if ax is not None:
        if spec['main_log']:
            ax.set_yscale('log')
        ax.set_ylabel(spec['main_label'])
        if b is None:
            ax.set_xlabel(xl)
    if b is not None:
        b.axhline(1.0, color='0.55', lw=1.0, zorder=0)
        _style_ratio_axis(b, rvals, log=spec['ratio_log'], ylim=ratio_ylim)
        b.set_xlabel(xl)
        b.set_ylabel(spec['ratio_W_label'])
    head = ax if ax is not None else b
    if title is not None:
        head.set_title(title, fontsize=14)

    # colours (density panels) above the figure, line styles (web types)
    # inside the axis
    qs = sorted(used_q)
    handles = [Line2D([], [], color=cols[q], lw=kw['lw']) for q in qs]
    labs = [labels[q] for q in qs]
    if show_all and ax is not None:
        handles.append(Line2D([], [], color='0.55', lw=2.4))
        labs.append('All densities')
    _fig_legend(np.array([head], dtype=object), handles, labs,
                ncol=min(legend_ncol, len(labs)))
    if web_legend is not None:
        head.legend([Line2D([], [], color='0.15', ls=web_ls[w], lw=kw['lw'])
                     for w in used_w], [WEB_NAMES[w] for w in used_w],
                    loc=web_legend, fontsize=11, frameon=False,
                    handlelength=2.8)
    return axs_out


def plot_spin_fixed_density(res, axs=None, **kw):
    r"""Spin-mass relation at fixed density (result of
    :func:`spin_by_web_fixed_density`); same options and defaults as
    :func:`plot_zf_fixed_density` (mass_cut='smoothing', null for
    stat='mean')."""
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_density(res, _zf_spec(res), axs, kind='main', **kw)


def plot_spin_fixed_web(res, axs=None, **kw):
    r"""Spin-mass relation at fixed web type; options as
    :func:`plot_zf_fixed_web`."""
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_web(res, _zf_spec(res), axs, **kw)


def plot_c_fixed_density(res, axs=None, **kw):
    r"""Concentration-mass relation at fixed density (result of
    :func:`concentration_by_web_fixed_density`); same options and defaults
    as :func:`plot_zf_fixed_density`."""
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_density(res, _zf_spec(res), axs, kind='main', **kw)


def plot_c_fixed_web(res, axs=None, **kw):
    r"""Concentration-mass relation at fixed web type; options as
    :func:`plot_zf_fixed_web`."""
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_web(res, _zf_spec(res), axs, **kw)


@_paper_figure('single')
def _plot_by_density(res, spec, axs=None, *, ratio_panel=True,
                     only_ratio=False, n_min=30, cmap='viridis_r',
                     cmap_range=(0.15, 1.0), show_all=True, shade_self=None,
                     title_in=None, height_ratios=(2.2, 1.0),
                     panel_size=(6.2, 3.8), ratio_size=None, ratio_ylim=None,
                     err_style='band', err_alpha=0.18, mass_cut=None,
                     mass_cut_at='lower', R_s=None, omega_m=None,
                     mass_cut_offset=None, all_mass_cut=True,
                     title=None, legend_ncol=2,
                     all_label='All haloes', **line_kw):
    """One cell: the all-web curve of every density panel (colour, as in the
    fixed-web figures) and the curve of all haloes (grey).  Lower panel:
    each density panel relative to all haloes."""
    import functools
    _band_ = functools.partial(_band, style=err_style)
    from matplotlib.lines import Line2D

    multi, _, _ = _as_samples(res, spec['main'])
    if multi:
        raise ValueError('this view takes one sample')
    lc = res['lm_c']
    pan_lo, pan_hi = np.asarray(res['x_lo']), np.asarray(res['x_hi'])
    n_q = len(pan_lo)
    N_q = np.nansum(res['N_mW'], axis=2)                    # (n_q, n_m)
    k_x, k_all, k_rat = (spec.get('by_x', 'z_x'), spec.get('by_all', 'z_all'),
                         spec.get('by_ratio', 'ratio_x_all'))
    if k_all in res:
        z_all, z_all_e = res[k_all], res.get(k_all + '_err')
        rat, rat_e = res[k_rat], res.get(k_rat + '_err')
    elif k_all == 'n_all':      # HMF from before n_all existed: no errors
        warnings.warn('result has no n_all (computed with an older version):'
                      ' overall HMF rebuilt from the panels, without '
                      'jackknife errors; recompute for errors')
        dlm = float(np.diff(res['lm_edges'])[0])
        with np.errstate(divide='ignore', invalid='ignore'):
            z_all = N_q.sum(0) / (np.nansum(res['V_Wx']) * dlm)
            rat = res[k_x] / z_all[None]
        z_all_e = rat_e = None
    else:                       # result from before z_all existed: no errors
        if res.get('stat', 'mean') != 'mean':
            raise ValueError('recompute the result: the overall median '
                             'cannot be rebuilt from the panel medians')
        warnings.warn('result has no z_all (computed with an older version):'
                      ' overall curve rebuilt from the panels, without '
                      'jackknife errors; recompute for errors')
        with np.errstate(divide='ignore', invalid='ignore'):
            z_all = (np.nansum(N_q * res['z_x'], 0) / N_q.sum(0))
            rat = res['z_x'] / z_all[None]
        z_all_e = rat_e = None
    z_x, z_x_e = res[k_x], res.get(k_x + '_err')
    keep_m, keep_all, _ = _mass_cut_masks(res, mass_cut, mass_cut_at,
                                    R_s=R_s, omega_m=omega_m,
                                    mass_cut_offset=mass_cut_offset)
    if shade_self is None:
        shade_self = mass_cut is None
    t_in = title_in or res.get('edges_in', 'x')
    kw = dict(lw=1.6)
    kw.update(line_kw)
    lMs = np.log10(res['M_s'])
    cm = plt.get_cmap(cmap)
    cols = [cm(v) for v in np.linspace(cmap_range[0], cmap_range[1],
                                       max(n_q, 2))][:n_q]

    draw_main = not only_ratio
    draw_ratio = ratio_panel or only_ratio
    if axs is None:
        if draw_main and draw_ratio:
            top, bot, _, _ = _cell_grid(1, None, True, height_ratios,
                                        panel_size)
            ax, b = top[0], bot[0]
            axs_out = np.array([ax, b], dtype=object)
        else:
            size = (ratio_size or (panel_size[0],
                                   max(1.8, panel_size[1] * 0.62))
                    if only_ratio else panel_size)
            top, _, _, _ = _cell_grid(1, None, False, height_ratios, size)
            ax, b = (None, top[0]) if only_ratio else (top[0], None)
            axs_out = top[0]
    else:
        a_ = np.atleast_1d(np.asarray(axs, dtype=object)).ravel()
        if draw_main and draw_ratio:
            if len(a_) != 2:
                raise ValueError('axs must be (main, ratio) axes')
            ax, b = a_[0], a_[1]
        elif only_ratio:
            ax, b = None, a_[0]
        else:
            ax, b = a_[0], None
        axs_out = axs

    good_all = (N_q.sum(0) >= n_min) & (keep_all if all_mass_cut
                                          else True)
    good_any = good_all.copy() if show_all else np.zeros(len(lc), bool)
    rvals, used_q = [], []
    for q in range(n_q):
        good = (N_q[q] >= n_min) & keep_m[q]
        if not good.any():
            continue
        used_q.append(q)
        good_any |= good
        if ax is not None:
            y = np.where(good, z_x[q], np.nan)
            ax.plot(lc, y, color=cols[q], **kw)
            if z_x_e is not None:
                _band_(ax, lc, y, z_x_e[q], good, cols[q],
                       log=spec['main_log'], alpha=err_alpha)
        if b is not None:
            g2 = good & good_all
            r = np.where(g2, rat[q], np.nan)
            b.plot(lc, r, color=cols[q], **kw)
            rvals.append(r)
            if rat_e is not None:
                _band_(b, lc, r, rat_e[q], g2, cols[q],
                       log=spec['ratio_log'], alpha=err_alpha)
    if show_all and ax is not None and good_all.any():
        ya = np.where(good_all, z_all, np.nan)
        ax.plot(lc, ya, color='0.35', lw=2.6, zorder=0)
        if z_all_e is not None:
            _band_(ax, lc, ya, z_all_e, good_all, '0.35',
                   log=spec['main_log'], alpha=err_alpha)
    if not good_any.any():
        raise ValueError(f'no mass bin with >= {n_min} haloes below the cut')

    xl = spec.get('xlabel', r'$\log(M_{\rm h}/h^{-1}{\rm M}_\odot)$')
    x0, x1 = lc[good_any][0] - 0.05, lc[good_any][-1] + 0.05
    for a in (ax, b):
        if a is None:
            continue
        if shade_self:
            a.axvspan(lMs, lc[-1] + 0.5, color='0.9', zorder=-1, lw=0)
        a.set_xlim(x0, x1)
    if ax is not None:
        if spec['main_log']:
            ax.set_yscale('log')
        ax.set_ylabel(spec['all_label'])
        if b is None:
            ax.set_xlabel(xl)
    if b is not None:
        b.axhline(1.0, color='0.35', lw=1.0, zorder=0)
        _style_ratio_axis(b, rvals, log=spec['ratio_log'], ylim=ratio_ylim)
        b.set_xlabel(xl)
        b.set_ylabel(spec['ratio_x_all_label'])
    head = ax if ax is not None else b
    if title is not None:
        head.set_title(title, fontsize=14)
    handles = [Line2D([], [], color=cols[q], lw=kw['lw']) for q in used_q]
    labs = [_panel_label(pan_lo[q], pan_hi[q], t_in) for q in used_q]
    if show_all and ax is not None:
        handles.append(Line2D([], [], color='0.35', lw=2.6))
        labs.append(all_label)
    _fig_legend(np.array([head], dtype=object), handles, labs,
                ncol=min(legend_ncol, len(labs)))
    return axs_out


_BY_DENSITY_DOC = """
    One panel, no split by web type: the curve of every density panel (all
    web types together; colours as in the fixed-web figures) and, grey, the
    curve of all haloes (all densities and web types).  Lower panel: each
    density panel relative to all haloes, with jackknife errors of the ratio
    (results computed before this function existed have no z_all / n_all:
    the grey curve is then rebuilt from the panels without errors --
    recompute).

    Options: xlim, ratio_panel, only_ratio, n_min (per density panel, all webs),
    mass_cut (default 'smoothing': log10 M_h < log10 M_s - 0.5 +
    log10(1 + delta), M_s from R_s), mass_cut_offset, R_s, omega_m,
    mass_cut_at (grey curve: loosest panel),
    show_all, all_label, cmap, cmap_range, err_style, err_alpha,
    panel_size (default (6.2, 3.8)), height_ratios, ratio_size, ratio_ylim,
    title_in ('x' or 'delta' for the legend), title, legend_ncol,
    shade_self, line keywords.  axs: (main, ratio) or one axis.
    Returns the axes."""


def plot_hmf_by_density(res, axs=None, **kw):
    r"""HMF per density panel, all web types together: n_delta(M_h) = N(M_h,
    delta) / (V(delta) dlog M) in the volume of each density panel (result
    of :func:`hmf_by_web_fixed_density`) and, grey, n(M_h) over the union of
    the panels.  Lower panel: n_delta / n (log axis).  n_min applies to the
    halo count of a panel's mass bin."""
    kw.setdefault('mass_cut', MASS_CUT)
    kw.setdefault('mass_cut_at', 'center')
    return _plot_by_density(res, _HMF_SPEC, axs, **kw)


def plot_zf_by_density(res, axs=None, **kw):
    r"""Formation time vs mass per density panel, all web types together."""
    kw.setdefault('mass_cut', MASS_CUT)
    kw.setdefault('mass_cut_at', 'center')
    return _plot_by_density(res, _zf_spec(res), axs, **kw)


def plot_spin_by_density(res, axs=None, **kw):
    r"""Spin vs mass per density panel, all web types together."""
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_by_density(res, _zf_spec(res), axs, **kw)


def plot_c_by_density(res, axs=None, **kw):
    r"""Concentration vs mass per density panel, all web types together."""
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_by_density(res, _zf_spec(res), axs, **kw)


for _f in (plot_hmf_by_density, plot_zf_by_density, plot_spin_by_density,
           plot_c_by_density):
    _f.__doc__ = _f.__doc__ + _BY_DENSITY_DOC


_SINGLE_DOC = """
    One cell instead of one per web type: colour = density panel (as in
    the multi-panel version), line style = web type (``web_ls``, default
    void ':', sheet '-.', filament '--', knot '-'); grey = each web's
    all-density curve, in that web's line style (``show_all``).  The ratio
    panel is <.>_{W,q} / <.>_W as in the multi-panel version, so the
    reference of every web is the line y = 1.

    Accepts the arguments of the multi-panel function (``xlim`` included;
    ``layout`` and
    ``ratio_sharey`` are ignored; one sample only), plus
        web_ls     four line styles, indexed by web type
        web_legend location of the line-style legend inside the axis
                   (default 'upper right'; None = no legend)
        legend_ncol columns of the colour legend above the figure (2)
        err_alpha  opacity of the error bands (default 0.12)
        title      optional title
        panel_size (width, height) of the main axis (default (6.4, 4.0))
    ``axs``: (main, ratio) axes, or one axis with only_ratio /
    ratio_panel=False.  Returns the axes."""


def plot_zf_fixed_web_single(res, axs=None, **kw):
    r"""Fixed web, formation time, all web types in one panel."""
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_web_single(res, _zf_spec(res), axs, **kw)


def plot_hmf_fixed_web_single(res, axs=None, **kw):
    r"""Fixed web, HMF, all web types in one panel."""
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_web_single(res, _HMF_SPEC, axs, **kw)


def plot_spin_fixed_web_single(res, axs=None, **kw):
    r"""Spin-mass relation, all web types in one panel."""
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_web_single(res, _zf_spec(res), axs, **kw)


def plot_c_fixed_web_single(res, axs=None, **kw):
    r"""Concentration-mass relation, all web types in one panel."""
    kw.setdefault('mass_cut', MASS_CUT)
    return _plot_fixed_web_single(res, _zf_spec(res), axs, **kw)


for _f in (plot_zf_fixed_web_single, plot_hmf_fixed_web_single,
           plot_spin_fixed_web_single, plot_c_fixed_web_single):
    _f.__doc__ = _f.__doc__ + _SINGLE_DOC
for _f in (plot_spin_fixed_density, plot_spin_fixed_web, plot_c_fixed_density,
           plot_c_fixed_web):
    _f.__doc__ = _f.__doc__ + _COMMON_DOC



def plot_fix_mass_pdf(res, axs=None, *, mass_panels=None, joint=False,
                      show_total=True, show_median=True, n_min=50,
                      sharey=True, colors=WEB_COLORS, **line_kw):
    r"""Fix m: density distribution of the haloes of each web type -- one
    panel per mass bin, four curves.

    Each curve is the 1-D projection of the corresponding heat map of
    :func:`plot_fix_mass` onto the density axis:

        joint=False (default):  p(x | W, m in bin)         [dex^-1]
                                every curve integrates to 1, so the shapes
                                of the four web types are compared directly.
        joint=True:             P(W | m in bin) p(x | W, m in bin)
                                the curves add up to the grey total
                                p(x | m in bin), so relative abundance shows.

    Errors are the jackknife ones of ``mass_density_web`` (band); in the
    joint form only the p(x | W) part of the error is propagated.  Web types
    with fewer than ``n_min`` haloes in a bin are not drawn (noted in the
    legend).  Dashed vertical lines mark each curve's median.
    """
    lp, lm_e, xc = res['lm_panels'], res['lm_edges'], res['x_c']
    x_e = res['x_edges']
    _, m_pan = _snap(np.asarray(lp, float), lm_e)
    n_all = len(lp) - 1
    rows = [p for p in (range(n_all) if mass_panels is None else mass_panels)
            if res['T'][m_pan == p].sum() > 0]
    if not rows:
        raise ValueError('no populated mass panel to show')
    if axs is None:
        _, axs = plt.subplots(1, len(rows), figsize=(2.9 * len(rows), 2.9),
                              sharey=sharey, constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    kw = dict(lw=1.7)
    kw.update(line_kw)
    has_err = 'fixM_px_w_err' in res
    dx = np.diff(x_e)
    ymax = np.zeros(len(rows))

    for j, p in enumerate(rows):
        ax = axs[j]
        Tp = res['T'][m_pan == p].sum(0)                  # (x, W)
        n_w = Tp.sum(0)
        P_w = n_w / max(n_w.sum(), 1e-30)
        tot = Tp.sum(1) / max(Tp.sum(), 1e-30) / dx       # p(x | m in bin)
        lo, hi = _hist_quantiles(Tp.sum(1), x_e, [0.003, 0.997])
        if show_total and joint:
            ax.plot(xc, tot, color='0.6', lw=2.4, zorder=0, label='All')
            ymax[j] = max(ymax[j], np.nanmax(tot))
        for w in range(N_WEB):
            lab = f'{WEB_NAMES[w]} ({100 * P_w[w]:.1f}%)'
            if n_w[w] < n_min:
                ax.plot([], [], color=colors[w], lw=kw['lw'],
                        label=f'{WEB_NAMES[w]} (N={int(n_w[w])}, not shown)')
                continue
            y = res['fixM_px_w'][p][:, w]
            scale = P_w[w] if joint else 1.0
            ax.plot(xc, scale * y, color=colors[w], label=lab, **kw)
            ymax[j] = max(ymax[j], np.nanmax(scale * y))
            if has_err:
                e = res['fixM_px_w_err'][p][:, w]
                ax.fill_between(xc, scale * (y - e), scale * (y + e),
                                color=colors[w], alpha=0.2, lw=0)
            if show_median:
                med = _hist_quantiles(Tp[:, w], x_e, [0.5])[0]
                ax.axvline(med, color=colors[w], ls='--', lw=0.9, alpha=0.7)
        ax.set_xlim(lo - 0.05, hi + 0.05)
        ax.set_title(rf'$\log_{{10}}M\in[{lp[p]:.1f},{lp[p+1]:.1f})$',
                     fontsize=9)
        ax.set_xlabel(r'$\log_{10}(1+\delta)$')
        if j == 0:
            ax.set_ylabel(r'$P(W|m)\,p(x\,|\,W, m)$' if joint
                          else r'$p(x\,|\,W, m)\ [{\rm dex}^{-1}]$')
        ax.legend(frameon=False, fontsize=6.8, loc='best',
                  handlelength=1.4)
    # y-limits only after everything is drawn (an early set_ylim on a shared
    # axis would freeze the top at the first panel's maximum)
    for j in range(len(rows)):
        top = (ymax.max() if sharey else ymax[j]) * 1.08
        axs[j].set_ylim(0.0, top if top > 0 else 1.0)
    return axs
