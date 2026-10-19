"""
Geometric diagnostics of the tidal eigen-frame.

Two questions this module answers, both of which set the range of separations
over which the 3D-WCCF carries a local, physical meaning:

1. Frame coherence.  How far can one walk away from a web cell before the
   eigen-frame carried by that cell stops describing the local structure?
   ``plot_frame_coherence`` measures

       C_m(r; p) = < | e_m(x) . e_m(x + r * e_p(x)) | >

   i.e. the eigenvector m is compared at two points separated by r *along the
   local eigen-direction p*.  Isotropically random directions give
   <|cos|> = 1/2, so C = 1 means perfect coherence and C = 0.5 means the frame
   has fully decorrelated.  The same walk also records the web-type
   persistence P(r; p) = P[ type(x + r e_p) = type(x) ], which quantifies the
   "we have walked out of the filament" effect.

2. Eigenvalue degeneracy.  When two eigenvalues are close the corresponding
   eigenvectors are only defined up to a rotation in their degenerate
   subspace, so any measured anisotropy between them is meaningless.
   ``plot_eigen_degeneracy`` shows the distribution of

       T = (l2 - l3) / (l1 - l3)   in [0, 1]

   (T -> 0: e2, e3 degenerate; T -> 1: e1, e2 degenerate; T = 0.5: no
   degeneracy) and, empirically, how the one-cell eigenvector stability
   depends on it.

Both functions take a ``pywccf.samples.TidalField`` (only ``.mesh``,
``.lams``, ``.n_lams`` and ``.recon_mask`` are used, so any object with those
attributes works) plus the eigenvector array, which ``TidalField.from_file``
does not currently read -- use ``load_evec(path)`` for that.

Eigenvector layout follows ``pyccb.process.web_web``:
``evec[i, j, k, :, m]`` is e_{m+1}, with l1 >= l2 >= l3 (m = 0 -> e1).
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['load_evec', 'measure_frame_coherence', 'plot_frame_coherence',
           'plot_web_transition', 'plot_eigen_degeneracy']

# index = number of eigenvalues above the threshold = tf.n_lams
WEB_NAMES = ('Void', 'Sheet', 'Filament', 'Knot')
WEB_ORDER = (3, 2, 1, 0)                       # rows, top to bottom
WEB_COLORS = ('#8abcd1', '#b9dec9', '#f9cb8b', '#f2cac9')
E_LABELS = (r'$\mathbf{e}_1$', r'$\mathbf{e}_2$', r'$\mathbf{e}_3$')
E_COLORS = ('#f33b1f', '#00ac38', '#5e76ff')


def load_evec(path, dtype=np.float32):
    """Read the 'evec' dataset, shape (n, n, n, 3, 3), from a field file."""
    with h5py.File(path, 'r', locking=False) as f:
        return np.asarray(f['evec'][()], dtype=dtype)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _grid(tf):
    return int(tf.mesh.n_grids), float(tf.mesh.l_grid)


def _sample_cells(tf, web_type, n_sample, rng):
    """Random grid indices (N, 3) of cells of the given web type."""
    mask = (tf.n_lams == web_type)
    if getattr(tf, 'recon_mask', None) is not None:
        mask = mask & tf.recon_mask
    flat = np.flatnonzero(mask.ravel())
    if flat.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if flat.size > n_sample:
        flat = rng.choice(flat, n_sample, replace=False)
    return np.stack(np.unravel_index(flat, mask.shape), axis=-1)


def _check_lam_order(tf, n_check=10_000):
    """This module (like ``pyccb.process.web_web``) assumes ``tf.lams`` is
    sorted descending, l1 >= l2 >= l3, and that column m of ``evec`` is the
    eigenvector belonging to ``tf.lams[..., m]``.  Cheap guard against a file
    written with the opposite (numpy.linalg.eigh) convention."""
    lam = np.asarray(tf.lams).reshape(-1, 3)[:n_check]
    if not np.all(np.diff(lam, axis=-1) <= 1e-6):
        raise ValueError(
            'tf.lams is not in descending order (l1 >= l2 >= l3). The '
            'eigen-frame labelling used here assumes it is; re-sort lams and '
            'the matching columns of evec before calling.')


def _gather(arr, ijk):
    return arr[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def _crossing(rs, c, level):
    """First r at which the curve c(r) drops below `level` (linear interp)."""
    below = np.flatnonzero(c < level)
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return rs[0]
    return float(np.interp(level, [c[i], c[i - 1]], [rs[i], rs[i - 1]]))


# ----------------------------------------------------------------------------
# 1. frame coherence
# ----------------------------------------------------------------------------

def measure_frame_coherence(tf, evec, *, r_max=20.0, n_r=21,
                            n_sample=100_000, seed=42):
    """Measure C_m(r; p) and the web-type persistence P(r; p).

    Returns a dict with
        'r'       : (n_r,)                separations [Mpc/h]
        'coh'     : (4, 3, 3, n_r)        [web type, probe dir p, evec m, r]
        'persist' : (4, 3, n_r)           [web type, probe dir p, r]
        'trans'   : (4, 3, n_r, 4)        landing-type fractions
        'n_cell'  : (4,)                  cells actually used
    Web-type index is tf.n_lams: 0 void, 1 sheet, 2 filament, 3 knot.
    """
    _check_lam_order(tf)
    n, h = _grid(tf)
    rs = np.linspace(0.0, r_max, n_r)
    rng = np.random.default_rng(seed)

    coh = np.full((4, 3, 3, n_r), np.nan)
    per = np.full((4, 3, n_r), np.nan)
    trn = np.full((4, 3, n_r, 4), np.nan)
    n_cell = np.zeros(4, dtype=np.int64)

    for t in range(4):
        ijk = _sample_cells(tf, t, n_sample, rng)
        n_cell[t] = len(ijk)
        if len(ijk) == 0:
            continue
        e0 = _gather(evec, ijk).astype(np.float64)          # (N, 3, 3)
        x0 = ijk * h
        for p in range(3):
            step = e0[:, :, p]                              # local direction
            for ir, r in enumerate(rs):
                ijk_r = np.rint((x0 + r * step) / h).astype(np.int64) % n
                e_r = _gather(evec, ijk_r)
                coh[t, p, :, ir] = np.abs(
                    np.einsum('nim,nim->nm', e0, e_r)).mean(0)
                t_r = _gather(tf.n_lams, ijk_r)
                per[t, p, ir] = (t_r == t).mean()
                trn[t, p, ir] = np.bincount(t_r, minlength=4) / len(t_r)

    return {'r': rs, 'coh': coh, 'persist': per, 'trans': trn,
            'n_cell': n_cell}


def plot_frame_coherence(tf, evec, *, save_path=None, r_max=20.0, n_r=21,
                         n_sample=100_000, seed=42, c_th=0.75, res=None,
                         r_s=None, figsize=(11.0, 12.0), dpi=150):
    """Rows = web type, columns = probe direction e_p, curves = C_m plus the
    web-type persistence (black dashed).

    `c_th` defines the quoted coherence length L_m (where C_m falls to it).
    `r_s` (the smoothing scale) is only drawn as a guide line.
    Pass a previous `measure_frame_coherence` output as `res` to re-plot
    without re-measuring.  `save_path=None` does not save.
    Returns (fig, res).
    """
    if res is None:
        res = measure_frame_coherence(tf, evec, r_max=r_max, n_r=n_r,
                                      n_sample=n_sample, seed=seed)
    rs, coh, per = res['r'], res['coh'], res['persist']
    if r_s is None:
        r_s = getattr(tf, 'smoothing_scale', np.nan)
    r_s = float(r_s)

    fig, axes = plt.subplots(4, 3, figsize=figsize, dpi=dpi,
                             sharex=True, sharey=True, constrained_layout=True)
    for row, t in enumerate(WEB_ORDER):
        for p in range(3):
            ax = axes[row, p]
            ax.axhline(0.5, color='0.6', lw=0.8, ls=':')
            ax.axhline(c_th, color='0.6', lw=0.8, ls='--')
            if np.isfinite(r_s):
                ax.axvline(r_s, color='0.6', lw=0.8, ls='-')
            for m in range(3):
                ax.plot(rs, coh[t, p, m], color=E_COLORS[m], lw=1.6,
                        label=rf'$C_{{{m + 1}}}$')
            ax.plot(rs, per[t, p], color='k', lw=1.2, ls='--',
                    label='same type')
            lab = []
            for m in range(3):
                L = _crossing(rs, coh[t, p, m], c_th)
                lab.append(rf'$L_{{{m + 1}}}>{rs[-1]:.0f}$' if np.isnan(L)
                           else rf'$L_{{{m + 1}}}={L:.1f}$')
            ax.text(0.97, 0.97, '\n'.join(lab), transform=ax.transAxes,
                    ha='right', va='top', fontsize=8)
            if row == 0:
                ax.set_title(rf'Along {E_LABELS[p]}')
            if row == 3:
                ax.set_xlabel(r'$r\ [h^{-1}\mathrm{Mpc}]$')
            if p == 0:
                ax.set_ylabel(f'{WEB_NAMES[t]}\n' r'$\langle|\mathbf{e}_i\cdot \mathbf{e}_j|\rangle,\ P_{\rm same}$')
    axes[0, 0].set_xlim(0, rs[-1])
    axes[0, 0].set_ylim(0.0, 1.02)
    # axes[0, 2].legend(fontsize=8, ncol=2, loc='lower left')
    # fig.suptitle('Coherence of the tidal eigen-frame and web-type persistence'
    #              + (rf'  ($R_{{\rm s}}={r_s:g}\ h^{{-1}}$Mpc)'
    #                 if np.isfinite(r_s) else ''))
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig, res


def plot_web_transition(res, *, save_path=None, figsize=(11.0, 12.0),
                        dpi=150, r_s=None):
    """Where does the walk actually end up?  Stacked fractions
    P[type(x + r e_p) = t' | type(x) = t] from a ``measure_frame_coherence``
    result.  Rows = starting web type, columns = probe direction.

    This is the quantitative version of "at r = 8 Mpc/h, half of the signal
    attributed to filaments no longer comes from filaments": the thickness of
    the same-type band is the fraction of the 3D-WCCF at that separation that
    is still sampling the environment it is labelled with.  `save_path=None`
    does not save.  Returns fig.
    """
    rs, trn = res['r'], res['trans']
    fig, axes = plt.subplots(4, 3, figsize=figsize, dpi=dpi,
                             sharex=True, sharey=True, constrained_layout=True)
    for row, t in enumerate(WEB_ORDER):
        for p in range(3):
            ax = axes[row, p]
            ax.stackplot(rs, *[trn[t, p, :, tp] for tp in range(4)],
                         colors=WEB_COLORS, labels=WEB_NAMES, alpha=0.85)
            ax.plot(rs, trn[t, p, :, t], color='k', lw=1.4)
            if r_s is not None:
                ax.axvline(float(r_s), color='w', lw=1.0, ls='--')
            if row == 0:
                ax.set_title(rf'Along {E_LABELS[p]}')
            if row == 3:
                ax.set_xlabel(r'$r\ [h^{-1}\mathrm{Mpc}]$')
            if p == 0:
                ax.set_ylabel(f'{WEB_NAMES[t]}')
    axes[0, 0].set_xlim(rs[0], rs[-1])
    axes[0, 0].set_ylim(0, 1)
    # axes[0, 2].legend(fontsize=8, ncol=2, loc='lower left', framealpha=0.9)
    # fig.suptitle('Web-type transition along the local eigen-frame '
    #              '(black: same type)')
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# 2. eigenvalue degeneracy
# ----------------------------------------------------------------------------

def plot_eigen_degeneracy(tf, evec=None, *, save_path=None, n_sample=200_000,
                          r_probe=None, n_T_bins=20, seed=42,
                          figsize=(11.0, 4.2), dpi=150):
    """Left: PDF of T = (l2 - l3)/(l1 - l3) per web type.  Right (needs
    `evec`): median eigenvector stability |e_m(x).e_m(x + d)| as a function of
    T, for a displacement of length `r_probe` [Mpc/h] in an isotropically
    random direction (default: the smoothing scale, else 3 grid cells).

    Returns (fig, res) where res holds the sampled T values per web type and
    the degenerate fractions f(T < 0.1) and f(T > 0.9).
    """
    _check_lam_order(tf)
    n, h = _grid(tf)
    rng = np.random.default_rng(seed)
    if r_probe is None:
        r_probe = float(getattr(tf, 'smoothing_scale', np.nan))
        if not np.isfinite(r_probe):
            r_probe = 3.0 * h
    res = {'T': {}, 'f_lo': np.full(4, np.nan), 'f_hi': np.full(4, np.nan)}

    n_panel = 1 if evec is None else 2
    fig, axes = plt.subplots(1, n_panel, figsize=(figsize[0] / 2 * n_panel,
                                                  figsize[1]),
                             dpi=dpi, constrained_layout=True)
    axes = np.atleast_1d(axes)

    ijk_all, T_all = [], []
    for t in range(4):
        ijk = _sample_cells(tf, t, n_sample, rng)
        if len(ijk) == 0:
            continue
        lam = _gather(tf.lams, ijk).astype(np.float64)      # (N, 3), l1>=l2>=l3
        spread = lam[:, 0] - lam[:, 2]
        ok = spread > 1e-12
        T = (lam[ok, 1] - lam[ok, 2]) / spread[ok]
        res['T'][WEB_NAMES[t]] = T
        res['f_lo'][t] = (T < 0.1).mean()
        res['f_hi'][t] = (T > 0.9).mean()
        axes[0].hist(T, bins=50, range=(0, 1), density=True, histtype='step',
                     lw=1.6, color=WEB_COLORS[t], label=f'{WEB_NAMES[t]} '
                     rf'($f_{{T<0.1}}={res["f_lo"][t]:.2f}$)')
        ijk_all.append(ijk[ok])
        T_all.append(T)

    axes[0].axvline(0.5, color='0.6', lw=0.8, ls=':')
    axes[0].set_xlabel(r'$T=(\lambda_2-\lambda_3)/(\lambda_1-\lambda_3)$')
    axes[0].set_ylabel('PDF')
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=8)
    axes[0].set_title(r'$T\to0$: $\mathbf{e}_2,\mathbf{e}_3$ degenerate;'
                      '\n' r'$T\to1$: $\mathbf{e}_1,\mathbf{e}_2$ degenerate')

    if evec is not None:
        ijk = np.concatenate(ijk_all)
        T = np.concatenate(T_all)
        d = rng.normal(size=(len(ijk), 3))
        d *= r_probe / np.linalg.norm(d, axis=1, keepdims=True)
        ijk_d = np.rint((ijk * h + d) / h).astype(np.int64) % n
        dots = np.abs(np.einsum('nim,nim->nm',
                                _gather(evec, ijk).astype(np.float64),
                                _gather(evec, ijk_d)))
        edges = np.linspace(0, 1, n_T_bins + 1)
        which = np.clip(np.digitize(T, edges) - 1, 0, n_T_bins - 1)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        for m in range(3):
            med = np.array([np.median(dots[which == b, m])
                            if np.any(which == b) else np.nan
                            for b in range(n_T_bins)])
            axes[1].plot(ctr, med, color=E_COLORS[m], lw=1.6,
                         label=rf'$m={m + 1}$')
        axes[1].axhline(0.5, color='0.6', lw=0.8, ls=':')
        axes[1].set_xlabel(r'$T$')
        axes[1].set_ylabel(r'median $|\mathbf{e}_m(x)\cdot'
                           r'\mathbf{e}_m(x+\mathbf{d})|$')
        axes[1].set_xlim(0, 1)
        axes[1].legend(fontsize=8)
        axes[1].set_title('eigenvector stability vs degeneracy '
                          rf'($|\mathbf{{d}}|={r_probe:.1f}\ h^{{-1}}$Mpc)')
        res['stability'] = {'T': ctr, 'dots': dots, 'bin': which}

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig, res
