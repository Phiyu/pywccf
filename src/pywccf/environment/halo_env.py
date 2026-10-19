"""
Attach the NGP tidal-field environment to every halo once, store it with the
halo catalogue, and stop loading the (large, slow) TidalField for halo-level
statistics.

Part of ``pywccf`` (split out of the analysis scripts of the 3D-WCCF paper).  The
output file is written with ``HaloSample.to_file`` itself, so it is read back
with ``HaloSample.from_file`` like any other catalogue; :func:`load_halo_env`
does that and also returns the file-level metadata.

Columns added (prefix ``env_``, one value per halo, taken at the halo's own
grid node: pyhipp convention, node = floor(x / l_grid + 0.5) mod n_grids)

    env_delta     float32      delta of the field TidalField reads
                               ('delta_sm_x', i.e. smoothed on R_s)
    env_web       int8         W = number of eigenvalues >= lambda_th
                               (0 void, 1 sheet, 2 filament, 3 knot);
                               identical to tf.n_lams at the node
    env_in_recon  bool         node inside the reconstruction mask
    env_lams      float32 (3,) eigenvalues at the node, lambda_1 >= lambda_2
                               >= lambda_3 -- W can be recomputed for any
                               threshold with :func:`web_from_lams`
    env_node      int32   (3,) node indices (i, j, k)
    env_evec      float32 (3,3) optional; [:, :, m] = e_{m+1}, the same
                               layout as ``web_geometry.load_evec``
    env_<name>    float32      optional extra grids (e.g. delta at another
                               smoothing scale from ``smooth_field``)

File attributes (HDF5 attrs, invisible to ``HaloSample.from_file``)

    env_lam_off, env_n_grids, env_l_box, env_l_grid, env_V_eff [(h^-1 Mpc)^3],
    env_f_vol (4,) = P_V(W) volume filling fractions of the reconstructed
    volume, env_R_s (if given), env_assignment = 'ngp', env_grid_convention,
    env_field_file, env_halo_file, env_created, env_keys.

What still needs the TidalField
    Anything normalised by *volume* or built on the grid: n_W(m), n_{W,x}(m),
    P_V(W | x), volume-weighted PDFs, gridding haloes for the WCCF, and
    re-smoothing.  (P_V(W) itself is stored, see ``env_f_vol``.)
    Everything that only needs (m, x, W) per halo -- ``mass_density_web``,
    ``density_pdf_by_sample_web``, ``density_matched_weights``,
    ``density_bin_labels``, ``summarize_density_bins``,
    ``ar_r0_from_one_point`` -- accepts ``tf=None`` and reads these columns.

Example
-------
>>> from pywccf.environment.halo_env import build_halo_env_file, load_halo_env
>>> build_halo_env_file('halos.hdf5', 'tidal_field.hdf5', 'halos_env.hdf5',
...                     with_evec=True, R_s=3.0)
>>> hs = load_halo_env('halos_env.hdf5')
>>> hs = hs.subset_by_val('is_backsplash', eq=False)
>>> mdw = mass_density_web(hs, None)                    # no TidalField
"""

from __future__ import annotations

import datetime
from pathlib import Path

import h5py
import numpy as np

from . import web_environment_stats as wes

__all__ = ['attach_halo_environment', 'save_halo_env', 'load_halo_env',
           'build_halo_env_file', 'web_from_lams', 'has_env', 'ENV_PREFIX',
           'ENV_REQUIRED']

ENV_PREFIX = 'env_'
ENV_REQUIRED = ('delta', 'web', 'in_recon')
GRID_CONVENTION = ('pyhipp: field values on nodes at i*l_grid; '
                   'halo node = floor(x/l_grid + 0.5) mod n_grids (NGP)')


def web_from_lams(lams, lam_th=0.0):
    """Web type for another eigenvalue threshold: #(lambda >= lam_th)."""
    return (np.asarray(lams) >= lam_th).sum(-1).astype(np.int8)


def has_env(h_samp, prefix=ENV_PREFIX):
    """True if `h_samp` carries the columns :func:`attach_halo_environment`
    writes (enough for every tf=None code path)."""
    return all((prefix + k) in h_samp for k in ENV_REQUIRED)


def _gather_evec_from_file(path, ijk, n_grids, chunk=16, dtype=np.float32):
    """evec at the given nodes, read slab by slab along axis 0 so that the
    full (n, n, n, 3, 3) array (4.5 GB at 500^3 in float32) is never loaded."""
    out = np.empty((len(ijk), 3, 3), dtype=dtype)
    order = np.argsort(ijk[:, 0], kind='stable')
    i_sorted = ijk[order, 0]
    with h5py.File(path, 'r', locking=False) as f:
        if 'evec' not in f:
            raise KeyError(f"no 'evec' dataset in {path}")
        ds = f['evec']
        if tuple(ds.shape[:3]) != (n_grids,) * 3 or tuple(ds.shape[3:]) != (3, 3):
            raise ValueError(f"'evec' has shape {ds.shape}, expected "
                             f"({n_grids},)*3 + (3, 3)")
        for i0 in range(0, n_grids, chunk):
            i1 = min(i0 + chunk, n_grids)
            lo, hi = np.searchsorted(i_sorted, [i0, i1])
            if lo == hi:
                continue
            slab = np.asarray(ds[i0:i1])
            idx = order[lo:hi]
            out[idx] = slab[ijk[idx, 0] - i0, ijk[idx, 1], ijk[idx, 2]]
    return out


def attach_halo_environment(h_samp, tf, *, evec=None, evec_path=None,
                            extra_fields=None, with_lams=True, with_node=True,
                            position_key='position', prefix=ENV_PREFIX,
                            R_s=None, compute_f_vol=True, overwrite=False,
                            inplace=False, evec_chunk=16, verbose=True):
    """Add the NGP environment of every halo as columns of a HaloSample.

    Parameters
    ----------
    h_samp : pywccf.samples.halo.HaloSample (positions in cMpc/h).
    tf : pywccf.samples.field.TidalField.  Load it with ``recon_only=True`` so
         that ``env_in_recon`` is the real reconstruction mask.
    evec : optional in-memory eigenvector grid (n, n, n, 3, 3), or
    evec_path : the field file to read 'evec' from, slab by slab.
    extra_fields : {name: grid} of further scalar grids on the same mesh, each
         stored as ``prefix + name`` (e.g. ``{'delta_R8': smooth_field(...)}``).
    with_lams, with_node : store the eigenvalues / node indices.
    R_s : smoothing scale, recorded in the metadata only.
    compute_f_vol : also record P_V(W) and V_eff of the reconstructed volume
         (one pass over the grid).
    overwrite : allow replacing existing ``prefix*`` columns.
    inplace : modify `h_samp` instead of returning a copy.

    Returns
    -------
    HaloSample with the new columns and a ``.meta`` dict (what
    :func:`save_halo_env` writes as file attributes).
    """
    xs = np.asarray(h_samp[position_key], dtype=np.float64)
    n_h = len(xs)
    n, h = wes._grid_spec(tf)
    ijk = wes._cell_index(tf, xs)                 # pyhipp NGP node
    i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]

    new = {
        prefix + 'delta': np.asarray(tf.delta)[i, j, k].astype(np.float32),
        prefix + 'web': np.asarray(tf.n_lams)[i, j, k].astype(np.int8),
        prefix + 'in_recon': np.asarray(wes._recon_mask(tf))[i, j, k]
        .astype(bool),
    }
    if with_lams:
        new[prefix + 'lams'] = np.asarray(tf.lams)[i, j, k].astype(np.float32)
    if with_node:
        new[prefix + 'node'] = ijk.astype(np.int32)
    if evec is not None:
        ev = np.asarray(evec)
        if ev.shape != (n, n, n, 3, 3):
            raise ValueError(f'evec has shape {ev.shape}, expected '
                             f'({n},)*3 + (3, 3)')
        new[prefix + 'evec'] = ev[i, j, k].astype(np.float32)
    elif evec_path is not None:
        new[prefix + 'evec'] = _gather_evec_from_file(
            evec_path, ijk, n, chunk=evec_chunk)
    for name, grid in (extra_fields or {}).items():
        g = np.asarray(grid)
        if g.shape != (n, n, n):
            raise ValueError(f'extra field {name!r} has shape {g.shape}, '
                             f'expected ({n},)*3')
        new[prefix + name] = g[i, j, k].astype(np.float32)

    clash = [kk for kk in new if kk in h_samp]
    if clash and not overwrite:
        raise KeyError(f'{clash} already in the sample; pass overwrite=True')

    # consistency: the stored web type must reproduce tf.n_lams from the
    # stored eigenvalues and threshold
    lam_off = float(getattr(tf, 'lam_off', 0.0))
    if with_lams:
        chk = web_from_lams(new[prefix + 'lams'], lam_off)
        if not np.array_equal(chk, new[prefix + 'web']):
            raise RuntimeError('env_web does not match the eigenvalues and '
                               'lam_off -- tf.n_lams / tf.lams inconsistent?')

    out = h_samp if inplace else h_samp.subset(np.arange(n_h))
    out.data.update(new)

    meta = {
        prefix + 'lam_off': lam_off,
        prefix + 'n_grids': int(n),
        prefix + 'l_grid': float(h),
        prefix + 'l_box': float(n * h),
        prefix + 'assignment': 'ngp',
        prefix + 'grid_convention': GRID_CONVENTION,
        prefix + 'created': datetime.datetime.now().isoformat(timespec='seconds'),
        prefix + 'keys': ','.join(sorted(new)),
    }
    if R_s is not None:
        meta[prefix + 'R_s'] = float(R_s)
    if compute_f_vol:
        cells = wes._web_cell_counts(tf)[0]
        meta[prefix + 'f_vol'] = (cells / cells.sum()).astype(np.float64)
        meta[prefix + 'V_eff'] = float(cells.sum() * h**3)
    old = getattr(h_samp, 'meta', None)
    out.meta = {**(old or {}), **meta}

    if verbose:
        w = new[prefix + 'web']
        ins = new[prefix + 'in_recon']
        print(f'attach_halo_environment: {n_h} haloes, {int(ins.sum())} in the '
              f'reconstructed volume; lam_th = {lam_off:g}; '
              f'grid {n}^3, l_grid = {h:g}')
        print('  P(W | haloes in recon) = '
              + np.array2string(np.bincount(w[ins], minlength=4)
                                / max(ins.sum(), 1), precision=3)
              + ('   P_V(W) = ' + np.array2string(meta[prefix + 'f_vol'],
                                                  precision=3)
                 if compute_f_vol else ''))
        print('  columns: ' + ', '.join(sorted(new)))
    return out


def save_halo_env(h_env, path, *, compression='gzip', overwrite=False):
    """Write a sample from :func:`attach_halo_environment` to HDF5.

    Datasets are written by ``HaloSample.to_file`` (so ``from_file`` reads the
    file unchanged); ``h_env.meta`` goes into file attributes.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} exists; pass overwrite=True')
    h_env.to_file(path, compression=compression)
    meta = getattr(h_env, 'meta', {}) or {}
    with h5py.File(path, 'r+', locking=False) as f:
        for kk, v in meta.items():
            f.attrs[kk] = v
    return path


def _py(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    if isinstance(v, np.generic):
        return v.item()
    return v


def load_halo_env(path, **init_kw):
    """Read a file written by :func:`save_halo_env`.

    Returns a HaloSample (``pywccf.samples.halo.HaloSample.from_file``) with the
    file attributes attached as ``.meta``.  Note that ``HaloSample.subset``
    returns a new object without ``.meta``; functions that need a piece of it
    (e.g. the box size for the jackknife) take it as an argument too.
    """
    from ..samples.halo import HaloSample
    hs = HaloSample.from_file(path, **init_kw)
    with h5py.File(path, 'r', locking=False) as f:
        hs.meta = {kk: _py(v) for kk, v in f.attrs.items()}
    if not has_env(hs):
        raise KeyError(f'{path} has no env_* columns; was it written by '
                       'save_halo_env?')
    return hs


def build_halo_env_file(halo_path, field_path, out_path, *, with_evec=False,
                        extra_fields=None, R_s=None, compression='gzip',
                        overwrite=False, verbose=True, **attach_kw):
    """One call: read the halo catalogue and the tidal field, attach the NGP
    environment, write ``out_path``.

    The TidalField is read with the reconstruction mask whenever the file has
    one, so that ``env_in_recon`` is meaningful.  ``with_evec=True`` also
    stores the eigenvectors, read slab by slab from `field_path`.
    Returns the enriched HaloSample (also written to disk).
    """
    from ..samples.field import TidalField
    from ..samples.halo import HaloSample

    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f'{out_path} exists; pass overwrite=True')
    with h5py.File(field_path, 'r', locking=False) as f:
        has_mask = 'reconstruction_mask' in f
    tf = TidalField.from_file(field_path, recon_only=has_mask,
                              verbose=verbose)
    hs = HaloSample.from_file(halo_path, verbose=verbose)
    h_env = attach_halo_environment(
        hs, tf, evec_path=field_path if with_evec else None,
        extra_fields=extra_fields, R_s=R_s, verbose=verbose, **attach_kw)
    h_env.meta['env_field_file'] = str(Path(field_path).resolve())
    h_env.meta['env_halo_file'] = str(Path(halo_path).resolve())
    if not has_mask:
        h_env.meta['env_note'] = ('field file has no reconstruction_mask: '
                                  'env_in_recon is True everywhere')
    save_halo_env(h_env, out_path, compression=compression,
                  overwrite=overwrite)
    if verbose:
        print(f'build_halo_env_file: wrote {out_path}')
    return h_env
