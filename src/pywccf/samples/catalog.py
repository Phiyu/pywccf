"""ELUCID halo catalogue used in the paper (Section 2.3).

Two steps, run once on the machine that holds the simulation:

1. :func:`build_elucid_halo_catalog` -- z = 0 central halos of every FoF group
   (``FirstSubOfGroup``), ``Len >= 800``, with M_200m, R_200m, V_max,
   concentration ``V_max/V_200`` and spin ``lambda = |j|/(sqrt(2) V_200 R_200)``
   (former ``code/sample.py``).
2. :func:`join_tree_properties` -- attach the formation redshift ``zf`` and the
   backsplash flags from the merger-tree property file by the strict key
   ``(FileNr, SubhaloIndex)`` (former cell 0 of ``elucid2.ipynb``).

The resulting file (``halo_0701.hdf5`` in the paper) is read with
:func:`load_halo_sample`, which applies the paper cuts.
"""
from __future__ import annotations

import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from .halo import HaloSample

G_CODE = 43.0187   # (km/s)^2 (h^-1 Mpc) / (10^10 h^-1 Msun)
LEN_MIN = 800      # resolution cut (Wang et al. 2021)

__all__ = ["build_elucid_halo_catalog", "join_tree_properties",
           "load_halo_sample", "count_mass_bins"]


# ---------------------------------------------------------------------------
# step 1: central halos from the SUBFIND tables
# ---------------------------------------------------------------------------
def _as_n3(dset, n):
    """SubPos/SubSpin may be stored flat (3N,); reshape to (N, 3)."""
    arr = dset[:]
    if arr.ndim == 1 and arr.size == 3 * n:
        return arr.reshape(n, 3)
    if arr.ndim == 2 and arr.shape == (n, 3):
        return arr
    raise ValueError(f'Unexpected shape {arr.shape} for n={n}')


def _process_central_chunk(i_c, sim_file, snap):
    """Read one SUBFIND chunk and keep the central subhalo of each group."""
    with h5py.File(sim_file, 'r') as f:
        sf = f[f'Snapshots/{snap}/{i_c}/SubfindTable']
        first_sub = sf['FirstSubOfGroup'][:]
        n_per_grp = sf['NumSubPerGroup'][:]
        M_mean200 = sf['Halo_M_Mean200'][:]          # 10^10 h^-1 Msun
        R_mean200 = sf['Halo_R_Mean200'][:]          # h^-1 Mpc
        SubVmax = sf['SubVmax'][:]
        SubLen = sf['SubLen'][:]
        SubMBID = sf['SubMostBoundID'][:]
        n_sub = SubVmax.shape[0]
        SubPos = _as_n3(sf['SubPos'], n_sub)
        SubSpin = _as_n3(sf['SubSpin'], n_sub)

    valid = (n_per_grp > 0) & (first_sub >= 0) & (first_sub < n_sub)
    cen_sidx = first_sub[valid].astype(np.int32)
    n_cen = int(valid.sum())
    return dict(
        chunk_id=i_c,
        FileNr=np.full(n_cen, i_c, dtype=np.int32),
        SubhaloIndex=cen_sidx,
        GroupIndex=np.where(valid)[0].astype(np.int32),
        SubMostBoundID=SubMBID[cen_sidx].astype(np.int64),
        M_Mean200=M_mean200[valid].astype(np.float32),
        R_Mean200=R_mean200[valid].astype(np.float32),
        Vmax=SubVmax[cen_sidx].astype(np.float32),
        Len=SubLen[cen_sidx].astype(np.int32),
        position=SubPos[cen_sidx].astype(np.float32),
        spin_vec=SubSpin[cen_sidx].astype(np.float32),
    )


def build_elucid_halo_catalog(sim_file, out_file, *, snap=99,
                              len_min=LEN_MIN, n_proc=None):
    """Write the z = 0 central-halo catalogue (former ``code/sample.py``).

    Parameters
    ----------
    sim_file : ELUCID ``simulation.hdf5`` (SUBFIND tables per chunk).
    out_file : output HDF5 file (``halo_0701.hdf5`` in the paper).
    snap : snapshot number (99 is z = 0).
    len_min : minimum number of particles of the central subhalo.
    n_proc : worker processes (default ``min(32, cpu_count)``).

    Datasets written: FileNr, SubhaloIndex, GroupIndex, SubMostBoundID,
    position [h^-1 Mpc], mass [log M_200m / h^-1 Msun], R200m, V200, Vmax,
    concentration (= Vmax/V200), spin_vec, lambda.
    """
    t0 = time.time()
    n_proc = n_proc or min(32, os.cpu_count())
    with h5py.File(sim_file, 'r') as f:
        n_chunks = int(f['Header'].attrs['NumFilesPerSnapshot'])
        h_param = float(f['Header'].attrs['HubbleParam'])
    print(f'n_chunks = {n_chunks}, using {n_proc} processes')

    bufs = [None] * n_chunks
    worker = partial(_process_central_chunk, sim_file=sim_file, snap=snap)
    with Pool(n_proc) as pool:
        for done, res in enumerate(
                pool.imap_unordered(worker, range(n_chunks), chunksize=4), 1):
            bufs[res['chunk_id']] = res
            if done % 100 == 0 or done == n_chunks:
                print(f'  {done:4d}/{n_chunks}   ({time.time()-t0:.1f}s)')

    def cat(k):
        return np.concatenate([b[k] for b in bufs])
    keys = ('FileNr', 'SubhaloIndex', 'GroupIndex', 'SubMostBoundID',
            'M_Mean200', 'R_Mean200', 'Vmax', 'Len', 'position', 'spin_vec')
    cols = {k: cat(k) for k in keys}
    n_all = len(cols['FileNr'])
    print(f'\nTotal centrals (all groups w/ subs): {n_all:,}')

    key = (cols['FileNr'].astype(np.int64) * (1 << 32)
           + cols['SubhaloIndex'].astype(np.int64))
    assert len(np.unique(key)) == n_all, '(FileNr, SubhaloIndex) not unique!'
    assert len(np.unique(cols['SubMostBoundID'])) == n_all, \
        'SubMostBoundID has duplicates!'

    M200, R200, Len = cols['M_Mean200'], cols['R_Mean200'], cols['Len']
    mask = (M200 > 0) & (R200 > 0) & (Len >= len_min)
    print(f'After Len>={len_min} & M,R>0 : {mask.sum():,} / {n_all:,}')
    cols = {k: v[mask] for k, v in cols.items()}
    M200, R200 = cols['M_Mean200'], cols['R_Mean200']

    V200 = np.sqrt(G_CODE * M200 / R200).astype(np.float32)        # km/s
    concentration = (cols['Vmax'] / V200).astype(np.float32)
    j_norm = np.linalg.norm(cols['spin_vec'], axis=1)
    lam = (j_norm / (np.sqrt(2.0) * V200 * R200)).astype(np.float32)
    logM = (np.log10(M200) + 10.0).astype(np.float32)             # log h^-1 Msun

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, 'w', locking=False) as f:
        f.attrs['snap'] = snap
        f.attrs['z'] = 0.0
        f.attrs['HubbleParam'] = h_param
        f.attrs['Len_min'] = len_min
        f.attrs['mass_def'] = ('M_Mean200 (mean density = 200 * mean matter '
                               'density)')
        f.attrs['note'] = ('Central halos at z=0. Match the merger tree '
                           'strictly by (FileNr, SubhaloIndex).')
        for k in ('FileNr', 'SubhaloIndex', 'GroupIndex', 'SubMostBoundID',
                  'position'):
            f.create_dataset(k, data=cols[k])
        f.create_dataset('mass', data=logM)
        f.create_dataset('R200m', data=R200.astype(np.float32))
        f.create_dataset('V200', data=V200)
        f.create_dataset('Vmax', data=cols['Vmax'].astype(np.float32))
        f.create_dataset('concentration', data=concentration)
        f.create_dataset('spin_vec', data=cols['spin_vec'])
        f.create_dataset('lambda', data=lam)
    print(f'\nsaved -> {out_file}   (total {time.time()-t0:.1f}s)')
    return out_file


# ---------------------------------------------------------------------------
# step 2: formation redshift and backsplash flags from the merger tree
# ---------------------------------------------------------------------------
def join_tree_properties(halo_file, tree_file, *, save=True):
    """Attach ``zf``, ``backsplash`` (loose), ``is_backsplash`` (strict) and,
    when present, ``M_exit``, ``exit_snap``, ``M_exit_code`` to the halo file.

    The join key is ``(FileNr << 32) | SubhaloIndex``.  Unmatched halos get
    ``zf = NaN`` and flag ``-1``.  Returns the dict of joined arrays.
    """
    with h5py.File(halo_file, 'r') as f:
        h_fnr = f['FileNr'][:]
        h_sidx = f['SubhaloIndex'][:]
    n_h = len(h_fnr)

    with h5py.File(tree_file, 'r') as f:
        t_fnr = f['FileNr'][:]
        t_sidx = f['SubhaloIndex'][:]
        t_zf = f['zf'][:]
        t_bs = f['backsplash'][:]
        t_bs_s = f['is_backsplash'][:]
        t_extra = {k: f[k][:] for k in ('M_exit', 'exit_snap', 'M_exit_code')
                   if k in f}

    h_key = (h_fnr.astype(np.int64) << 32) | h_sidx.astype(np.int64)
    t_key = (t_fnr.astype(np.int64) << 32) | t_sidx.astype(np.int64)
    assert len(np.unique(h_key)) == n_h, 'duplicated (FileNr, SubhaloIndex)'

    order = np.argsort(t_key)
    t_key_s = t_key[order]
    idx = np.searchsorted(t_key_s, h_key)
    hit = (idx < len(t_key_s)) & (
        t_key_s[np.clip(idx, 0, len(t_key_s) - 1)] == h_key)
    sel = order[idx[hit]]
    print(f'halo total       : {n_h:,}')
    print(f'matched with tree: {hit.sum():,}   ({100*hit.mean():.2f}%)')

    out = {
        'zf': np.full(n_h, np.nan, dtype=np.float32),
        'backsplash': np.full(n_h, -1, dtype=np.int8),
        'is_backsplash': np.full(n_h, -1, dtype=np.int8),
    }
    out['zf'][hit] = t_zf[sel]
    out['backsplash'][hit] = t_bs[sel].astype(np.int8)
    out['is_backsplash'][hit] = t_bs_s[sel].astype(np.int8)
    if 'M_exit' in t_extra:
        out['M_exit'] = np.full(n_h, np.nan, dtype=np.float32)
        out['M_exit'][hit] = t_extra['M_exit'][sel]
    for k in ('exit_snap', 'M_exit_code'):
        if k in t_extra:
            out[k] = np.full(n_h, -1, dtype=np.int32)
            out[k][hit] = t_extra[k][sel].astype(np.int32)

    n_hit = max(int(hit.sum()), 1)
    print(f'finite zf           : {np.isfinite(out["zf"]).sum():>10,d}')
    print(f'loose  backsplash   : {(out["backsplash"] == 1).sum():>10,d}   '
          f'({100*(out["backsplash"] == 1).sum()/n_hit:.2f}% of matched)')
    print(f'strict backsplash   : {(out["is_backsplash"] == 1).sum():>10,d}   '
          f'({100*(out["is_backsplash"] == 1).sum()/n_hit:.2f}% of matched)')

    if save:
        with h5py.File(halo_file, 'a') as f:
            for k, v in out.items():
                if k in f:
                    del f[k]
                f.create_dataset(k, data=v)
            f.attrs['tree_joined_on'] = '(FileNr, SubhaloIndex)'
        print(f'[saved {", ".join(out)} into {halo_file}]')
    return out


# ---------------------------------------------------------------------------
# paper sample
# ---------------------------------------------------------------------------
def load_halo_sample(halo_file, *, zf_range=(0.01, 20.0),
                     exclude_backsplash=True, **init_kw) -> HaloSample:
    """Halo sample with the cuts of the paper (identical to the selection in
    :func:`pywccf.pipeline.runner._select_mass_samples`): finite ``zf`` in
    ``[zf_lo, zf_hi)`` and, optionally, ``is_backsplash == 0`` (strict flag;
    halos without a tree match, flag ``-1``, are removed as well)."""
    hs = HaloSample.from_file(halo_file, **init_kw)
    if zf_range is not None:
        hs = hs.subset_by_val('zf', lo=zf_range[0], hi=zf_range[1])
    if exclude_backsplash:
        hs = hs.subset_by_val('is_backsplash', eq=False)
    return hs


def count_mass_bins(halo_sample: HaloSample, edges=(11.5, 12.0, 12.5, 13.0,
                                                   13.5, 14.0, 14.5)):
    """Number of halos per ``[lo, hi)`` bin of ``log M_h`` (Table 1)."""
    edges = np.asarray(edges, dtype=float)
    m = np.asarray(halo_sample['mass'])
    counts = [int(((m >= lo) & (m < hi)).sum())
              for lo, hi in zip(edges[:-1], edges[1:])]
    return {'edges': edges, 'counts': np.array(counts)}
