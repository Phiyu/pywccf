from __future__ import annotations
import h5py
from pyhipp.io import h5
from pyhipp.core import abc
from pyhipp.field.cubic_box import Mesh
import numpy as np
from pathlib import Path
from sklearn.neighbors import KDTree
from functools import cached_property

class TidalField(abc.HasLog):
    def __init__(self, lams: np.ndarray, delta: np.ndarray,
                 mesh: Mesh, lam_off: float = 0.,
                 recon_mask: np.ndarray = None, 
                 verbose = True):

        super().__init__(verbose=verbose)

        self.log(f'TidalField: threshold={lam_off}')
        if recon_mask is None:
            self.log('All field volume are used.')
            recon_mask = np.ones_like(delta, dtype=bool)

        self.lams = lams
        self.delta = delta
        self.mesh = mesh
        self.lam_off = lam_off
        self.recon_mask = recon_mask
        self._xs_web_t_cache = {}
        self.n_lams = (lams >= lam_off).sum(-1)

    @classmethod
    def from_file(cls, path: Path | str, recon_only=True, lam_off: float = 0, **init_kw):
        with h5py.File(path, 'r', locking=False) as f:
            lams, delta, n_grids, l_box = tuple(f[key][()] for key in [
                'lam', 'delta_sm_x', 'n_grids', 'l_box'])
            if recon_only:
                recon_mask = f['reconstruction_mask'][()]
                if recon_mask.shape != delta.shape:
                    recon_mask = align_mask_to_grid(recon_mask, delta.shape)
            else:
                recon_mask = None
            mesh = Mesh.new(n_grids, l_box)
            lams = np.array(lams, dtype=np.float32)
            if f.keys().__contains__('lambda_th'):
                lam_off = f['lambda_th'][()]
            else :
                lam_off = 0.0
        return cls(lams, delta, mesh, recon_mask=recon_mask, lam_off=lam_off, **init_kw)

    @cached_property
    def xs_grids(self):
        mesh = self.mesh
        n, h = mesh.n_grids, mesh.l_grid

        idx_1d = np.arange(n)
        x_1d = idx_1d * h
        x, y, z = np.meshgrid(x_1d, x_1d, x_1d, indexing='ij')
        X = np.stack([x, y, z], axis=-1)

        return X

    @cached_property
    def xs_recon(self):
        xs = self.xs_grids[self.recon_mask]
        return xs.reshape(-1, 3)

    @cached_property
    def kdt_recon(self):
        return KDTree(self.xs_recon)

    @cached_property
    def kdt_grids(self):
        xs = self.xs_grids.reshape(-1, 3)
        return KDTree(xs)

    def xs_web_t(self, n_lam):
        cache = self._xs_web_t_cache
        if n_lam in cache:
            return cache[n_lam]
        mask = self._mask_web_t(n_lam)
        xs = self.xs_grids[mask].reshape(-1, 3)
        cache[n_lam] = xs
        return xs

    def wet_t_at(self, xs: np.ndarray):
        xs = np.asarray(xs)
        n_xs = len(xs)
        assert xs.shape == (n_xs, 3)
        xis = self.mesh._impl.x_to_xi_nd(xs)
        xi0, xi1, xi2 = xis.T
        return self.n_lams[xi0, xi1, xi2]

    def _mask_web_t(self, n_lam):
        mask = (self.lams >= self.lam_off).sum(-1) == n_lam
        mask &= self.recon_mask
        return mask
    
    def __getitem__(self, key: str | tuple[str, ...]):
        '''
        Get the attribute of the field sample.

        @key: key of the attribute.
        '''
        return self.data[key]
    
def align_mask_to_grid(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    src_shape = np.array(mask.shape)
    tgt_shape = np.array(target_shape)

    i_tgt, j_tgt, k_tgt = np.meshgrid(
        np.arange(tgt_shape[0]),
        np.arange(tgt_shape[1]),
        np.arange(tgt_shape[2]),
        indexing='ij'
    )
    
    scale = tgt_shape / src_shape
    i_src = np.round(i_tgt / scale[0]).astype(int)
    j_src = np.round(j_tgt / scale[1]).astype(int)
    k_src = np.round(k_tgt / scale[2]).astype(int)
    
    valid = (
        (i_src >= 0) & (i_src < src_shape[0]) &
        (j_src >= 0) & (j_src < src_shape[1]) &
        (k_src >= 0) & (k_src < src_shape[2])
    )
    
    aligned_mask = np.zeros(target_shape, dtype=bool)
    aligned_mask[valid] = mask[i_src[valid], j_src[valid], k_src[valid]]
    
    return aligned_mask

class FieldSample(abc.HasLog, abc.HasDictRepr):
    '''
    A simple field sample.
    
    @l_box: size of the simulation box in Mpc/h.
    @N_grids: number of grid points along each axis.
    @data: arrays contain field data, e.g. {'temperature': ..., 'density': ...} .
    @verbose: whether to print log messages.
    @copy: whether to copy the data.
    '''
    def __init__(self, boxsize: float, N_grids: int, data: dict[str, np.ndarray], verbose=True, copy=True):

        keys = tuple(data.keys())
        for k, v in data.items():
            assert v.shape == (N_grids, N_grids, N_grids), f"Size of {k} != ({N_grids}, {N_grids}, {N_grids})"
        
        super().__init__(verbose=verbose)
        self.boxsize = float(boxsize)
        self.N_grids = int(N_grids)
        self.data = data.copy()
        self.log(f'FieldSample: {boxsize=}, {N_grids=}, {keys=}')
        
    def __getitem__(self, key: str | tuple[str, ...]):
        '''
        Get the attribute of the field sample.
        
        @key: key of the attribute.
        '''
        return self.data[key]
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __contains__(self, key: str):
        return key in self.data
    

    def to_file(self, path: Path | str, *, compression: str | None = None):
        """
        Save the sample to an HDF5 file so it can be reloaded with
        ``FieldSample.from_file(path)``.

        - path: target file path (directories will be created if needed).
        - compression: optional h5py compression name (e.g. 'gzip', 'lzf').
        """
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        # Write datasets. Handle numeric arrays and string/object arrays.
        with h5py.File(path, 'w', locking=False) as f:
            f.create_dataset('l_box', data=self.l_box)
            f.create_dataset('N_grids', data=self.N_grids)
            for k, v in self.data.items():
                arr = np.asarray(v)
                # handle string-like or object arrays specially
                if arr.dtype.kind in ('U', 'S') or arr.dtype == object:
                    dt = h5py.string_dtype(encoding='utf-8')
                    # ensure a numpy array of Python strings
                    str_arr = arr.astype('U').astype(object)
                    f.create_dataset(k, data=str_arr, dtype=dt)
                else:
                    if compression:
                        f.create_dataset(k, data=arr, compression=compression)
                    else:
                        f.create_dataset(k, data=arr)
    
    @classmethod
    def from_file(cls, path: Path | str, **init_kw):
        '''
        Create the sample from a file. The file should be in HDF5 format,
        with datasets containing the field attributes.
        
        @path: path to the file.
        @init_kw: additional keyword arguments passed to __init__().
        '''
        with h5py.File(path, 'r', locking=False) as f:
            boxsize = f['l_box'][()]
            N_grids = f['n_grids'][()]
            data = {k: f[k][:] for k in f.keys() if k not in ('l_box', 'n_grids')}
        return cls(boxsize=boxsize, N_grids=N_grids, data=data, copy=False, **init_kw)