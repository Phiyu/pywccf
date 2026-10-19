from __future__ import annotations
from typing import Self
import numpy as np
from pathlib import Path
import h5py
from pyhipp.core import abc


class HaloSample(abc.HasLog, abc.HasDictRepr):
    
    repr_attr_keys = ('n_objs',)
    
    '''
    A simple halo sample.
    
    @data: attributes of the halos, e.g. {'ra': ..., 'dec': ...}.
    @verbose: whether to print log messages.
    @copy: whether to copy the data.
    '''
    def __init__(self, data: dict[str, np.ndarray], verbose=True, copy=True):

        keys = tuple(data.keys())
        n_objs = len(data[keys[0]])
        for k, v in data.items():
            assert len(v) == n_objs, f"Size of {k} != {n_objs}"
        
        super().__init__(verbose=verbose)
        self.data = data.copy()
        self.log(f'HaloSample: {n_objs=}, {keys=}')
        
    def __getitem__(self, key: str | tuple[str, ...]):
        '''
        Get the attribute of the halo sample.
        
        @key: key of the attribute.
        '''
        return self.data[key]
    
    @property
    def n_objs(self):
        key = next(iter(self.keys()))    
        val = self[key]
        return len(val)
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __contains__(self, key: str):
        return key in self.data
    
    def subset(self, args: np.ndarray | slice, copy=True) -> Self:
        '''
        Row-wise subsetting by `args`, applied to each value. Return a new 
        (copied) sample.
        
        @copy: if False, values are not guaranteed to be copied.
        '''
        return HaloSample({
            k: v[args] for k, v in self.data.items()    
        }, copy=copy, verbose=self.verbose)
        
    def subset_by_val(self, key: str, lo=None, hi=None, eq=None) -> Self:
        sel = np.ones(self.n_objs, dtype=bool)
        val = self[key]
        if lo is not None:
            sel &= val >= lo
        if hi is not None:
            sel &= val < hi
        if eq is not None:
            sel &= val == eq
        return self.subset(sel, copy=False)
    
    def subset_by_p(self, key: str, p_lo=None, p_hi=None) -> Self:
        val = self[key]
        lo, hi = None, None
        if p_lo is not None:
            lo = np.quantile(val, p_lo)
        if p_hi is not None:
            hi = np.quantile(val, p_hi)
        return self.subset_by_val(key, lo=lo, hi=hi)    
    
    def add_property(self, func: function):
        add = func(self)
        for k, v in add.items():
            assert len(v) == self.n_objs, f"Size of {k} != {self.n_objs}"
            self.data[k] = v

    def to_file(self, path: Path | str, *, compression: str | None = None):
        """
        Save the sample to an HDF5 file so it can be reloaded with
        ``HaloSample.from_file(path)``.

        - path: target file path (directories will be created if needed).
        - compression: optional h5py compression name (e.g. 'gzip', 'lzf').
        """
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        # Write datasets. Handle numeric arrays and string/object arrays.
        with h5py.File(path, 'w', locking=False) as f:
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
        with datasets containing the halo attributes.
        
        @path: path to the file.
        @init_kw: additional keyword arguments passed to __init__().
        '''
        with h5py.File(path, 'r', locking=False) as f:
            data = {k: f[k][:] for k in f.keys()}
        return cls(data, copy=False, **init_kw)