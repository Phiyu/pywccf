# pywccf — cosmic-web cross-correlation function

[![Last commit](https://img.shields.io/github/last-commit/Phiyu/pywccf/main)](https://github.com/Phiyu/pywccf/commits/main)
[![MIT License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/ChenYangyao/dwarf_assembly_bias/blob/master/LICENSE)

Companion code of

> H. Yu, H. Wang, et al., *Halo formation and distribution around cosmic web* (in prep.)

`pywccf` measures the cross-correlation between the cells of a cosmic-web type (T-web) and a tracer or a field — isotropically (1D) or in the local tidal eigenframe of each cell, `r_i = |(x − x_c) · e_i(x_c)|` (3D).

## Installation

```bash
pip install -e ".[fast]"        # numpy, scipy, h5py; fast = numba kernel for the 3D pair counts
```

## Quick start

```python
import numpy as np
from pywccf import TidalField, HaloSample, WCCF3D

tf = TidalField.from_file("field_099.2.hdf5", load_evec="lazy")
halos = HaloSample.from_file("halo_0701.hdf5", l_box=tf.l_box).subset_by_val("mass", lo=11.5, hi=12.0)
young, old = halos.quantile_masks("zf", 0.2)

res = WCCF3D(tf, halos, workers=8).compute(
    "filament", {"all": np.ones(len(halos), bool), "old": old, "young": young}, error="jackknife")
r, xi, sd = res["old"].profile(3)          # xi_3D along e_3
```

## Contents

| module | content |
|---|---|
| `samples` | tracers `HaloSample`, `SubhaloSample`, `GalaxySample` (base `Tracer`); fields `MatterField`, `TidalField`, `GasField` (base `Field`) |
| `wccf` | `WCCF1D` (pair counts, or FFT convolution for fields), `WCCF3D` (pair counts in the eigenframe); jackknife / bootstrap errors |
| `core` | `Mesh`, NGP / CIC / TSC assignment, separation bins, resampling, `Log` |
| `io` | HDF5 helpers, `save_results` / `load_results` |
| `utils` | old / young selections, mass-matching weights, synthetic boxes |

Tracer keys: `position` (N, 3) [h⁻¹ Mpc], `mass` (log10 M [h⁻¹ M☉]), optional `id`; subhalos and galaxies also `central` (bool) and `host_id` (int, `id` of the host halo). Converting simulation output to these keys is left to the user.

## Examples

| `examples/` | |
|---|---|
| `samples.ipynb` | building and using the sample classes |
| `halo_web.ipynb`, `halo_web.py` | halo – cosmic-web 1D / 3D WCCF, old vs. young halos |
| `custom_sample.ipynb` | writing a new tracer or field class |

All examples run on a synthetic box (`pywccf.utils.synthetic_box`) when no simulation data are given.

## License

MIT — see `LICENSE`.
