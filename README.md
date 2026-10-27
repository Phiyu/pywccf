# pywccf — cosmic-web cross-correlation function

[![Last commit](https://img.shields.io/github/last-commit/Phiyu/pywccf/main)](https://github.com/Phiyu/pywccf/commits/main)
[![MIT License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/ChenYangyao/dwarf_assembly_bias/blob/master/LICENSE)

Companion code of

> Yu, H., Wang, H., et al. (2026). Halo formation and distribution around cosmic web. arXiv preprint arXiv:2610.34042.


`pywccf` measures the cross-correlation between the cells of a cosmic-web type (T-web) and a discrete tracer or a field, isotropically (1D) or in the local tidal eigenframe of each cell (3D, defined in the paper above).

## Installation

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from pywccf import TidalField, HaloSample, WCCF3D

tf = TidalField.from_file("field.hdf5", load_evec="lazy")
halos = HaloSample.from_file("halo.hdf5", l_box=tf.l_box).subset_by_val("mass", lo=11.5, hi=12.0)
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

Tracer keys: `position` (N, 3) [$h^{-1} {\rm Mpc}$], `mass` ($log_{10} M/ (h^{-1} M_\odot)$), optional `id`; subhalos and galaxies also `central` (bool) and `host_id` (int, `id` of the host halo). You can also add any keys beyond these.

## Examples

| `examples/` | |
|---|---|
| `samples.ipynb` | building and using the sample classes |
| `halo_web.ipynb`, `halo_web.py` | halo – cosmic-web 1D / 3D WCCF, old vs. young halos |
| `custom_sample.ipynb` | writing a new tracer or field class |

SandBox Test: All examples run on a synthetic box (`pywccf.utils.synthetic_box`) when no simulation data are given.

## Acknowledgements

Thanks to Yangyao Chen for [his code](https://github.com/ChenYangyao/dwarf_assembly_bias) for the paper ***Unexpected clustering pattern in dwarf galaxies challenges formation models*** (Ziwen Zhang et al. [Nature 2025, 642, 47-52](https://www.nature.com/articles/s41586-025-08965-5); [arXiv:2504.03305](https://arxiv.org/abs/2504.03305)).

> After long-term development, our initial python package had become quite messy. We thank [Claude Code Opus 5.5](https://www.anthropic.com/claude-opus-5-5) for helping us clean up and reorganize it. The authors used Claude Code legally in WI, USA.