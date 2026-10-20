# pywccf — cosmic-web cross-correlation function

[![Last commit](https://img.shields.io/github/last-commit/Phiyu/pywccf/main)](https://github.com/Phiyu/pywccf/commits/main)
[![MIT License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/ChenYangyao/dwarf_assembly_bias/blob/master/LICENSE)

Companion code of

> H. Yu, H. Wang, et al., *Halo formation and distribution around cosmic web* (in prep.)

The **3D-WCCF** cross-correlates the cells of one cosmic-web type with a
tracer (or a field), expressing every separation in the local tidal
eigenframe of the cell, `r_i = |(x_h − x_c) · e_i(x_c)|`.  The package
provides the sample classes, the tidal-field / T-web machinery and the 1D
and 3D estimators; it depends only on NumPy, SciPy and h5py (numba
optional).

## Installation

```bash
pip install -e ".[fast,test]"      # from the repository root; fast = numba kernel
pytest
```

## Quick start

```python
import numpy as np
from pywccf import TidalField, HaloSample, WCCF1D, WCCF3D, save_results

tf = TidalField.from_file("field_099.2.hdf5", load_evec="lazy")   # lam, evec, delta_sm_x
halos = HaloSample.from_file("halo_0701.hdf5", l_box=tf.l_box).select_paper()
halos = halos.subset_by_val("mass", lo=11.5, hi=12.0)

young, old = halos.quantile_masks("zf", 0.2)          # bottom / top 20% in z_f
samples = {"all": np.ones(len(halos), bool), "old": old, "young": young}

w3 = WCCF3D(tf, halos, max_cells=1_000_000, workers=8)      # paper bins by default
res = w3.compute("filament", samples, error="jackknife")    # one pass for all samples
r, xi, sd = res["old"].profile(3)                           # xi_3D along e_3
save_results("filament.h5", {"filament": res})

w1 = WCCF1D(tf, halos)                                       # PH estimator, shells
w1m = WCCF1D(tf, field=matter_field)                         # field: FFT convolution
```

`examples/halo_web.py` runs the whole halo–cosmic-web measurement (1D + 3D,
all web types, whole bin and old / young tails); `--synthetic` uses a
generated box and needs no data.

## Package layout

```
src/pywccf/
├── core/        Log/HasLog, Mesh, NGP/CIC/TSC assignment + interpolation,
│                RadialBins / FrameBins, jackknife & bootstrap
├── samples/
│   ├── tracer.py    Tracer, HostedTracer  (column table, selections, to_field)
│   ├── halo.py      HaloSample
│   ├── subhalo.py   SubhaloSample   (host: f_in_grp; central iff f_in_grp == subhalo_id)
│   ├── galaxy.py    GalaxySample    (host: host_id; central flag is_c)
│   ├── field.py     Field, MatterField, GasField  (smoothing, interpolation, to_tracer)
│   └── tidal.py     TidalField  (tidal tensor from a density, T-web, eigenframes)
├── wccf/
│   ├── estimators.py  WCCF1D, WCCF3D
│   ├── cells.py       WebCells (centres, weights, eigenframes)
│   ├── _pairs.py      pair counting (NumPy, KD-tree)
│   ├── _numba.py      compiled 3D kernel (optional)
│   └── result.py      WCCFResult (profiles, planes, dict round trip)
├── io/          hdf5 dict io, save/load results, reader of the paper's combined files
└── utils/       old/young tails, mass-matching weights, synthetic boxes
```

### Samples

| class | kind | notes |
|---|---|---|
| `Tracer` | discrete | `positions`, `masses`, `weights('number'│'mass'│key)`, `subset*`, `quantile_masks`, `to_field(mesh, scheme='ngp'│'cic'│'tsc', weight=...)` |
| `HaloSample` | `Tracer` | `select_paper()`, `mass_bins()` |
| `SubhaloSample`, `GalaxySample` | `HostedTracer` | `host_ids`, `is_central`, `centrals()`, `satellites()`, `host_index(hosts)`, `attach_host(hosts, keys)` |
| `Field` | gridded | dict of node arrays on a `Mesh`, mask, `values_at`, `smoothed`, `to_tracer` |
| `MatterField` | `Field` | `delta`; `from_tracer`; `tidal_field(R_s, lam_th)` |
| `TidalField` | `Field` | `lam` (descending), `evec`, `delta`; `web_mask`, `volume_fractions`, `with_threshold`, `from_density`, `from_file` / `to_file` (ELUCID layout) |
| `GasField` | `Field` | `rho`, `temperature`, …; `delta`, `mass_weighted_mean` |

Grid values live on nodes `x = (i, j, k) · L/N` (NGP = nearest node), the
convention of the ELUCID field files.

### Estimators

| | input | estimator |
|---|---|---|
| `WCCF3D` | `TidalField` (+ eigenvectors) and a tracer **or** a field | Peebles–Hauser in folded eigenframe bins, `V_B = 8 Δr₁Δr₂Δr₃` |
| `WCCF1D` | `TidalField` and a tracer | Peebles–Hauser in spherical shells |
| `WCCF1D` | `TidalField` and a field on the same mesh | FFT convolution of the web-cell weight field with the field's overdensity, averaged over the grid displacements of each shell |

Cells of a web type are the centres, weighted by `(1 + δ)₊` by default
(`cell_weight`), optionally sub-sampled (`max_cells`, as in the paper).
Several tracer sub-samples (masks or weights over the parent) are measured
in one pair-counting pass.  Errors: `'none'`, `'jackknife'` (delete-one
over `n_jk_side³` sub-cubes of cells) or `'bootstrap'` (weighted cell
resampling, PH only).  The central value is always the deterministic
estimator.

A field enters the PH estimators through its nodes weighted by
`1 + value` (`field_stride` thins them); use separation bins wider than the
node spacing.

## Repository

| path | content |
|---|---|
| `src/` | the package |
| `examples/halo_web.py` | halo–cosmic-web example |
| `tests/` | unit and consistency tests |
| `docs/` | paper figures, measured correlation functions (`docs/data`, read with `pywccf.io.load_combined_h5`), format notes |
| `notebook/` | per-section notebooks of the paper (still written for the v0.1 API) |
| `configs/` | YAML files of the paper runs (v0.1 pipeline; provenance) |

## License

MIT — see `LICENSE`.
