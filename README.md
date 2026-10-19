# pywccf — cosmic-web cross-correlation function

[![Last commit](https://img.shields.io/github/last-commit/Phiyu/pywccf/main)](https://github.com/Phiyu/pywccf/commits/main)
[![MIT License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/ChenYangyao/dwarf_assembly_bias/blob/master/LICENSE)

Companion code and data products of

> H. Yu, H. Wang, et al., *Halo formation and distribution around cosmic web* (in prep.)

The **3D-WCCF** cross-correlates cosmic-web cells with a tracer population,
expressing every cell–tracer separation in the local tidal eigenframe of the
cell, `r_i = |(x_h - x_c) · e_i(x_c)|`.  This repository contains the
estimator, the measurement pipeline used for the ELUCID halo sample at z = 0,
the analysis and plotting code of every figure, and the measured correlation
functions.

## Layout

```
3DWCCF/
├── src/pywccf/            package
│   ├── samples/           TidalField, HaloSample (from CosmicCoreBox) + ELUCID catalogue
│   ├── estimator/         3D-WCCF estimators: halo_web (reference), halo_web_fast, numba kernel
│   ├── pipeline/          YAML runner (pywccf-run) and legacy result combiner (pywccf-combine)
│   ├── io.py              readers for the combined result files
│   ├── ratio.py           scale-averaged amplitudes, amplitude ratio (AR)
│   ├── geometry.py        directional walk: frame coherence, web-type transitions
│   ├── environment/       HMF and z_f vs. local density and web type (Section 3)
│   └── plotting/          paper figures (xi, ratio, field)
├── notebook/              one notebook per paper section (N_*.ipynb, A–C = appendices)
├── configs/               YAML files of the paper runs
├── docs/                  main outputs: figures/, data/ (measured WCCFs), format notes
├── tests/
├── LICENSE  pyproject.toml  requirements.txt
```

## Installation

Python >= 3.11.  `pyhipp` must come from GitHub (the PyPI package of that
name is an old, unrelated release):

```bash
pip install "git+https://github.com/ChenYangyao/pyhipp.git"
pip install -e ".[fast,notebook,test]"      # from the repository root
pytest
```

## Quick start

```bash
pywccf-run configs/sec6_age.yaml --dry-run   # validate, count records
pywccf-run configs/sec6_age.yaml             # measure -> one combined HDF5 file
```

```python
from pywccf.io import load_combined_h5
from pywccf.plotting import plot_2_main_contour

f = "docs/data/fiducial/final-data_s2_lth0_combined.h5"
res = load_combined_h5(f, dimension="3d", sample="age", tail="high",
                       web="filament", mass_bin_index=1)
res["xi"].shape                      # (17, 17, 17) bins in (r1, r2, r3)
plot_2_main_contour(f, mass_bin_idx=1)
```

`notebook/4_wccf_method.ipynb` runs the whole pipeline on a small synthetic
box and needs no simulation data.

## Reproducing the paper

| notebook | paper | needs simulation products? |
|---|---|---|
| `2_simulation.ipynb` | §2: tidal-field slice, halo catalogue, Table 1, z_f–M_h | yes |
| `3_halo_environment.ipynb` | §3: HMF and z_f at fixed density / web type | yes |
| `4_wccf_method.ipynb` | §4: configuration, synthetic end-to-end run, reading results | no |
| `5_web_halo.ipynb` | §5: directional slices, contours, web transitions | §5.3 only |
| `6_halo_age.ipynb` | §6: amplitude-ratio maps and AR–mass relation | no |
| `A_smoothing_scale.ipynb` | App. A | no |
| `B_web_threshold.ipynb` | App. B | volume fractions only |
| `C_frame_locality.ipynb` | App. C | yes |

Paths are set once in `notebook/_paths.py` (or with the environment
variables `PYWCCF_SIM_DIR`, `PYWCCF_FIG_DIR`).  Figures are written to
`docs/figures/`; `docs/README.md` maps every figure to its function and input
file.  The cells that start from the measured correlation functions
reproduce the figures pixel-for-pixel with the original scripts.

Simulation products (not distributed here): the ELUCID tidal-field files
`field_099.<R_s>.hdf5` and the halo catalogue `halo_0701.hdf5`
(`docs/data_format.md`).

## Where the code came from

| former file | now |
|---|---|
| `CosmicCoreBox/src/pyccb/samples/{field,halo}.py` | `pywccf/samples/{field,halo}.py` (unchanged) |
| `CosmicCoreBox/src/pyccb/process/web_geometry.py` | `pywccf/geometry.py` |
| `3dweb/halo_web.py`, `halo_web_fast.py`, `halo_web_numba.py` | `pywccf/estimator/` |
| `3dweb/code/run.py` | `pywccf/pipeline/runner.py` (`pywccf-run`) |
| `3dweb/code/combine_results_h5.py` | `pywccf/pipeline/combine.py` (`pywccf-combine`) |
| `3dweb/code/sample.py`, `elucid2.ipynb` (tree join) | `pywccf/samples/catalog.py` |
| `3dweb/func2.py` | `pywccf/io.py` (readers) + `pywccf/ratio.py` (amplitudes) |
| `3dweb/func3.py` | `pywccf/plotting/ratio.py` |
| `3dweb/plot.py` | `pywccf/plotting/xi.py` |
| `figure.ipynb` (Fig. 1, z_f–M_h), `remote.ipynb` (volume fractions) | `pywccf/plotting/field.py` |
| `3dweb/analysis/{web_environment_stats,age_density_control,halo_env,mass_density_web}.py` | `pywccf/environment/` |

Only imports were changed in the ported modules; results are identical to
the original scripts (checked on a synthetic box for both estimator engines
and on all figures that start from `docs/data`).

## Notes

* **Central values with bootstrap errors.**  With `error: bootstrap` (the
  paper runs), the stored `xi` is the mean of the `n_bootstrap` resamples,
  not the deterministic weighted estimator; with `error: jackknife` or
  `n_bootstrap: 0` it is the deterministic estimator.
* **Cell weights.**  `w_c = (1 + delta)_+` uses the field-file dataset
  `delta_sm_x` (see `docs/data_format.md`).
* The `legacy` and `fast` engines agree to rounding for the same error mode.

## License

MIT — see `LICENSE`.
