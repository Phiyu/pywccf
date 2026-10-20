"""Halo -- cosmic-web cross-correlation, end to end.

Measures, for one halo mass bin, the 1D and 3D WCCF of every web type for
the whole bin and for its early- / late-forming tails (top / bottom 20% in
z_f), and writes everything to one HDF5 file.

Synthetic box (runs anywhere in about a minute)::

    python examples/halo_web.py --synthetic --out halo_web_synthetic.h5

ELUCID (paper setup: R_s = 2, lambda_th = 0, 10^6 cells per web type)::

    python examples/halo_web.py \
        --field /path/to/ELUCID/field_099.2.hdf5 \
        --halo  /path/to/ELUCID/halo_0701.hdf5 \
        --mass 11.5 12.0 --workers 8 --out halo_web_m115_120.h5

Reading the output::

    from pywccf import load_results
    res = load_results('halo_web_synthetic.h5')
    r, xi, sd = res['filament']['3d_old'].profile(3)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import pywccf
from pywccf import WCCF1D, WCCF3D, HaloSample, TidalField, save_results
from pywccf.core import PAPER_EDGES
from pywccf.utils import synthetic_box


def load_inputs(args):
    if args.synthetic:
        _, tf, halos = synthetic_box(64, 128.0, n_halos=40_000, R_s=args.R_s,
                                     lam_th=args.lam_th, seed=args.seed,
                                     verbose=True)
        edges = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        return tf, halos, edges
    tf = TidalField.from_file(args.field, load_evec="lazy",
                              lam_th=args.lam_th)
    halos = HaloSample.from_file(args.halo, l_box=tf.l_box)
    halos = halos.select_paper(zf_range=(0.01, 20.0), exclude_backsplash=True)
    return tf, halos, PAPER_EDGES


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--synthetic", action="store_true",
                   help="use a synthetic box instead of --field / --halo")
    p.add_argument("--field", help="tidal-field HDF5 file")
    p.add_argument("--halo", help="halo catalogue HDF5 file")
    p.add_argument("--mass", nargs=2, type=float, default=(11.5, 12.0),
                   metavar=("LO", "HI"), help="log10 halo-mass bin")
    p.add_argument("--R_s", type=float, default=2.0,
                   help="smoothing scale (synthetic box only)")
    p.add_argument("--lam_th", type=float, default=0.0)
    p.add_argument("--tail", type=float, default=0.2,
                   help="tail fraction in z_f for old / young")
    p.add_argument("--webs", nargs="+", default=list(pywccf.WEB_TYPES))
    p.add_argument("--max-cells", type=int, default=1_000_000)
    p.add_argument("--error", default="jackknife",
                   choices=("none", "jackknife", "bootstrap"))
    p.add_argument("--n-jk-side", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="halo_web.h5")
    args = p.parse_args(argv)
    if not args.synthetic and not (args.field and args.halo):
        p.error("give --synthetic, or both --field and --halo")

    t0 = time.time()
    tf, halos, edges = load_inputs(args)

    # one mass bin; old / young = upper / lower tails in z_f inside it
    lo, hi = args.mass
    halos = halos.subset_by_val("mass", lo=lo, hi=hi)
    young, old = halos.quantile_masks("zf", args.tail)
    samples = {"all": np.ones(halos.n_objs, dtype=bool),
               "old": old, "young": young}
    print(f"mass bin [{lo}, {hi}): {halos.n_objs:,} halos, "
          f"{old.sum():,} old, {young.sum():,} young")

    kw = dict(edges=edges, max_cells=args.max_cells, workers=args.workers)
    w1 = WCCF1D(tf, halos, **kw)
    w3 = WCCF3D(tf, halos, **kw)
    results = {}
    for web in args.webs:
        r1 = w1.compute(web, samples, error=args.error,
                        n_jk_side=args.n_jk_side, seed=args.seed)
        r3 = w3.compute(web, samples, error=args.error,
                        n_jk_side=args.n_jk_side, seed=args.seed)
        results[web] = ({f"1d_{k}": v for k, v in r1.items()}
                        | {f"3d_{k}": v for k, v in r3.items()})
        r, xi, _ = r3["all"].profile(3)
        print(f"{web:9s} xi_3D(e3) at r={r[1]:.1f}: {xi[1]:+.3f}; "
              f"A(old/young) at the origin bin: "
              f"{r3['old'].xi[0, 0, 0] / r3['young'].xi[0, 0, 0]:.2f}")

    run_info = {"pywccf_version": pywccf.__version__,
                "mass_bin": [lo, hi], "tail": args.tail,
                "R_s": tf.R_s if tf.R_s is not None else np.nan,
                "lambda_th": tf.lam_th, "error": args.error,
                "synthetic": bool(args.synthetic),
                "field": args.field or "", "halo": args.halo or ""}
    save_results(args.out, results, run_info=run_info)
    print(f"saved {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
