"""Plot the singular-value spectra of the dataset U matrices (log-y).

Reads the pre-computed ``*_U_singvals.txt`` files in ``data/datasets`` (one
singular value per line, descending) and draws all eight requested datasets in
a single log-y figure.

Styling follows the convention already used in ``2d-pdes/plot_singvals.py``:
  * colour     encodes the problem family (heat-sine, heat-grf, adv-diff,
    Burgers, KdV); within the multi-curve KdV family the shade grades with tau.
  * linestyle  encodes the dimension: 2D = solid, 1D = dashed.
Colours come from ``colors_cb.txt`` (colour-blind-safe) where a fixed hue is
needed, so identity never rests on colour alone.

The spectra all decay into the double-precision noise floor (sigma_i/sigma_1
~ 1e-15 and below), which on a log axis would otherwise waste 15+ decades on
numerical zeros. The y-axis is therefore clipped at ``--floor`` (relative to
the largest sigma_1) by default; pass ``--full-range`` to see everything.

Usage
-----
    python src/analysis/plot_dataset_singvals.py
    python src/analysis/plot_dataset_singvals.py --normalize
    python src/analysis/plot_dataset_singvals.py --save figures/pngs/singvals.png
    python src/analysis/plot_dataset_singvals.py --full-range
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATADIR = os.path.join(REPO_ROOT, "data", "datasets")

# The eight datasets, in the order they were requested: 2D first, then 1D.
# `family` -> colour, `dim` -> linestyle, `key` -> shade within a family.
DATASETS = [
    # --- 2D ---
    # m5000: spectra recomputed from the 5000-sample matrices (4096x5000)
    dict(file="heat2d_grf_nx64_l0.05_D1_m5000_t0.004_U_singvals.txt",
         label="2D heat, GRF ($\\ell=0.05$)", family="heat-grf", dim="2D", key=0.05),
    #dict(file="heat2d_grf_nx64_l0.1-0.2_D1_m5000_t0.004_U_singvals.txt",
    #     label="2D heat, GRF ($\\ell=0.1$–$0.2$)", family="heat-grf", dim="2D", key=0.15),
    dict(file="heat2d_sine_nx64_K8_D1_m5000_t0.004_U_singvals.txt",
         label="2D heat, sine ($K=8$)", family="heat-sine", dim="2D"),
    # --- 1D ---
    dict(file="advdiffnx201_dt0.0005_nc20_m1000_1000_U_singvals.txt",
             label="1D AD ($\\tau=0.5$)", family="adv-diff", dim="1D"),
    #dict(file="kdvnx401_dt0.0001_nc5_m5000_10_U_singvals.txt",
    #     label="1D KdV ($\\tau=10$)", family="kdv", dim="1D", key=10),
    dict(file="kdvnx401_dt0.0001_nc5_m5000_1999_U_singvals.txt",
         label="1D KdV ($\\tau=0.2$)", family="kdv", dim="1D", key=1999),
    #dict(file="kdvnx401_dt0.0001_nc5_m5000_5999_U_singvals.txt",
    #     label="1D KdV ($\\tau=5999$)", family="kdv", dim="1D", key=5999),
    dict(file="burgers_dt0.0001_nc10_m3800_100_U_singvals.txt",
         label="1D Burgers ($\\tau=0.1$)", family="burgers", dim="1D"),
]

# Fixed hues taken from the repo's colour-blind-safe palette (colors_cb.txt).
FAMILY_STYLE = {
    "heat-grf":  dict(cmap="Purples", lo=0.55, hi=0.95),  # gradient over ell
    "heat-sine": dict(color="#377eb8"),                   # blue
    "kdv":       dict(cmap="YlOrBr", lo=0.45, hi=0.90),   # gradient over tau
    "adv-diff":  dict(color="#4daf4a"),                   # green
    "burgers":   dict(color="#e41a1c"),                   # red
}
DIM_STYLE = {
    "2D": dict(linestyle="-"),
    "1D": dict(linestyle="--"),
}


def assign_colors(entries):
    """Fixed colour per single-member family; a shade ramp within multi-member ones."""
    by_family = {}
    for e in entries:
        by_family.setdefault(e["family"], []).append(e)

    for family, members in by_family.items():
        style = FAMILY_STYLE[family]
        if "cmap" in style and len(members) > 1:
            cmap = plt.get_cmap(style["cmap"])
            lo, hi = style.get("lo", 0.4), style.get("hi", 0.9)
            ranked = sorted({m["key"] for m in members})
            for m in members:
                frac = ranked.index(m["key"]) / (len(ranked) - 1) if len(ranked) > 1 else 1.0
                m["color"] = cmap(lo + (hi - lo) * frac)
        elif "cmap" in style:
            m = members[0]
            m["color"] = plt.get_cmap(style["cmap"])(style.get("hi", 0.9))
        else:
            for m in members:
                m["color"] = style["color"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default=DATADIR,
                    help=f"directory holding the *_singvals.txt files (default: {DATADIR})")
    ap.add_argument("--normalize", action="store_true",
                    help="plot sigma_i / sigma_1 instead of the raw singular values")
    ap.add_argument("--floor", type=float, default=1e-16,
                    help="clip the y-axis this far below the largest sigma_1 "
                         "(default: 1e-16, i.e. the double-precision noise floor)")
    ap.add_argument("--full-range", action="store_true",
                    help="do not clip the y-axis; show the numerical-zero tails too")
    ap.add_argument("--logx", action="store_true",
                    help="log-scale the index axis too; spreads out the fast-decaying "
                         "1D spectra, which otherwise bunch up near i=0")
    ap.add_argument("--xmax", type=float, default=None,
                    help="truncate the index axis at this value")
    ap.add_argument("--save", metavar="PATH", default=None,
                    help="save the figure to PATH instead of showing it interactively")
    args = ap.parse_args()

    entries = [dict(d, path=os.path.join(args.datadir, d["file"])) for d in DATASETS]
    missing = [e["file"] for e in entries if not os.path.exists(e["path"])]
    if missing:
        raise SystemExit("missing singvals files in {}:\n  {}".format(
            args.datadir, "\n  ".join(missing)))

    assign_colors(entries)

    if args.save:                      # headless: no interactive backend needed
        plt.switch_backend("Agg")

    fig, ax = plt.subplots(figsize=(7, 6))   # width restored; it was empty and broke parsing
    top = 0.0
    for e in entries:
        s = np.atleast_1d(np.loadtxt(e["path"]))
        s1 = s[0]
        if args.normalize:
            s = s / s1
        top = max(top, s[0])
        x = np.arange(1, len(s) + 1)
        ax.semilogy(x, s, label=e["label"], color=e["color"], lw=1.8,
                    **DIM_STYLE[e["dim"]])
        print(f"{e['label']:32s} n={len(s):5d}  sigma_1={s1:.4e}  "
              f"sigma_n/sigma_1={s[-1] / s[0]:.3e}")

    if not args.full_range:
        ax.set_ylim(bottom=top * args.floor, top=top * 3)
    if args.logx:
        ax.set_xscale("log")
    if args.xmax is not None:
        ax.set_xlim(right=args.xmax)

    ax.set_xlim((-4, 150))
    ax.set_ylim((6e-5, 1.5))

    ax.set_xlabel("index $i$")
    ax.set_ylabel("$\\sigma_i / \\sigma_1$" if args.normalize else "singular value $\\sigma_i$")
    #ax.set_title("Singular-value spectra of the dataset $U$ matrices")
    ax.grid(True, which="major", ls=":", alpha=0.45)
    ax.grid(True, which="minor", ls=":", alpha=0.2)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    fig.tight_layout()

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print("saved:", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
