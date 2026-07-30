"""Plot singular-value spectra of the U matrices (interactive).

Loads pre-computed *_singvals.txt files (one singular value per line) and shows
a log-y plot of the (normalized) spectra, in a window you can pan/zoom.

Styling of the default curves is chosen for readability:
  * colour   encodes the problem class (heat-sine, heat-grf, adv-diff, Burgers,
    KdV); within the multi-curve GRF and KdV families the shade grades with the
    parameter (length scale l / time tau).
  * marker + linestyle encode the dimension: 2D = solid line + circles,
    1D = dashed line + triangles.

Usage
-----
    python plot_singvals.py                 # all default curves, styled
    python plot_singvals.py *_singvals.txt  # arbitrary files (label = stem)
    python plot_singvals.py --save spectra.png   # headless
    python plot_singvals.py --linear             # linear y-axis
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Default curves. Each entry carries the file, a human label, the problem
# `family` (-> colour) and `dim` (-> marker/linestyle); `key` orders the shade
# within a multi-curve family (GRF length scale, KdV time tau).
DEFAULTS = [
    # --- 2D heat: sine ---
    dict(file="heat2d_sine_nx64_K8_D1_U_t0.004_singvals.txt",
         label="2D heat: sine (K=8)", family="heat-sine", dim="2D"),
    # --- 2D heat: GRF (single fixed length scale, + the mixed l0.1/0.2 set) ---
    dict(file="heat2d_grf_nx64_l0.05_D1_U_t0.004_singvals.txt",
         label="2D heat: grf (l=0.05)", family="heat-grf", dim="2D", key=0.05),
    dict(file="heat2d_grf_nx64_l0.1_D1_U_t0.004_singvals.txt",
         label="2D heat: grf (l=0.1)", family="heat-grf", dim="2D", key=0.10),
    dict(file="heat2d_grf_nx64_l0.1-0.2_D1_U_t0.004_singvals.txt",
         label="2D heat: grf (l=0.1/0.2)", family="heat-grf", dim="2D", key=0.15),
    dict(file="heat2d_grf_nx64_l0.3_D1_U_t0.004_singvals.txt",
         label="2D heat: grf (l=0.3)", family="heat-grf", dim="2D", key=0.30),
    dict(file="heat2d_grf_nx64_l0.5_D1_U_t0.004_singvals.txt",
         label="2D heat: grf (l=0.5)", family="heat-grf", dim="2D", key=0.50),
    dict(file="heat2d_grf_nx64_l0.7_D1_U_t0.004_singvals.txt",
         label="2D heat: grf (l=0.7)", family="heat-grf", dim="2D", key=0.70),
    # --- 1D datasets ---
    dict(file="advdiffnx201_dt0.0005_nc20_m1000_1000_U_singvals.txt",
         label="1D adv-diff (nx201, tau=1000)", family="adv-diff", dim="1D"),
    dict(file="burgers_dt0.0001_nc10_m3800_100_U_singvals.txt",
         label="1D Burgers (tau=100)", family="burgers", dim="1D"),
    dict(file="kdvnx401_dt0.0001_nc5_m5000_10_U_singvals.txt",
         label="1D KdV (nx401, tau=10)", family="kdv", dim="1D", key=10),
    dict(file="kdvnx401_dt0.0001_nc5_m5000_1999_U_singvals.txt",
         label="1D KdV (nx401, tau=1999)", family="kdv", dim="1D", key=1999),
    dict(file="kdvnx401_dt0.0001_nc5_m5000_5999_U_singvals.txt",
         label="1D KdV (nx401, tau=5999)", family="kdv", dim="1D", key=5999),
]

# family -> colour. Single-member families get a fixed colour; multi-member
# families get a sequential colormap sampled by `key` rank (dark = larger param).
FAMILY_STYLE = {
    "heat-sine": dict(color="#1f77b4"),   # blue
    "heat-grf":  dict(cmap="Purples"),    # gradient over l
    "adv-diff":  dict(color="#2ca02c"),   # green
    "burgers":   dict(color="#d62728"),   # red
    "kdv":       dict(cmap="YlOrBr"),     # gradient over tau
}
# dimension -> line/marker style
DIM_STYLE = {
    "2D": dict(linestyle="-",  marker="o"),
    "1D": dict(linestyle="--", marker="^"),
}


def assign_colors(entries):
    """Fill entry['color'] from FAMILY_STYLE (gradient within cmap families)."""
    members = defaultdict(list)
    for e in entries:
        members[e.get("family")].append(e)
    for fam, ms in members.items():
        style = FAMILY_STYLE.get(fam)
        if style is None:                 # unknown/CLI family -> default cycle
            continue
        if "cmap" in style:
            cmap = plt.get_cmap(style["cmap"])
            uniq = sorted({m.get("key", i) for i, m in enumerate(ms)})
            lo, hi = 0.45, 0.95           # avoid too-faint light end
            for m in ms:
                frac = (uniq.index(m.get("key")) / (len(uniq) - 1)
                        if len(uniq) > 1 else 1.0)
                m["color"] = cmap(lo + (hi - lo) * frac)
        else:
            for m in ms:
                m["color"] = style["color"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*",
                    help="*_singvals.txt files to plot (default: the styled set)")
    ap.add_argument("--save", metavar="PATH", default=None,
                    help="save the figure to PATH instead of showing it interactively")
    ap.add_argument("--linear", action="store_true",
                    help="use a linear y-axis instead of log")
    args = ap.parse_args()

    if args.files:
        entries = [dict(path=f, label=os.path.splitext(os.path.basename(f))[0])
                   for f in args.files]
    else:
        entries = [dict(d, path=os.path.join(HERE, d["file"])) for d in DEFAULTS]

    assign_colors(entries)

    if args.save:               # headless: don't need an interactive backend
        plt.switch_backend("Agg")

    fig, ax = plt.subplots(figsize=(9, 6))
    plotter = ax.plot if args.linear else ax.semilogy
    for e in entries:
        s = np.atleast_1d(np.loadtxt(e["path"]))
        s = (s / s[0]) ** 2
        x = np.arange(1, len(s) + 1)

        kw = dict(label=e["label"], lw=1.6)
        if e.get("color") is not None:
            kw["color"] = e["color"]
        ds = DIM_STYLE.get(e.get("dim"), {})
        if ds.get("linestyle"):
            kw["linestyle"] = ds["linestyle"]
        if ds.get("marker"):
            kw["marker"] = ds["marker"]
            kw["markevery"] = max(1, len(s) // 12)   # ~12 markers per curve
            kw["markersize"] = 5.5
            kw["markeredgecolor"] = "white"
            kw["markeredgewidth"] = 0.4

        plotter(x, s, **kw)
        print(f"{e['label']}: {len(s)} singular values, "
              f"first={s[0]:.4e}, last={s[-1]:.4e}")

    ax.set_xlabel("index $i$")
    ax.set_ylabel("normalized energy $(\\sigma_i/\\sigma_1)^2$")
    ax.set_title("Singular value spectra of the $U$ matrices")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=1)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print("saved:", args.save)
    else:
        plt.show()   # interactive: pan/zoom with the matplotlib toolbar


if __name__ == "__main__":
    main()
