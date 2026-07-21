"""Animate 2D heat-equation solutions from a generated dataset.

The datasets store only a handful of save times, so instead of interpolating
between them this script reconstructs each initial condition and re-evolves it on
a dense time grid with the exact spectral solver -- giving a genuinely smooth
animation of the diffusion. Any dataset ``.npz`` written by ``generate_dataset``
works (its stored D and side are reused).

Examples
--------
    # both families (sine + grf) side by side, sample 0, default output
    python animate_solutions.py

    # a specific dataset and sample, per-frame contrast stretch, faster
    python animate_solutions.py --dataset data2d/heat2d_grf_nx64_l0.1-0.2_D1.npz \
        --sample 3 --normalize --fps 20

By default frames use a fixed color scale taken from u0, so the physical
amplitude decay is visible (later frames fade toward zero). Pass --normalize to
rescale each frame to its own max instead, which keeps the spatial structure
visible all the way down.
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from diffusion_spectral import diffusion_spectral


def load_panel(npz_path, sample, frames, tmax):
    """Return (label, times, stack) where stack is (frames, n, n)."""
    d = np.load(npz_path)
    u0 = d["u0"][sample]
    D = float(d["D"])
    side = float(d["side"])
    if tmax is None:
        tmax = float(np.max(d["times"]))
    times = np.linspace(0.0, tmax, frames)
    stack = np.stack([
        u0 if t == 0.0 else diffusion_spectral(u0, D=D, side=side, t=t)
        for t in times
    ])
    label = os.path.basename(npz_path).replace(".npz", "")
    # A short human label: family + key parameter.
    if "grf" in label and "l_per_sample" in d:
        tag = f"GRF  (l={float(d['l_per_sample'][sample]):g}, D={D:g})"
    elif "sine" in label:
        k = int(d["n_modes"]) if "n_modes" in d else "?"
        tag = f"sine  (K={k}, D={D:g})"
    else:
        tag = label
    return tag, times, stack, d["X"], d["Y"]


def animate(npz_paths, sample, frames, tmax, fps, normalize, out):
    panels = [load_panel(p, sample, frames, tmax) for p in npz_paths]
    npan = len(panels)

    fig, axes = plt.subplots(1, npan, figsize=(5.2 * npan, 5.0), squeeze=False)
    axes = axes[0]
    ims = []
    for ax, (tag, times, stack, X, Y) in zip(axes, panels):
        vmax = np.abs(stack[0]).max() or 1.0
        im = ax.imshow(stack[0], origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=[0, 1, 0, 1],
                       interpolation="bilinear")
        ax.set_title(tag, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append((im, ax))

    suptitle = fig.suptitle("", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    times0 = panels[0][1]

    def update(k):
        for (im, ax), (tag, times, stack, X, Y) in zip(ims, panels):
            frame = stack[k]
            im.set_data(frame)
            if normalize:
                v = np.abs(frame).max() or 1.0
                im.set_clim(-v, v)
        peak = max(np.abs(p[2][k]).max() for p in panels)
        mode = "per-frame scaled" if normalize else "fixed scale"
        suptitle.set_text(
            f"t = {times0[k]:.4f}    |    max|u| = {peak:.3e}    ({mode})")
        return [im for im, _ in ims] + [suptitle]

    anim = FuncAnimation(fig, update, frames=len(times0), blit=False)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {out}  ({len(times0)} frames, {fps} fps, sample {sample})")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", default=None,
                   help="one or more dataset .npz files "
                        "(default: sine + grf found in --outdir)")
    p.add_argument("--outdir", default="data2d",
                   help="directory to auto-discover datasets in")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--tmax", type=float, default=None,
                   help="end time (default: max save time in the dataset)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--normalize", action="store_true",
                   help="rescale each frame to its own max (keeps structure "
                        "visible as amplitude decays)")
    p.add_argument("--out", default=None,
                   help="output .gif path (default: <outdir>/animation.gif)")
    args = p.parse_args()

    datasets = args.dataset
    if datasets is None:
        datasets = sorted(glob.glob(os.path.join(args.outdir, "heat2d_*.npz")))
        if not datasets:
            p.error(f"no datasets found in {args.outdir}; pass --dataset")
    out = args.out or os.path.join(args.outdir, "animation.gif")
    animate(datasets, args.sample, args.frames, args.tmax, args.fps,
            args.normalize, out)


if __name__ == "__main__":
    main()
