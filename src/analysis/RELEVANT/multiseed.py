"""Seed-averaging helpers shared by the *_multiseed analysis scripts.

The originals each read one net directory ending in "_v0".  Their _multiseed
counterparts read all ten seeds "_v0".."_v9" and plot the mean instead.

MEAN CONVENTION -- arithmetic, taken on the plotted quantity.
    don_code writes np.log10(error) into log.txt and the plot scripts convert
    with 10**(col/2).  Averaging the stored column would give a GEOMETRIC mean;
    we deliberately convert first and average afterwards, so what is plotted is
    the arithmetic mean of the ten relative errors.  In practice: pass a `fn`
    that returns the already-converted quantity.

Missing or unreadable seeds are skipped rather than fatal, so a partial sweep
still plots; every helper reports how many seeds it actually used.
"""

import os

from ... import don_code

np = don_code.np

SEEDS = list(range(10))

# Every *_multiseed script exposes its own SHOW_BAND; this is the default.
# When True, the scripts additionally shade min..max across seeds behind the
# mean line.  The band is always computed, so switching it on costs nothing.
SHOW_BAND = False

BAND_ALPHA = 0.18


def seed_names(name, seeds=SEEDS):
    """'..._v0' -> ['..._v0', '..._v1', ...] (splits on the trailing '_v')."""
    base = name.rsplit("_v", 1)[0]
    return ["%s_v%d" % (base, s) for s in seeds]


def present(name, seeds=SEEDS):
    """The seed directories that actually exist under don_code.nets_dir."""
    return [n for n in seed_names(name, seeds)
            if os.path.isdir(os.path.join(don_code.nets_dir, n))]


def _stack(vals):
    """Stack per-seed arrays, truncating to the common shape if they differ."""
    arrs = [np.asarray(v, dtype=float) for v in vals]
    shapes = {a.shape for a in arrs}
    if len(shapes) > 1:
        ndim = min(a.ndim for a in arrs)
        cut = tuple(min(a.shape[d] for a in arrs) for d in range(ndim))
        print("multiseed: seed arrays disagree in shape %s, truncating to %s"
              % (sorted(shapes), cut))
        arrs = [a[tuple(slice(0, c) for c in cut)] for a in arrs]
    return np.stack(arrs)


def mean_over_seeds(fn, name, seeds=SEEDS):
    """Call fn(net_dir_name) for each seed and average the results.

    fn should return an array (or None / raise, if that seed is unusable).
    Returns (mean, lo, hi, n_used); (None, None, None, 0) if no seed worked.
    lo/hi are the elementwise min/max across seeds, for SHOW_BAND.
    """
    vals = []
    for nm in seed_names(name, seeds):
        try:
            v = fn(nm)
        except Exception as exc:                      # noqa: BLE001
            print("multiseed: skipping %s (%s)" % (nm, exc))
            continue
        if v is None:
            continue
        vals.append(v)

    if not vals:
        print("multiseed: NO usable seed for", name)
        return None, None, None, 0

    A = _stack(vals)
    return A.mean(axis=0), A.min(axis=0), A.max(axis=0), len(vals)


def mean_of_keys(fn, name, keys, seeds=SEEDS):
    """Like mean_over_seeds but fn returns a dict; average the given keys.

    Returns (means, los, his, n_used) as three dicts keyed by `keys`.
    """
    per_key = {k: [] for k in keys}
    n = 0
    for nm in seed_names(name, seeds):
        try:
            d = fn(nm)
        except Exception as exc:                      # noqa: BLE001
            print("multiseed: skipping %s (%s)" % (nm, exc))
            continue
        if d is None:
            continue
        n += 1
        for k in keys:
            per_key[k].append(d[k])

    if n == 0:
        print("multiseed: NO usable seed for", name)
        return None, None, None, 0

    means, los, his = {}, {}, {}
    for k in keys:
        A = _stack(per_key[k])
        means[k], los[k], his[k] = A.mean(axis=0), A.min(axis=0), A.max(axis=0)
    return means, los, his, n


def load_mean(name, filename, transform=None, seeds=SEEDS):
    """Mean of np.loadtxt(<net>/<filename>) across seeds.

    `transform` is applied to each seed's array BEFORE averaging -- use it to
    convert log10 columns to errors so the mean stays arithmetic in the plotted
    quantity.
    """
    def one(nm):
        path = os.path.join(don_code.nets_dir, nm, filename)
        if not os.path.isfile(path):
            return None
        a = np.loadtxt(path)
        return transform(a) if transform is not None else a

    return mean_over_seeds(one, name, seeds)


def band(ax, x, lo, hi, color, show=None, alpha=BAND_ALPHA):
    """Shade lo..hi behind a mean line, if the script's SHOW_BAND is on."""
    if show is None:
        show = SHOW_BAND
    if not show or lo is None or hi is None:
        return
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=0)


def note(fig, n_used, seeds=SEEDS):
    """Small footer recording how many seeds went into the figure."""
    if n_used and n_used != len(seeds):
        fig.text(0.01, 0.01, "mean over %d/%d seeds" % (n_used, len(seeds)),
                 fontsize=6, color="0.4")
