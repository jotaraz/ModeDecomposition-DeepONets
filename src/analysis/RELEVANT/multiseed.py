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

# Band drawing is driven by the environment so one set of scripts produces both
# output sets:  MULTISEED_BAND=1 python produce_all_figures_multiseed.py
# Each script also exposes SHOW_BAND at the top, which the env var overrides.
SHOW_BAND = os.environ.get("MULTISEED_BAND", "") not in ("", "0", "false")

# "std"    -> mean -/+ 1 standard deviation across seeds  (what the band set uses)
# "minmax" -> elementwise min..max across seeds
BAND_MODE = os.environ.get("MULTISEED_BAND_MODE", "std")

BAND_ALPHA = 0.35

# On log-scaled axes mean-std can fall at or below zero, which fill_between
# cannot draw.  Where that happens the lower edge is clipped to this fraction of
# the mean, so the band is TRUNCATED AT THE BOTTOM rather than dropped -- keep it
# in mind when reading any panel whose band reaches this floor.
BAND_FLOOR_FRAC = 1e-2


def suffix():
    """'_band' when bands are on, so the two output sets never collide."""
    return "_band" if SHOW_BAND else ""


def _spread(A):
    """(lo, hi) across the seed axis, per BAND_MODE."""
    m = A.mean(axis=0)
    if BAND_MODE == "minmax":
        return A.min(axis=0), A.max(axis=0)
    sd = A.std(axis=0)
    lo, hi = m - sd, m + sd
    # keep the lower edge positive so log axes can render it
    floor = np.abs(m) * BAND_FLOOR_FRAC
    lo = np.where(lo > 0, lo, floor)
    return lo, hi


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
    lo/hi delimit the band (mean -/+ 1 std by default; see BAND_MODE).
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
    lo, hi = _spread(A)
    return A.mean(axis=0), lo, hi, len(vals)


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
        means[k] = A.mean(axis=0)
        los[k], his[k] = _spread(A)
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
    """Shade lo..hi behind a mean line, if the script's SHOW_BAND is on.

    A thin edge is drawn as well: on panels spanning several decades the fill
    alone is only a few pixels tall and effectively invisible.
    """
    if show is None:
        show = SHOW_BAND
    if not show or lo is None or hi is None:
        return
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=0)
    ax.plot(x, lo, '-', color=color, alpha=min(1.0, alpha * 2), linewidth=0.4, zorder=0)
    ax.plot(x, hi, '-', color=color, alpha=min(1.0, alpha * 2), linewidth=0.4, zorder=0)


def is_endpoint_stage(epoch, stages):
    """True for the first/last plotted training stage.

    The stage-ramp panels draw ~8 curves in a colour ramp; banding every one
    turns into an opaque wash, so those scripts band only the endpoints.
    """
    return epoch == min(stages) or epoch == max(stages)


def note(fig, n_used, seeds=SEEDS):
    """Small footer recording how many seeds went into the figure."""
    if n_used and n_used != len(seeds):
        fig.text(0.01, 0.01, "mean over %d/%d seeds" % (n_used, len(seeds)),
                 fontsize=6, color="0.4")
