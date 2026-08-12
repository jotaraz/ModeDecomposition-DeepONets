"""Summarise the 10-seed Fig. 6 nets: final train/test error per seed, and
whether the per-mode error ordering (the spectral-bias claim) holds for every
seed. Reads only log.txt / log_modes.txt; no dataset is loaded."""
import glob
import os
import re

import numpy as np

NETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "nets")
LLW = 5

dirs = sorted(glob.glob(os.path.join(NETS, "*synthv09082026*")))
rows = {}
for d in dirs:
    b = os.path.basename(d)
    tag = re.search(r"m5000_(fs[^_]+)_numd", b).group(1)
    seed = int(re.search(r"_v(\d+)$", b).group(1))
    log = np.loadtxt(os.path.join(d, "log.txt"))
    modes = np.loadtxt(os.path.join(d, "log_modes.txt"))
    train = 10 ** log[-1, 1]
    test = 10 ** log[-1, 4]
    final_modes = modes[-1, 1:1 + LLW]
    rows.setdefault(tag, []).append((seed, train, test, final_modes))

for tag in ["fs0.2ss-0.01", "fs-0.2ss-0.01", "fs0.2ss-0.5", "fs-0.2ss-0.5"]:
    key = tag + "ns0.2"
    if key not in rows:
        print(f"{key}: MISSING")
        continue
    rs = sorted(rows[key])
    seeds = [r[0] for r in rs]
    tr = np.array([r[1] for r in rs])
    te = np.array([r[2] for r in rs])
    increasing = "fs0.2" in tag and not tag.startswith("fs-")
    # Three views of "does the error follow the frequency ordering?", from
    # strictest to loosest: every consecutive step monotone; the rank
    # correlation of error vs mode index has the right sign; the last mode is
    # on the right side of the first.
    ok = trend = ends = 0
    ratios = []
    for _, _, _, m in rs:
        d = np.diff(m)
        if (d > 0).all() if increasing else (d < 0).all():
            ok += 1
        rank = np.corrcoef(np.arange(len(m)), np.argsort(np.argsort(m)))[0, 1]
        if (rank > 0) if increasing else (rank < 0):
            trend += 1
        r = m[-1] / m[0]
        ratios.append(r if increasing else 1.0 / r)
        if ratios[-1] > 1.0:
            ends += 1
    ratios = np.array(ratios)
    print(f"\n{key}   ({'increasing' if increasing else 'decreasing'} freq.)")
    print(f"  seeds present : {seeds}")
    print(f"  train  min/med/max : {tr.min():.3e} / {np.median(tr):.3e} / {tr.max():.3e}")
    print(f"  test   min/med/max : {te.min():.3e} / {np.median(te):.3e} / {te.max():.3e}")
    print(f"  test   spread max/min : {te.max()/te.min():.2f}x")
    print(f"  strictly monotone modes : {ok}/{len(rs)} seeds")
    print(f"  rank-corr right sign    : {trend}/{len(rs)} seeds")
    print(f"  hardest/easiest mode on right side : {ends}/{len(rs)} seeds"
          f"  (ratio med {np.median(ratios):.2f}x, min {ratios.min():.2f}x)")
    med = np.median(np.array([r[3] for r in rs]), axis=0)
    print(f"  median final mode errors : {' '.join(f'{x:.2e}' for x in med)}")
