"""Where does mean-std go non-positive, i.e. where is the band clipped to the
BAND_FLOOR_FRAC floor rather than showing the real lower edge?"""
import glob
import os
import re

import numpy as np

NETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "nets")
LLW = 5

for tag in ["fs0.2ss-0.01", "fs-0.2ss-0.01", "fs0.2ss-0.5", "fs-0.2ss-0.5"]:
    pat = os.path.join(NETS, f"*_{tag}ns0.2_numd5000_lrAdam40_v*")
    ds = sorted(glob.glob(pat), key=lambda p: int(re.search(r"_v(\d+)$", p).group(1)))
    first, last = [], []
    for d in ds:
        m = np.loadtxt(os.path.join(d, "log_modes.txt"))
        first.append(m[0, 1:1 + LLW])
        last.append(m[-1, 1:1 + LLW])
    for label, A in (("initial", np.array(first)), ("final", np.array(last))):
        mu, sd = A.mean(axis=0), A.std(axis=0)
        clipped = sd >= mu
        print(f"{tag:16s} {label:8s} n={len(ds):2d}  "
              f"cv=[{' '.join(f'{s/m:.2f}' for s, m in zip(sd, mu))}]  "
              f"clipped at modes {list(np.where(clipped)[0])}")
