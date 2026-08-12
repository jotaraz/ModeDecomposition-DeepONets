"""Run run_synthetic_data_gen.py unmodified, but with numpy's global RNG seeded
first, so a successful draw is reproducible.

The generator rejection-samples V directions against innerprod_threshold and
gives up (returns False) if no candidate clears it, which makes an unseeded run
a coin flip. Seeding lets us record which draw produced the shipped dataset.

    python gen_synth_seeded.py <seed> <path/to/run_synthetic_data_gen.py>
"""
import os
import runpy
import sys

import numpy as np

seed = int(sys.argv[1])
script = os.path.abspath(sys.argv[2])

np.random.seed(seed)
# The script does `from synthetic_gramschmidt import *`, which normally resolves
# because running a file puts its directory on sys.path; runpy does not.
sys.path.insert(0, os.path.dirname(script))

runpy.run_path(script, run_name="__main__")
