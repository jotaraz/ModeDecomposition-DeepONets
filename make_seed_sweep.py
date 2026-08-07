"""
Generate the HTCondor manifests for the 10-seed sweep.

Every DeepONet needed by the figures of arXiv:2602.21910 -- except the four
Figure 6 (synthv) nets, whose datasets do not exist -- trained from seeds
vtag = 0..9.  See list_nets_seeds.txt for where each config comes from and
SEED_SWEEP.md for the calibration behind the packing.

Run from the repo root:
    python make_seed_sweep.py [n_jobs]        # default 30

Writes:
    condor/sweep_manifests/job_XX.txt   one TSV line per training run:
                                        <expected_dir> <produced_dir> <20 args>
    condor/sweep_jobs.txt               the manifest list sweep.sub queues over
"""

import os
import sys

N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "condor", "sweep_manifests")

KDV = ("kdvnx401_dt0.0001_nc5_m5000", "1999")
KDV6 = ("kdvnx401_dt0.0001_nc5_m5000", "5999")
AD = ("advdiffnx201_dt0.0005_nc20_m1000", "1000")
BUR = ("burgers_dt0.0001_nc10_m3800", "100")

# lrstag -> (init_lr, decay); src/run.py lrs()
LRS = {32: ("1e-4", "0.95"), 40: ("2e-3", "0.95")}


def cfg(nep, d, w, llw, bat, lrstag, whichT, dosigma, sisc, exp, doadam,
        stacked=0, adaptive=0, momentum=1, numd=1000):
    batch_name, uendtag = bat
    init_lr, decay = LRS[lrstag]
    return dict(nep=nep, d=d, w=w, llw=llw, batch_name=batch_name, uendtag=uendtag,
                lrstag=lrstag, init_lr=init_lr, decay=decay, numd=numd,
                whichT=whichT, dosigma=dosigma, sisc=sisc, exp=exp,
                doadam=doadam, stacked=stacked, adaptive=adaptive,
                momentum=momentum)


def stem(c, vtag, produced):
    """Reproduce the directory name built by src/execute_don.py (line 95ff).

    produced=True gives the name execute_don actually writes, which for
    adaptiveLR=1 and exponent != 0 carries '_expA' instead of '_exp'.
    """
    s = "whichT%d_doStacked%s_doSigma%d" % (
        c["whichT"], "True" if c["stacked"] else "False", c["dosigma"])
    tag = "expA" if (produced and c["adaptive"] and abs(c["exp"]) > 1e-12) else "exp"
    s += "_sisc%s_aT0.0_aB0.0_%s%s" % (c["sisc"], tag, c["exp"])
    s += "_Nep%d_d%d_w%d_llw%d" % (c["nep"], c["d"], c["w"], c["llw"])
    s += "_bat%s_%s_numd%d" % (c["batch_name"], c["uendtag"], c["numd"])
    if c["doadam"]:
        s += "_lrAdam%d" % c["lrstag"] if c["momentum"] else "_lrAda%d" % c["lrstag"]
    else:
        s += "_lrSGD%d" % c["lrstag"]
    return s + "_v%d" % vtag


def args(c, vtag):
    """The 20 positional arguments of execute_don.py, in order."""
    return " ".join(str(x) for x in [
        c["nep"], vtag, c["d"], c["w"], c["llw"], 0, c["batch_name"],
        c["lrstag"], c["init_lr"], c["decay"], c["numd"], c["whichT"],
        c["dosigma"], c["uendtag"], c["sisc"], c["exp"], c["doadam"],
        c["stacked"], c["adaptive"], c["momentum"]])


# --------------------------------------------------------------------------
# The 89 configs, grouped as in list_nets_seeds.txt.  Duplicates across
# figures are removed below by directory name.
# --------------------------------------------------------------------------
configs = []

# Fig 1 (top right) + Fig 2: five trunk bases x N in arange(2,100,8)
for whichT in (-1, 0, 1, 2, 7):
    for llw in range(2, 100, 8):
        configs.append(cfg(5000, 5, 100, llw, KDV, 40, whichT, 0, "First", 0.0, 1))

# Fig 1 (bottom right)
configs.append(cfg(5000, 5, 100, 50, KDV, 40, 0, 1, "1.0", 0.0, 1))

# Fig 3 / Fig 4 e=0 / Fig 9a w=335 / Fig 9b w=335  (one and the same net)
configs.append(cfg(10000, 5, 335, 50, KDV, 32, 0, 1, "1.0", 0.0, 0))

# Fig 4: the four e != 0 nets, GD.  adaptiveLR=1 -> alpha_1 = 1e-4*sigma_1^(-2e);
# they are written as '_expA<e>' and renamed to '_exp<e>' by the runner.
for e in (-1.0, -0.5, 0.5, 1.0):
    configs.append(cfg(4000, 5, 335, 50, KDV, 32, 0, 1, "1.0", e, 0, adaptive=1))

# Fig 5
configs.append(cfg(10000, 5, 335, 50, KDV, 32, 0, 1, "1.0", 0.0, 1))

# Fig 7: Adam only (decided -- the GD counterparts are not part of the sweep)
configs.append(cfg(4000, 5, 332, 20, AD,   32, 0, 1, "1.0", 0.0, 1))
configs.append(cfg(4000, 5, 335, 50, KDV,  32, 0, 1, "1.0", 0.0, 1))
configs.append(cfg(4000, 5, 335, 50, KDV6, 32, 0, 1, "1.0", 0.0, 1))
configs.append(cfg(4000, 5, 337, 50, BUR,  32, 0, 1, "1.0", 0.0, 1))

# Fig 8: stacked (w42) vs unstacked (w495), matched parameter count
configs.append(cfg(10000, 5, 42,  50, KDV, 32, 0, 1, "1.0", 0.0, 1, stacked=1))
configs.append(cfg(10000, 5, 495, 50, KDV, 32, 0, 1, "1.0", 0.0, 1))

# Fig 9a: width sweep, GD
for w in (50, 100, 220, 335, 495):
    configs.append(cfg(10000, 5, w, 50, KDV, 32, 0, 1, "1.0", 0.0, 0))

# Fig 9b: the additional w=50 / Nep4000 net
configs.append(cfg(4000, 5, 50, 50, KDV, 32, 0, 1, "1.0", 0.0, 0))

# Fig 10: exponents with Adam, alpha_1 fixed at 1e-4 (Table 3) -> adaptiveLR=0
for e in (-1.0, -0.5, 0.0, 0.5, 1.0):
    configs.append(cfg(4000, 5, 335, 50, KDV, 32, 0, 1, "1.0", e, 1))

# Fig 11: AdaGrad (momentum=0) and the GD net; the Adam one is Fig 10's e=0.0
configs.append(cfg(4000, 5, 335, 50, KDV, 32, 0, 1, "1.0", 0.0, 1, momentum=0))
configs.append(cfg(5000, 5, 335, 50, KDV, 32, 0, 1, "1.0", 0.0, 0))

# Deduplicate by the directory name at seed 0.
seen, uniq = set(), []
for c in configs:
    k = stem(c, 0, produced=False)
    if k not in seen:
        seen.add(k)
        uniq.append(c)

# Guard: this net must never exist -- with both it and the Nep10000 one present,
# analyze_mode_losses_rotate.py draws a sixth column in fig. 4.
FORBIDDEN = ("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000"
             "_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v")
for c in uniq:
    assert not stem(c, 0, produced=False).startswith(FORBIDDEN), \
        "forbidden fig-4 net in sweep: " + stem(c, 0, produced=False)

assert len(uniq) == 89, "expected 89 configs, got %d" % len(uniq)

# --------------------------------------------------------------------------
# Cost model from the calibration runs (4 CPUs).  See SEED_SWEEP.md.
# --------------------------------------------------------------------------
# Startup is pinned by two w100 runs that happened to land on the same node
# (g103): 52 s at Nep500 and 134 s at Nep2000 give the same 0.0546 s/epoch, so
# STARTUP = 52 - 500*0.0546.  Per-epoch costs use the LONGEST run measured for
# each width, which is least sensitive to startup.
#
# Caveat: nodes differ by ~30%.  The two w495 runs landed on g203 and g166 and
# implied 0.2046 vs 0.2722 s/epoch; the slower figure is used here.  The w335
# and stacked anchors are single runs, so they carry the same uncertainty and
# real job lengths may exceed the estimate by up to a third.
STARTUP = 24.7
ANCH = {100: (134 - STARTUP) / 2000, 335: (107 - STARTUP) / 500,
        495: (569 - STARTUP) / 2000}
STACKED = (454 - STARTUP) / 500


def per_epoch(w, stacked):
    if stacked:
        return STACKED
    ws = sorted(ANCH)
    if w <= ws[0]:
        return ANCH[ws[0]] * (w / ws[0]) ** 0.5
    for a, b in zip(ws, ws[1:]):
        if w <= b:
            return ANCH[a] + (w - a) / (b - a) * (ANCH[b] - ANCH[a])
    return ANCH[ws[-1]]


runs = []
for c in uniq:
    cost = STARTUP + c["nep"] * per_epoch(c["w"], c["stacked"])
    for vtag in range(10):
        runs.append((cost, stem(c, vtag, False), stem(c, vtag, True), args(c, vtag)))

# Longest-processing-time-first: assign each run to the currently emptiest job.
bins = [[0.0, []] for _ in range(N_JOBS)]
for r in sorted(runs, key=lambda x: -x[0]):
    b = min(bins, key=lambda x: x[0])
    b[0] += r[0]
    b[1].append(r)

os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

listing = []
for i, (load, items) in enumerate(bins):
    name = "job_%02d.txt" % i
    with open(os.path.join(OUT, name), "w") as fh:
        for _, expected, produced, a in items:
            fh.write("%s\t%s\t%s\n" % (expected, produced, a))
    listing.append("condor/sweep_manifests/" + name)

with open(os.path.join(REPO, "condor", "sweep_jobs.txt"), "w") as fh:
    fh.write("\n".join(listing) + "\n")

total = sum(b[0] for b in bins)
print("%d configs x 10 seeds = %d runs" % (len(uniq), len(runs)))
print("estimated %.1f CPU-hours total" % (total / 3600))
print("%d jobs: longest %.2f h, shortest %.2f h, %d-%d runs per job" % (
    N_JOBS, max(b[0] for b in bins) / 3600, min(b[0] for b in bins) / 3600,
    min(len(b[1]) for b in bins), max(len(b[1]) for b in bins)))
print("wrote %s/job_*.txt and condor/sweep_jobs.txt" % OUT)
