# 10-seed sweep on the MPI-IS cluster

Every DeepONet the figures of arXiv:2602.21910 need — except the four Figure 6
(synthv) nets — trained from ten seeds, `vtag = 0..9`.

**89 configs × 10 seeds = 890 runs**, packed into 30 HTCondor jobs. Which nets,
and where each comes from, is in `list_nets_seeds.txt`.

Estimated at 137 CPU-hours / ~4.6 h per job; it actually took **152 CPU-hours
and 9.55 h wall clock** — see *Measured outcome* below, and read the calibration
section knowing it under-modelled how much nodes differ.

## Running it

```bash
python make_seed_sweep.py 30                 # regenerate manifests (repo root)
# copy condor/ to the cluster, then there:
condor_submit_bid 100 condor/sweep.sub
```

`condor/` is gitignored, so the wrappers live only on the cluster; the
*generator* (`make_seed_sweep.py`) is tracked and reproduces them.

Resubmitting is safe and is the intended recovery path — see the guard below.

## What varies with the seed

`vtag` is `argv[2]` of `execute_don.py` and feeds `jax.random.PRNGKey(vtag)`,
i.e. **only the network initialisation**. `load_dataset` takes a deterministic
90/10 prefix split with no shuffling, so all ten seeds see identical data. That
is intended.

## Calibration

Timed on 4 CPUs with short runs (`condor/calib.sub`, `vtag=99`, since deleted).
Two points at w100 separate the fixed cost from the per-epoch cost:

| config | Nep500 | Nep2000 | node(s) |
|---|---|---|---|
| w100 unstacked | 52 s | 134 s | g103, g103 |
| w335 unstacked | 107 s | | g132 |
| w495 unstacked | 127 s | 569 s | g203, g166 |
| w42 **stacked** | 454 s | | g158 |

→ **startup 24.7 s** (venv + JAX import + JIT + data load + SVD), and per-epoch
0.055 s (w100), 0.165 s (w335), 0.272 s (w495), **0.859 s (stacked w42)**.

**Nodes differ by ~30%, and that dominates the error bar.** The two w100 runs
happened to land on the same node (g103) and imply the same 0.0546 s/epoch to
four decimals — so the linear `t = startup + Nep · c(w)` model itself is sound.
The two w495 runs landed on *different* nodes and implied 0.2046 vs 0.2722
s/epoch. The slower figure is used. The w335 and stacked anchors are single
runs and carry the same ±30%, so real job lengths may run up to a third over.

**This turned out to be a large underestimate: the real spread is up to 4×.**
Worse, the stacked anchor was measured on g158, which the run revealed to be the
slowest node on the cluster, so that config was overestimated by ~1.7×. See
*Measured outcome*.

The stacked net is the outlier: it has essentially the same parameter count as
w495 unstacked (1,205,450 vs 1,205,375 — the parameter-matching Fig. 8 claims)
but runs **3.6× slower**, because 50 separate narrow branch MLPs vectorise far
worse than one wide one. At Nep10000 that is 2.4 h, the longest single run in
the sweep, and it sets the floor on how short any job can be.

Resulting per-run times: 5 min (w100/Nep5000, 660 of the runs), 11 min
(w335/Nep4000), 28 min (w335/Nep10000), 35 min (w495/Nep10000), 2.4 h (stacked).

## Why 30 jobs and not 890

890 separate procs is not a problem for HTCondor — one `queue from file` submit
is a single cluster ID. The real argument is preemption: this is a bidding
cluster, and **`execute_don.py` skips any net whose directory exists without
checking that it finished** (`execute_don.py:118`). A run killed mid-training
leaves a partial directory that every later attempt silently skips, surfacing
much later as a missing curve or a net quietly dropped from the Fig. 9
computation.

Longest-processing-time-first packing is near-perfect here (30 jobs span
4.54–4.56 h), so no straggler decides the wall clock: if all 30 start together
the sweep finishes in ~4.6 h (~6 h if they land on slow nodes). Fewer jobs is worse (20 → 6.8 h each, more exposure
per preemption); many more is pointless (the 2.4 h stacked run means asking for
≤2 h/job degenerates to 401 jobs).

## The completeness guard

`condor/run_manifest.sh` does what `execute_don.py` does not: it requires the
**last** checkpoint, `<Nep-99>cur_chp` (checkpoints are written at `i%100==0` as
`<i+1>`, so 9901 for Nep10000, 4901 for Nep5000, 3901 for Nep4000), and deletes
anything incomplete before retraining.

That makes the sweep idempotent — resubmit the same 30 jobs and only unfinished
work runs again, so a preemption costs at most the single run in flight, never a
silently half-written net.

## Two special cases the manifests encode

- **Figure 4** trains with `adaptiveLR=1`, so `execute_don.py` applies
  α₁ = 1e-4·σ₁^(−2e) itself and names the directory `_expA<e>`. The runner
  renames it to `_exp<e>`, which is what the analysis scripts look for. Each
  manifest line therefore carries both the expected and the produced name.
- **Figure 11's AdaGrad net** is the only one with `momentum=0` (→ `_lrAda32`).

The generator also asserts that the fig. 4 `exp0.0/Nep4000/w335/lrSGD32` net is
never produced: with both it and the Nep10000 one present,
`analyze_mode_losses_rotate.py` draws a sixth column.

## Measured outcome (run of 2026-08-07, cluster 17443885)

Ran 15:15:02 → 00:48:23, i.e. **9.55 h wall clock** against a predicted ~4.6 h.

| | predicted | actual |
|---|---|---|
| total | 137 CPU-h | **152 CPU-h** (+12%) |
| job wall time | 4.54–4.56 h | **3.25 / 5.01 / 9.55 h** (min/median/max) |
| checkpoints | ~420 GB | **+349 GB** (461 → 810 GB) |

Correctness was clean: `TRAIN=889 DONE=889 SKIP=1 FAIL=0 RENAME=40 REPAIR=0`,
30/30 jobs finished, 0 evicted. All 890 directories were verified independently
of the logs to exist and carry their final checkpoint. `RENAME=40` is exactly the
four fig. 4 configs × 10 seeds, with no `_expA` left over and no instance of the
forbidden fig. 4 net. `REPAIR=0` — nothing was preempted, so the guard never had
to fire.

**The one problem was scheduling, and it was the cost model's fault, not the
packer's.** Aggregate cost was fine (+12%); the *distribution* was not. Nodes
differ by up to 4× on identical work:

| config | fastest (g203) | slowest (g158) | ratio |
|---|---|---|---|
| Nep5000 w100 | 240 s | 1007 s | **4.2×** |
| Nep10000 w100 | 503 s | 1642 s | 3.3× |
| Nep4000 w335 | 617 s | 1400 s | 2.3× |
| Nep10000 stacked | 3516 s | 8194 s | 2.3× |

Two compounding errors. The stacked anchor came from g158 — the slowest node —
so that config was predicted at 8600 s against an actual median of 4979 s.
And LPT balanced *predicted* load perfectly, but predicted load is not actual
load when nodes vary 4×. Job 5 drew both a stacked run and g158 and became a
9.55 h straggler that alone set the wall clock, while job 3 finished in 3.25 h
and left its slot idle for six hours.

## What to change next time: pull, don't push

Static bin-packing commits each job to a fixed list up front, so one unlucky
pairing sets the wall clock. Replace it with **one shared queue that workers pull
from**: a fast node then takes more items and a slow node fewer, nobody idles
while work remains, and **no cost model is needed at all**.

Claim items with `mkdir`, which is atomic on Lustre — not a lockfile, because
`/fast` does not support file locking:

```bash
[[ -f "${NETS}/${EXPECTED}/${FINAL_CHP}" ]] && continue    # already done
mkdir "${CLAIMS}/${EXPECTED}" 2>/dev/null || continue      # someone else has it
train...
```

Order the shared list longest-first anyway (dynamic assignment + LPT ordering is
the classic near-optimal pairing) so the 2.3 h stacked runs are not picked up
last. Crash recovery already works: a dead worker leaves a stale claim and a
partial directory, and the completeness guard deletes and retrains anything
without its final checkpoint — so resubmitting still fixes everything, you just
clear the claims directory first. On this sweep it would plausibly have cut
9.55 h to roughly 4–4.5 h.

**Only worth it when runs greatly outnumber slots**, i.e. a re-run of the full
890 or a larger seed set. For the 30 fig. 9 post-processing runs the right layout
is simply one run per job, where balance is a non-issue.

## Not included

- **Figure 9 post-processing** — now specified, see below.
- **Storage**: ~420 GB of checkpoints. `/fast` had 219 TB free.
- **The analysis and plotting scripts still hardcode `_v0`** and must be adapted
  before any seed-averaged figure can be produced.

## Figure 9 post-processing

`log_diagoffdiag*.txt` is not written by training. Exactly **one** script computes
it — `src/analysis/RELEVANT/compute_components_fixedindices.py`, which takes
`bid nepstr exponent w [vtag]` and writes `log_diagoffdiag_new.txt`, 40 × 5111
(`11 + 2·llw + 2·llw²`).

`show_components_2x2.py` (fig. 9b) reads that file directly.
`show_components_mult_multsizes.py` (fig. 9a) looks for
`log_diagoffdiag_big1e-08.txt` instead — and on HuggingFace those two files are
**byte-identical**, i.e. the published `_big1e-08` is a *copy* of `_new`, not a
separate computation. `condor/components_one.sh` therefore computes once and
copies.

**`old-stuff/compute_components.py` is not the fig. 9 path.** It allocates
`11 + 2·llw + 4·llw²` = 10111 columns, matching neither the published files nor
the readers. (One published file, w220's, is a third format again — 10109 =
`9 + 2·llw + 4·llw²` — a leftover of an older code state. Fig. 9a only reads
columns 0–8, which all three formats share, which is why the inconsistency never
showed.) It has been annotated in place rather than used.

**The jobs.** 5 widths × 10 seeds = **50, one net per job**:

```
condor_submit_bid 100 condor/components.sub     # args: 3 10000 0.0 <w> <v>
```

w ∈ {50, 100, 220, 335, 495}, all Nep10000, KdV τ=0.2, `lrSGD32`. Measured at
267 s / net on 4 CPUs with 16 GB (no OOM), so ~3.7 CPU-hours total. Idempotent:
`fixedindices` skips any net that already has `log_diagoffdiag_new.txt`.

**Fig. 9b needs no extra net.** It used to take w50 at Nep4000, the only net not
shared with fig. 9a. `show_components_2x2.py` plots `plot_epoch_ids = [1, 11, …,
71]` and caps `lossdata` at 80 rows, while `bigdata` is always the 40 rows at
epochs 1…3901 — so it never reads past ~epoch 3550. Since the LR schedule is
epoch-indexed, a Nep10000 run's first 4000 epochs are the same trajectory as a
Nep4000 run. So `produce_all_figures.py` now passes `neptag1 = 10000`, reusing
fig. 9a's w50 net: one argument changed, no script edited, figure unchanged.
