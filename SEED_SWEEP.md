# 10-seed sweep on the MPI-IS cluster

Every DeepONet the figures of arXiv:2602.21910 need — except the four Figure 6
(synthv) nets — trained from ten seeds, `vtag = 0..9`.

**89 configs × 10 seeds = 890 runs ≈ 137 CPU-hours**, packed into 30 HTCondor
jobs of ~4.6 h each. Which nets, and where each comes from, is in
`list_nets_seeds.txt`.

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

## Not included

- **Figure 9 post-processing.** `log_diagoffdiag*.txt` is not written by
  training; it comes from `compute_components_fixedindices.py` (fig. 9b) and
  `old-stuff/compute_components.py` (fig. 9a), both of which now take the seed
  as a trailing argument. 10 + 20 runs, each ~4000 jitted gradient evaluations,
  not yet calibrated — give them 16 GB (the `(num_pars × llw)` gradient matrix
  alone is 482 MB at w495, held twice).
- **Storage**: ~420 GB of checkpoints. `/fast` had 219 TB free.
- **The analysis and plotting scripts still hardcode `_v0`** and must be adapted
  before any seed-averaged figure can be produced.
