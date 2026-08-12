"""Left panel of fig. 2 (analyze_whichTs_newlayout2), for the 2D heat problems.

Per dataset: total relative error delta (circles), trunk error delta_T (dashed)
and branch error delta_B = delta - delta_T (solid), over the inner dimension N,
for the learned (whichT=-1) and SVD (whichT=0) trunks.

delta   : 10**(log.txt col4 / 2), i.e. ||A-Atilde||_F/||A||_F on test data
delta_T : ||(I - T T^+) U||_F / ||U||_F, from the trunk matrix T
delta_B : delta - delta_T, as in the original script
"""
import os, re, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
from matplotlib.lines import Line2D
sys.path.insert(0, "/fast/jtaraz/MISC/ModeDecomposition-DeepONets/src")
import don_code
sys.path.insert(0, "/fast/jtaraz/MISC/ModeDecomposition-DeepONets/src/analysis")
import plot_dataset_singvals as sv          # local version, rsynced
import optax

NETS = don_code.nets_dir
CACHE = {}

EXCLUDE = set()   # nothing excluded: the retrained net (17449054) is complete
FIGS = "/fast/jtaraz/MISC/ModeDecomposition-DeepONets/figures"
DS   = ["heat2d_sine_nx64_K8_D1", "heat2d_grf_nx64_l0.05_D1"]   # GRF l=0.1-0.2 excluded
SHORT= {DS[0]: r"sine, $K=8$", DS[1]: r"GRF, $\ell = 0.05$"}
lr_schedule = optax.exponential_decay(init_value=1e-4, transition_steps=500,
                                      decay_rate=0.95, staircase=True)
optimizer = optax.sgd(learning_rate=lr_schedule)

def latest_chp(direc):
    """highest <n>cur_chp present"""
    ns = [int(f[:-len("cur_chp")]) for f in os.listdir(os.path.join(NETS, direc))
          if f.endswith("cur_chp") and f[:-len("cur_chp")].isdigit()]
    return str(max(ns)) if ns else None

def trunk_of(direc, epoch, rtrain, ptrain, nb, nt, llw, uu_train, batch_name, ScaledSigma):
    depth, width, llw_, whichT, _, _, _ = don_code.get_dwllw(direc)
    if whichT >= 0:
        return don_code.get_fixed_trunk(whichT, llw_, rtrain[:, 0], batch_name, uu_train)
    model = don_code.DeepONet(nb, nt, depth, width, llw_)
    init_params = model.init(don_code.jax.random.PRNGKey(0), ptrain, rtrain, ScaledSigma)
    params, _ = don_code.load_checkpoint(init_params, init_params,
                                         path=os.path.join(NETS, direc, epoch + "cur_chp"))
    state = don_code.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    _, Ttr, _ = state.apply_fn(params, ptrain, rtrain, ScaledSigma)
    return np.asarray(Ttr)

def run(tag, title, want, ns_expected):
    RX = re.compile(r"^whichT(-?\d+)_doStackedFalse_doSigma\d_sisc[^_]+_aT0\.0_aB0\.0_exp0\.0_"
                    r"Nep(\d+)_d(\d+)_w(\d+)_llw(\d+)_bat(.+)_t0\.004_numd(\d+)_lrAdam32_v0$")
    found = {}
    for name in os.listdir(NETS):
        if name in EXCLUDE: continue
        m = RX.match(name)
        if not m: continue
        T, nep, d, w, N, bat, numd = m.groups()
        key = want(int(nep), int(d), int(w), bat, int(numd))
        if key is None: continue
        found.setdefault((key, T), {})[int(N)] = name
    print("%s: matched %d (dataset,trunk) groups" % (tag, len(found)))

    # leftmost panel: normalized singular-value spectra (its own y-axis); the two
    # error panels to its right share theirs, as before.
    # nested: the singular-value panel needs its own y-axis and therefore a gap;
    # only the two error panels abut (wspace=0), as in analyze_whichTs_newlayout2.
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(11.6, 3.3))
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.15, 2.0], wspace=0.28)
    ax_sv = fig.add_subplot(outer[0, 0])
    gs_err = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], wspace=0.0)
    axs = [fig.add_subplot(gs_err[0, 0])]
    axs.append(fig.add_subplot(gs_err[0, 1], sharey=axs[0]))

    entries = [dict(d, path=os.path.join(sv.DATADIR, d["file"])) for d in sv.DATASETS]
    entries = [e for e in entries if os.path.isfile(e["path"])]
    sv.assign_colors(entries)
    # Only one 2D GRF and one 2D sine set are left in the figure, so the
    # (ell=0.05) / (K=8) qualifiers no longer distinguish anything. Overridden
    # here rather than in plot_dataset_singvals.py, whose standalone plot may
    # still want them.
    SV_LABEL = {"heat2d_grf": "2D heat, GRF", "heat2d_sine": "2D heat, sine"}
    for e in entries:
        for pre, lab in SV_LABEL.items():
            if e["file"].startswith(pre):
                e["label"] = lab
    for e in entries:
        y = np.atleast_1d(np.loadtxt(e["path"]))
        y = y / y[0]                                  # --normalize
        ax_sv.semilogy(np.arange(1, len(y)+1), y, label=e["label"],
                       color=e["color"], lw=1.8, **sv.DIM_STYLE[e["dim"]])
    ax_sv.set_xlim((-4, 150)); ax_sv.set_ylim((6e-5, 1.5))
    ax_sv.set_xlabel(r"index $i$", fontsize=13)
    ax_sv.set_ylabel(r"Normalized singular values $\sigma_i/\sigma_1$", fontsize=13)
    ax_sv.tick_params(labelsize=11)
    ax_sv.legend(fontsize=10, loc="upper right", framealpha=0.92)
    for ax, ds in zip(axs, DS):
        finite = []
        # colours as in analyze_whichTs_newlayout2: colors_cb.txt[0]=#984ea3 purple
        # goes to whichT=-1 (learned), [1]=#ff7f00 orange to whichT=0 (SVD)
        for T, col, lbl in (("0", "#ff7f00", "SVD"), ("-1", "#984ea3", "learned")):
            grp = found.get((ds, T), {})
            if not grp: continue
            batch_name, uend = None, "t0.004"
            Ns, dl, dT, dB = [], [], [], []
            rank = None
            for N in sorted(grp):
                direc = grp[N]
                dd, ww, llw, whichT, batch_name, num_data, endtag = don_code.get_dwllw(direc)
                ck = (batch_name, endtag, num_data)
                if ck not in CACHE:                      # loading is 100-500 MB of text + a
                    print("   loading %s ..." % str(ck), flush=True)   # 4096 x N_train SVD;
                    dat = don_code.load_dataset(*ck)     # do it once per dataset, not per net
                    uu, ss, _ = np.linalg.svd(np.asarray(dat[6]), full_matrices=False)
                    CACHE[ck] = (dat, uu, ss)
                (nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest), uu_train, ss_train = CACHE[ck]
                rank = int(np.sum(ss_train > ss_train[0] * 1e-12))
                ep = latest_chp(direc)
                if ep is None: continue
                Tm = trunk_of(direc, ep, rtrain, ptrain, nb, nt, llw, uu_train,
                              batch_name, np.eye(llw))
                if Tm is None: continue
                proj = np.eye(np.shape(Tm)[0]) - Tm @ np.linalg.pinv(Tm)
                ute = np.asarray(utest)
                dT_ = np.sqrt(np.sum((proj @ ute)**2) / np.sum(ute**2))
                L = np.loadtxt(os.path.join(NETS, direc, "log.txt"))
                d_ = 10**(L[:, 4].min()/2)          # best test delta over training
                Ns.append(N); dl.append(d_); dT.append(dT_); dB.append(max(d_-dT_, 1e-12))
                print("   %-14s %-8s N=%-4d  delta %.4f  delta_T %.4f  delta_B %.4f  (chp %s)"
                      % (SHORT[ds], lbl, N, d_, dT_, d_-dT_, ep))
            if not Ns: continue
            # delta_T can be EXACTLY zero once N exceeds the rank of the data (the
            # K=8 sine set is rank-deficient), which a log axis cannot show. Mask
            # those points and annotate instead of letting them crush the limits.
            dTm = [v if v > 1e-12 else np.nan for v in dT]
            zeroN = [n for n, v in zip(Ns, dT) if v <= 1e-12]
            # delta_T hits exactly 0 once N reaches the rank of the data (the K=8
            # sine set is rank-deficient). Rather than let the dashed line stop
            # dead, continue it to 1e-16 at N = rank, so it leaves the frame with
            # the slope implied by the collapse. y-limits are unchanged, so this
            # is only visible as the line running off the bottom.
            Nt, dTt = list(Ns), list(dTm)
            if zeroN and rank:
                keep = [(n, v) for n, v in zip(Ns, dT) if v > 1e-12]
                if keep:
                    Nt  = [n for n, _ in keep] + [rank]
                    dTt = [v for _, v in keep] + [1e-16]
            ax.plot(Ns, dl, 'o', linestyle='none', fillstyle='none', markeredgewidth=2,
                    color=col, label=r"%s: $\delta$" % lbl)
            ax.plot(Nt, dTt, '--', color=col, lw=2, label=r"%s: $\delta_T$" % lbl)
            ax.plot(Ns, dB, '-',  color=col, lw=2, label=r"%s: $\delta_B$" % lbl)
            finite += [v for v in dl + dTm + dB if v == v and v > 0]
        ax.set_yscale("log")          # x stays linear
        ax.set_ylim(1e-3, 1.0)
        ax.set_title(SHORT[ds], fontsize=13)
        ax.tick_params(labelsize=11)
    # --- styling after analyze_whichTs_newlayout2 -----------------------------
    for k, ax in enumerate(axs):
        if k > 0:
            ax.tick_params(labelleft=False)          # shared y across the row
    axs[0].set_ylabel(r"relative error $\delta$", fontsize=13)
    GREY = "0.45"
    handles = [Line2D([], [], marker="o", ls="none", mfc="none", mec=GREY, mew=2, label=r"$\delta$"),
               Line2D([], [], ls=(0, (3.5, 2.2)), color=GREY, lw=2, label=r"$\delta_T$"),
               Line2D([], [], ls="-",  color=GREY, lw=2, label=r"$\delta_B$"),
               Line2D([], [], marker="s", ls="none", color="#984ea3", label="Learned"),
               Line2D([], [], marker="s", ls="none", color="#ff7f00", label="SVD")]
    axs[-1].legend(handles=handles, fontsize=11, ncol=2, loc="lower left",
                   labelspacing=0.25, handlelength=2.4, handletextpad=0.5,
                   borderpad=0.2, borderaxespad=0.5, columnspacing=1.0,
                   framealpha=1.0).set_zorder(10)
    fig.text(0.71, 0.02, r"inner dimension $N$", fontsize=13)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.17)
    for ext in ("pdf", "png"):
        fig.savefig("%s/%ss/2d_errdecomp_%s.%s" % (FIGS, ext, tag, ext), dpi=150)
    print("wrote 2d_errdecomp_%s" % tag)

run("m1000", "2D heat: error decomposition over N  (m=1000, w=335, d=10, Nep4000)",
    lambda nep,d,w,bat,numd: bat if (nep==4000 and d==10 and w==335 and numd==1000 and bat in DS) else None,
    [50,100,150,200])
run("m5000", "2D heat: error decomposition over N  (m=5000, w=500, d=10, Nep10000)",
    lambda nep,d,w,bat,numd: (bat[:-len("_m5000")] if (nep==10000 and d==10 and w==500 and numd==5000
        and bat.endswith("_m5000") and bat[:-len("_m5000")] in DS) else None),
    [10,20,40,60,80,100,120])
