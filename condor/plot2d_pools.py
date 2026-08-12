"""Two figures in the form of 2d_bestnets: one over ALL m=1000 nets (grid +
width sweeps + older runs), one over the m=5000 nets. Per dataset and trunk
type: (a) lowest minimum test error, (b) lowest final train error."""
import os, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/fast/jtaraz/MISC/ModeDecomposition-DeepONets"
NETS = REPO + "/data/nets"
DS = ["heat2d_sine_nx64_K8_D1", "heat2d_grf_nx64_l0.1-0.2_D1", "heat2d_grf_nx64_l0.05_D1"]
SHORT = {DS[0]: "sine, K=8", DS[1]: "GRF, l=0.1-0.2", DS[2]: "GRF, l=0.05"}
TRUNK = {"-1": "learned", "0": "SVD"}

RX = re.compile(r"^whichT(-?\d+)_doStacked(\w+)_doSigma(\d)_sisc([^_]+)_aT0\.0_aB0\.0_"
                r"exp(-?[\d.]+)_Nep(\d+)_d(\d+)_w(\d+)_llw(\d+)_bat(.+)_numd(\d+)_lr(\D+)(\d+)_v(\d+)$")

def scan(want_numd, ds_suffix):
    out = []
    for name in os.listdir(NETS):
        m = RX.match(name)
        if not m: continue
        whichT, stk, dosig, sisc, exp, nep, d, w, llw, bat, numd, opt, lrs, v = m.groups()
        if int(numd) != want_numd or int(nep) != 4000 or stk != "False": continue
        if not bat.endswith("_t0.004"): continue
        dsname = bat[:-len("_t0.004")]
        base = dsname[:-len("_m5000")] if dsname.endswith("_m5000") else dsname
        if base not in DS: continue
        if (dsname.endswith("_m5000")) != ds_suffix: continue
        p = os.path.join(NETS, name, "log.txt")
        if not os.path.isfile(p): continue
        L = np.loadtxt(p)
        if L.ndim != 2 or len(L) < 10: continue
        out.append(dict(name=name, ds=base, N=int(llw), d=int(d), w=int(w), exp=float(exp),
                        trunk=whichT, ep=L[:, 0], tr=10**(L[:, 1]/2), te=10**(L[:, 4]/2)))
    return out

COL = {("0","a"):"#1b6ca8", ("0","b"):"#63b7e6", ("-1","a"):"#b3202c", ("-1","b"):"#e8836b"}
CRIT = {"a": "lowest min test error", "b": "lowest final train error"}

def make(pool, tag, title):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    chosen = []
    for ax, ds in zip(axs, DS):
        sub_ds = [n for n in pool if n["ds"] == ds]
        for t in ("0", "-1"):
            sub = [n for n in sub_ds if n["trunk"] == t]
            if not sub:
                continue
            picks = {"a": min(sub, key=lambda n: n["te"].min()),
                     "b": min(sub, key=lambda n: n["tr"][-1])}
            same = picks["a"]["name"] == picks["b"]["name"]
            for crit in ("a", "b"):
                if same and crit == "b": continue
                n = picks[crit]
                lc = "(a)+(b) both criteria" if same else "(%s) %s" % (crit, CRIT[crit])
                extra = "" if abs(n["exp"]) < 1e-9 else ", exp=%g" % n["exp"]
                lab = ("%s trunk | %s\nN=%d, d=%d, w=%d%s | min test %.3f, final train %.3f"
                       % (TRUNK[t], lc, n["N"], n["d"], n["w"], extra, n["te"].min(), n["tr"][-1]))
                c = COL[(t, crit)]
                ax.plot(n["ep"], n["te"], "-", color=c, lw=1.9, label=lab)
                ax.plot(n["ep"], n["tr"], "--", color=c, lw=1.4, alpha=0.85)
                chosen.append((ds, TRUNK[t], lc, n))
        ax.set_yscale("log"); ax.set_xlabel("epoch")
        ax.set_title("%s   (%d nets)\n(solid = test, dashed = train)"
                     % (SHORT[ds], len(sub_ds)), fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.2, loc="upper right", framealpha=0.95)
    axs[0].set_ylabel(r"relative error  $\delta = \|A-\tilde{A}\|_F / \|A\|_F$")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        fig.savefig("%s/figures/%ss/2d_bestnets_%s.%s" % (REPO, ext, tag, ext), dpi=150)
    out = "%s/figures/2d_bestnets_%s_netdirs.txt" % (REPO, tag)
    with open(out, "w") as f:
        f.write(title + "\n\n")
        f.write("(a) lowest minimum test error over training; (b) lowest train error at epoch 4000.\n")
        f.write("delta = ||A-Atilde||_F/||A||_F = 10**(log.txt col/2); col1 train, col4 test.\n\n")
        for ds, tr, crit, n in chosen:
            f.write("dataset : %s\ntrunk   : %s   %s\n" % (ds, tr, crit))
            f.write("N=%d d=%d w=%d exp=%g\n" % (n["N"], n["d"], n["w"], n["exp"]))
            f.write("min test %.6f   final train %.6f   final test %.6f\n"
                    % (n["te"].min(), n["tr"][-1], n["te"][-1]))
            f.write("%s\n\n" % n["name"])
    print("wrote", out)
    for ds, tr, crit, n in chosen:
        print("   %-15s %-8s %-22s N=%-4d d=%-3d w=%-4d minte=%.3f fintr=%.3f"
              % (SHORT[ds], tr, crit[:20], n["N"], n["d"], n["w"], n["te"].min(), n["tr"][-1]))

p1 = scan(1000, False); print("m=1000 pool: %d nets" % len(p1))
make(p1, "m1000", "2D heat, ALL m=1000 nets (4x4 grid + width sweeps + earlier runs, "
                  "Nep4000, vtag=0): best per dataset and trunk")
p5 = scan(5000, True); print("m=5000 pool: %d nets" % len(p5))
make(p5, "m5000", "2D heat, m=5000 nets (w=335, Nep4000, vtag=0): best per dataset and trunk")
