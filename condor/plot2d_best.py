"""Per dataset: for each trunk type, the net with (a) the lowest minimum test
error over training and (b) the lowest final train error. Train+test curves."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/fast/jtaraz/MISC/ModeDecomposition-DeepONets"
NETS, SUB = REPO + "/data/nets", REPO + "/condor/run_2d_sweep2.sub"

DS = ["heat2d_sine_nx64_K8_D1", "heat2d_grf_nx64_l0.1-0.2_D1", "heat2d_grf_nx64_l0.05_D1"]
SHORT = {DS[0]: "sine, K=8", DS[1]: "GRF, l=0.1-0.2", DS[2]: "GRF, l=0.05"}
TRUNK = {"-1": "learned", "0": "SVD"}

nets = []
for l in open(SUB):
    if not l.startswith("arguments"): continue
    a = l.split("=", 1)[1].split()
    nep, d, w, llw, batch = a[0], a[2], a[3], a[4], a[6]
    whichT, dosig, uend, sisc, lrs = a[11], a[12], a[13], a[14], a[7]
    name = ("whichT%s_doStackedFalse_doSigma%s_sisc%s_aT0.0_aB0.0_exp0.0_Nep%s_d%s_w%s_llw%s"
            "_bat%s_%s_numd1000_lrAdam%s_v0" % (whichT, dosig, sisc, nep, d, w, llw, batch, uend, lrs))
    p = os.path.join(NETS, name, "log.txt")
    if not os.path.isfile(p):
        print("MISSING", name); continue
    L = np.loadtxt(p)
    nets.append(dict(name=name, ds=batch, N=int(llw), d=int(d), w=int(w), nep=int(nep),
                     trunk=whichT, ep=L[:, 0],
                     tr=10**(L[:, 1] / 2), te=10**(L[:, 4] / 2)))
print("loaded %d nets" % len(nets))

COL = {("0", "a"): "#1b6ca8", ("0", "b"): "#63b7e6",
       ("-1", "a"): "#b3202c", ("-1", "b"): "#e8836b"}
CRIT = {"a": "lowest min test error", "b": "lowest final train error"}

fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
chosen = []
for ax, ds in zip(axs, DS):
    pool = [n for n in nets if n["ds"] == ds]
    for t in ("0", "-1"):
        sub = [n for n in pool if n["trunk"] == t]
        picks = {"a": min(sub, key=lambda n: n["te"].min()),
                 "b": min(sub, key=lambda n: n["tr"][-1])}
        same = picks["a"]["name"] == picks["b"]["name"]
        for crit in ("a", "b"):
            if same and crit == "b":
                continue
            n = picks[crit]
            lab_crit = ("(a)+(b) both criteria" if same else "(%s) %s" % (crit, CRIT[crit]))
            lab = ("%s trunk | %s\nN=%d, d=%d, w=%d | min test %.3f, final train %.3f"
                   % (TRUNK[t], lab_crit, n["N"], n["d"], n["w"], n["te"].min(), n["tr"][-1]))
            c = COL[(t, crit)]
            ax.plot(n["ep"], n["te"], "-", color=c, lw=1.9, label=lab)
            ax.plot(n["ep"], n["tr"], "--", color=c, lw=1.4, alpha=0.85)
            chosen.append((ds, TRUNK[t], lab_crit, n))
    ax.set_yscale("log"); ax.set_xlabel("epoch")
    ax.set_title("%s\n(solid = test, dashed = train)" % SHORT[ds], fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.2, loc="upper right", framealpha=0.95)
axs[0].set_ylabel(r"relative error  $\delta = \|A-\tilde{A}\|_F / \|A\|_F$")
fig.suptitle("2D heat: best nets per dataset and trunk type "
             "(run_2d_sweep2.sub, 72 nets, Nep4000, w=335, Adam lr 1e-4, vtag=0)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("pdf", "png"):
    fig.savefig("%s/figures/%ss/2d_bestnets.%s" % (REPO, ext, ext), dpi=150)

out = REPO + "/figures/2d_bestnets_netdirs.txt"
with open(out, "w") as f:
    f.write("Nets shown in figures/pdfs/2d_bestnets.pdf\n")
    f.write("Pool: the 72 nets of condor/run_2d_sweep2.sub (Nep4000, w=335, vtag=0,\n")
    f.write("lrAdam32 = 1e-4/0.95, numd1000, t0.004, exp0.0, unstacked).\n")
    f.write("(a) = lowest minimum test error over training; (b) = lowest train error at epoch 4000.\n")
    f.write("delta = ||A-Atilde||_F/||A||_F, i.e. 10**(log.txt col/2); col1 train, col4 test.\n\n")
    for ds, tr, crit, n in chosen:
        f.write("dataset : %s\n" % ds)
        f.write("trunk   : %s   %s\n" % (tr, crit))
        f.write("N=%d d=%d w=%d Nep=%d\n" % (n["N"], n["d"], n["w"], n["nep"]))
        f.write("min test delta = %.6f   final train delta = %.6f   final test delta = %.6f\n"
                % (n["te"].min(), n["tr"][-1], n["te"][-1]))
        f.write("%s\n\n" % n["name"])
print("wrote", out)
for ds, tr, crit, n in chosen:
    print("  %-28s %-8s %-24s N=%-4d d=%-3d minte=%.3f fintr=%.3f"
          % (SHORT[n["ds"]], tr, crit[:22], n["N"], n["d"], n["te"].min(), n["tr"][-1]))
