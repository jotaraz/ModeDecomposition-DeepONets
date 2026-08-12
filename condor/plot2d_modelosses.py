"""Mode losses for the 2D SVD-DeepONets at N=60, m=5000, one figure per width.

Layout follows plot_modelosses_with_out_sigma.py: the train/test members of a
pair abut (wspace=0), rows abut (hspace=0), the row is named by the y-label of
the leftmost panel, and only the bottom row carries the x-label.

log_modes.txt columns: 0 = epoch, then llw each of
   mode_losses_train | min_mode_losses_train | mode_losses_test | min_mode_losses_test
unweighted: L_j * sf     weighted: sigma_j^2 * L_j * sf
sf = 1 (train), mtrain/mtest (test)
base: sigma_j^2 (train), ||T^T u_test||^2 / mtest * mtrain (test)
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
sys.path.insert(0, "/fast/jtaraz/MISC/ModeDecomposition-DeepONets/src")
import don_code

NETS = don_code.nets_dir
FIGS = "/fast/jtaraz/MISC/ModeDecomposition-DeepONets/figures"
LLW  = 60
WIDTHS = [100, 200, 300, 400, 500]
STAGE_ROWS = [1, 2, 8, 16, 32, 64, 128]
CASES = [("heat2d_sine_nx64_K8_D1",   r"sine, $K=8$"),
         ("heat2d_grf_nx64_l0.05_D1", r"GRF, $\ell = 0.05$")]   # GRF l=0.1-0.2 excluded
NAME = ("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d10_w%d"
        "_llw%d_bat%s_m5000_t0.004_numd5000_lrAdam32_v0")
FS = 13                                     # base font size
CACHE = {}

def ramp(k, n, end):
    f = k/(n-1) if n > 1 else 1.0
    return (0.6*(1-f) + (1.0 if end == "red"  else 0.0)*f,
            0.6*(1-f),
            0.6*(1-f) + (1.0 if end == "blue" else 0.0)*f)

def make(W):
    nrow = len(CASES)
    fig = plt.figure(figsize=(12.5, 1.85*nrow))
    outer = gridspec.GridSpec(nrow, 2, figure=fig, wspace=0.16, hspace=0.0)
    axs = [[None]*4 for _ in range(nrow)]
    for r in range(nrow):
        for p in range(2):                                   # the two pairs
            gp = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[r, p], wspace=0.0)
            for c in range(2):
                axs[r][2*p+c] = fig.add_subplot(gp[0, c])

    for row, (ds, short) in enumerate(CASES):
        direc = NAME % (W, LLW, ds)
        _, _, llw, _, batch_name, num_data, endtag = don_code.get_dwllw(direc)
        ck = (batch_name, endtag, num_data)
        if ck not in CACHE:
            dat = don_code.load_dataset(*ck)
            uu, ss, _ = np.linalg.svd(np.asarray(dat[6]), full_matrices=False)
            CACHE[ck] = (dat, uu, ss)
        (nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest), uu, ss = CACHE[ck]
        utrain = np.asarray(utrain); utest = np.asarray(utest)
        mtrain, mtest = utrain.shape[1], utest.shape[1]
        VT_test = uu[:, :llw].T @ utest
        base_te = np.array([np.sum(VT_test[i, :]**2) for i in range(llw)]) / mtest * mtrain
        sig2 = ss[:llw]**2
        base_tr = sig2

        M = np.loadtxt(os.path.join(NETS, direc, "log_modes.txt"))
        epochs = M[:, 0]
        stages = [r_ for r_ in STAGE_ROWS if r_ < len(M)]
        tr_modes = M[:, 1:1+llw]; te_modes = M[:, 1+2*llw:1+3*llw]
        print("w=%d %s: rows %s -> epochs %s" % (W, ds, stages, epochs[stages].astype(int)))

        x = np.arange(1, llw+1)
        panels = [(r"unweighted train $L_j$",           tr_modes, 1.0,          base_tr/sig2, "red",  1.0),
                  (r"unweighted test $L_j$",            te_modes, mtrain/mtest, base_te/sig2, "blue", 1.0),
                  (r"weighted train $\sigma_j^2 L_j$",  tr_modes, 1.0,          base_tr,      "red",  sig2),
                  (r"weighted test $\sigma_j^2 L_j$",   te_modes, mtrain/mtest, base_te,      "blue", sig2)]
        lims = {}
        for col, (title, modes, sf, base, end, w) in enumerate(panels):
            ax = axs[row][col]
            vals = []
            for k, s_ in enumerate(stages):
                y = w*modes[s_, :]*sf
                ax.plot(x, y, "--", color=ramp(k, len(stages), end), lw=1.3, alpha=0.9)
                vals.append(y)
            ax.plot(x, base, ".-", color="k", lw=1.3, ms=3)
            ax.set_yscale("log"); ax.set_xticks([1, 20, 40, 60])
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
            ax.tick_params(labelsize=FS-2)
            if row == 0:
                ax.set_title(title, fontsize=FS)
            if row == nrow-1:
                ax.set_xlabel(r"mode index $j$", fontsize=FS)
            else:
                ax.tick_params(labelbottom=False)
            v = np.concatenate(vals + [base]); v = v[v > 0]
            lims[col] = (v.min(), v.max())
        for a, b in ((0, 1), (2, 3)):                        # train/test share y
            lo = min(lims[a][0], lims[b][0]); hi = max(lims[a][1], lims[b][1])
            for c in (a, b):
                axs[row][c].set_ylim(lo/1.6, hi*1.6)
            axs[row][b].tick_params(labelleft=False)
        # the row is named by the y-label of its leftmost panel
        axs[row][0].set_ylabel(short + "\n" + r"mode loss", fontsize=FS)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.925, bottom=0.135)
    for ext in ("pdf", "png"):
        fig.savefig("%s/%ss/2d_modelosses_N60_width%d.%s" % (FIGS, ext, W, ext), dpi=150)
    plt.close(fig)
    print("wrote 2d_modelosses_N60_width%d" % W)

for W in WIDTHS:
    make(W)
