import os, numpy as np
NETS = "/fast/jtaraz/MISC/ModeDecomposition-DeepONets/data/nets"
BASE = ("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w335"
        "_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v%d")
llw = 50
rows = [1] + list(range(11, 80, 10))          # plot_epoch_ids
tr, te = [], []
for s in range(10):
    a = np.loadtxt(os.path.join(NETS, BASE % s, "log_modes.txt"))
    tr.append(a[:, 1:1+llw]); te.append(a[:, 1+2*llw:1+3*llw])
tr = np.stack(tr); te = np.stack(te)
print("stacked:", tr.shape, "(seeds, epochs, modes)")
for nm, A in (("train (panels A/C)", tr), ("test  (panels B/D)", te)):
    sub = A[:, rows, :]
    m, sd = sub.mean(0), sub.std(0)
    rel = np.where(m > 0, sd / np.abs(m), np.nan)
    print("%s  relative std across seeds:  median %.4f  p90 %.4f  max %.4f"
          % (nm, np.nanmedian(rel), np.nanpercentile(rel, 90), np.nanmax(rel)))
    # how much of a decade does +/-1 std span on a log axis?
    dec = np.log10((m + sd) / np.maximum(m - sd, m * 1e-2))
    print("      band height in decades:      median %.4f  p90 %.4f  max %.4f"
          % (np.nanmedian(dec), np.nanpercentile(dec, 90), np.nanmax(dec)))
