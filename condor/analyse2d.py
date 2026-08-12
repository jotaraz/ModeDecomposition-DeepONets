import os, re, itertools, numpy as np
from scipy.stats import spearmanr

NETS = "/fast/jtaraz/MISC/ModeDecomposition-DeepONets/data/nets"
SUB  = "/fast/jtaraz/MISC/ModeDecomposition-DeepONets/condor/run_2d_sweep2.sub"

rows = []
for l in open(SUB):
    if not l.startswith("arguments"): continue
    a = l.split("=",1)[1].split()
    nep,_,d,w,llw,_,batch,lrs,*_ = a
    whichT, dosig, uend, sisc = a[11], a[12], a[13], a[14]
    name = ("whichT%s_doStackedFalse_doSigma%s_sisc%s_aT0.0_aB0.0_exp0.0_Nep%s_d%s_w%s_llw%s"
            "_bat%s_%s_numd1000_lrAdam%s_v0" % (whichT,dosig,sisc,nep,d,w,llw,batch,uend,lrs))
    p = os.path.join(NETS, name, "log.txt")
    if not os.path.isfile(p): rows.append((batch,int(llw),int(d),whichT,None,None,0)); continue
    L = np.loadtxt(p)
    tr = 10**(L[:,1]/2); te = 10**(L[:,4]/2)
    rows.append((batch,int(llw),int(d),whichT, tr[-1], te[-1], len(L)))

print("nets found: %d / %d" % (sum(1 for r in rows if r[4] is not None), len(rows)))
bad = [r for r in rows if r[4] is None]
if bad: print("MISSING:", bad[:5])
lens = {r[6] for r in rows if r[4] is not None}
print("log lengths:", sorted(lens))

DS = ["heat2d_sine_nx64_K8_D1","heat2d_grf_nx64_l0.1-0.2_D1","heat2d_grf_nx64_l0.05_D1"]
SHORT = {DS[0]:"sine K8", DS[1]:"grf l0.1-0.2", DS[2]:"grf l0.05"}

print("\n=== final relative TEST error, per dataset ===")
for ds in DS:
    v = np.array([r[5] for r in rows if r[0]==ds and r[5] is not None])
    print("  %-13s n=%2d  min %.4f  median %.4f  max %.4f  (max/min = %.1fx)"
          % (SHORT[ds], len(v), v.min(), np.median(v), v.max(), v.max()/v.min()))

print("\n=== best 3 configs per dataset ===")
for ds in DS:
    sub = sorted([r for r in rows if r[0]==ds and r[5] is not None], key=lambda r:r[5])
    print("  %s:" % SHORT[ds])
    for r in sub[:3]:
        print("      N=%-4d d=%-3d %-8s test %.4f" % (r[1],r[2],"learned" if r[3]=="-1" else "SVD", r[5]))

print("\n=== does the ranking transfer? Spearman rho over the 24 shared configs ===")
key = lambda r:(r[1],r[2],r[3])
tab = {ds:{key(r):r[5] for r in rows if r[0]==ds and r[5] is not None} for ds in DS}
common = sorted(set.intersection(*[set(t) for t in tab.values()]))
print("  shared configs:", len(common))
for a,b in itertools.combinations(DS,2):
    x=[tab[a][k] for k in common]; y=[tab[b][k] for k in common]
    print("  %-13s vs %-13s  rho = %+.3f" % (SHORT[a], SHORT[b], spearmanr(x,y).correlation))

print("\n=== marginal effect of each hyperparameter (median test error) ===")
for ds in DS:
    print("  %s:" % SHORT[ds])
    for lbl, idx, vals in (("trunk",3,["-1","0"]), ("N",1,[50,100,150,200]), ("depth",2,[5,10,15,20])):
        parts=[]
        for v in vals:
            vv=[r[5] for r in rows if r[0]==ds and r[idx]==v and r[5] is not None]
            if vv: parts.append("%s=%s:%.3f" % (lbl, "learned" if v=="-1" else ("SVD" if v=="0" and lbl=="trunk" else v), np.median(vv)))
        print("     " + "  ".join(parts))
