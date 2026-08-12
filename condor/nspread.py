import os, re, numpy as np
NETS="/fast/jtaraz/MISC/ModeDecomposition-DeepONets/data/nets"
RX=re.compile(r"^whichT(-?\d+)_doStackedFalse_doSigma\d_sisc[^_]+_aT0\.0_aB0\.0_exp0\.0_"
              r"Nep(\d+)_d(\d+)_w(\d+)_llw(\d+)_bat(.+)_t0\.004_numd(\d+)_lrAdam32_v0$")
rows=[]
for name in os.listdir(NETS):
    m=RX.match(name)
    if not m: continue
    T,nep,d,w,N,bat,numd=m.groups()
    if "heat2d" not in bat or int(numd)!=5000: continue
    p=os.path.join(NETS,name,"log.txt")
    if not os.path.isfile(p): continue
    L=np.loadtxt(p)
    if L.ndim!=2 or len(L)<5: continue
    rows.append(dict(T=T,nep=int(nep),N=int(N),bat=bat,te=10**(L[:,4]/2)))
new=[r for r in rows if r["nep"]==10000]; old=[r for r in rows if r["nep"]==4000]
cut=min(len(r["te"]) for r in new)
print("common epoch window: rows 0..%d (epoch %d)\n" % (cut-1,(cut-1)*50))
NS=[10,20,40,60,80,100,120]
print("min test error by N (m=5000, w=500, d=10), and the single old w=335 net:")
print("  %-22s %-8s %s  | old   | best-of-7 | median-of-7" % ("dataset","trunk"," ".join("N=%-5d"%n for n in NS)))
for bat in sorted({r["bat"] for r in old}):
    for T in ("0","-1"):
        o=[r for r in old if r["bat"]==bat and r["T"]==T]
        n={r["N"]:r["te"][:cut].min() for r in new if r["bat"]==bat and r["T"]==T}
        if not o or not n: continue
        vals=[n.get(x) for x in NS]
        med=float(np.median([v for v in vals if v is not None]))
        ov=o[0]["te"][:cut].min()
        print("  %-22s %-8s %s  | %.3f | %.3f     | %.3f  (best %+.0f%%, median %+.0f%%)"
              % (bat.replace("_m5000","").replace("heat2d_",""),"SVD" if T=="0" else "learned",
                 " ".join(("%-7.3f"%v if v is not None else "  --   ") for v in vals),
                 ov,min(v for v in vals if v),med,
                 100*(min(v for v in vals if v)-ov)/ov,100*(med-ov)/ov))
