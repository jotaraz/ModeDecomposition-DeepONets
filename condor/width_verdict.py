import os, re, numpy as np
NETS = "/fast/jtaraz/MISC/ModeDecomposition-DeepONets/data/nets"
RX = re.compile(r"^whichT(-?\d+)_doStackedFalse_doSigma\d_sisc[^_]+_aT0\.0_aB0\.0_exp0\.0_"
                r"Nep(\d+)_d(\d+)_w(\d+)_llw(\d+)_bat(.+)_t0\.004_numd(\d+)_lrAdam32_v0$")
def load(name):
    p = os.path.join(NETS, name, "log.txt")
    if not os.path.isfile(p): return None
    L = np.loadtxt(p)
    if L.ndim != 2 or len(L) < 5: return None
    return L
rows=[]
for name in os.listdir(NETS):
    m = RX.match(name)
    if not m: continue
    T,nep,d,w,N,bat,numd = m.groups()
    if "heat2d" not in bat: continue
    L = load(name)
    if L is None: continue
    rows.append(dict(T=T, nep=int(nep), d=int(d), w=int(w), N=int(N), bat=bat,
                     numd=int(numd), ep=L[:,0], tr=10**(L[:,1]/2), te=10**(L[:,4]/2)))
print("parsed %d heat2d nets\n" % len(rows))

# ---------- (A) the CLEAN width evidence: m=1000 width sweep, N/d fixed -------
print("=== (A) m=1000 width sweep: min test error vs w, N and d held fixed ===")
CFG = [("heat2d_sine_nx64_K8_D1","0",50,15),("heat2d_sine_nx64_K8_D1","-1",150,10),
       ("heat2d_grf_nx64_l0.1-0.2_D1","0",200,15),("heat2d_grf_nx64_l0.1-0.2_D1","-1",150,10),
       ("heat2d_grf_nx64_l0.05_D1","0",150,15),("heat2d_grf_nx64_l0.05_D1","-1",150,15)]
WS=[100,200,335,500,600,700]
print("  %-30s %-8s %s" % ("dataset","trunk"," ".join("w=%-6d"%w for w in WS)))
best_w=[]
for bat,T,N,d in CFG:
    line={}
    for w in WS:
        c=[r for r in rows if r["numd"]==1000 and r["nep"]==4000 and r["bat"]==bat
           and r["T"]==T and r["N"]==N and r["d"]==d and r["w"]==w]
        line[w]=c[0]["te"].min() if c else None
    vals={w:v for w,v in line.items() if v is not None}
    bw=min(vals,key=vals.get); best_w.append(bw)
    print("  %-30s %-8s %s   best w=%d" % (bat.replace("heat2d_",""),
          "SVD" if T=="0" else "learned",
          " ".join(("%-8.3f"%line[w]) if line[w] else "  --    " for w in WS), bw))
from collections import Counter
print("  -> best width per config:", Counter(best_w))

# ---------- (B) new m=5000 sweep vs existing m=5000 nets, at a matched epoch --
print("\n=== (B) m=5000: new sweep (w=500,d=10,Nep10000) vs existing (w=335,Nep4000) ===")
new=[r for r in rows if r["numd"]==5000 and r["nep"]==10000]
old=[r for r in rows if r["numd"]==5000 and r["nep"]==4000]
print("  new nets: %d (still training)   existing: %d" % (len(new), len(old)))
if new:
    cut=min(len(r["ep"]) for r in new)
    ep_cut=new[0]["ep"][cut-1]
    print("  comparing at epoch %d (row %d), the furthest all new nets have reached\n" % (ep_cut, cut))
    for bat in sorted({r["bat"] for r in old}):
        base=bat.replace("_m5000","").replace("heat2d_","")
        for T in ("0","-1"):
            o=[r for r in old if r["bat"]==bat and r["T"]==T]
            n=[r for r in new if r["bat"]==bat and r["T"]==T]
            if not o or not n: continue
            om=min(r["te"][:cut].min() for r in o)
            oc=o[0]
            nb=min(n,key=lambda r:r["te"][:cut].min())
            print("   %-22s %-8s existing w=335 N=%-4d d=%-3d -> %.4f | new w=500 d=10 best N=%-4d -> %.4f  (%+.0f%%)"
                  % (base,"SVD" if T=="0" else "learned",oc["N"],oc["d"],om,nb["N"],
                     nb["te"][:cut].min(),100*(nb["te"][:cut].min()-om)/om))
