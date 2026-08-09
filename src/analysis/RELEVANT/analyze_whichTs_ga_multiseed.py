# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 2026
The figure shows delta, delta_T and delta_B for test data of various bases.
tag:10_02_2026
@author: J_Taraz
"""

from ... import don_code

nets_dir = don_code.nets_dir

## ---------------------------------------------------------------------------
## MULTISEED VARIANT of analyze_whichTs_ga.py
## Each plotted point is the ARITHMETIC MEAN over seeds v0..v9 of the quantity
## that the original plots for v0 alone.  Set SHOW_BAND = True to shade the
## min..max spread across seeds behind each mean line.
## ---------------------------------------------------------------------------
from . import multiseed
SHOW_BAND = False
multiseed.SHOW_BAND = SHOW_BAND



def get_colors():
    f = open(don_code.colors_path, "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors_list0 = get_colors()
#full, trunk, branch
colors_list = [colors_list0[3], colors_list0[1], colors_list0[8]]


bid   = 3 # int(don_code.sys.argv[1]) #3
lrtag = 40 #int(don_code.sys.argv[2]) #40
LLW   = -1 #int(don_code.sys.argv[3]) #-1
dw    = "d5_w100" #don_code.sys.argv[4]      #"d5_w100"
nep = "5000"

don_code.plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

check_run = False

all_svd_llw_max = {0: 20, 1: 20, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100}

def llw_ranges(whT):
    if LLW > 0:
        llw_range_svd = [LLW]
        llw_range_gen = [LLW]
    else:
        llw_range_gen = don_code.np.arange(2, 100, 8)
        if bid < 3:
            llw_range_svd = [2, 10, 18, 20]
        else:
            llw_range_svd = llw_range_gen
    
    if whT == 0:
        return llw_range_svd
    return llw_range_gen
#svd_llw_range = np.arange(2, svd_llw_max, 8)

plotinfo = {}

#colors = ["red", "blue", "green", "orange", "cyan", "gray", "fuchsia", "k"]
#colors = {-1: "red", 0: "blue", 1: "green", 2: "orange", 5: "cyan", 7: "gray", 9: "black"}

whichTids = [-1] #, 0, 1, 2, 7]
for i,whT in enumerate(whichTids):
    if whT <= 0:
        axid = 0
    else:
        axid = 1
    plotinfo[whT] = {"color": colors_list[i], "ax": axid}

all_whTs = whichTids


lr_schedule = don_code.optax.exponential_decay(
    init_value=2e-3,  
    transition_steps=500,  
    decay_rate=0.95,
    staircase=True  # Set to True if you want discrete decay steps
)
optimizer = don_code.optax.sgd(learning_rate=lr_schedule)

def array_to_tuple(raw_color):
    return (raw_color[0], raw_color[1], raw_color[2])

def error_indices(whT):
    if whT < 0:
        return don_code.np.array([0, 2], dtype=int)
    return don_code.np.array([0, 1, 2, 3], dtype=int)

def run_check_DON(direc, epoch, T, ScaledSigma, 
                  rtrain, ptrain, utrain, ptest, utest, 
                  nb, nt, truesigma, vh_train, 
                  exponent=0.0, chptag="cur"):
    
    tmp_str     = nets_dir+"/"+direc+"/"+epoch

    tmp_dir = don_code.os.listdir(nets_dir+"/"+direc)
    if epoch+chptag+"_chp" not in tmp_dir:
        print(epoch+"_chp", "not in", tmp_dir)
        return None 

    depth, width, llw, whichT, _, _, _ = don_code.get_dwllw(direc)

    model    = None 
    VT_train = None 
    VT_test  = None
    init_params = None
    if whichT < 0:
        if "doStackedFalse" in direc:
            model = don_code.DeepONet(nb, nt, depth, width, llw)
        else:
            model = don_code.StackedDeepONet(nb, nt, depth, width, depth, width, llw)
        VT_train = vh_train[:llw, :]
        VT_test  = don_code.jnp.matmul(don_code.jnp.diag(1/truesigma), don_code.jnp.matmul(uu_train[:, :llw].T, utest))
        init_params = model.init(don_code.jax.random.PRNGKey(0), ptrain, rtrain, ScaledSigma)
    else:
        if "doStackedFalse" in direc:
            model = don_code.TDeepONet(nb, depth, width, llw)
        else:
            model = don_code.StackedTDeepONet(nb, depth, width, llw)
        VT_train = don_code.jnp.matmul(don_code.jnp.diag(1/truesigma), don_code.jnp.matmul(don_code.np.linalg.pinv(T), utrain))
        VT_test  = don_code.jnp.matmul(don_code.jnp.diag(1/truesigma), don_code.jnp.matmul(don_code.np.linalg.pinv(T), utest))
        init_params = model.init(don_code.jax.random.PRNGKey(0), ptrain, T, ScaledSigma)
    
    params, _   = don_code.load_checkpoint(init_params, init_params, path=tmp_str+chptag+"_chp")

    state  = don_code.TrainState.create(
                    apply_fn=model.apply,
                    params=params,
                    tx=optimizer  
                )
    
    if whichT < 0:
        upredtrain, _, Btrain = state.apply_fn(params, ptrain, rtrain, ScaledSigma)
        upredtest,  _, Btest  = state.apply_fn(params, ptest,  rtrain, ScaledSigma)
    else:
        upredtrain, _, Btrain = state.apply_fn(params, ptrain, T, ScaledSigma)
        upredtest,  _, Btest  = state.apply_fn(params, ptest,  T, ScaledSigma)


    
    utrain_norm   = don_code.jnp.mean(utrain**2)
    utest_norm    = don_code.jnp.mean(utest**2)
    utrainE_norm  = don_code.jnp.mean(don_code.jnp.matmul(don_code.jnp.diag(truesigma**(1+exponent)), VT_train)** 2)
    utestE_norm   = don_code.jnp.mean(don_code.jnp.matmul(don_code.jnp.diag(truesigma**(1+exponent)), VT_test)** 2)

    utrain_error  = don_code.jnp.mean( (upredtrain - utrain)**2 ) / utrain_norm
    utest_error   = don_code.jnp.mean( (upredtest  - utest )**2 ) / utest_norm

    if whichT < 0:
        utrainE_error = 1.0
        utestE_error  = 1.0
    else:
        A = don_code.jnp.matmul(don_code.jnp.diag(truesigma**(1+exponent)), VT_train)
        B = don_code.jnp.matmul(don_code.jnp.diag(truesigma**exponent), don_code.jnp.matmul(ScaledSigma, Btrain.T))
        utrainE_error = don_code.jnp.mean((A - B)** 2) / utrainE_norm
        A = don_code.jnp.matmul(don_code.jnp.diag(truesigma**(1+exponent)), VT_test)
        B = don_code.jnp.matmul(don_code.jnp.diag(truesigma**exponent), don_code.jnp.matmul(ScaledSigma, Btest.T))
        utestE_error  = don_code.jnp.mean((A - B)** 2) / utestE_norm

    return don_code.np.array([utrain_error, utest_error, utrainE_error, utestE_error])

def get_T(direc, epoch, ScaledSigma, 
            rtrain, ptrain, ptest, nb, nt, 
            uu_train, batch_name,
            chptag="cur"):
    
    depth, width, llw, whichT, _, _, _ = don_code.get_dwllw(direc)    
    if whichT >= 0:
        T = don_code.get_fixed_trunk(whichT, llw, rtrain[:,0], batch_name, uu_train)
        return T
    tmp_str     = nets_dir+"/"+direc+"/"+epoch
    tmp_dir = don_code.os.listdir(nets_dir+"/"+direc)
    if epoch+chptag+"_chp" not in tmp_dir:
        print(epoch+"_chp", "not in", tmp_dir)
        return None 


    model    = None 
    init_params = None
    if "doStackedFalse" in direc:
        model = don_code.DeepONet(nb, nt, depth, width, llw)
    else:
        model = don_code.StackedDeepONet(nb, nt, depth, width, depth, width, llw)
    init_params = model.init(don_code.jax.random.PRNGKey(0), ptrain, rtrain, ScaledSigma)
    
    params, _   = don_code.load_checkpoint(init_params, init_params, path=tmp_str+chptag+"_chp")

    state  = don_code.TrainState.create(
                    apply_fn=model.apply,
                    params=params,
                    tx=optimizer  
                )
    
    upredtrain, Ttr, Btrain = state.apply_fn(params, ptrain, rtrain, ScaledSigma)
    upredtest,  Tte, Btest  = state.apply_fn(params, ptest,  rtrain, ScaledSigma)
    if don_code.np.linalg.norm(Ttr - Tte) < 1e-6:
        return Ttr 
    else:
        print("Ttr != Tte")
        return None
    
def read_logfile(d):
    if batch_name not in d:
        print("Check whether bid fits dir!")
        print(batch_name)
        print(d)
        
    if d in all_dirs:
        logdata = don_code.np.loadtxt(nets_dir+"/"+d+"/log.txt")
        epochs = logdata[:,0]
        err_tr = 10**(logdata[:,1]/2)
        err_te = 10**(logdata[:,4]/2)
        erE_tr = 10**(logdata[:,8]/2)
        erE_te = 10**(logdata[:,9]/2)
        
        tmp = {}
        tmp["epochs"] = epochs
        tmp["err_tr"] = err_tr
        tmp["err_te"] = err_te
        tmp["erE_tr"] = erE_tr
        tmp["erE_te"] = erE_te
        tmp["last_values"] = don_code.np.array([err_tr[-2], err_te[-2], erE_tr[-2], erE_te[-2]])
        mintrid = don_code.np.argmin(err_tr)
        tmp["atbesttr_values"] = don_code.np.array([err_tr[mintrid], err_te[mintrid], erE_tr[mintrid], erE_te[mintrid]])

        return tmp #don_code.np.array([epochs, err_tr, err_te, erE_tr, erE_te])
    
    return None

def process_data(d):
    ret_dict = {}
    save_return = read_logfile(d)
    if not save_return:
        print(d, ": No saved file found.")
        return None 
    
    ret_dict["save"] = save_return

    #epochs = save_return["epochs"]
    _, _, llw, whichT, _, _, _ = don_code.get_dwllw(d)
    epoch = "4901"
    ScaledSigma = don_code.np.eye(llw)
    T = get_T(d, epoch, ScaledSigma, rtrain, ptrain, ptest, nb, nt, uu_train, batch_name)
        
    proj             = don_code.np.eye(don_code.np.shape(T)[0]) - don_code.np.matmul(T, don_code.np.linalg.pinv(T))
    proj_error_tr_sq = don_code.np.sum( don_code.np.matmul(proj, utrain)**2 ) / don_code.np.sum(utrain**2)
    proj_error_te_sq = don_code.np.sum( don_code.np.matmul(proj, utest)**2 ) / don_code.np.sum(utest**2)
    
    #if proj_error > 1e-4:
    #    don_code.plt.plot(epochs, 0*epochs + don_code.np.log10(proj_error), '--', color=color1, alpha=0.5)

    ret_dict["proj_error_tr"] = don_code.np.sqrt(proj_error_tr_sq)
    ret_dict["proj_error_te"] = don_code.np.sqrt(proj_error_te_sq)


    if check_run:
        errors_calc = run_check_DON(d, epoch, T, ScaledSigma, 
                                    rtrain, ptrain, utrain, ptest, utest, 
                                    nb, nt, ss_train[:llw], vh_train, exponent=0.0)
        if errors_calc is None:
            return None
        ids = error_indices(whichT)
        if don_code.np.max( don_code.np.abs( (errors_calc[ids]-save_return["last_values"][ids])/errors_calc[ids] ) ) > 1e-6:
            return None 
        
    return ret_dict

def process_data_seeds(name, showlast):
    """Arithmetic mean over seeds of the four quantities the figure plots.

    process_data() already returns relative errors (read_logfile converts the
    stored log10 columns with 10**(x/2)), so averaging here is arithmetic in the
    plotted quantity, which is what we want.
    """
    key = "last_values" if showlast else "atbesttr_values"

    def one(nm):
        r = process_data(nm)
        if r is None:
            return None
        v = r["save"][key]
        return {"etr": v[0], "ete": v[1],
                "proj_tr": r["proj_error_tr"], "proj_te": r["proj_error_te"],
                "bran_tr": v[0] - r["proj_error_tr"],
                "bran_te": v[1] - r["proj_error_te"]}

    return multiseed.mean_of_keys(
        one, name, ("etr", "ete", "proj_tr", "proj_te", "bran_tr", "bran_te"))


def plot_multLLW(showlast=True):
    minval = 100
    maxval = 0
    fig, axs = don_code.plt.subplots(1, 1, figsize=(2.5,2), sharey=True)
    grand_dict = {}
    band_dict = {}
    n_seeds_used = []
    for whT in all_whTs:
        grand_dict[whT] = {}
        band_dict[whT] = {}
    
    for i in nets.keys():
        whT = nets[i]["whT"]
        llw = nets[i]["llw"]
        name = nets[i]["name"]
        means, los, his, nseed = process_data_seeds(name, showlast)
        if means is None:
            continue
        n_seeds_used.append(nseed)
        print("multiseed: %s -> mean over %d seeds" % (name, nseed))
        grand_dict[whT][llw] = (means["etr"], means["ete"],
                                means["proj_tr"], means["proj_te"])
        band_dict[whT][llw] = (los["ete"], his["ete"],
                               los["proj_te"], his["proj_te"],
                               los["bran_te"], his["bran_te"])
            
    for whT in all_whTs:
        xax = grand_dict[whT].keys()
        etr     = []
        ete     = []
        proj_tr = []
        proj_te = []
        bran_tr = []
        bran_te = []

        for x in xax:
            a,b,c,d = grand_dict[whT][x]
            etr.append(a)
            ete.append(b)
            proj_tr.append(c)
            proj_te.append(d)
            bran_tr.append(a-c)
            bran_te.append(b-d)


        xl = list(xax)
        multiseed.band(axs, xl, [band_dict[whT][x][0] for x in xl],
                       [band_dict[whT][x][1] for x in xl], colors_list[0])
        multiseed.band(axs, xl, [band_dict[whT][x][2] for x in xl],
                       [band_dict[whT][x][3] for x in xl], colors_list[1])
        multiseed.band(axs, xl, [band_dict[whT][x][4] for x in xl],
                       [band_dict[whT][x][5] for x in xl], colors_list[2])

        axs.plot(xax, ete, 'o', linestyle='none', fillstyle='none', markeredgewidth=2, color=colors_list[0], linewidth=3, label="total error")
        axs.plot(xax, proj_te, '--', color=colors_list[1], linewidth=3, label="trunk error")
        axs.plot(xax, bran_te, '-', color=colors_list[2], linewidth=3, label="branch error")


        minval = min(minval, don_code.np.min(etr), don_code.np.min(ete))
        maxval = max(maxval, don_code.np.max(etr), don_code.np.max(ete))
        #if whT >= 0:
        #axs[plotinfo[whT]["ax"]].plot(xax, proj_tr, ':', color=plotinfo[whT]["color"], linewidth=3)
        #axs[plotinfo[whT]["ax"]].plot(xax, proj_te, '-.', color=plotinfo[whT]["color"], linewidth=3)


    #axs.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.03, 1.01))
    axs.tick_params(labelsize=10)
    axs.set_xticks([10, 50, 90])
    
    axs.set_ylim((minval/10, maxval*2))
    
    axs.set_xlabel(r"inner dimension $N$", fontsize=10)
    axs.set_ylabel(r"relative errors", fontsize=10)
    #axs.set_ylabel(r"relative error $\delta$", fontsize=10)
    
    return fig, axs

all_dirs = don_code.os.listdir(nets_dir)

batch_name, uendtag, _ = don_code.dic(bid)

nets = {}

i = 0

for whT in all_whTs:
     for llw in llw_ranges(whT):
        tmp = {}
        if whT == 0 and llw > all_svd_llw_max[bid]:
            tmp_llw = all_svd_llw_max[bid]
        else:
            tmp_llw = llw
        tmp["name"] = "whichT"+str(whT)+"_doStackedFalse_doSigma0_siscFirst_aT0.0_aB0.0_exp0.0_Nep5000_"+dw+"_llw"+str(tmp_llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0"
        tmp["llw"]  = tmp_llw 
        tmp["whT"]  = whT
        nets[i] = tmp
        i += 1

num_data = 1000
nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, uendtag, num_data)
uu_train, ss_train, vh_train = don_code.jnp.linalg.svd(utrain, full_matrices=False)

fig, axs = 0,0

showlast = False

nametag = don_code.figures_dir + "/"
# if LLW > 0:
#     nametag += "errorcurves_N"+str(LLW)
#     plot_errorcurves()
# else:
nametag += "error_overN"
fig, axs = plot_multLLW(showlast=showlast)

axs.set_yscale("log")

#fig.subplots_adjust(wspace=0.0, hspace=0.0, right=0.95, left=0.24, top=0.95, bottom=0.21)
fig.subplots_adjust(left=0.23, right=0.99, bottom=0.21, top=0.85)

#tmp = nametag+"_bid"+str(bid)+"_lr"+str(lrtag)+"_"+dw
#if not showlast:
#    tmp += "_besttr"
path = don_code.figures_dir
tmp = "Fig1_top_right_multiseed"
fig.savefig(path+"/pdfs/"+tmp+".pdf")
fig.savefig(path+"/pngs/"+tmp+".png")

#don_code.plt.show()



