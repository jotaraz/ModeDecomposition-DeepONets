#tag:10_02_2026

from ... import don_code

nets_dir = don_code.nets_dir

llws = [2, 10, 18, 50, 90]
llw_strs = ["_llw"+str(x)+"_" for x in llws]
tags = ["whichT-1", "Nep5000", "Adam40", llw_strs]

alldirs = don_code.os.listdir(nets_dir)

def filter(tags):
    nets = {}
    for d in alldirs:
        #print(d)
        allthere = True
        key = None
        for tag in tags:
            if type(tag) == str:
                if tag not in d:
                    allthere = False 
            else:
                tmp = False
                for t in tag:
                    if t in d:
                        key = int(t[4:-1])
                        #print(t, key, d)
                        tmp = True
                allthere = tmp
        if allthere:
            nets[key] = d
    return nets

def run_check_DON(direc, epoch, T, ScaledSigma, 
                  rtrain, ptrain, utrain, ptest, utest, 
                  nb, nt, truesigma, vh_train, uu_train,
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

    Tpinv = don_code.np.linalg.pinv(T)
    TpinvA = don_code.jnp.matmul(Tpinv, utest)
    TTpinvA = don_code.jnp.matmul(T, TpinvA)

    eps   = don_code.jnp.sum( (utest - upredtest)**2 )
    eps_T = don_code.jnp.sum( (TTpinvA - utest)**2 )
    eps_B = don_code.jnp.sum( (TTpinvA - upredtest)**2 )
    eps_C = don_code.jnp.sum( (Btest.T - TpinvA)**2 )
    Tnorm = don_code.jnp.linalg.norm(T, ord=2)
    eps_D = Tnorm**2 * eps_C
    ret = don_code.np.array([eps, eps_T, eps_B, eps_D, eps_D/eps_B])
    #print(eps_T + eps_B - eps, eps)
    if abs(eps_T + eps_B - eps)/eps > 1e-6:
        print("Not adding up!", ret)

    return don_code.np.log10(ret)

    # utrain_norm   = don_code.jnp.mean(utrain**2)
    # utest_norm    = don_code.jnp.mean(utest**2)
    # utrainE_norm  = don_code.jnp.mean(don_code.jnp.matmul(don_code.jnp.diag(truesigma**(1+exponent)), VT_train)** 2)
    # utestE_norm   = don_code.jnp.mean(don_code.jnp.matmul(don_code.jnp.diag(truesigma**(1+exponent)), VT_test)** 2)

    # utrain_error  = don_code.jnp.mean( (upredtrain - utrain)**2 ) / utrain_norm
    # utest_error   = don_code.jnp.mean( (upredtest  - utest )**2 ) / utest_norm

    # if whichT < 0:
    #     utrainE_error = 1.0
    #     utestE_error  = 1.0
    # else:
    #     A = don_code.jnp.matmul(don_code.jnp.diag(truesigma**(1+exponent)), VT_train)
    #     B = don_code.jnp.matmul(don_code.jnp.diag(truesigma**exponent), don_code.jnp.matmul(ScaledSigma, Btrain.T))
    #     utrainE_error = don_code.jnp.mean((A - B)** 2) / utrainE_norm
    #     A = don_code.jnp.matmul(don_code.jnp.diag(truesigma**(1+exponent)), VT_test)
    #     B = don_code.jnp.matmul(don_code.jnp.diag(truesigma**exponent), don_code.jnp.matmul(ScaledSigma, Btest.T))
    #     utestE_error  = don_code.jnp.mean((A - B)** 2) / utestE_norm

    # return don_code.np.array([utrain_error, utest_error, utrainE_error, utestE_error])

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

lr_schedule = don_code.optax.exponential_decay(
    init_value=2e-3,  
    transition_steps=500,  
    decay_rate=0.95,
    staircase=True  # Set to True if you want discrete decay steps
)
optimizer = don_code.optax.sgd(learning_rate=lr_schedule)


pde_names = ["adv", "kdv", "burgers"]
pde_names2 = ["AD", "KdV", "Burgers"]
bids      = [1, 3, 6]
taus      = ["0.5", "0.2", "0.1"]
num_data  = 1000
epoch     = "4901"

for i in range(len(pde_names)):
    pde = pde_names[i]
    pde2 = pde_names2[i]
    bid = bids[i]
    tau = taus[i]
    batch_name, uendtag, _ = don_code.dic(bid)
    nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, uendtag, num_data)
    uu_train, ss_train, vh_train = don_code.jnp.linalg.svd(utrain, full_matrices=False)
    complete_tags = tags + [pde]
    nets = filter(complete_tags)
    #print("nets", nets)

    nets = {k:v for k,v in sorted(nets.items(), key=lambda x: x[0])}

    for llw, direc in nets.items(): #zip(llws, nets):
        #print(pde, llw, direc)
        #direc = nets_dir + "/" + net
        ScaledSigma = don_code.np.eye(llw)
        T = get_T(direc, epoch, ScaledSigma, rtrain, ptrain, ptest, nb, nt, uu_train, batch_name)

        ret = run_check_DON(direc, epoch, T, ScaledSigma, 
                      rtrain, ptrain, utrain, ptest, utest, 
                      nb, nt, ss_train[:llw], vh_train, uu_train, 
                      exponent=0.0, chptag="cur")
        s = pde2 + " & " + str(tau) + " & " + str(llw)
        for x in ret:
            s +=  " & " + str(round(x, 2))
        print(s + r"\\")




