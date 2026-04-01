#import os
#import sys
#import jax
#import numpy as np
#import jax.numpy as jnp
#import optax
#from jax import grad, jit, vmap
#import flax.linen as nn
from functools import partial
import matplotlib.pyplot as plt
from flax.training import train_state
import time

#from read_data import *

from don_code import *

jax.config.update('jax_enable_x64', True)

"""
see don_code for more details.
trunk_names[-1] = "Learned"
trunk_names[0]  = "SVD"
trunk_names[1] = "Leg."
trunk_names[2] = "Cheb."
trunk_names[3] = "Trig."
trunk_names[4] = "EvenSin"
trunk_names[5] = "Sin"
trunk_names[6] = "EvenCos"
trunk_names[7] = "Cos"
trunk_names[8] = "BatSpecTrig"
trunk_names[9] = "Cos1"
trunk_names[10] = "LegN"
trunk_names[11] = "CheN"

"""

Nepochs     = int(sys.argv[1]) # number of epochs
vtag        = int(sys.argv[2]) # version tag: usually set to 0, used to indicate differet instances of DeepONets who share all hyperparameters
d           = int(sys.argv[3]) # depth
w           = int(sys.argv[4]) # width of the hidden lazyers
llw         = int(sys.argv[5]) # number of output neurons of branch and trunk net (“inner dimension”), N in the paper
doplot      = bool(int(sys.argv[6])) # should a plot of the errorcurve be displayed? (is saved in any case)
batch_name  = sys.argv[7] # which PDE? choose from 'ModeDecomposition-DeepONets/data/datasets', e.g., 'kdvnx401_dt0.0001_nc5_m5000' (do not include tau here)
lrstag      = sys.argv[8] # just a string for you, to identify learning rate and decay factor in file name
init_lr     = float(sys.argv[9])  # Initial learning rate, e.g., 2e-3, or 1e-4
decay_rate  = float(sys.argv[10])  # Decay factor (e.g., 0.99 means lr decreases by 1% every 500 epochs), usually 0.95
num_data    = int(sys.argv[11]) # usually 1000: number of samples, these are split 90/10 as train/test
which_T     = int(sys.argv[12]) # indicates which basis, see comments above
dotruesigma = int(sys.argv[13]) # should \tilde{A} be $factor * \sigma_1 * T B^T$ (set to 0), or $factor * T \Sigma B^T$ (set to 1)
uendtag     = sys.argv[14] # this is basically tau/dt, default/standard example is 1999
sigmascale  = sys.argv[15] # this is the $factor$, from dotruesigma: if you want $\tilde{A} = T B^T$, set sigmascale="First", if you want $\tilde{A} = T \Sigma B^T$, set sigmascale="1.0"
exponent    = float(sys.argv[16]) # the exponent from reweighting
doadam      = bool(int(sys.argv[17])) # Adam or GD? (if you want AdaGrad set this to True as well, and set momentum=False)
dostacked   = bool(int(sys.argv[18])) # stacked or unstacked network?
adaptive_init_lr = False # if you use exponent != 0 and SGD, set this to True, otherwise dont
momentum    = True # only set to False, for AdaGrad. Adam needs True, for (normal) GD it does not matter

transfinp   = False 
alphaT      = 0.0 #float(sys.argv[8])
alphaB      = 0.0 #float(sys.argv[9])
lambdarange = 0.0



print("doadam", doadam, init_lr, decay_rate)

key = jax.random.PRNGKey(vtag) #int(time.time()))



# get correct, splitted dataset
nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = load_dataset(batch_name, uendtag, num_data)

r = 0
if transfinp:
    up, sp, vhp = np.linalg.svd(ptrain)
    r = len(np.where(sp > sp[0]*1e-10)[0])
    vhp_cut = vhp[:r,:]
    nb = r
    ptrain = np.matmul(vhp_cut, ptrain.T).T
    ptest = np.matmul(vhp_cut, ptest.T).T
    print("Input was transformed and projected on", r, "most relevant eigenmodes.")



print("nt", nt, "nb", nb, np.shape(rtrain), np.shape(ptrain), np.shape(utrain))

# do svd, ..., build model

llw = min(np.shape(utrain)[0], int(np.shape(utrain)[1]), llw)


"""Define name of new net-directory."""
stem = "whichT"+str(which_T)+"_doStacked"+str(dostacked)+"_doSigma"+str(dotruesigma)
if abs(exponent) > 1e-12 and adaptive_init_lr:
    stem += "_sisc"+str(sigmascale)+"_aT"+str(alphaT)+"_aB"+str(alphaB)+"_expA"+str(exponent)
else:
    stem += "_sisc"+str(sigmascale)+"_aT"+str(alphaT)+"_aB"+str(alphaB)+"_exp"+str(exponent)
stem += "_Nep"+str(Nepochs)+"_d"+str(d)+"_w"+str(w)+"_llw"+str(llw)
stem += "_bat"+batch_name+"_"+uendtag+"_numd"+str(num_data)
if doadam:
    if momentum:
        stem += "_lrAdam"+str(lrstag)
    else:
        stem += "_lrAda"+str(lrstag)
else:
    stem += "_lrSGD"+str(lrstag)

if transfinp:
    stem += "_tf"+str(r)
stem += "_v"+str(vtag)

dirs = os.listdir("nets")

print("stem :", stem)


if stem in dirs:
    print("stem :", stem in dirs)
    #print("stem2:", stem2 in dirs)
    #print(stem)
    #print(stem2)
else:
    T = 0
    lambdas = np.zeros(llw)
    uu_train, ss_train, vh_train = jnp.linalg.svd(utrain, full_matrices=False)
    n_train, m_train = np.shape(utrain)
    VT_train = vh_train[:llw, :]
    ts = ss_train[:llw] 
    VT_test  = jnp.matmul(jnp.diag(1/ts), jnp.matmul(uu_train[:, :llw].T, utest))

    ScaledSigma = np.ones(llw) * ss_train[0]

    if dotruesigma:
        ScaledSigma = ss_train[:llw] # / np.sqrt(n_train * m_train)

    factor = 1.0
    if sigmascale == "First":
        factor = 1.0/ss_train[0]
    elif sigmascale == "Size":
        factor = 1.0/np.sqrt(n_train * m_train)
    else: # sigmascale in (int, float):
        factor = float(sigmascale)

    print(sigmascale, factor)
    print(ScaledSigma*factor)
    ScaledSigma_diagvalues = factor * ScaledSigma.copy()
    ScaledSigma            = jnp.diag(ScaledSigma_diagvalues)



    if which_T < 0:
        if not dostacked:
            model = DeepONet(nb, nt, d, w, llw)
        else:
            model = StackedDeepONet(nb, nt, d, w, d, w, llw)
        params = model.init(key, ptrain, rtrain, ScaledSigma)

    else:
        T = get_fixed_trunk(which_T, llw, rtrain[:,0], batch_name, uu_train)
        
            
        VT0_train = jnp.matmul(jnp.diag(1/ts), jnp.matmul(np.linalg.pinv(T), utrain))
        VT0_test  = jnp.matmul(jnp.diag(1/ts), jnp.matmul(np.linalg.pinv(T), utest))
        
        #print(np.sum((VT0_train-VT_train)**2), np.sum((VT0_test-VT_test)**2))
        #utrain2 = np.matmul(T, np.matmul(ScaledSigma, VT0_train))
        #utest2  = np.matmul(T, np.matmul(ScaledSigma, VT0_test))
        #print("rec error", np.sum( (utrain2-utrain)**2 ) / np.sum(utrain**2), np.sum( (utest2-utest)**2 ) / np.sum(utest**2) )
        
        VT_train = VT0_train
        VT_test  = VT0_test    
        
        lambdas = 10**(lambdarange * (2*np.random.rand(llw)-1))
        if not dostacked:
            model = TDeepONet(nb, d, w, llw)
        else:
            model = StackedTDeepONet(nb, d, w, llw)
        params = model.init(key, ptrain, T, ScaledSigma)


    if adaptive_init_lr:
        init_lr *= ss_train[0]**(-2*exponent)

    lr_schedule = optax.exponential_decay(
        init_value=init_lr,  
        transition_steps=500,  
        decay_rate=decay_rate,
        staircase=True  # Set to True if you want discrete decay steps
    )

    if doadam:
        if momentum:
            optimizer = optax.adam(learning_rate=lr_schedule)
        else:
            optimizer = optax.adam(learning_rate=lr_schedule, b1=0.0)    
    else:
        optimizer = optax.sgd(learning_rate=lr_schedule)

    #state = TrainState.create(
    state = TrainStateWithUpdates.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer  # Uses learning rate decay automatically
    )





    #print("stem2:", stem2)

    stem = "nets/"+stem
    os.mkdir(stem)
    param_base = stem+"/"
    np.savetxt(param_base+"lambdas.txt", lambdas)



    out = all_mult_updates(Nepochs, state, model, 
                    ptrain, rtrain, utrain, ptest, rtest, utest, 
                    optimizer, alphaT, alphaB, lr_schedule, 
                    param_base, ScaledSigma, T, ss_train[:llw], 
                    VT_train, VT_test, exponent, llw, which_T, uu_train[:,:llw])

    state, opt_params, errors_data_train, errors_orthoT_train, errors_orthoB_train, errors_data_test, errors_orthoT_test, errors_orthoB_test = out


    # make figures
    colors = ["black","red","blue","green","yellow","orange","cyan","pink","brown","gray"]

    plt.figure()
    plt.plot(errors_data_train, label="L_data train")
    plt.plot(errors_data_test,  label="L_data test")

    if alphaT > 0:
        plt.plot(alphaT*errors_orthoT_train, label=str(alphaT)+"*L_orthoT train")
        plt.plot(alphaT*errors_orthoT_test, label=str(alphaT)+"*L_orthoT test")
    else:
        plt.plot(errors_orthoT_train, label="L_orthoT train")
        plt.plot(errors_orthoT_test, label="L_orthoT test")
        
    if alphaB > 0:
        plt.plot(alphaB*errors_orthoB_train, label=str(alphaB)+"*L_orthoB train")
        plt.plot(alphaB*errors_orthoB_test, label=str(alphaB)+"*L_orthoB test")
    else:
        plt.plot(errors_orthoB_train, label="L_orthoB train")
        plt.plot(errors_orthoB_test, label="L_orthoB test")


    plt.legend()
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(param_base+"errorcurve.png")



    if doplot:
        plt.show()








