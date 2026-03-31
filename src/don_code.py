import os
import sys
import jax
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import flax.linen as nn
import matplotlib.pyplot as plt
from flax.training import train_state
import optax
from typing import Any
import flax.serialization
import scipy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nets_dir = os.path.join(REPO_ROOT, "data", "nets")
data_dir = os.path.join(REPO_ROOT, "data", "datasets")
colors_path = os.path.join(REPO_ROOT, "colors_cb.txt")
figures_dir = os.path.join(REPO_ROOT, "figures")


jax.config.update('jax_enable_x64', True)


bid_to_name = {}



def dic(tag):
    if tag == 0:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1000", 20
    if tag == 1:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1999", 20
    if tag == 2:
        return "kdvnx401_dt0.0001_nc5_m5000", "10", 50
    if tag == 3:
        return "kdvnx401_dt0.0001_nc5_m5000", "1999", 50
    if tag == 4:
        return "kdvnx401_dt0.0001_nc5_m5000", "5999", 50
    if tag == 5:
        return "kdvnx401_dt0.0001_nc5_m5000", "9999", 50    
    if tag == 6:
        return "burgers_dt0.0001_nc10_m3800", "100", 50
    if tag == 7:
        return "burgers_dt0.0001_nc10_m3800", "999", 50

    tag0 = 21
    for sign in [+1.0, -1.0]:
        for sigm in [-1.0,-0.5,-0.01]:
            for fs in [0.05, 0.1, 0.2, 0.4]:
                bs = sign*fs
                ss = sigm
                if tag0 == tag:
                    return "synthv2_n100_N20_m5000", f"fs{bs}ss{ss}", 20

                tag0 += 1
    
    tag0 = 51
    for sign in [+1.0]:
        for sigm in [-1.0,-0.5]:
            for fs in [0.05, 0.1, 0.2, 0.4, 1.0]:
                bs = sign*fs
                ss = sigm
                if tag0 == tag:
                    return "synthv3_n100_N5_m5000", f"fs{bs}ss{ss}", 5

                tag0 += 1

    tag0 = 61 
    freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
    sigmaexpss    = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
    norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv4_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    
    tag0 = 71        
    freqincscales = [ -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0, -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0]
    sigmaexpss    = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [ 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0), 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0)]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            if tag0 >= 81:
                tmp = "v6"
            else:
                tmp = "v5"
            return "synth"+tmp+"_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(round(norm0scales[i], 4)), 5
        tag0 += 1
    
    tag0 = 91
    freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
    sigmaexpss    = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv7_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    
    tag0 = 101
    freqincscales = [-0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0] #, -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0]
    sigmaexpss    = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1] #, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [0.2*np.exp(4*0.2), 0.4*np.exp(4*0.2), 1.0*np.exp(4*0.2), 2.0*np.exp(4*0.2), 0.2*np.exp(4*0.4), 0.2*np.exp(4*1.0)] #, 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0)]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv8_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(round(norm0scales[i], 4)), 5
        tag0 += 1
    
    tag0 = 111
    freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
    sigmaexpss    = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
    norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv9_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    
    tag0 = 121
    freqincscales = [-0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0] #, -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0]
    sigmaexpss    = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01] #, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [0.2*np.exp(4*0.2), 0.4*np.exp(4*0.2), 1.0*np.exp(4*0.2), 2.0*np.exp(4*0.2), 0.2*np.exp(4*0.4), 0.2*np.exp(4*1.0)] #, 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0)]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv10_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(round(norm0scales[i], 4)), 5
        tag0 += 1
    
    return None


for tag in range(127):
    tmp = dic(tag)
    if tmp:
        bid_to_name[tag] = tmp



def get_dwllw(name):
    #tmp = name.split("/")[1]
    tmp = name.split("_")
    whichT = int(tmp[0][6:])
    d = int(tmp[8][1:])
    w = int(tmp[9][1:])
    llw = int(tmp[10][3:])
    #where_data = 0
    batch_name = tmp[11][3:]
    num_data = 0
    for j,x in enumerate(tmp[12:]):
        if "numd" in x:
            where_data = j
            num_data = x[4:]
            break
        else:
            batch_name += "_"+x 

    tmp = batch_name.split("_")
    batch_name = tmp[0]
    #print(batch_name)
    for t in tmp[1:-1]:
        #print("t", t)
        batch_name += "_"+t 
    
    return d, w, llw, whichT, batch_name, int(num_data), tmp[-1]


# if all elements of tags are in direc, return True
# else, return False
def contains_all(direc, tags):
    for tag in tags:
        if tag not in direc:
            #print(tag, "not in", direc)
            return False 
    return True


# subtracts all elements of tags from direc and returns (remaining text, True)
def contains_all_remove(direc, tags):
    remain = direc
    for tag in tags:
        if tag not in direc:
            #print(tag, "not in", direc)
            return False, ""
        else:
            tmp = remain.split(tag)
            remain = ""
            for t in tmp:
                remain += t

    return True, remain

def get_scaled_r(R, batch_name):
    if "adv" in batch_name:
        return R / (2*np.pi)
    elif "kdv" in batch_name:
        return R / (2*np.pi)
    elif "burgers" in batch_name:
        return R 
    else:
        return R

def generate_legendre_trunk(llw, R, batch_name):
    new_R = get_scaled_r(R, batch_name)
    n = len(R)
    T = np.zeros((n, llw))
    for i in range(llw):
        T[:,i] = scipy.special.eval_legendre(i, 2.0*new_R - 1.0)
    return T 

def generate_chebyshev_trunk(llw, R, batch_name):
    new_R = get_scaled_r(R, batch_name)
    n = len(R)
    T = np.zeros((n, llw))
    for i in range(llw):
        T[:,i] = scipy.special.eval_chebyt(i, 2.0*new_R - 1.0)
    return T     
    
def generate_legendre_trunk_norm(llw, R, batch_name):
    new_R = get_scaled_r(R, batch_name)
    n = len(R)
    T = np.zeros((n, llw))
    for i in range(llw):
        y = scipy.special.eval_legendre(i, 2.0*new_R - 1.0)
        T[:,i] = y / np.sqrt(np.sum(y**2))
    return T 

def generate_chebyshev_trunk_norm(llw, R, batch_name):
    new_R = get_scaled_r(R, batch_name)
    n = len(R)
    T = np.zeros((n, llw))
    for i in range(llw):
        y = scipy.special.eval_chebyt(i, 2.0*new_R - 1.0)
        T[:,i] = y / np.sqrt(np.sum(y**2)) 
    return T     

def generate_trigonometric_trunk(llw, R, batch_name):
    new_R = get_scaled_r(R, batch_name)
    n = len(R)
    T = np.zeros((n, llw))
    for i in range(llw):
        if i%2 == 0:
            T[:,i] = np.cos(np.pi*int(i/2)*new_R) #scipy.special.eval_chebyt(i, 2.0*new_R - 1.0)
        else:
            T[:,i] = np.sin(np.pi*int((i+1)/2)*new_R)
    return T     

def generate_evensine_trunk(llw, R, batch_name):
    new_R = get_scaled_r(R, batch_name)
    n = len(R)
    T = np.zeros((n, llw))
    for i in range(llw):
        T[:,i] = np.sin(2*np.pi*(i+1)*new_R)
    return T     

def generate_sine_trunk(llw, R, batch_name):
    new_R = get_scaled_r(R, batch_name)
    n = len(R)
    T = np.zeros((n, llw))
    for i in range(llw):
        T[:,i] = np.sin(np.pi*(i+1)*new_R)
    return T     

def generate_evencos_trunk(llw, R, batch_name):
    new_R = get_scaled_r(R, batch_name)
    n = len(R)
    T = np.zeros((n, llw))
    for i in range(llw):
        T[:,i] = np.cos(2*np.pi*(i+1)*new_R)
    return T     

def generate_cos_trunk(llw, R, batch_name):
    new_R = get_scaled_r(R, batch_name)
    n = len(R)
    T = np.zeros((n, llw))
    for i in range(llw):
        T[:,i] = np.cos(np.pi*(i+1)*new_R)
    return T     

def generate_cos1_trunk(llw, R, batch_name):
    new_R = get_scaled_r(R, batch_name)
    n = len(R)
    T = np.zeros((n, llw))
    for i in range(llw):
        T[:,i] = np.cos(np.pi*(i)*new_R)
    return T     

def generate_batspec_trig_trunk(llw, r, batch_name):
    if "adv" in batch_name:
        return generate_evencos_trunk(llw, r, batch_name)
    elif "kdv" in batch_name:
        return generate_evensine_trunk(llw, r, batch_name)
    elif "burgers" in batch_name:
        return generate_sine_trunk(llw, r, batch_name)
    return None

def get_fixed_trunk(which_T, llw, r, batch_name, uu_train):
    n = len(r)
    T = np.zeros((n, llw))
    if which_T == 0:
        T = uu_train[:, :llw]
    elif which_T == 1:
        T = generate_legendre_trunk(llw, r, batch_name)
    elif which_T == 2:
        T = generate_chebyshev_trunk(llw, r, batch_name)
    elif which_T == 3:
        T = generate_trigonometric_trunk(llw, r, batch_name)
    elif which_T == 4:
        T = generate_evensine_trunk(llw, r, batch_name)
    elif which_T == 5:
        T = generate_sine_trunk(llw, r, batch_name)
    elif which_T == 6:
        T = generate_evencos_trunk(llw, r, batch_name)
    elif which_T == 7:
        T = generate_cos_trunk(llw, r, batch_name)
    elif which_T == 8:
        T = generate_batspec_trig_trunk(llw, r, batch_name)
    elif which_T == 9:
        T = generate_cos1_trunk(llw, r, batch_name)
    elif which_T == 10:
        T = generate_legendre_trunk_norm(llw, r, batch_name)
    elif which_T == 11:
        T = generate_chebyshev_trunk_norm(llw, r, batch_name)
    
    return T
    
trunk_names = {}
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



### load data

def stringify(xx):
    s = ""
    for x in xx:
        s += str(x)+" "
    return s 

def load_dataset(batch_name0, uendtag, num_data):
    # load data
    batch_name = data_dir+"/"+batch_name0
    R0 = np.loadtxt(batch_name+"_R.txt")
    P0 = np.loadtxt(batch_name+"_P.txt")
    U0 = np.loadtxt(batch_name+"_"+uendtag+"_U.txt")
    print(np.shape(R0), np.shape(P0))

    if len(np.shape(R0)) == 1:
        R1 = np.zeros((len(R0),1))
        R1[:,0] = R0 
        R0 = R1 

    if len(np.shape(P0)) == 1:
        P1 = np.zeros((len(P0),1))
        P1[:,0] = P0 
        P0 = P1

    print(np.shape(R0), np.shape(P0))
    
    P = P0[:num_data, :]
    U = U0[:, :num_data]

    # split into test and train
    num_train = int(0.9 * num_data)
    rtrain = R0 
    rtest  = R0
    ptrain = P[:num_train, :]
    ptest  = P[num_train:, :]
    utrain = U[:, :num_train]
    utest  = U[:, num_train:]

    return np.shape(rtrain)[1], np.shape(ptrain)[1], rtrain, rtest, ptrain, ptest, utrain, utest

### save params

def save_checkpoint(params, update, path="checkpoint.msgpack"):
    to_save = {
        "params": params,
        "update": update,
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(to_save))

def load_checkpoint(init_params, init_update, path="checkpoint.msgpack"):
    with open(path, "rb") as f:
        byte_data = f.read()

    template = {
        "params": init_params,
        "update": init_update,
    }
    restored = flax.serialization.from_bytes(template, byte_data)
    return restored["params"], restored["update"]

def save_params(params, base):
    for p in params.keys():
        #print("p", p)
        for net in params[p].keys():
            #print("net", net)
            for layer in params[p][net].keys():
                #print("layer", layer)
                n = base+net+"_"+layer+".txt"
                f = open(n, "w")
                for a in params[p][net][layer].keys():
                    x = params[p][net][layer][a]
                    #print("a", a, jnp.shape(x))
                    #print(x)
                    for i in range(jnp.shape(x)[0]):
                        s = ""
                        if len(jnp.shape(x)) == 2:
                            for j in range(jnp.shape(x)[1]):
                                s = s+str(x[i,j])+" " 
                        else:
                            s = str(x[i])        
                        f.write(s+"\n")
                    f.write("\n\n")
                f.close()


### train state
class TrainState(train_state.TrainState):
    pass

class TrainStateWithUpdates(train_state.TrainState):
    def apply_gradients_with_updates(
        self, *, grads
    ):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state
        ), updates


### normal DON

def get_params(direc, w, llw, lastlayer):
    files = os.listdir(direc)
    b = {}
    t = {}
    #print(files)
    for n in files:
        if "test" not in n and "train" not in n:
            if lastlayer in n:
                tmp_w = llw
            else:
                tmp_w = w

            bias = jnp.array(np.loadtxt(direc+"/"+n, max_rows=tmp_w))
            kern0 = np.loadtxt(direc+"/"+n, skiprows=tmp_w+2)
            
            if len(np.shape(kern0)) == 1:
                kern = jnp.zeros((1, len(kern0)))
                kern = kern.at[0,:].set(kern0)
            else:
                kern = jnp.array(kern0)

            #print(n, jnp.shape(kern), jnp.shape(bias))    

            tmp = n.split("_")
            layer_name = tmp[2]+"_"+tmp[3].split(".")[0]

            if "branch_net" in n:
                b[layer_name] = {"kernel": kern, "bias": bias}

            elif "trunk_net" in n:
                t[layer_name] = {"kernel": kern, "bias": bias}

    p = {"branch_net": b, "trunk_net": t}
    params = {"params": p}
    return params

def get_model(direc, last_layer, d, w, llw, nt, F):
    model = DeepONet(F, nt, d, w, llw)
    params = get_params(direc, w, llw, last_layer)
    return model, params

class DeepONet(nn.Module):
    nb: int  # Branch network input size
    nt: int  # Trunk network input size
    d: int   # Number of layers (including output layer)
    w: int   # Number of neurons per layer
    llw: int

    def setup(self):
        """ Initialize branch and trunk networks with d layers of w neurons each. """
        self.branch_net = self._build_mlp(self.nb)
        self.trunk_net = self._build_mlp(self.nt)

    def _build_mlp(self, input_dim):
        """ Helper function to create an MLP with d layers of w neurons each. """
        layers = [nn.Dense(self.w), nn.gelu]  # First hidden layer
        for _ in range(self.d - 1):  # Additional hidden layers
            layers.append(nn.Dense(self.w))
            layers.append(nn.gelu)
        layers.append(nn.Dense(self.llw))  # Final output layer (w neurons)
        return nn.Sequential(layers)

    def __call__(self, p, r, ScaledSigma):
        """ Compute DeepONet output as a dot product of branch and trunk outputs. """
        B = self.branch_net(p)  # Shape: (batch_size, w)
        T = self.trunk_net(r)   # Shape: (batch_size, w)
        #print(np.shape(p), np.shape(r))
        #print(np.shape(B), np.shape(T))
        #return jnp.sum(B * T, axis=-1, keepdims=True), T  # Shape: (batch_size, 1)
        G = jnp.matmul(T, jnp.matmul(ScaledSigma, B.T)) #jnp.sum(B * T, axis=-1, keepdims=True)
        return G, T, B

@jax.jit
def all_losses(params, state, p, r, G_true, ScaledSigma, scaleT, scaleB):
    G_pred, T, B = state.apply_fn(params, p, r, ScaledSigma)

    L_data = jnp.mean((G_pred - G_true) ** 2)

    gram_matrixT = jnp.matmul(T.T, jnp.matmul(T, ScaledSigma))
    L_orthoT = jnp.mean((gram_matrixT - scaleT * ScaledSigma)**2)
        
    gram_matrixB = jnp.matmul(B.T, jnp.matmul(B, ScaledSigma))
    L_orthoB = jnp.mean((gram_matrixB - scaleB * ScaledSigma)**2)

    return L_data, L_orthoT, L_orthoB

@jax.jit
def train_step(state, p, r, G_true, alphaT, alphaB, ScaledSigma, scaleT, scaleB):
    """ Compute loss, gradients, and update parameters. """
    def loss_fn(params):
        L_data, L_orthoT, L_orthoB = all_losses(params, state, p, r, G_true, ScaledSigma, scaleT, scaleB)

        # Total loss
        return L_data + alphaT * L_orthoT + alphaB * L_orthoB


    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    #state = state.apply_gradients(grads=grads)
    state, updates = state.apply_gradients_with_updates(grads=grads)
    return state, loss, updates

@jax.jit
def all_loss_components(params, state, p, r, VT, ScaledSigma, truesigma, leftsingvec):
    G_pred, T, B = state.apply_fn(params, p, r, ScaledSigma)
    VTapprox = jnp.matmul(jnp.diag(1.0/truesigma), jnp.matmul(leftsingvec.T, G_pred))
    return jnp.sum( (VTapprox.T-VT.T)**2, axis=0 )

def mult_updates(Nepochs, state, model, 
                ptrain, rtrain, utrain, ptest, rtest, utest, 
                optimizer, alphaT, alphaB, lr_schedule, 
                param_base, ScaledSigma):
    return 0
    print(param_base)
    n_train, m_train = np.shape(utrain)
    n_test,  m_test  = np.shape(utest)
    scaleT_train = n_train
    scaleB_train = m_train
    scaleT_test  = n_test
    scaleB_test  = m_test

    utrain_norm = jnp.mean(utrain**2)
    utest_norm = jnp.mean(utest**2)
    

    opt_state, loss, update = train_step(state, ptrain, rtrain, utrain, 
                                 alphaT, alphaB, ScaledSigma, scaleT_train, scaleB_train)
    opt_params = opt_state.params
    opt_update = update
    
    minloss = 1e8
    errors_data_train   = jnp.zeros(Nepochs)
    errors_orthoT_train = jnp.zeros(Nepochs)
    errors_orthoB_train = jnp.zeros(Nepochs)
    errors_data_test    = jnp.zeros(Nepochs)
    errors_orthoT_test  = jnp.zeros(Nepochs)
    errors_orthoB_test  = jnp.zeros(Nepochs)
    last_opt_it = 0
    for i in range(Nepochs):
        # gradient step
        state, total_loss_train, update = train_step(state, ptrain, rtrain, utrain, 
                                   alphaT, alphaB, ScaledSigma, scaleT_train, scaleB_train)
        # get different training loss contributions
        L_data_train, L_orthoT_train, L_orthoB_train = all_losses(state.params, state, ptrain, rtrain, utrain, 
                                                                  ScaledSigma, scaleT_train, scaleB_train)
        # get different test loss contributions
        L_data_test,  L_orthoT_test,  L_orthoB_test  = all_losses(state.params, state, ptest, rtest, utest, 
                                                                  ScaledSigma, scaleT_test, scaleB_test)
        
        #total_loss_train = L_data_train + alphaT*L_orthoT_train + alphaB*L_orthoB_train
        #total_loss_test  = L_data_test  + alphaT*L_orthoT_test  + alphaB*L_orthoB_test

        if L_data_test < minloss:
            opt_params = state.params.copy()
            opt_update = update
            last_opt_it = i
            minloss = L_data_test

        # save errors in arrays
        errors_data_train   = errors_data_train.at[i].set(L_data_train)
        errors_orthoT_train = errors_orthoT_train.at[i].set(L_orthoT_train)
        errors_orthoB_train = errors_orthoB_train.at[i].set(L_orthoB_train)
        errors_data_test    = errors_data_test.at[i].set(L_data_test)
        errors_orthoT_test  = errors_orthoT_test.at[i].set(L_orthoT_test)
        errors_orthoB_test  = errors_orthoB_test.at[i].set(L_orthoB_test)
        

        # print errors into log-file
        if (i+1)%100 == 0 or i in [0, 10, 50]:
            file = open(param_base+"log.txt", "a")
            tmp  = str(i)+" "
            tmp += str(np.log10(errors_data_train[i]/utrain_norm))+" "+str(np.log10(errors_orthoT_train[i]/utrain_norm))+" "+str(np.log10(errors_orthoB_train[i]/utrain_norm))+" "
            tmp += str(np.log10(errors_data_test[i]/utest_norm))+ " "+str(np.log10(errors_orthoT_test[i]/utest_norm))+ " "+str(np.log10(errors_orthoB_test[i]/utest_norm))+" "
            tmp += str(np.log10(minloss/utest_norm))
            file.write(tmp+"\n")
            file.close()

        # print errors to screen
        if (i+1)%int(Nepochs/100) == 0:
            print(i,"/",Nepochs,": L total train",jnp.log10(total_loss_train/utrain_norm),
                  "L data train",jnp.log10(L_data_train/utrain_norm),"L data test",jnp.log10(L_data_test/utest_norm),
                  "min L data test",jnp.log10(minloss/utest_norm))
            
        # save (so-far) optimal and current parameters
        if (i+1)%1000 == 0 or (i+1) in [0, 100, 500]:
            os.mkdir(param_base+str(i+1)+"opt")
            save_params(opt_params, param_base+str(i+1)+"opt/")

            os.mkdir(param_base+str(i+1)+"cur")
            save_params(state.params, param_base+str(i+1)+"cur/")            
    
            os.mkdir(param_base+str(i+1)+"upd_opt")
            save_params(opt_update, param_base+str(i+1)+"upd_opt/")

            os.mkdir(param_base+str(i+1)+"upd_cur")
            save_params(update, param_base+str(i+1)+"upd_cur/")            
    
    #os.mkdir(param_base+"END")
    #save_params(opt_params, param_base+"END/")
    
    return state, opt_params, errors_data_train, errors_orthoT_train, errors_orthoB_train, errors_data_test, errors_orthoT_test, errors_orthoB_test

### normal stacked

def get_STparams(direc, depth, w):
    return {}
    files = os.listdir(direc)
    lastlayer = "Dense_"+str(depth)
    branch_nets = {}
    for j in range(llw):
        bn = {}
        for n in files:
            if "branch_nets_"+str(j)+"_Dense" in n:
                if lastlayer in n:
                    tmp_w = 1
                else:
                    tmp_w = w
                bias0 = jnp.array(np.loadtxt(direc+"/"+n, max_rows=tmp_w))
                kern0 = np.loadtxt(direc+"/"+n, skiprows=tmp_w+2)
                if len(np.shape(kern0)) == 1 and lastlayer in n:
                    kern = jnp.zeros((len(kern0), 1))
                    kern = kern.at[:,0].set(kern0)
                elif len(np.shape(kern0)) == 1 and lastlayer not in n:
                    kern = jnp.zeros((1, len(kern0)))
                    kern = kern.at[0,:].set(kern0)
                else:
                    kern = jnp.array(kern0)

                if len(np.shape(bias0)) == 0: # in (float,int):
                    bias = jnp.zeros((1,))
                    bias = bias.at[0].set(bias0)
                else:
                    bias = bias0
                tmp = n.split("_")
                layer_name = "Dense_"+tmp[4].split(".")[0]

                bn[layer_name] = {"kernel": kern, "bias": bias}
        branch_nets["branch_nets_"+str(j)] = bn

    params = {"params": branch_nets}
    return params

def get_STmodel(direc, last_layer, d, w, llw, nt, F):
    #model = StackedDeepONet(F, nt, d, w, d, w, llw)
    #params = get_STparams(direc, d, w)
    return 0, 0 #model, params

class FeedforwardNet(nn.Module):
    input_dim: int
    hidden_dim: int
    depth: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return x  # shape: (batch_size, 1)

class FeedforwardNet2(nn.Module):
    input_dim: int
    hidden_dim: list
    depth: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth):
            x = nn.Dense(self.hidden_dim[i])(x)
            x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return x  # shape: (batch_size, 1)


class TrunkNet(nn.Module):
    input_dim: int
    hidden_dim: int
    depth: int
    output_dim: int  # same as number of branch subnets

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(self.output_dim)(x)
        return x  # shape: (batch_size, p)


class StackedDeepONet(nn.Module):
    branch_input_dim: int
    trunk_input_dim: int
    depth_tr: int
    hidden_dim_tr: int
    depth_br: int
    hidden_dim_br: int
    p: int  # number of branch subnets

    def setup(self):
        self.branch_nets = [
            FeedforwardNet(
                input_dim=self.branch_input_dim,
                hidden_dim=self.hidden_dim_br,
                depth=self.depth_br
            )
            for _ in range(self.p)
        ]
        self.trunk_net = TrunkNet(
            input_dim=self.trunk_input_dim,
            hidden_dim=self.hidden_dim_tr,
            depth=self.depth_tr,
            output_dim=self.p
        )

    def __call__(self, branch_input, trunk_input, ScaledSigma):
        # Evaluate each branch net separately
        branch_outputs = [net(branch_input) for net in self.branch_nets]
        #print(jnp.shape(branch_outputs[0]))
        # shape: [(batch_size, 1), ..., (batch_size, 1)]
        B = jnp.concatenate(branch_outputs, axis=1)  # shape: (batch_size, p)
        #print(jnp.shape(branch))

        T = self.trunk_net(trunk_input)  # shape: (batch_size, p)

        # Inner product over the p dimension
        #result = jnp.sum(branch_vec * trunk_vec, axis=1, keepdims=True)  # shape: (batch_size, 1)
        return jnp.matmul(T, jnp.matmul(ScaledSigma, B.T)), T, B

### DONT and DONTsvd

def get_Tparams(direc, w, llw, lastlayer):
    files = os.listdir(direc)
    b = {}
    #print(files)
    for n in files:
        if "test" not in n and "train" not in n:
            if lastlayer in n:
                tmp_w = llw
            else:
                tmp_w = w

            bias = jnp.array(np.loadtxt(direc+"/"+n, max_rows=tmp_w))
            kern0 = np.loadtxt(direc+"/"+n, skiprows=tmp_w+2)
            
            if len(np.shape(kern0)) == 1:
                kern = jnp.zeros((1, len(kern0)))
                kern = kern.at[0,:].set(kern0)
            else:
                kern = jnp.array(kern0)

            #print(n, jnp.shape(kern), jnp.shape(bias))    

            tmp = n.split("_")
            layer_name = tmp[2]+"_"+tmp[3].split(".")[0]

            b[layer_name] = {"kernel": kern, "bias": bias}

    p = {"branch_net": b}
    params = {"params": p}
    return params

def get_Tmodel(direc, last_layer, d, w, llw, F):
    model = TDeepONet(F, d, w, llw)
    params = get_Tparams(direc, w, llw, last_layer)
    return model, params

class TDeepONet(nn.Module):
    nb: int  # Branch network input size
    d: int   # Number of layers (including output layer)
    w: int   # Number of neurons per layer
    llw: int

    def setup(self):
        """ Initialize branch and trunk networks with d layers of w neurons each. """
        self.branch_net = self._build_mlp(self.nb)
        #self.trunk_net = self._build_mlp(self.nt)

    def _build_mlp(self, input_dim):
        """ Helper function to create an MLP with d layers of w neurons each. """
        layers = [nn.Dense(self.w), nn.gelu]  # First hidden layer
        for _ in range(self.d - 1):  # Additional hidden layers
            layers.append(nn.Dense(self.w))
            layers.append(nn.gelu)
        layers.append(nn.Dense(self.llw))  # Final output layer (w neurons)
        return nn.Sequential(layers)

    def __call__(self, p, T, ScaledSigma):
        """ Compute DeepONet output as a dot product of branch and trunk outputs. """
        B = self.branch_net(p)  # Shape: (batch_size, w)
        #return jnp.sum(B * T, axis=-1, keepdims=True), T  # Shape: (batch_size, 1)
        G = jnp.matmul(T, jnp.matmul(ScaledSigma, B.T)) #jnp.sum(B * T, axis=-1, keepdims=True)
        return G, T, B

@jax.jit
def Tall_losses(params, state, p, T, G_true, ScaledSigma, scaleB):
    G_pred, T, B = state.apply_fn(params, p, T, ScaledSigma)

    L_data = jnp.mean((G_pred - G_true) ** 2)

    gram_matrixB = jnp.matmul(B.T, jnp.matmul(B, ScaledSigma))
    L_orthoB = jnp.mean((gram_matrixB - scaleB * ScaledSigma)**2)

    return L_data, L_orthoB

@jax.jit
def Ttrain_step(state, p, T, G_true, alphaB, ScaledSigma, scaleB):
    """ Compute loss, gradients, and update parameters. """
    def loss_fn(params):
        L_data, L_orthoB = Tall_losses(params, state, p, T, G_true, ScaledSigma, scaleB)

        # Total loss
        return L_data + alphaB * L_orthoB


    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state, updates = state.apply_gradients_with_updates(grads=grads)
    return state, loss, updates

@jax.jit
def Tall_loss_components(params, state, p, T, VT, ScaledSigma):
    G_pred, T, B = state.apply_fn(params, p, T, ScaledSigma)
    return jnp.sum( (B-VT.T)**2, axis=0 )

@jax.jit
def TEall_losses(params, state, p, T, truesigma, VT, exponent, ScaledSigma, scaleB):
    G_pred, T, B = state.apply_fn(params, p, T, ScaledSigma)

    L_data = jnp.mean((jnp.matmul(jnp.diag(truesigma**(1+exponent)), VT) - jnp.matmul(jnp.diag(truesigma**exponent), jnp.matmul(ScaledSigma, B.T)))** 2)

    gram_matrixB = jnp.matmul(B.T, jnp.matmul(B, ScaledSigma))
    L_orthoB = jnp.mean((gram_matrixB - scaleB * ScaledSigma)**2)

    return L_data, L_orthoB

@jax.jit
def TEtrain_step(state, p, T, truesigma, VT, exponent, alphaB, ScaledSigma, scaleB):
    """ Compute loss, gradients, and update parameters. """
    def loss_fn(params):
        L_data, L_orthoB = TEall_losses(params, state, p, T, truesigma, VT, exponent, ScaledSigma, scaleB)

        # Total loss
        return L_data + alphaB * L_orthoB


    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    #state = state.apply_gradients(grads=grads)
    state, updates = state.apply_gradients_with_updates(grads=grads)
    return state, loss, updates

### stacked DONT 

def get_TSTparams(direc, depth, w):
    files = os.listdir(direc)
    lastlayer = "Dense_"+str(depth)
    branch_nets = {}
    for j in range(llw):
        bn = {}
        for n in files:
            if "branch_nets_"+str(j)+"_Dense" in n:
                if lastlayer in n:
                    tmp_w = 1
                else:
                    tmp_w = w
                bias0 = jnp.array(np.loadtxt(direc+"/"+n, max_rows=tmp_w))
                kern0 = np.loadtxt(direc+"/"+n, skiprows=tmp_w+2)
                if len(np.shape(kern0)) == 1 and lastlayer in n:
                    kern = jnp.zeros((len(kern0), 1))
                    kern = kern.at[:,0].set(kern0)
                elif len(np.shape(kern0)) == 1 and lastlayer not in n:
                    kern = jnp.zeros((1, len(kern0)))
                    kern = kern.at[0,:].set(kern0)
                else:
                    kern = jnp.array(kern0)

                if len(np.shape(bias0)) == 0: # in (float,int):
                    bias = jnp.zeros((1,))
                    bias = bias.at[0].set(bias0)
                else:
                    bias = bias0
                tmp = n.split("_")
                layer_name = "Dense_"+tmp[4].split(".")[0]

                bn[layer_name] = {"kernel": kern, "bias": bias}
        branch_nets["branch_nets_"+str(j)] = bn

    params = {"params": branch_nets}
    return params

def get_TSTmodel(direc, last_layer, d, w, llw, F):
    model = StackedTDeepONet(F, d, w, llw)
    params = get_TSTparams(direc, d, w)
    return model, params

class StackedTDeepONet(nn.Module):
    branch_input_dim: int
    depth_br: int
    hidden_dim_br: int
    p: int  # number of branch subnets

    def setup(self):
        self.branch_nets = [
            FeedforwardNet(
                input_dim=self.branch_input_dim,
                hidden_dim=self.hidden_dim_br,
                depth=self.depth_br
            )
            for _ in range(self.p)
        ]

    def __call__(self, branch_input, T, ScaledSigma):
        # Evaluate each branch net separately
        branch_outputs = [net(branch_input) for net in self.branch_nets]
        #print(jnp.shape(branch_outputs[0]))
        # shape: [(batch_size, 1), ..., (batch_size, 1)]
        B = jnp.concatenate(branch_outputs, axis=1)  # shape: (batch_size, p)
        #print(jnp.shape(branch))

        # Inner product over the p dimension
        #result = jnp.sum(branch_vec * trunk_vec, axis=1, keepdims=True)  # shape: (batch_size, 1)
        return jnp.matmul(T, jnp.matmul(ScaledSigma, B.T)), T, B

## updates

def all_mult_updates(Nepochs, state, model, 
                ptrain, rtrain, utrain, ptest, rtest, utest, 
                optimizer, alphaT, alphaB, lr_schedule, 
                param_base, ScaledSigma, T, truesigma, 
                VT_train, VT_test, exponent, llw, whichT, leftsingvec):

    print(param_base)
    n_train, m_train = np.shape(utrain)
    n_test,  m_test  = np.shape(utest)
    scaleT_train = n_train**2
    scaleT_test  = n_test**2
    scaleB_train = m_train**2
    scaleB_test  = m_test**2

    utrain_norm  = jnp.mean(utrain**2)
    utest_norm   = jnp.mean(utest**2)
    utrainE_norm = jnp.mean(jnp.matmul(jnp.diag(truesigma**(1+exponent)), VT_train)** 2)
    utestE_norm  = jnp.mean(jnp.matmul(jnp.diag(truesigma**(1+exponent)), VT_test)** 2)

    if whichT < 0:
        opt_state, _, update = train_step(state, ptrain, rtrain, utrain, 
                                          alphaT, alphaB, ScaledSigma, scaleT_train, scaleB_train)
    else: #if whichT == 0:
        opt_state, _, update = TEtrain_step(state, ptrain, T, truesigma, VT_train, exponent, 
                                            alphaB, ScaledSigma, scaleB_train)
    #else:
    #    if abs(exponent) > 1e-6:
    #        print("T \neq U not implemented with exp != 0")
    #        return False
    #    opt_state, _, update = Ttrain_step(state, ptrain, T, utrain, alphaB, ScaledSigma, scaleB_train)
        
    

    opt_params = opt_state.params
    opt_update = update
    last_params = opt_state.params
    last_update = update
    after_opt_params = opt_state.params
    after_opt_update = update
    
    min_mode_losses_train = 1e8*np.ones(llw)
    min_mode_losses_test  = 1e8*np.ones(llw)

    minloss = 1e8
    errors_data_train   = jnp.zeros(Nepochs)
    errorsE_data_train  = jnp.zeros(Nepochs)
    errors_orthoT_train = jnp.zeros(Nepochs)
    errors_orthoB_train = jnp.zeros(Nepochs)
    errors_data_test    = jnp.zeros(Nepochs)
    errorsE_data_test   = jnp.zeros(Nepochs)
    errors_orthoT_test  = jnp.zeros(Nepochs)
    errors_orthoB_test  = jnp.zeros(Nepochs)
    last_opt_it = 0
    for i in range(Nepochs):
        # gradient step
        if whichT < 0:
            state, _, update = train_step(state, ptrain, rtrain, utrain, alphaT, alphaB, ScaledSigma, scaleT_train, scaleB_train)
            LE_data_train = 0
            LE_data_test  = 0
            
            # get different training loss contributions
            L_data_train, L_orthoT_train, L_orthoB_train = all_losses(state.params, state, ptrain, rtrain, utrain, 
                                                                    ScaledSigma, scaleT_train, scaleB_train)
            # get different test loss contributions
            L_data_test,  L_orthoT_test,  L_orthoB_test  = all_losses(state.params, state, ptest, rtest, utest, 
                                                                    ScaledSigma, scaleT_test, scaleB_test)
            
            mode_losses_train = all_loss_components(state.params, state, ptrain, rtrain, VT_train, ScaledSigma, truesigma, leftsingvec)
            mode_losses_test  = all_loss_components(state.params, state, ptest, rtest, VT_test, ScaledSigma, truesigma, leftsingvec)


        else:
            state, _, update = TEtrain_step(state, ptrain, T, truesigma, VT_train, exponent, 
                                            alphaB, ScaledSigma, scaleB_train)
            LE_data_train, _ = TEall_losses(state.params, state, ptrain, T, truesigma, VT_train, exponent, 
                                                                    ScaledSigma, scaleB_train)
            LE_data_test,  _  = TEall_losses(state.params, state, ptest, T, truesigma, VT_test, exponent, 
                                                                    ScaledSigma, scaleB_test)

            L_data_train, L_orthoB_train = Tall_losses(state.params, state, ptrain, T, utrain, 
                                                                    ScaledSigma, scaleB_train)
            L_data_test,  L_orthoB_test  = Tall_losses(state.params, state, ptest, T, utest, 
                                                                    ScaledSigma, scaleB_test)

            mode_losses_train = Tall_loss_components(state.params, state, ptrain, T, VT_train, ScaledSigma)
            mode_losses_test  = Tall_loss_components(state.params, state, ptest,  T, VT_test, ScaledSigma)

            L_orthoT_train = 0
            L_orthoT_test  = 0
            
        min_mode_losses_train = np.min(np.vstack((mode_losses_train,min_mode_losses_train)), axis=0)
        min_mode_losses_test  = np.min(np.vstack((mode_losses_test, min_mode_losses_test)), axis=0)

        
        # save errors in arrays
        errors_data_train   = errors_data_train.at[i].set(L_data_train/utrain_norm)
        errorsE_data_train  = errorsE_data_train.at[i].set(LE_data_train/utrainE_norm)
        errors_orthoT_train = errors_orthoB_train.at[i].set(L_orthoT_train)
        errors_orthoB_train = errors_orthoB_train.at[i].set(L_orthoB_train)
        errors_data_test    = errors_data_test.at[i].set(L_data_test/utest_norm)
        errorsE_data_test   = errorsE_data_test.at[i].set(LE_data_test/utestE_norm)
        errors_orthoT_test  = errors_orthoB_test.at[i].set(L_orthoT_test)
        errors_orthoB_test  = errors_orthoB_test.at[i].set(L_orthoB_test)

        if L_data_test < minloss:
            opt_params = state.params.copy()
            last_opt_it = i
            minloss = L_data_test
            opt_update = update
        
        if last_opt_it+1 == i:
            after_opt_params = state.params.copy()
            after_opt_update = update
    

        # 0: id, 1: log(rel train data), 2: 0, 3: log(rel train oB)
        #        4: log(rel test  data), 5: 0, 6: log(rel test  oB)
        #        7: log(rel minloss test data), 8: log LE data train, 9: log LE data test

        # print errors into log-file
        if i%50 == 0: #(i+1)%100 == 0 or (i+1) in [0, 10, 50]:
            # print errors to log file
            file = open(param_base+"log.txt", "a")
            tmp  = str(i)+" "
            for error in [errors_data_train[i], errors_orthoT_train[i], errors_orthoB_train[i], 
                          errors_data_test[i], errors_orthoT_test[i], errors_orthoB_test[i], 
                          minloss/utest_norm, errorsE_data_train[i], errorsE_data_test[i]]:
                tmp += str(np.log10(error))+" "
            tmp += str(last_opt_it)+" "
            file.write(tmp+"\n")
            file.close()

            # print errors to screen
            #if (i+1)%int(Nepochs/100) == 0 or (i+1) in [0, 10, 50]:
            print(tmp)
            print(errorsE_data_train[i], errorsE_data_test[i])

            # print mode errors to log file
            file = open(param_base+"log_modes.txt", "a")
            file.write(str(i)+" "+stringify(mode_losses_train)+stringify(min_mode_losses_train)+stringify(mode_losses_test)+stringify(min_mode_losses_test)+"\n")    
            file.close()

            
        # save (so-far) optimal and current parameters
        if i%100 == 0: #(i+1)%1_000 == 0 or (i+1) in [0, 100, 500]:
            save_checkpoint(state.params,     update,           path=param_base+str(i+1)+"cur_chp")
            save_checkpoint(opt_params,       opt_update,       path=param_base+str(i+1)+"opt_chp")
            save_checkpoint(after_opt_params, after_opt_update, path=param_base+str(i+1)+"aop_chp")
            save_checkpoint(last_params,      last_update,      path=param_base+str(i+1)+"las_chp")

            '''
            os.mkdir(param_base+str(i+1)+"opt")
            save_params(opt_params, param_base+str(i+1)+"opt/")

            os.mkdir(param_base+str(i+1)+"cur")
            save_params(state.params, param_base+str(i+1)+"cur/")            

            os.mkdir(param_base+str(i+1)+"aop")
            save_params(after_opt_params, param_base+str(i+1)+"aop/")            


            os.mkdir(param_base+str(i+1)+"upd_opt")
            save_params(opt_update, param_base+str(i+1)+"upd_opt/")

            os.mkdir(param_base+str(i+1)+"upd_cur")
            save_params(update, param_base+str(i+1)+"upd_cur/")      

            os.mkdir(param_base+str(i+1)+"upd_aop")
            save_params(after_opt_update, param_base+str(i+1)+"upd_aop/")      
            '''

        last_update = update.copy()
        last_params = state.params.copy()

    #os.mkdir(param_base+"END")
    #save_params(opt_params, param_base+"END/")
    
    return state, opt_params, errors_data_train, errors_orthoT_train, errors_orthoB_train, errors_data_test, errors_orthoT_test, errors_orthoB_test




