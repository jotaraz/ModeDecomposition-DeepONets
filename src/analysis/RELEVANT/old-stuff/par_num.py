## 1708thesisrelevant (kinda)
# used to compute the numbers of parameters in stacked vs unstacked SVDONets


import sys 
import numpy as np 
import jax.numpy as jnp
import jax

from ... import don_code

bid = int(sys.argv[1])


def get_Ntheta_unstacked(w, m, d, N):
    return (d-1)*w**2 + (m+d+N)*w + N 

def get_Ntheta_stacked(w, m, d, N):
    #return N*(d-1)*w**2 + N*(m+d+1)*w + N 
    return N*get_Ntheta_unstacked(w, m, d, 1)


def flatten_pars(params):
    #vals, _ = jax.tree.flatten(params) #np.hstack([np.concatenate((params['params']['branch_net'][key]['bias'].flatten(), params['params']['branch_net'][key]['kernel'].flatten())) for key in params['params']['branch_net'].keys()])
    leaves, _ = jax.tree_util.tree_flatten(params)

    # Step 2: Convert to NumPy and flatten each leaf
    flat_leaves = [np.ravel(np.asarray(leaf)) for leaf in leaves]

    # Step 3: Concatenate into one array
    return np.concatenate(flat_leaves)
    #return np.array(vals)

def get_num_pars(params):
    tmp = flatten_pars(params)
    return np.shape(tmp)[0]

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

llw = 0
w_stacked = []
w_unstacked = []

if bid < 2:
    w_stacked = [36, 58, 89] # 65 40, 45]
    w_unstacked = [222, 332, 495]
    llw = 20
    d = 5

elif bid < 6:
    w_stacked = [13, 24, 42]
    w_unstacked = [42, 220, 335, 495]
    llw = 50
    d = 5

elif bid < 8:
    w_stacked   = [29, 43, 65] #12, 13, 24, 40, 41, 42, 55]
    w_unstacked = [43, 237, 337, 494] #200, 350, 400, 500, 600]
    llw = 50
    d = 5

key = jax.random.PRNGKey(0)

for bid in [bid]: #, 3, 4, 5]:
    batch_name, uendtag, _ = dic(bid)
    num_data = 1000

    nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, uendtag, num_data)

    print("nt", nt, "nb", nb, np.shape(rtrain), np.shape(ptrain), np.shape(utrain))


    llw = min(np.shape(utrain)[0], int(np.shape(utrain)[1]), llw)


    lambdas = np.zeros(llw)

    uu_train, ss_train, vh_train = jnp.linalg.svd(utrain, full_matrices=False)
    T = uu_train[:,:llw]
    n_train, m_train = np.shape(utrain)
    #VT_train = vh_train[:llw, :]
    #ts = ss_train[:llw] 
    #VT_test  = jnp.matmul(jnp.diag(1/ts), jnp.matmul(uu_train[:, :llw].T, utest))

    ScaledSigma = jnp.diag(np.ones(llw)) * ss_train[0]


    for w in w_stacked:
        model = don_code.StackedTDeepONet(nb, d, w, llw)
        params = model.init(key, ptrain, T, ScaledSigma)
        nsta_emp = get_num_pars(params)
        nsta_theo = get_Ntheta_stacked(w, nb, d, llw)
        print("Stacked  ", w, nsta_emp, nsta_theo, nsta_emp-nsta_theo)

    for w in w_unstacked:
        model = don_code.TDeepONet(nb, d, w, llw)
        params = model.init(key, ptrain, T, ScaledSigma)
        print("Unstacked", w, get_num_pars(params), get_Ntheta_unstacked(w, nb, d, llw), get_num_pars(params)-get_Ntheta_unstacked(w, nb, d, llw))


## here you find the number of parameters for the widths above
"""
bid = 0
Stacked   36 252740 252740 0
Stacked   58 509260 509260 0
Stacked   89 1002160 1002160 0
Unstacked 222 247328 247328 0
Unstacked 332 515948 515948 0
Unstacked 495 1091990 1091990 0

bid = 3
Stacked   13 298400 298400 0
Stacked   24 603650 603650 0
Stacked   42 1207550 1207550 0
Unstacked 220 293970 293970 0
Unstacked 335 601710 601710 0
Unstacked 495 1205870 1205870 0

bid = 6
Stacked   29 249450 249450 0
Stacked   43 490250 490250 0
Stacked   65 1027050 1027050 0
Unstacked 43 11961 11961 0
Unstacked 237 249611 249611 0
Unstacked 337 489711 489711 0
Unstacked 494 1028064 1028064 0
"""




