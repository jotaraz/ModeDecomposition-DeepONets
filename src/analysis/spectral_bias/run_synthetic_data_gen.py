# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:39:05 2025

@author: J_Taraz
"""

## burgers_dt0.0001_nc10_m3800_100_U


from synthetic_gramschmidt import *

import os


n = 100
m = 5000
N = 5
rf_x = 0.1



basescales = [0.05, +0.2, +0.1, +0.4, +1.0, +0.05, +0.2, +0.1, +0.4, +1.0] # +0.2, +0.1,  -0.2, -0.1, -0.2, -0.1,  +0.2,  +0.1,  +0.01, +0.01]
sigmaexpss = [-1.0, -1.0, -1.0, -1.0, -1.0, -0.5,  -0.5, -0.5, -0.5, -0.5] # -0.5, -0.5,  -1.0, -1.0, -0.5, -0.5,  -0.01, -0.01, -1.0,  -0.5]

#v4
vtag = "4?"
freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
sigmaexpss    = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]


# write into the repo's dataset directory, i.e. the one don_code.load_dataset reads from
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "datasets")
os.makedirs(data_dir, exist_ok=True)

path = os.path.join(data_dir, "synth")
path += "v"+vtag
path += f"_n{n}_N{N}_m{m}"


X = generate_X(n, m, rf_x)
P = X.T
R = np.linspace(0, 1, n)


np.savetxt(path+"_P.txt", P)
np.savetxt(path+"_R.txt", R)

#for sign in [+1.0, -1.0]:
#    for sigm in [-1.0,-0.5,-0.01]:
#        for fs in [0.05, 0.1, 0.2, 0.4]:
#            basescales.append(sign*fs)
#            sigmaexpss.append(sigm)
        


for i in range(len(freqincscales)):
    basescale  = freqincscales[i]
    sigmaexp   = sigmaexpss[i]
    norm0scale = norm0scales[i]
    print("\n\nfs", basescale, "ss", sigmaexp, "n0", norm0scale, "\n\n")
    _, _, _, _, A = generate_usvh_varfreq(N, X, basescale, sigmaexp, 
                                          norm0scale=norm0scale,
                                          verbose=True, 
                                          compare_to_true_svd=True)
    np.savetxt(path+"_fs"+str(basescale)+"ss"+str(sigmaexp)+"ns"+str(round(norm0scale, 4))+"_U.txt", A)




