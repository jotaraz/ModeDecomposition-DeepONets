# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 18:32:53 2025

@author: J_Taraz
"""

from ... import don_code


def format(d):
    s = ""
    for k,v in d.items():
        s += str(k)+": "+str(round(don_code.np.log10(v), 4))+", "
    return s

bid = int(don_code.sys.argv[1])



whTs = [0, 1, 2, 5, 7, 9, 10, 11]

whTs = [1, 10, 2, 11]

llws = [10, 30, 50, 70, 100] #, 150, 190, 200]

batch_name, uendtag, _ = don_code.dic(bid)

num_data = 1000
nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, uendtag, num_data)
uu_train, ss_train, vh_train = don_code.jnp.linalg.svd(utrain, full_matrices=False)


for whichT in whTs:
    errors = {}
    kappas = {}
    sigma1 = {}
    sigmaN = {}
    diag1  = {}
    diagN  = {}
    orthoe = {}
    for llw in llws:
        T = don_code.get_fixed_trunk(whichT, llw, rtrain[:,0], batch_name, uu_train)
        
        proj = don_code.np.eye(don_code.np.shape(T)[0]) - don_code.np.matmul(T, don_code.np.linalg.pinv(T))
        proj_error = don_code.np.sum( don_code.np.matmul(proj, utrain)**2 ) / don_code.np.sum(utrain**2)
        errors[llw] = proj_error

        TTT = don_code.np.matmul(T.T, T)
        orthoe[llw] = don_code.np.sum( (TTT-don_code.np.eye(llw))**2 )
        #print(TTT[0,0], TTT[llw-1, llw-1])
        diag1[llw] = TTT[0,0]
        diagN[llw] = TTT[llw-1, llw-1]

        SigmaT = don_code.np.linalg.svd(T, compute_uv=False)
        #print(don_code.np.max(don_code.np.abs(T)))
        kappas[llw] = SigmaT[0]/SigmaT[llw-1]
        #sigma1[llw] = SigmaT[0]
        #sigmaN[llw] = SigmaT[llw-1]

    print(don_code.trunk_names[whichT])
    print("log errors", format(errors))
    #print("log sigma1", format(sigma1))
    #print("log sigmaN", format(sigmaN))
    print("log kappas", format(kappas))
    print("log diag1 ", format(diag1))
    print("log diagN ", format(diagN))
    print("log orthoe", format(orthoe))
    print("")

