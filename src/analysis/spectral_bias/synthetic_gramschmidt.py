import numpy as np 
import matplotlib.pyplot as plt 
import sys 

#freqincscale = float(sys.argv[1])
#lowerbound = float(sys.argv[1])
#upperbound = 
#m        = int(sys.argv[2])
#sigmaexp = float(sys.argv[3])
#rf_x     = float(sys.argv[4])
#rf_k  = float(sys.argv[3])
#n = 100
#m = 100
# fix M = n (number of eval coords in input and output is equal)
#N = 20


def bettersign(x, y):
    err1 = np.sum( (x - (+y))**2 )
    err2 = np.sum( (x - (-y))**2 )
    if err1 < err2:
        return +y
    else:
        return -y
    
def lambda_j(x, j):
    return np.sin(2*np.pi * (j+1) * x)

def rho_j(x, kj):
    y = np.sin(2*np.pi * np.sum(kj*x))
    return y

def generate_X(n, m, rf_x):
    X = np.zeros((n, m))
    for i in range(m):
        X[:,i] = np.random.rand() * (np.ones(n)  + rf_x*(np.random.rand(n)*2 - 1.0))
    
    return X

def generate_Xusvh_varfreq(N, n, m, freqincscale, sigmaexp, rf_x, norm0scale=1.0, innerprod_threshold=0.05, verbose=False):
    X = generate_X(n, m, rf_x)
    
    #X = np.random.randn(n,m)
    compare_to_true_svd  = True
    check_orthogonality  = True
    explicit_gramschmidt = True
    doplot = False
    
    freqs, Phi, sigmas, VT, A = generate_usvh_varfreq(
                                                N, X, freqincscale, sigmaexp, 
                                                norm0scale=norm0scale,
                                                innerprod_threshold=innerprod_threshold, 
                                                verbose=verbose, 
                                                explicit_gramschmidt=explicit_gramschmidt,
                                                check_orthogonality=check_orthogonality,
                                                compare_to_true_svd=compare_to_true_svd,
                                                doplot=doplot)

    
    return freqs, X, Phi, sigmas, VT, A
    

def generate_usvh_varfreq(
        N, X, freqincscale, sigmaexp, 
        norm0scale=1.0,
        innerprod_threshold=0.05, 
        verbose=False, 
        explicit_gramschmidt=True,
        check_orthogonality=False,
        compare_to_true_svd=False,
        doplot=False):
    n,m = X.shape
    freqincscaleabs = abs(freqincscale)
    
    
    rs = np.linspace(0, 1, n)
    #X = 2.0*np.random.rand(n,m)-1.0
    
    num_direcs = 200
    num_norms  = 1
    

    ## set Phi
    Phi = np.zeros((n, N))
    for j in range(N):
        tmp = lambda_j(rs, j)    
        Phi[:,j] = tmp / np.sqrt(np.sum(tmp**2))


    ## set ks and V
    V = np.zeros((m, N))
    freqs = np.zeros(N)
    for j in range(N):
        found = False
        kdirs = []
        for i in range(num_direcs):
            tmp = np.ones(n) + np.random.randn(n) #- 1.0
            #tmp = np.sin(2*np.pi*(j+i+1)*rs)
            #tmp = np.ones(n) + rf_k*(np.random.rand(n)*2 - 1.0)
            kdirs.append(tmp/np.sqrt(np.sum(tmp**2)))
        
        knorms = [norm0scale*np.exp(freqincscaleabs*(j + k/(2*num_norms))) for k in range(num_norms)]
        #knorms = [np.exp(freqincscale*(j+1 + k/150)) for k in range(50)]
        minmaxip = 1.0
        #for ikn in range(num_norms): #, knorm in enumerate(knorms):
        for ikn, knorm in enumerate(knorms):
            #knorm = norm0scale * np.exp(freqincscaleabs * (j + 0.5*np.random.rand()))
    
            if not found:
                for ikd, kdir in enumerate(kdirs):
                    if not found:
                        vtmp = np.zeros(m)
                        for i in range(m):
                            vtmp[i] = rho_j(X[:,i], knorm*kdir)    
                        vtmp /= np.sqrt(np.sum(vtmp**2))
                        
                        ## compare to previous v-vectors
                        maxip = 0.0
                        for jj in range(j):
                            ip = np.sum(vtmp * V[:,jj])
                            maxip = max(maxip, np.abs(ip))
                        #if verbose:
                        #    print(j, knorm, ikd, maxip)
                        if maxip < minmaxip:
                            minmaxip = maxip
                        if maxip < innerprod_threshold:
                            V[:,j] = vtmp
                            if verbose:
                                print(j, "Yes", "ikd", ikd, "ikn", ikn)
                            found = True
                            freqs[j] = knorm
                if verbose:
                    print(ikn, minmaxip)
        if verbose:
            print(j, "minmaxip", minmaxip)
        if not found:
            if verbose:
                print(j, "no new direction found")
            return False
    
    if explicit_gramschmidt:
        Vnew = np.zeros((m,N))
        for j in range(N):
            tmp = V[:,j]
            
            for i in range(j):
                a = np.sum(tmp * Vnew[:,i])
                tmp -= a * Vnew[:,i]
            
            Vnew[:,j] = tmp / np.sqrt(np.sum(tmp**2))
        
        V = Vnew.copy()
        
    if freqincscale < 0:
        V = np.flip(V, axis=1)
        freqs = np.flip(freqs)

    sigmas = np.exp(sigmaexp * np.arange(N))
    Sigmas = np.diag(sigmas)
    A = np.real( np.matmul( np.matmul(Phi, Sigmas), V.T ) )

    if doplot:
        fig, axs = plt.subplots(1, 4)
    if check_orthogonality:
        ## ortho check Phi
        print("Phi^T Phi")
        PhiTPhi = np.matmul(Phi.T, Phi)
        print("Phi^TPhi-I", np.max(np.abs(PhiTPhi - np.eye(N))))
        
        for i in range(N):
            Phi[:,i] /= np.sqrt(PhiTPhi[i,i])
        
        if doplot:
            axs[0].imshow( PhiTPhi )
        
        ## ortho check V
        print("V^T V")
        VTV = np.matmul(V.T, V)
        print("V^TV-I", np.max(np.abs(VTV - np.eye(N))))
        VTV_abs = VTV**2
        print("min VTV", np.min(VTV_abs))
        if doplot:
            axs[1].imshow( VTV_abs )
        mask_diag = np.eye(N)
        mask_offdiag = np.ones((N,N)) - mask_diag
        print("max diag", np.max(np.diag(VTV_abs)), "min diag", np.min(np.diag(VTV_abs)), "diag min / offdiag max : ", np.min( np.diag(VTV_abs) ) / np.max( VTV_abs * mask_offdiag ) )
    
    if compare_to_true_svd:
        u, s, vh = np.linalg.svd(A)
        
        threshold = 1e-14
        N_emp = np.where(s > threshold)[0][-1] + 1
        
        print("N", N, "N_emp", N_emp)
        
        max_u_diff = 0
        max_v_diff = 0
        for i in range(N):
            if sigmaexp < 0:
                j = i
            else:
                j = N-i-1
            max_u_diff = max(max_u_diff, np.sum(Phi[:,j] -  bettersign(Phi[:,j], u[:,i]) )**2)    
            max_v_diff = max(max_v_diff, np.sum(V[:,j] -  bettersign(V[:,j], vh[i,:]) )**2)
        
            if doplot:
                axs[2].plot(Phi[:,j]-i, color="blue")
                axs[2].plot(bettersign(Phi[:,j],u[:,i])-i, '--', color="red")
                axs[3].plot(V[:,j]-i, color="blue")
                axs[3].plot(bettersign(V[:,j],vh[i,:])-i, '--', color="red")
            
        print(s[:N_emp])
        if sigmaexp < 0:
            print("s diff", np.max(np.abs(s[:N_emp] - sigmas)))
        else:
            print("s diff", np.max(np.abs(np.flip(s[:N_emp]) - sigmas)))
            
        print("u diff", max_u_diff)
        print("v diff", max_v_diff)

    return freqs, Phi, sigmas, V.T, A
    


