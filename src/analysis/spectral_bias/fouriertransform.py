import numpy as np 
import matplotlib.pyplot as plt 
import sys

uend = sys.argv[1]

base = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/data"

xx = np.loadtxt(base+"/"+"kdvnx401_dt0.0001_nc5_m5000_P.txt")
yy = np.loadtxt(base+"/"+"kdvnx401_dt0.0001_nc5_m5000_"+uend+"_U.txt")
uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
vv = vvhh.T

def doit(num_samples, modeids, ks):
    print(".")
    transforms = np.zeros((len(modeids), len(ks), len(num_samples)))

    #print(np.shape(x), np.shape(y), np.shape(vvhh))

    for modeid_index, modeid in enumerate(modeids):
        for num_id,m in enumerate(num_samples):
            #XX = x[:m,:]
            #YY = y[:m,:]
            for i in range(m):
                for j,k in enumerate(ks):
                    ytmp = vv[i,modeid]
                    xtmp = xx[i,:]
                    transforms[modeid_index,j,num_id] += ytmp * np.exp(2*np.pi * 1j*np.sum(k*xtmp)) / m

    return transforms


# one k, multiple m
'''
k0 = np.random.rand(401)*2.0 - 1.0    

num_samples = [100, 1000, 2000, 3000, 4000, 4200, 4400, 4600, 4800, 4999]
modeids = [0, 1, 2, 3, 4, 5]
transforms = doit(num_samples, modeids, [k0])
for modeid_index, modeid in enumerate(modeids):
    plt.plot(num_samples, transforms[modeid_index,0,:], '.-', label=str(modeid))
'''

# multiple k, one m

num_ks = 70
ks = []
k_norms = []

for i in range(num_ks):
    x = np.random.rand(401)*2.0 - 1.0
    x = x/np.sqrt(np.sum(x**2))
    ktmp = (i+1)/num_ks*1000 * x
    ks.append(ktmp)   
    k_norms.append(np.sqrt(np.sum(ktmp**2)))

xax = np.linspace(0, 2*np.pi, 401)

modeids = [0, 2, 4, 6, 8] 

vecs = np.zeros((len(modeids), len(k_norms)))

fig, axs = plt.subplots(1) #len(modeids))

for jjj in [1, 2, 3, 4, 5]:

    ks = []
    k_norms = []

    for j in range(jjj,jjj+1):
        ytmp = np.sin(j*xax)
        ytmp = ytmp / np.sqrt(np.sum(ytmp**2))
        for i in range(num_ks):
            fac = 2**((i+1)/5 - 10)
            ks.append(ytmp * fac)
            k_norms.append(fac) #1, 2, 3, 4, 5]

    num_samples = [4999] #100, 1000, 2000, 3000, 4000, 4200, 4400, 4600, 4800, 4999]
    #range(0, 100) #[0, 1, 2, 3, 4, 5]
    transforms = doit(num_samples, modeids, ks)

    for modeid_index, modeid in enumerate(modeids):
        vecs[modeid_index,:] += np.abs(transforms[modeid_index,:,0])**2
        #axs[modeid_index].plot(k_norms, transforms[modeid_index,:,0], '.-', label=str(modeid)+" "+str(jjj))

        #axs[modeid_index].set_xscale("log")
        #axs[modeid_index].legend()

for modeid_index, modeid in enumerate(modeids):
    axs.plot(k_norms, np.zeros(len(k_norms)), '--', color="grey")
    axs.plot(k_norms, vecs[modeid_index,:]/len(modeids), 'o-', label=str(modeid)+" "+str(jjj))
    axs.set_xscale("log")
    axs.legend()
        

fig.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/fourier/v_kdv_"+uend+".pdf")

plt.show()

