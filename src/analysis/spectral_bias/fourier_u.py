import numpy as np 
import matplotlib.pyplot as plt 
import sys

bid = int(sys.argv[1])

base = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/data"

#xx = np.loadtxt(base+"/"+"kdvnx401_dt0.0001_nc5_m5000_P.txt")

def dic(tag):
    if tag == 0:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1000", 20
    if tag == 1:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1999", 20
    if tag == 2:
        return "kdvnx401_dt0.0001_nc5_m5000", "10", 100
    if tag == 3:
        return "kdvnx401_dt0.0001_nc5_m5000", "1999", 100
    if tag == 4:
        return "kdvnx401_dt0.0001_nc5_m5000", "5999", 100
    if tag == 5:
        return "kdvnx401_dt0.0001_nc5_m5000", "9999", 100    
    if tag == 6:
        return "burgers_dt0.0001_nc10_m3800", "100", 100
    if tag == 7:
        return "burgers_dt0.0001_nc10_m3800", "999", 100




name, uend, num_modes = dic(bid)

num_samples = int(name.split("_m")[-1])


rinp = np.loadtxt(base+"/"+name+"_R.txt")
M = np.shape(rinp)[0]

xx = np.zeros((M, 1))
xx[:,0] = np.linspace(0, 1, M)
yy = np.loadtxt(base+"/"+name+"_"+uend+"_U.txt")
uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
vv = vvhh.T
vv = uu

def count_switches(x):
    n = len(x)
    k = 0
    for i in range(1, n):
        if np.sign(x[i]) != np.sign(x[i-1]):
            k +=1 
    return k 

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

num_ks = 60
ks = np.linspace(1e-3, 1e2, 300) #2**(np.linspace(-10, 10, num_ks))
k_norms = ks


#modeids = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26] 
modeids = np.array(np.arange(0, num_modes), dtype=np.int64)

fig, axs = plt.subplots(1,3) #len(modeids))


num_samples = [M] #100, 1000, 2000, 3000, 4000, 4200, 4400, 4600, 4800, 4999]
#range(0, 100) #[0, 1, 2, 3, 4, 5]
transforms = doit(num_samples, modeids, ks)

maxvs = []
avgvs = []
switches = []

for modeid_index, modeid in enumerate(modeids):
    colval = modeid_index / len(modeids)
    axs[0].plot(k_norms, transforms[modeid_index,:,0], '.-', color=(0.1+0.9*colval, 0.0, 1.0-colval)) #, label=str(modeid)+" "+str(0))

    axs[1].plot(0*vv[:,modeid_index] - modeid_index, '--', color='gray')
    axs[1].plot(0.5*vv[:,modeid_index]/np.max(np.abs(vv[:,modeid_index])) - modeid_index)

    vec = transforms[modeid_index,:,0]
    maxvs.append(np.argmax(np.abs(vec)))
    avgvs.append(np.sum(np.arange(0, len(vec)) * np.abs(vec) ) / len(vec))
    switches.append(count_switches(vv[:,modeid_index]))

maxvs = np.array(maxvs)
avgvs = np.array(avgvs)

axs[2].plot(modeids, maxvs/np.max(maxvs), '.-')
axs[2].plot(modeids, avgvs/np.max(avgvs), '.-')
ax3 = axs[2].twinx()
ax3.plot(modeids, switches, '.-', color="green")

#axs[0].set_xscale("log")
#axs[0].legend()

fig.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/fourier/u_"+name[:3]+"_"+uend+".pdf")

#plt.show()



