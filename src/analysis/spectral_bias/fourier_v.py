import numpy as np 
import matplotlib.pyplot as plt 
import sys

#uend = sys.argv[1]
bid = int(sys.argv[1])
kmode = int(sys.argv[2])
num_modes = int(sys.argv[3])
num_direcs = int(sys.argv[4])

base = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/data"

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




name, uend, _ = dic(bid)

num_samples = int(name.split("_m")[-1])


xx = np.loadtxt(base+"/"+name+"_P.txt")
yy = np.loadtxt(base+"/"+name+"_"+uend+"_U.txt")
uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
vv = vvhh.T

m = np.shape(xx)[0]
M = np.shape(xx)[1]


print("num_samples", num_samples, "M", M)

def norm2(x):
    return np.sum(x**2)

def reconstruct(num_samples, modeids, ks, transforms):
    for num_id,m in enumerate(num_samples):
        for modeid_index, modeid in enumerate(modeids):
            target = vv[:,modeid]
            approx = np.zeros(len(target))
            for ix in range(m):
                xi = xx[ix,:]
                for ik, k in enumerate(ks):
                    approx[ix] += transforms[modeid_index, ik, num_id]*np.exp(-2*np.pi*2j*np.sum(k*xi)) / len(ks)

                approx = approx / np.sqrt(norm2(approx))

            print(modeid, "||t-a||", norm2(target-approx), "||t||", norm2(target), "||a||", norm2(approx), "<t,a>", np.sum(target*approx))



def doit(num_samples, modeids, ks):
    print(".")
    transforms = np.zeros((len(modeids), len(ks), len(num_samples)))

    #print(np.shape(x), np.shape(y), np.shape(vvhh))

    for modeid_index, modeid in enumerate(modeids):
        print(modeid)
        for num_id,m in enumerate(num_samples):
            #XX = x[:m,:]
            #YY = y[:m,:]
            for j,k in enumerate(ks):
                #print(np.shape(xx), np.shape(k))
                tmp = np.matmul(xx, k)
                transforms[modeid_index,j,num_id] = np.mean(vv[:,modeid] * np.exp(2*np.pi*1j*tmp))

            #for i in range(m):
            #    for j,k in enumerate(ks):
            #        ytmp = vv[i,modeid]
            #        xtmp = xx[i,:]
            #        transforms[modeid_index,j,num_id] += ytmp * np.exp(2*np.pi * 1j*np.sum(k*xtmp)) / m

    return transforms


xax = np.linspace(0, 2*np.pi, M)

k_norms = np.linspace(1e-3, 100, 1000) #10**np.linspace(-3, 2, 30)
ks = []
k_norm_ids = []
j = -1
for kn in k_norms:
    tmp = []
    
    if kmode == 0:
        for i in range(1,6):
            ytmp = np.sin(i*xax)
            ytmp = ytmp / np.sqrt(np.sum(ytmp**2))
            ks.append(kn * ytmp)
            j += 1
            tmp.append(j)
    else:    
        for i in range(num_direcs):
            ytmp = 2.0*np.random.rand(len(xax)) - 1.0#np.sin(i*xax)
            ytmp = ytmp / np.sqrt(np.sum(ytmp**2))
            ks.append(kn * ytmp)
            j += 1
            tmp.append(j)

    k_norm_ids.append(tmp)


modeids = range(num_modes) #[0, 2, 4, 6, 8] 

fig, axs = plt.subplots(1,2)

axs[0].plot(k_norms, np.zeros(len(k_norms)), '--', color="gray") # label=str(modeid))


maxvs = np.zeros(len(modeids))
avgvs = np.zeros(len(modeids))

num_samples = [num_samples-1] #100, 1000, 2000, 3000, 4000, 4200, 4400, 4600, 4800, 4999]
#range(0, 100) #[0, 1, 2, 3, 4, 5]
transforms = doit(num_samples, modeids, ks)
reconstruct(num_samples, modeids, ks, transforms)
#print(np.shape(transforms))
for modeid_index, modeid in enumerate(modeids):
    power_spectrum = np.zeros(len(k_norms))
    for i in range(len(k_norms)):
        #print(k_norms[i], k_norm_ids[i])
        for j in k_norm_ids[i]:
            power_spectrum[i] += transforms[modeid_index,j,0]**2
        power_spectrum[i] /= len(k_norm_ids[i])

    colval = modeid_index / len(modeids)
    axs[0].plot(k_norms, power_spectrum, '.-', color=(0.1+0.9*colval, 0.0, 1.0-colval)) #, label=str(modeid))
    maxvs[modeid_index] = k_norms[np.argmax(power_spectrum)]
    avgvs[modeid_index] = np.sum(power_spectrum * k_norms)





axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].legend()
        
axs[1].plot(maxvs/np.max(maxvs))
axs[1].plot(avgvs/np.max(avgvs))

#fig.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/fourier/v_"+name[:3]+"_"+uend+"_nm"+str(num_modes)+"_nd"+str(num_direcs)+".pdf")

#plt.show()

