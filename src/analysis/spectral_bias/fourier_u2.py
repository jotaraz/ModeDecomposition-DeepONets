import numpy as np 
import matplotlib.pyplot as plt 
import sys



# start ...

bid = int(sys.argv[1])
#N   = int(sys.argv[2])

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


# heart start


def count_switches(x):
    n = len(x)
    k = 0
    for i in range(1, n):
        if np.sign(x[i]) != np.sign(x[i-1]):
            k +=1 
    return k 

def dft(y, ks, M):
    xx = np.linspace(0, 1, M)

    W = np.zeros((len(ks),M))
    for i, k in enumerate(ks): # range(int(M/2)):
        for j in range(M):
            W[i,j] = np.exp(-2*np.pi * 1j * k * xx[j])
    
    return np.matmul(W, y)

    #transforms = np.zeros(len(ks)) #(len(modeids), len(ks), len(num_samples)))

    #for j,k in enumerate(ks):
    #    #ytmp = y[i]
    #    #xtmp = xx[i]
    #    transforms[j] = np.sum(y * np.exp(2*np.pi * 1j*k*xx)) / M 

    #return transforms


def all_characteristics(y):
    M = len(y)
    #kmin = 1 
    #kmax = M 
    #ks = 10**(np.linspace(-3, np.log10(M/2), M)) 
    ks = np.arange(0, int(M/2))
    #ks = np.linspace(1e-3, M/4, 300)
    fourier = dft(y, ks, M)
    #print(np.shape(y), np.shape(ks), np.shape(fourier))

    f_dom  = ks[np.argmax(np.abs(fourier))]
    f_mean = np.sum(fourier**2 * ks) / np.sum(fourier**2)
    c_swi  = count_switches(y)

    fourier_cumsum = np.cumsum(fourier**2)
    i_med  = np.where(fourier_cumsum > fourier_cumsum[-1]/2)[0][0]
    #print(i_med)
    #print(fourier_cumsum[max(0,i_med-1)], fourier_cumsum[i_med], fourier_cumsum[min(i_med+1, len(ks)-1)], "sum/2", np.sum(fourier**2)/2, "last cs/2", fourier_cumsum[-1]/2)
    #total_sum = np.sum(fourier**2)
    f_med = 0.5*(ks[max(0,i_med-1)] + ks[i_med])

    return (f_dom, f_mean, c_swi, f_med), fourier, ks


# heart end 


# run stuff

name, uend, _ = dic(bid)

print(name)

yy = np.loadtxt(base+"/"+name+"_"+uend+"_U.txt")
uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
N = min(100, len(np.where(ss > 1e-10)[0]))

#print(ss)

print("N=", N)


fig, axs = plt.subplots(1, 3, figsize=(16,8))


doms  = []
means = []
swits = []
medis = []


M = np.shape(yy)[0]

for i in range(N):
    print(name, i)
    chars, spectrum, ks = all_characteristics(uu[:,i])
    axs[0].plot(np.zeros(M)-i, '--', color='gray')
    axs[1].plot(ks, np.zeros(len(ks))-i, '--', color='gray')
    axs[0].plot(0.5*uu[:,i]/np.max(np.abs(uu[:,i]))-i)
    axs[1].plot(ks, 0.5*spectrum/np.max(np.abs(spectrum))-i)
    doms.append(chars[0])
    means.append(chars[1])
    swits.append(chars[2])
    medis.append(chars[3])



doms  = np.array(doms)
means = np.array(means)
swits = np.array(swits)
medis = np.array(medis)


axs[2].plot(doms / np.max(doms), label="dom  "+str(np.max(doms)))
axs[2].plot(means / np.max(means), label="mean "+str(np.max(means)))
axs[2].plot(swits / np.max(swits), label="swit "+str(np.max(swits)))
axs[2].plot(medis / np.max(medis), label="medi "+str(np.max(medis)))

axs[2].legend()

fig.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/fourier/u_"+name[:3]+"_"+uend+"_sb.pdf")

#plt.show()



