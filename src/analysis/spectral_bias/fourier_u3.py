import numpy as np 
import matplotlib.pyplot as plt 
import sys

def get_colors():
    f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/colors.txt", "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# start ...

#bid = int(sys.argv[1])
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

#bids   = [1, 5, 7]
bids   = [0, 3, 6]
nmodes = [10, 10, 10]
xmax = [2*np.pi, 2*np.pi, 1.0]
#colors = ["red", "blue", "green"] 

fig, axs = plt.subplots(1, len(bids)+1, figsize=(14,6))

for ib, bid in enumerate(bids):
    name, uend, _ = dic(bid)

    print(name)

    yy = np.loadtxt(base+"/"+name+"_"+uend+"_U.txt")
    yy = yy[:,:1000]
    uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
    N = min(50, len(np.where(ss > 1e-10)[0]))

    #print(ss)

    print("N=", N)


    doms  = []
    means = []
    swits = []
    medis = []

    n = np.shape(yy)[0]
    xax = np.linspace(0.0, 1.0, n)
        

    if ib == 0:
        #axs[0].text(-0.22, 0-0.4, r"$\phi_1$", horizontalalignment='left', verticalalignment='bottom', fontsize=28)
        #axs[0].text(-0.22, -nmodes[ib]+1-0.4, r"$\phi_{10}$", horizontalalignment='left', verticalalignment='bottom', fontsize=28)
        #axs[0].set_xlim((-0.24, 1.02))
        #axs[0].text(-0.22, -11.0, "A)", horizontalalignment='left', verticalalignment='bottom', fontsize=32)
        axs[0].set_xlim((-0.02, 1.02))
        axs[0].text(-0., -11.0, "A)", horizontalalignment='left', verticalalignment='bottom', fontsize=32)

    else:
        axs[ib].set_xlim((-0.02, 1.02))

        if ib == 1:
            axs[1].text(-0.0, -11.0, "B)", horizontalalignment='left', verticalalignment='bottom', fontsize=32)
        elif ib == 2:
            axs[2].text(-0.0, -11.0, "C)", horizontalalignment='left', verticalalignment='bottom', fontsize=32)

    axs[ib].set_ylim((-11.5, 1.0))

    M = np.shape(yy)[0]

    for i in range(N):
        print(name, i)
        chars, spectrum, ks = all_characteristics(uu[:,i])
        if i < nmodes[ib]:
            axs[ib].plot(xax, np.zeros(M)-i, '--', color='gray')
            #axs[1].plot(ks, np.zeros(len(ks))-i, '--', color='gray')
            axs[ib].plot(xax, 0.45*uu[:,i]/np.max(np.abs(uu[:,i]))-i, color=colors[ib], linewidth=3)
            #axs[1+ib].plot(xax, 0.5*uu[:,i]/np.max(np.abs(uu[:,i]))-i, color=colors[ib])
            #axs[1].plot(ks, 0.5*spectrum/np.max(np.abs(spectrum))-i)
        doms.append(chars[0])
        means.append(chars[1])
        swits.append(chars[2])
        medis.append(chars[3])

    

    axs[ib].set_xticks([])
    axs[ib].set_yticks([])

    axs[3].plot(means / np.max(means), '.-', color=colors[ib], linewidth=5)

    '''



    doms  = np.array(doms)
    means = np.array(means)
    swits = np.array(swits)
    medis = np.array(medis)


    axs[2].plot(doms / np.max(doms), label="dom  "+str(np.max(doms)))
    axs[2].plot(means / np.max(means), label="mean "+str(np.max(means)))
    axs[2].plot(swits / np.max(swits), label="swit "+str(np.max(swits)))
    axs[2].plot(medis / np.max(medis), label="medi "+str(np.max(medis)))

    axs[2].legend()
    '''

axs[3].set_xticks([0, 24, 49], ["1", "25", "50"], fontsize=13)

axs[3].set_xlabel(r"Mode index $i$", fontsize=16)
axs[3].set_ylabel(r"Relative frequency $f_i/(\max_j f_j)$", fontsize=16)

axs[3].set_ylim((-0.15, 1.05)) #r"Mode index $i$", fontsize=16)
axs[3].text(0.0, -0.102, "D)", horizontalalignment='left', verticalalignment='bottom', fontsize=32)

fig.text(0.01, 0.25, r"$\phi_{10}$", horizontalalignment='left', verticalalignment='bottom', fontsize=28)
fig.text(0.01, 0.83, r"$\phi_{1}$", horizontalalignment='left', verticalalignment='bottom', fontsize=28)

plt.subplots_adjust(wspace=0.22, hspace=0.0, left=0.05, right=0.99, bottom=0.1, top=0.9)
#plt.tight_layout()


fig.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/fourier/u_all.pdf")

plt.show()



