## 1708thesisrelevant

from ... import don_code
import os
import sys
import numpy as np 
import matplotlib.pyplot as plt 

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

sizeid = int(sys.argv[1])
doshow = bool(int(sys.argv[2]))
show_dips = False # this used to be true

methodids = [1, 4] #, 2, 3, 4]
bids = [0, 3, 4, 6]


colors_tr = ["red"] #"darkred", "red", "lightcoral"]
colors_te = ["blue"] #"darkblue", "blue", "skyblue"]
#colors_tr = ["darkred", "red", "lightcoral"]
#colors_te = ["darkblue", "blue", "skyblue"]
#colors_ra = ["darkmagenta", "fuchsia", "plum"]


nets_dir = don_code.nets_dir

alldirecs = os.listdir(nets_dir)

num_epochs = 80 #max(num_epochs, 80)



f0mat = np.zeros((5, len(bids)))
tr0mat = np.zeros((2,len(bids)))
te0mat = np.zeros((2,len(bids)))


def get_frequencies(bid):
    f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/fourier_v_2/data/bid"+str(bid)+"_abs.txt", "r")
    lines = f.readlines()
    f.close()

    labels  = []
    freqs   = []
    strings = []
    f0 = []

    for line in lines:
        xx = line[:-1].split(" ")
        labels.append(xx[0])
        vec = np.zeros(len(xx)-1)
        for i,x in enumerate(xx[1:]):
            vec[i] = x 

        maxfreq = np.max(vec)

        tmpstr = "method "+xx[0]+" min0 "+str(vec[0])+" maxf "+str(maxfreq)
        f0.append(vec[0])
        print(tmpstr)
        strings.append(tmpstr)
        freqs.append(vec / maxfreq)
        #print(labels[-1])
        #print(freqs[-1])
        #print("-")
    
    return freqs, labels, tmpstr, f0


fig, axs = plt.subplots(2, len(bids), figsize=(8,6)) #, constrained_layout=True)

for ib, bid in enumerate(bids):
    print("\n\n", bid)
    freqs, labels, tmpstr, f0 = get_frequencies(bid)
    f0mat[:, ib] = np.array(f0)

    if bid < 2:
        wun_s = [50, 100, 222, 332, 495]
        wun_s = [100, 222, 332, 495]
        #wun_s = [222]
        llw = 20

    elif bid < 6:
        wun_s = [100, 220, 335, 495]
        #wun_s = [220]
        llw = 50
        
    else:
        wun_s = [100, 237, 337, 494]
        #wun_s = [237]
        llw = 50

    
    wun_s = wun_s[sizeid:sizeid+1]

    batch_name, endtag, _ = don_code.dic(bid)
    nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, endtag, 1000)
    m_train = np.shape(utrain)[1]
    m_test  = np.shape(utest)[1]

    uu_train, ss_train, vvhh_train = np.linalg.svd(utrain, full_matrices=False)

    for irow, optim in enumerate(["SGD", "Adam"]):
        direcs = []
        for wun in wun_s:
            #if bid == 3 and irow == 2:
            #    direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lr"+optim+"40_v0")
            #    direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lr"+optim+"40_v0")
            #else:
            direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lr"+optim+"32_v0")
            direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lr"+optim+"32_v0")
        
        num_columns = len(wun_s)

        _, _, llw, _, batch_name, num_data, endtag = don_code.get_dwllw(direcs[0])

        axs[irow,ib].set_zorder(10)

        ax2 = axs[irow,ib].twinx()
        ax2.set_zorder(1)
        ax2.plot(1 / ss_train[:llw], '.-', color="k")

        k = 0

        dip_comps = []
        freq_comps = []

        ax3 = axs[irow,ib].twinx()
        ax3.set_zorder(2)
        ax3.plot([10], [0.5], 'o', color="white")
        for imj,im in enumerate(methodids):
            ax3.plot(freqs[im], '--', label=labels[im], color=colors[2+imj], linewidth=3)
            dip_comps.append(freqs[im])
            freq_comps.append(freqs[im])

        #ax3.legend(loc='lower right')

        for direc in direcs:
            if direc not in alldirecs:
                print(direc, "not there")
            else:
                modeloss_data = np.loadtxt(nets_dir+"/"+direc+"/log_modes.txt")
                train_modes = modeloss_data[:,1      :1+  llw]
                test_modes  = modeloss_data[:,1+2*llw:1+3*llw]
            

                #axs[irow,ib].plot(np.ones(llw), '--', color="k")
                #axs[irow,ib].plot(0.1*np.ones(llw), '--', color="k")
                tr = train_modes[num_epochs-1,:]
                axs[irow,ib].plot(tr, color=colors_tr[k])
                te = test_modes[num_epochs-1,:]/m_test*m_train
                axs[irow,ib].plot(te, color=colors_te[k])

                #if "Adam" in direc:
                dip_comps.append(tr) #[im])
                dip_comps.append(te)


                #axs[irow,ib].plot(1.0 - tr / te, color=colors_ra[k]) #"colors_te[k])
                if "SGD" in direc:
                    tr0mat[0,ib] = tr[0]
                    te0mat[0,ib] = te[0]
                elif "Adam" in direc:
                    tr0mat[1,ib] = tr[0]
                    te0mat[1,ib] = te[0]

                print(tr[0], te[0])



                k += 1
        
        if show_dips:
            purefreqdip = []
            dipids = []
            for i in range(1, llw-1):
                isdip = True
                for arr in dip_comps:
                    if arr[i] > min(arr[i-1], arr[i+1]):
                        isdip = False 
                if isdip:
                    dipids.append(i)
                    #print(i)
                    ax2.plot([i, i], [1/ss_train[0], 1/ss_train[llw-1]], '--', color="fuchsia")

                isfreqdip = True
                for arr in freq_comps:
                    if arr[i] > min(arr[i-1], arr[i+1]):
                        isfreqdip = False 
                if isfreqdip:
                    purefreqdip.append(i)

            print(optim, "dips", dipids, "len yes dips", len(dipids), "len pure freq dips", len(purefreqdip))

        axs[irow,ib].patch.set_visible(False) 

        ax2.set_yscale("log")
        ax3.set_yscale("log")
        axs[irow,ib].set_yscale("log")
        #axs[irow,ib].legend(loc='upper left')

        for axtmp in [axs[irow, ib], ax2, ax3]:
            axtmp.yaxis.set_major_locator(plt.NullLocator())
            axtmp.yaxis.set_minor_locator(plt.NullLocator())


    axs[0,ib].set_xticks([])
    if llw == 20:
        axs[1,ib].set_xticks([0, 9, 19], ["1", "10", "20"])
    else:
        axs[1,ib].set_xticks([0, 24, 49], ["1", "25", "50"])


axs[1,len(bids)-1].plot([], [], '.-', color="k", label=r"$1/\sigma_i$")
axs[1,len(bids)-1].plot([], [], '-', color="red", label=r"$L_{i,tr}$")
axs[1,len(bids)-1].plot([], [], '-', color="blue", label=r"$L_{i,te}$")
#axs[0,len(bids)-1].legend(loc='lower right', fontsize=12)

# labels for the paper:
axs[1,len(bids)-1].plot([], [], '--', linewidth=3, color=colors[2], label=r"TV $k=3:~f(\rho_i)$")
axs[1,len(bids)-1].plot([], [], '--', linewidth=3, color=colors[3], label=r"LE $k=50:~f(\rho_i)$")
# these are the original labels from the thesis:
#axs[1,len(bids)-1].plot([], [], '--', linewidth=3, color=colors[2], label=r"TV $k=3:~f_i$")
#axs[1,len(bids)-1].plot([], [], '--', linewidth=3, color=colors[3], label=r"LE $k=50:~f_i$")
leg=axs[1,len(bids)-1].legend(fontsize=13, bbox_to_anchor=(0.4, 0.6), framealpha=1.0)
leg.set_in_layout(False)
fig.canvas.draw()


axs[0,0].set_ylabel(r"GD", fontsize=12)
axs[1,0].set_ylabel(r"Adam", fontsize=12)

fig.text(0.4, 0.04, r"Mode index $i$", fontsize=12)
#fig.suptitle(r"Mode index $i$", fontsize=12, y=0.04)

plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.06, right=0.84, bottom=0.1, top=0.98)

#plt.tight_layout()

name = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/sb/bids0346_pink_size"+str(sizeid)
if not show_dips:
    name += "_nodips"


fig.savefig(name+".pdf")




#axs[1].arrow(32, 0.3, 0, 0.18, color="fuchsia")
#axs[1].arrow(36, 0.5, 0, 0.18, color="fuchsia")

#plt.figure()
#plt.plot(tr0mat[0,:], '.-', label="Tr GD")
#plt.plot(tr0mat[1,:], '.-', label="Tr Ad")

#plt.plot(te0mat[0,:], '.-', label="Te GD")
#plt.plot(te0mat[1,:], '.-', label="Te Ad")



#for i in range(5):
#    plt.plot(f0mat[i,:] / np.max(f0mat[i,:]), '--', label=str(i))
#plt.legend()

if doshow:
    plt.show()






