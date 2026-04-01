## 1708thesisrelevant
#tag:10_02_2026

from ... import don_code
import os
import sys
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
matplotlib.use("pgf")

def get_colors():
    f = open(don_code.colors_path, "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()
fontsize = 10 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage{xcolor} \definecolor{myorange}{HTML}{ec7f3c} \definecolor{myyellow}{HTML}{F5C527} \definecolor{darkred}{HTML}{8b0000}",
})

sizeid = int(sys.argv[1])
doshow = bool(int(sys.argv[2]))
show_dips = False # this used to be true

methodids = [1, 4] #, 2, 3, 4]
bids = [0, 3, 4, 6]
names = [r"AD $\tau=0.5$", r"KdV $\tau=0.2$", r"KdV $\tau=0.6$", r"Burgers $\tau=0.1$"]


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
    f = open(don_code.os.path.join(don_code.REPO_ROOT, "data", "fourier_data", "bid"+str(bid)+"_abs.txt"), "r")
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

freq_axs = [[], []]

min_y_gd   = 10
min_y_adam = 10
min_y_freq = 0.1
max_y_gd   = -10
max_y_adam = -10
max_y_freq = -10


fig1, axs1 = plt.subplots(1, len(bids), figsize=(6,2.5)) #, constrained_layout=True)
fig2, axs2 = plt.subplots(1, len(bids), figsize=(6,2.5)) #, constrained_layout=True)


for ib, bid in enumerate(bids):
    print("\n\n", bid)
    freqs, labels, tmpstr, f0 = get_frequencies(bid)
    f0mat[:, ib] = np.array(f0)

    # get w and llw
    if bid < 2:
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
        axs_tmp = axs1 if irow == 0 else axs2
        direcs = []
        for wun in wun_s:
            #if bid == 3 and irow == 2:
            #    direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lr"+optim+"40_v0")
            #    direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lr"+optim+"40_v0")
            #else:
            direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lr"+optim+"32_v0")
            #direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lr"+optim+"32_v0")
        
        num_columns = len(wun_s)

        _, _, llw, _, batch_name, num_data, endtag = don_code.get_dwllw(direcs[0])

        axs_tmp[ib].set_zorder(10)

        ax3 = axs_tmp[ib].twinx()
        ax3.set_zorder(2)
        #ax2 = axs_tmp[ib] #.twinx()
        ax3.set_zorder(1)
        sigma_exp = -0.2
        scaled_sigma = ss_train[:llw]**sigma_exp
        ax3.plot(scaled_sigma, '.-', color="k")

        min_y_freq = min(min_y_freq, np.min(scaled_sigma))
        max_y_freq = max(max_y_freq, np.max(scaled_sigma))
        if irow == 0:
            print(bid, "sigmas", 1/ss_train[0], 1/ss_train[llw-1])

        k = 0

        dip_comps = []
        freq_comps = []

        #ax3.plot([10], [0.5], 'o', color="white")
        for imj,im in enumerate(methodids):
            ax3.plot(freqs[im], '--', label=labels[im], color=colors[2+imj], linewidth=3)
            min_y_freq = min(min_y_freq, np.min(freqs[im]), np.min(freqs[im]))
            max_y_freq = max(max_y_freq, np.max(freqs[im]), np.max(freqs[im]))
            
            dip_comps.append(freqs[im])
            freq_comps.append(freqs[im])

        #ax3.legend(loc='lower right')
        print("direcs")
        for direc in direcs:
            if direc in alldirecs:
                print(direc)
        print(" ")


        for direc in direcs:
            if direc not in alldirecs:
                print(direc, "not there")
            else:
                modeloss_data = np.loadtxt(nets_dir+"/"+direc+"/log_modes.txt")
                train_modes = modeloss_data[:,1      :1+  llw]
                test_modes  = modeloss_data[:,1+2*llw:1+3*llw]


                #axs_tmp[ib].plot(np.ones(llw), '--', color="k")
                #axs_tmp[ib].plot(0.1*np.ones(llw), '--', color="k")
                tr = train_modes[num_epochs-1,:]
                axs_tmp[ib].plot(tr, color=colors_tr[k])
                te = test_modes[num_epochs-1,:]/m_test*m_train
                axs_tmp[ib].plot(te, color=colors_te[k])


                #axs_tmp[ib].plot(1.0 - tr / te, color=colors_ra[k]) #"colors_te[k])
                if "SGD" in direc:
                    tr0mat[0,ib] = tr[0]
                    te0mat[0,ib] = te[0]
                    min_y_gd = min(min_y_gd, np.min(tr), np.min(te))
                    max_y_gd = max(max_y_gd, np.max(tr), np.max(te))
                elif "Adam" in direc:
                    tr0mat[1,ib] = tr[0]
                    te0mat[1,ib] = te[0]
                    min_y_adam = min(min_y_adam, np.min(tr), np.min(te))
                    max_y_adam = max(max_y_adam, np.max(tr), np.max(te))

                print(direc, "yes", tr[0], te[0])



                k += 1
        
        axs_tmp[ib].patch.set_visible(False) 

        #ax2.set_yscale("log")
        ax3.set_yscale("log")
        freq_axs[irow].append(ax3)
        axs_tmp[ib].set_yscale("log")
        axs1[ib].set_title(names[ib], fontsize=fontsize)
        axs2[ib].set_title(names[ib], fontsize=fontsize)
        #axs_tmp[ib].legend(loc='upper left')

        #for axtmp in [axs_tmp[ ib], ax2, ax3]:
        #    axtmp.yaxis.set_major_locator(plt.NullLocator())
        #    axtmp.yaxis.set_minor_locator(plt.NullLocator())

        if llw == 20:
            axs_tmp[ib].set_xticks([0, 7, 15], ["1", "8", "16"])
        else:
            axs_tmp[ib].set_xticks([0, 19, 39], ["1", "20", "40"])


yfac = 1.5
for ib in range(len(bids)):
    axs1[ib].set_ylim((min_y_gd/yfac, max_y_gd*yfac))
    axs2[ib].set_ylim((min_y_adam/yfac, max_y_adam*yfac))
    freq_axs[0][ib].set_ylim((min_y_freq/yfac, max_y_freq*yfac))
    freq_axs[1][ib].set_ylim((min_y_freq/yfac, max_y_freq*yfac))

    
    if ib != len(bids)-1:
        freq_axs[0][ib].set_yticks([])
        freq_axs[1][ib].set_yticks([])
    else:
        freq_axs[0][ib].set_yticks([0.1, 1.0], [r"$10^{-1}$", r"$10^0$"])
        freq_axs[1][ib].set_yticks([0.1, 1.0], [r"$10^{-1}$", r"$10^0$"])

    
for axs in [axs1, axs2]:
    axs[len(bids)-1].plot([], [], '-', color="red", label=r"$L_{i,tr}$")
    axs[len(bids)-1].plot([], [], '-', color="blue", label=r"$L_{i,te}$")
    axs[len(bids)-1].plot([], [], '--', linewidth=3, color=colors[2], label=r"TV")
    axs[len(bids)-1].plot([], [], '--', linewidth=3, color=colors[3], label=r"LE")
    axs[len(bids)-1].plot([], [], '.-', color="k", label=r"$\sigma_i^{-0.2}$")
    axs[len(bids)-1].legend(loc="lower right", bbox_to_anchor=(1.05, -0.02), borderpad=0.0, fontsize=fontsize)


    axs[0].set_ylabel(r"unweighted mode loss $L_i$ \textbf{{\color{red}\!\!/}{\color{blue}\!\!/}}", fontsize=fontsize)

    ax_tmp = axs[3].twinx()
    ax_tmp.set_ylabel(r"frequency and singular values \textbf{{\color{teal}\!\!/}{\color{myyellow}\!\!/}{\color{black}\!\!/}}", fontsize=fontsize, labelpad=30)
    ax_tmp.set_yticks([])

    for ib in range(len(bids)):
        print("grid!", ib)
        axs[ib].grid(visible=True, axis='y')
        if ib != 0:
            axs[ib].set_yticklabels([])
            #axs2[ib].set_yticks([])

fig1.text(0.42, 0.04, r"mode index $i$", fontsize=fontsize)
fig2.text(0.42, 0.04, r"mode index $i$", fontsize=fontsize)


#fig.suptitle(r"Mode index $i$", fontsize=fontsize, y=0.04)



fig1.subplots_adjust(wspace=0.01, hspace=0.01, left=0.1, right=0.9, bottom=0.16, top=0.9)
fig2.subplots_adjust(wspace=0.01, hspace=0.01, left=0.1, right=0.9, bottom=0.16, top=0.9)


#plt.tight_layout()

#name = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/final_imgs/Fig7"+str(sizeid)


#fig1.savefig(name+"_GD.pdf")
#fig2.savefig(name+".pdf")

path = don_code.figures_dir
tmp = "Fig7"
fig2.savefig(path+"/pdfs/"+tmp+".pdf")
fig2.savefig(path+"/pngs/"+tmp+".png")

#if doshow:
#    plt.show()






