# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 15:07:55 2025

@author: J_Taraz
"""

# -*- coding: utf-8 -*-
"""
2025 09 27

@author: J_Taraz
"""

#colors = ["green", "orange", "purple", "blue", "pink", "red"]

def get_colors():
    f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/colors.txt", "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()[1:]

import numpy as np
import matplotlib.pyplot as plt
import sys 
import os 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})



sigmascale_num = float(sys.argv[1]) #"ss-0.0001"
alpha          = float(sys.argv[2]) #default should be 0.2 # or "fs"


sigmascale = "ss"+str(sigmascale_num)

sigmascale_to_vid = {
    "ss-0.0001": {"base":11, "trunc":12, "flip":13, "Nep":5000},
    "ss-0.01":   {"base": 9, "trunc":14, "flip":15, "Nep":2000},
    "ss-0.1":    {"base": 7, "trunc":16, "flip":17, "Nep":2000},
    "ss-0.5":    {"base": 4, "trunc":18, "flip":19, "Nep":2000},
    }

F0 = 0.2 # or "ns"
dw = "d5_w50"

truncated_id = sigmascale_to_vid[sigmascale]["trunc"]
flipped_id   = sigmascale_to_vid[sigmascale]["flip"]
nepochs      = sigmascale_to_vid[sigmascale]["Nep"]

path = r"/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/nets/"
llw = 5
direcs = []
labels = []

sharedbase = "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep"

direcs.append(sharedbase+str(nepochs)+"_"+dw+"_llw5_batsynthv"+str(truncated_id)+"_n100_N5_m5000_fs"+str(alpha)+sigmascale+"ns"+str(F0)+"_numd5000_lrAdam40_v0")
#labels.append(r"Inc. Freq., $\alpha > 0$")
labels.append(r"Inc. Freq., $\alpha = "+str(alpha)+"$")

direcs.append(sharedbase+str(nepochs)+"_"+dw+"_llw5_batsynthv"+str(flipped_id)+"_n100_N5_m5000_fs"+str(alpha)+sigmascale+"ns"+str(F0)+"_numd5000_lrAdam40_v0")
#labels.append(r"Dec. Freq., $\alpha < 0$")
labels.append(r"Dec. Freq., $\alpha = -"+str(alpha)+"$")
# the j-th right singular function rho_j has frequency ns * exp(fs * j) (Note that j starts counting at 0)
# the j-th singular value is given by sigma_j = exp(sigmascale*j) (Note that sigmascale must be negative)


    
fig, axs = plt.subplots(1, 1+len(direcs), figsize=(10,5))

minmode = 1 
maxmode = 0

alldirs = os.listdir(path)

k = 0
for d in direcs:
    if d in alldirs:
        loss_data = np.loadtxt(path+d+r"/log.txt")
        epochs     = loss_data[:,0]
        train_loss = loss_data[:,1]
        test_loss  = loss_data[:,4]
        
        tag = d.split("m5000_")[1].split("_numd5")[0]
        
        axs[0].plot(epochs, 10**train_loss, '--', color=colors[k])
        if len(labels) == len(direcs):
            tag = labels[k]
        
        axs[0].plot(epochs, 10**test_loss, '.-', color=colors[k], label=tag+" Loss")
        axs[0].set_yscale("log")

        mode_data = np.loadtxt(path+d+r"/log_modes.txt")
        num_saves = np.shape(mode_data)[0]
        for i in range(2, int(num_saves/3), 1):
            #print(i, 3.0*i/num_saves)
            axs[1+k].plot(mode_data[3*i, 1:1+llw],'--',color=(0.2 + 0.7*3.0*(i-2)/num_saves, 0, 0))
            minmode = min(minmode, np.min(mode_data[3*i, 1:1+llw]))
            maxmode = max(maxmode, np.max(mode_data[3*i, 1:1+llw]))
            #print(mode_data[3*i, 0])
        #axs[1+k].set_ylim((1e-2, 10))
        
        axs[1+k].set_yscale("log")
        axs[1+k].set_title(tag, color=colors[k], fontsize=20, fontweight='bold')
        print(d, num_saves)
        
        k += 1
    else:
        print(d, "not there")
    
xax = np.arange(0, llw)
for kk in range(k):
    ax_tmp = axs[1+kk].twinx()
    freqs = F0*np.exp( alpha * xax)
    if kk > 0:
        freqs = np.flip(freqs)
        ax_tmp.set_ylabel(r"Frequency $f(\rho_j)$", fontsize=20)
    ax_tmp.plot(xax, freqs, 'o-', color=colors[kk], linewidth=2)
    ax_tmp.set_ylim((-0.02, np.max(freqs)+0.02))
    ax_tmp.tick_params(axis='y', colors=colors[kk], labelsize=15)
    
    ax_tmp.set_zorder(0)      # low enough to be beneath
    axs[1+kk].set_zorder(1)  
    axs[1+kk].patch.set_alpha(0.0)     # make background transparent so image shows through
    

    axs[1+kk].set_ylim((0.5*minmode, 2*maxmode))

for i in range(3):
    axs[i].tick_params(axis='y', labelsize=15)

legid = 1
axs[legid].plot([], [], '--', color=(0.2, 0, 0), label="Initial Mode Losses")
axs[legid].plot([], [], '--', color=(0.2+0.7, 0, 0), label="Final Mode Losses")    
axs[legid].plot([], [], 'o-', color=colors[0], linewidth=2, label=labels[0])    
axs[legid].plot([], [], 'o-', color=colors[1], linewidth=2, label=labels[1])    
axs[legid].legend(loc='upper left', framealpha=0.9, fontsize=15, bbox_to_anchor=(-0.65, 1.03))

fig.text(0.58, 0.03, r"Mode index $j$", fontsize=20)
axs[0].set_title(r"$\beta = "+str(sigmascale_num)+"$", fontsize=20)
axs[0].set_xlabel(r"Epochs", fontsize=20)
axs[0].set_ylabel(r"Total Loss $\mathcal{L}$", fontsize=20)
axs[1].set_ylabel(r"Mode Losses $L_j$ ____", fontsize=20)

fig.subplots_adjust(left=0.083, right=0.924, top=0.88, bottom=0.11, wspace=0.4)

fig.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/sb/spectralbias_"+dw+sigmascale+"_fs"+str(alpha)+"_ns"+str(F0)+".pdf")

plt.show()
    
    
    
    