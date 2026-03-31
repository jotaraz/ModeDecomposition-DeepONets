#tag:10_02_2026
#colors = ["green", "orange", "purple", "blue", "pink", "red"]

import numpy as np

import matplotlib
#import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os

def get_colors():
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "colors_cb.txt"), "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors0 = get_colors()

colors = [colors0[0], colors0[1]]

def generate_flow_color(x):
    return (0.2 + 0.7*x, 0.1*(1-x), 0.1*(1-x))



matplotlib.use("pgf")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage{xcolor} \definecolor{myorange}{HTML}{ec7f3c} \definecolor{mygreen}{HTML}{2a9577} \definecolor{darkred}{HTML}{8b0000}",
})



#
# plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "text.latex.preamble": r"\usepackage{xcolor}"
#})

#print(plt.rcParams["text.latex.preamble"])  # Verify it's set

#rc('text.latex', preamble=r'\usepackage{color}')

fontsize=10

#do_inc = int()
#do_exp = bool(int(sys.argv[2])) 

#sigmascale_num = float(sys.argv[1]) #"-0.0001"
alpha          = float(sys.argv[1]) #default should be 0.2 # or "fs"



sigmascale_to_vid = {
    "ss-0.0001": {"base":11, "trunc":12, "flip":13, "Nep":5000},
    "ss-0.01":   {"base": 9, "trunc":14, "flip":15, "Nep":2000},
    "ss-0.1":    {"base": 7, "trunc":16, "flip":17, "Nep":2000},
    "ss-0.5":    {"base": 4, "trunc":18, "flip":19, "Nep":2000},
    }

F0 = 0.2 # or "ns"
dw = "d5_w50"



path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "nets") + os.sep

llw = 5

sharedbase = "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep"

#fig, axs = plt.subplots(2, 1+2, figsize=(6,4))

fig = plt.figure(figsize=(6, 4))

# Outer grid: two big blocks
outer = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.5)

# first block: one row
gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], hspace=0)

# Second block: two rows (row 2 and row 3) with hspace=0 between them
gs_right = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], hspace=0, wspace=0)

axs = []
axs.append(fig.add_subplot(gs_left[0]))
axs.append(fig.add_subplot(gs_left[1]))
for i in range(2):
    axs.append(fig.add_subplot(gs_right[0, i]))
    axs.append(fig.add_subplot(gs_right[1, i]))

# loops over rows and thus also sigmascales
for irow, sigmascale_num in enumerate([-0.01, -0.5]):
    direcs = []
    labels = []
    axs[irow].text(0.95, 0.95, r"$\beta="+str(sigmascale_num)+r"$", horizontalalignment='right',
        verticalalignment='top', fontsize=fontsize, transform=axs[irow].transAxes)


    sigmascale = "ss"+str(sigmascale_num)

    truncated_id = sigmascale_to_vid[sigmascale]["trunc"]
    flipped_id   = sigmascale_to_vid[sigmascale]["flip"]
    nepochs      = sigmascale_to_vid[sigmascale]["Nep"]

    direcs.append(sharedbase+str(nepochs)+"_"+dw+"_llw5_batsynthv"+str(truncated_id)+"_n100_N5_m5000_fs"+str(alpha)+sigmascale+"ns"+str(F0)+"_numd5000_lrAdam40_v0")
    labels.append(r"increasing freq., $\alpha="+str(alpha)+r"$")

    direcs.append(sharedbase+str(nepochs)+"_"+dw+"_llw5_batsynthv"+str(flipped_id)+"_n100_N5_m5000_fs"+str(alpha)+sigmascale+"ns"+str(F0)+"_numd5000_lrAdam40_v0")
    labels.append(r"decreasing freq., $\alpha=-"+str(alpha)+r"$")


    # the j-th right singular function rho_j has frequency ns * exp(fs * j) (Note that j starts counting at 0)
    # the j-th singular value is given by sigma_j = exp(sigmascale*j) (Note that sigmascale must be negative)
        
    minmode = 1 
    maxmode = 0


    alldirs = os.listdir(path)

    # k counts the net / column of the row: k=0 (second column)
    k = 0
    for d in direcs:
        if d in alldirs:
            loss_data = np.loadtxt(path+d+r"/log.txt")
            de = 2
            epochs     = loss_data[::de,0]
            train_loss = loss_data[::de,1]
            test_loss  = loss_data[::de,4]
            
            tag = d.split("m5000_")[1].split("_numd5")[0]
            
            axs[irow].plot(epochs, 10**(train_loss/2), '--', color=colors[k])
            if len(labels) == len(direcs):
                tag = labels[k]
            
            axs[irow].plot(epochs, 10**(test_loss/2), 'o', linestyle='none', fillstyle='none', color=colors[k]) #, label=tag+" Loss")

            mode_data = np.loadtxt(path+d+r"/log_modes.txt")
            num_saves = np.shape(mode_data)[0]
            for i in range(0, num_saves, 1):
                #print(i, 3.0*i/num_saves)
                axs[2*(k+1)+irow].plot(mode_data[i, 1:1+llw],'--',color=generate_flow_color(i/num_saves))
                minmode = min(minmode, np.min(mode_data[i, 1:1+llw]))
                maxmode = max(maxmode, np.max(mode_data[i, 1:1+llw]))
                #print(mode_data[3*i, 0])
            #axs[irow, 1+k].set_ylim((1e-2, 10))
            
            axs[2*(k+1)+irow].set_yscale("log")
            #if irow == 0:
            #    axs[2*(k+1)+irow].set_title(tag, color=colors[k])
            print(d, num_saves)
            
            k += 1
        else:
            print(d, "not there")
        
    xax = np.arange(0, llw)
    print("Sigma^2's =", np.exp(2*sigmascale_num*xax))
    for kk in range(k):
        ax_tmp = axs[2*(kk+1)+irow].twinx()
        freqs = F0 * np.exp( alpha * xax)
        if kk > 0:
            freqs = np.flip(freqs)

        ax_tmp.plot(xax, freqs, 'o-', color=colors[kk], linewidth=3)
        if kk == 0:
            ax_tmp.set_yticks([]) #tick_params(axis='y', colors=colors[kk])
        #else:
        #    ax_tmp.tick_params(axis='y', )
        
        # freq above:
        #ax_tmp.set_ylim((0,0.5))
        #ax_tmp.set_zorder(0)      # low enough to be beneath
        #axs[2*(kk+1)+irow].set_zorder(1)  
        #axs[2*(kk+1)+irow].patch.set_alpha(0.0)     # make background transparent so image shows through
        #axs[2*(kk+1)+irow].set_ylim((0.5*minmode, 1_000*maxmode))

        # freq below:
        ax_tmp.set_ylim((0.1,1.0))
        ax_tmp.set_zorder(0)      # low enough to be beneath
        axs[2*(kk+1)+irow].set_zorder(1)  
        axs[2*(kk+1)+irow].patch.set_alpha(0.0)     # make background transparent so image shows through
        axs[2*(kk+1)+irow].set_ylim((9e-5,20))#(0.02*minmode, 2*maxmode))


axs[0].set_yscale("log")
axs[1].set_yscale("log")
axs[0].set_xticks([])
axs[2].set_xticks([])
axs[4].set_xticks([])
axs[4].set_yticks([])
axs[5].set_yticks([])


axs[1].set_xticks([0, 1000, 2000])

axs[3].set_xticks([1, 3], [2, 4])
axs[5].set_xticks([1, 3], [2, 4])

#axs[1].set_xlabel("Epochs", fontsize=fontsize)
fig.text(0.2, 0.02, "epochs", fontsize=fontsize)
#axs[irow].set_ylabel("Loss")
fig.text(0.02, 0.4, r"relative error $\delta$", rotation=90, fontsize=fontsize)


fig.text(0.6, 0.02, r"mode index $j$", fontsize=fontsize)
fig.text(0.38, 0.32, r"unweighted mode loss $L_j$ \textbf{{\color{black}\!\!/\!\!/}{\color{darkred}\!\!/\!\!/}{\color{red}\!\!/\!\!/}}", rotation=90, fontsize=fontsize)
fig.text(0.95, 0.35, r"frequency $f(\rho_j)$ \textbf{{\color{myorange}/}{\color{mygreen}\!\!/}}", rotation=90, fontsize=fontsize)
#fig.text(0.95, 0.35, r"frequency $f(\rho_j)$ \textbf{{\color{myorange}\!\!/\!\!/\!\!/}{\color{mygreen}\!\!/\!\!/\!\!/}}", rotation=90, fontsize=fontsize)


legid = 0
axs[legid].plot([], [], 'o', linestyle='none', fillstyle='none', color="gray", label="test")    
axs[legid].plot([], [], '--', color="gray", label="train")
axs[legid].legend(loc='lower left', bbox_to_anchor=(0.0, 0.99), framealpha=1.0, fontsize=fontsize)


legid = 2
axs[legid].plot([], [], '--', color=generate_flow_color(0.0), label="initial mode losses")
axs[legid].plot([], [], '--', color=generate_flow_color(1.0), label="final mode losses")    
axs[legid].plot([], [], 'o-', color="gray", linewidth=3, label="frequency")    
#axs[legid].plot([], [], 'o-', color=colors[1], linewidth=2, label=labels[1])    
axs[legid].legend(loc='lower left', bbox_to_anchor=(-0.3, 0.99), framealpha=1.0, ncol=2, fontsize=fontsize)

legid = 3
axs[legid].plot([], [], 's', linestyle='none', label=labels[0], color=colors[0]) #, label="initial mode losses")
axs[legid].plot([], [], 's', linestyle='none', label=labels[1], color=colors[1]) #, label="initial mode losses")

axs[legid].legend(loc='lower left', bbox_to_anchor=(-1.65, 2.35), framealpha=1.0, fontsize=fontsize, ncol=2)

# legid = 2
# axs[legid].plot([], [], '--', color=generate_flow_color(0.0), label="initial mode losses")
# axs[legid].plot([], [], '--', color=generate_flow_color(1.0), label="final mode losses")    
# axs[legid].plot([], [], 'o-', color=colors[0], linewidth=2, label=labels[0])    
# axs[legid].plot([], [], 'o-', color=colors[1], linewidth=2, label=labels[1])    
# axs[legid].legend(loc='lower left', bbox_to_anchor=(-0.7, 0.99), framealpha=1.0, ncol=2, fontsize=fontsize)


#plt.subplots_adjust(left=0.12, bottom=0.09, right=0.88, top=0.87)
plt.subplots_adjust(left=0.12, bottom=0.1, right=0.88, top=0.79)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "figures")
tmp = "Fig6"
fig.savefig(path+"/pdfs/"+tmp+".pdf")
fig.savefig(path+"/pngs/"+tmp+".png")

#plt.show()
    
    
    
    