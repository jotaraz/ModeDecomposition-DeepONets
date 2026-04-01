## 1708thesisrelevant
# august 15, used for $\gamma$ over $-Delta L$ for multiple sizes and GD
# fancyshow is not used

from ... import don_code #import * 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
#import matplotlib.gridspec as gridspec
import matplotlib #.cm as cm
import sys

bid     = int(sys.argv[1])
size_id = int(sys.argv[2])
neptag  = sys.argv[3]
showplot = bool(int(sys.argv[4]))
dostacked = "False"
doarrows = True
#lrtag = int(sys.argv[2])
#size = 
plotmode = 1
ratioplot = True
offdiagplot = True


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

fontsize = 10 # used to be 12
fig3, axs3 = plt.subplots(figsize=(2.5,3)) #figsize used to be (7,5)

nets_dir = don_code.nets_dir


if neptag == "500":
    num_points = 100
else:
    num_points = 80

def relu(x):
    return x*(np.sign(x)+1.0)/2

def fancyshow(matrix, ax, fig):
    neg_mask = matrix < 0
    pos_mask = matrix >= 0

    # Prepare masked arrays
    neg_data = np.ma.masked_where(~neg_mask, -matrix)  # Flip sign for log scale
    pos_data = np.ma.masked_where(~pos_mask, matrix)

    # Create figure and axes
    #fig, ax = plt.subplots()

    # Plot negative values with custom normalization and colormap A
    #print(np.max(relu(-matrix)))
    cmap_neg = matplotlib.colormaps.get_cmap('Blues') #cm.get_cmap('Blues')  # Example for negatives
    norm_neg = LogNorm(vmin=1e-10, vmax=np.max(relu(-matrix)))  # Because we flipped -1 → 1

    im1 = ax.imshow(neg_data, cmap=cmap_neg, norm=norm_neg)

    # Plot positive values with colormap B
    cmap_pos = matplotlib.colormaps.get_cmap('Reds') #cm.get_cmap('hot')  # Example for positives
    norm_pos = LogNorm(vmin=1e-10, vmax=np.max(relu(matrix)))

    im2 = ax.imshow(pos_data, cmap=cmap_pos, norm=norm_pos)
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.2, label='Negative values (-log scale)')
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.06, label='Positive values (log scale)')


direcs  = []
direcsB = []

#direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_5999_numd1000_lrSGD32_v0"]
#direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_5999_numd1000_lrAdam32_v0"]
#direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w337_llw50_batburgers_dt0.0001_nc10_m3800_100_numd1000_lrAdam32_v0"]

if bid < 2:
    wun_s = [50, 100, 222, 332, 495] #20, 50, 100, 222, 332, 495]
    llw = 20
elif bid < 6:
    wun_s = [50, 100, 220, 335, 495] #50, 100, 220, 335, 495]
    llw = 50    
else:
    wun_s = [50, 100, 237, 337, 494] #50, 100, 237, 337, 494]
    llw = 50

if size_id != -1:
    wun_s = wun_s[size_id:size_id+1]

batch_name, uendtag, _ = don_code.dic(bid)

alldirecs = don_code.os.listdir(nets_dir)



for w in wun_s:
    if neptag == "500":
        direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep500_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0")
    else:
        #direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0")
        direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0")
    #direcs.append("whichT0_doStacked"+dostacked+"_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam32_v0")
    #direcs.append("whichT0_doStacked"+dostacked+"_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam32_v0")

#colors  = ["red",   "blue",      "orange",     "green","cyan","pink","brown","gray","purple","lightblue","lightgreen"]
def get_colors():
    f = open(don_code.colors_path, "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()

line_ids = [] #0, 1, 5, 10]

_, _, llw, _, batch_name, num_data, endtag = don_code.get_dwllw(direcs[0])
print(batch_name, endtag, num_data, llw)

nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, endtag, num_data)
n_train, m_train = np.shape(utrain)
n_test,  m_test  = np.shape(utest)

uu_train, ss_train, vh_train = don_code.jnp.linalg.svd(utrain, full_matrices=False)



modeids = [0,1,2]
if neptag == "500":
    bigname = "log_diagoffdiag"
else:
    bigname = "log_diagoffdiag_big1e-08"
#fac = 100 #00_000

#1e-08


#fig4, axs4 = plt.subplots()

offdiags_x = []
offdiags_y = []
offdiags_label = []

min_loss_red = 1
max_loss_red = 0

k = 0

num_epochs_loss = 80
k = 0

founddirs = []


for dir in direcs:
    if dir in alldirecs:
        print("YES found", dir)
        founddirs.append(dir)
        lossdata      = np.loadtxt(nets_dir+"/"+dir+"/log.txt")
        epochs0       = lossdata[:num_epochs_loss:2,0]
        tr_loss       = 10**(lossdata[:num_epochs_loss:2,1]/2)
        te_loss       = 10**(lossdata[:num_epochs_loss:2,4]/2)
        bigdata       = np.loadtxt(nets_dir+"/"+dir+"/"+bigname+".txt")
        epochs        = bigdata[:,0]
        tr_diags      = bigdata[:,1]
        tr_offdiags   = bigdata[:,2]
        tr_taylorgrad = bigdata[:,3]
        tr_actualdiff = bigdata[:,4]
        te_diags      = bigdata[:,5]
        te_offdiags   = bigdata[:,6]
        te_taylorgrad = bigdata[:,7]
        te_actualdiff = bigdata[:,8]


        print(epochs0[:10])
        print(epochs[:10])

        numsigndiff_tr = len(np.where(np.sign(tr_actualdiff[1:]) != np.sign(tr_taylorgrad[1:]))[0]) / len(epochs[1:])
        numsigndiff_te = len(np.where(np.sign(te_actualdiff[1:]) != np.sign(te_taylorgrad[1:]))[0]) / len(epochs[1:])
    
        print("Width", wun_s)
        print("Train : Diff sign taylor v act change "+str(round(numsigndiff_tr*100,2))+" %")
        print("Test  : Diff sign taylor v act change "+str(round(numsigndiff_te*100,2))+" %")
        tmp = dir.split("_bat")


        if plotmode == 1:

            if offdiagplot:

                min_loss_red = min(min_loss_red, np.min(-tr_taylorgrad[1:]))
                max_loss_red = max(max_loss_red, np.max(-tr_taylorgrad[1:]))
                offdiags_x.append(-tr_taylorgrad[1:])
                offdiags_y.append(tr_offdiags[1:])
                offdiags_label.append(dir.split("Nep")[1].split("bat")[0].split("_")[2])

        k += 1

    else:
        print("not found", dir)





#plt.subplots_adjust(wspace=0.4, hspace=0.2)

path = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/diag_offdiag"        

#fig.savefig(path+"/bid"+str(bid)+"_sizeid"+str(size_id)+"_nep"+neptag+"_pm"+str(plotmode)+".pdf")

#if len(line_ids) > 0:
#    fig2.savefig(path+"/vtv_bid"+str(bid)+"_sizeid"+str(size_id)+"_nep"+neptag+".pdf")

#if ratioplot:
#    fig3.savefig(path+"/ratio_bid"+str(bid)+"_sizeid"+str(size_id)+"_nep"+neptag+".pdf")

if offdiagplot:
    ymin = 1
    ymax = -1
    xmin = 1
    xmax = -1
    xax = 10**np.linspace(np.log10(min_loss_red), np.log10(max_loss_red), 100)
    #axs3.plot(xax, +1.0 + 0*xax, '--', color="gray", alpha=0.5, label=r"$\pm$ Loss change")
    #axs3.plot(xax, -1.0 + 0*xax, '--', color="gray", alpha=0.5)
    #axs3.plot(xax, +0.1 + 0*xax, '--', color="k", alpha=0.5, label=r"$\pm 0.1 \cdot$ Loss change")
    #axs3.plot(xax, -0.1 + 0*xax, '--', color="k", alpha=0.5)
    for i in range(len(offdiags_y)):
        print("offdiag", i)
        axs3.plot(offdiags_x[i][1:], offdiags_y[i][1:] / offdiags_x[i][1:], '.-', label=r"$w="+offdiags_label[i][1:]+"$", color=colors[i])
        ymin = min(ymin, np.min(offdiags_y[i][1:] / offdiags_x[i][1:]))
        ymax = max(ymax, np.max(offdiags_y[i][1:] / offdiags_x[i][1:]))
        xmin = min(xmin, np.min(offdiags_x[i][1:]))
        xmax = max(xmax, np.max(offdiags_x[i][1:]))
    
    # new stuff:
    #axs3.set_xlim((0.5*xmin, 2*xmax))
    axs3.set_ylim((0.05*ymin, 1.5*ymax))

    axs3.set_xlabel(r"loss reduction $-\Delta \mathcal{L}$", fontsize=fontsize)
    axs3.set_ylabel(r"negative relative coupling strength $-\gamma$", fontsize=fontsize)
    axs3.legend(loc='lower left', fontsize=fontsize, framealpha=1.0)
    axs3.set_xscale("log")
    axs3.set_yscale("log")
    #axs3.set_title(".")

    plt.subplots_adjust(left=0.3, bottom=0.16, right=0.964, top=0.92)

    # new stuff:
    #axs3.annotate("Training Progress", 
    #                xy=(0.4, 0.9), xycoords='axes fraction',
    #                xytext=(0.9, 0.7), textcoords='axes fraction',
    #                ha='left', va='top', 
    #                arrowprops=dict(facecolor='black', arrowstyle='->'),
    #                fontsize=15, color='black'
    #            )


    ymin = 1
    ymax = -1

    '''
    axs4.plot(xax, +xax, '--', color="gray", alpha=0.5, label=r"$\pm$ Loss change")
    axs4.plot(xax, -xax, '--', color="gray", alpha=0.5)
    axs4.plot(xax, +0.1*xax, '--', color="k", alpha=0.5, label=r"$\pm 0.1 \cdot$ Loss change")
    axs4.plot(xax, -0.1*xax, '--', color="k", alpha=0.5)
    for i in range(len(offdiags_y)):
        print("offdiag", i)
        axs4.plot(offdiags_x[i], offdiags_y[i], '.-', label=offdiags_label[i], color=colors[i])
        ymin = min(ymin, np.min(offdiags_y[i]))
        ymax = max(ymax, np.max(offdiags_y[i]))
    axs4.legend()        
    axs4.set_xscale("log")
    axs4.set_xlabel("-Loss change")
    axs4.set_ylabel(r"Off-diag $\omega$")
    axs4.set_ylim((ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin)))
    '''
    
    path = don_code.figures_dir
    tmp = "Fig9a"
    fig3.savefig(path+"/pdfs/"+tmp+".pdf")
    fig3.savefig(path+"/pngs/"+tmp+".png")


for d in founddirs:
    print(d)

#plt.tight_layout()
#if showplot:
#    plt.show()
