## 1708thesisrelevant
# august 15th, used for GD, one size, to make the (1) loss,loss change, gamma and (2) S_tr & mode losses plot
# I also included the mode-internal-overfitting indicator (in fuchsia)
from ... import don_code
#import numpy as np 
#import matplotlib.pyplot as don_code.plt.
from matplotlib.colors import Normalize, LogNorm
import matplotlib #.cm as cm

bid     = int(don_code.sys.argv[1])
size_id = int(don_code.sys.argv[2])
neptag  = don_code.sys.argv[3]
showplot = bool(int(don_code.sys.argv[4]))
exponent = float(don_code.sys.argv[5])
do_mio   = bool(int(don_code.sys.argv[6]))
dostacked = "False"
dotest  = True
#lrtag = int(don_code.sys.argv[2])
#size = 
plotmode = 0
ratioplot = True
offdiagplot = True

startind = 11

nets_dir = don_code.nets_dir

don_code.plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

if neptag == "500":
    num_points = 100
else:
    num_points = 80

def relu(x):
    return x*(don_code.np.sign(x)+1.0)/2

def fancyshow(matrix, ax, fig, minval, maxval):
    neg_mask = matrix < 0
    pos_mask = matrix >= 0

    # Prepare masked arrays
    neg_data = don_code.np.ma.masked_where(~neg_mask, -matrix)  # Flip sign for log scale
    pos_data = don_code.np.ma.masked_where(~pos_mask, matrix)

    # Create figure and axes
    #fig, ax = don_code.plt.subplots()

    # Plot negative values with custom normalization and colormap A
    #print(don_code.np.max(relu(-matrix)))
    cmap_neg = matplotlib.colormaps.get_cmap('Blues') #cm.get_cmap('Blues')  # Example for negatives
    norm_neg = LogNorm(vmin=1e-15, vmax=-minval) #np.max(relu(-matrix)))  # Because we flipped -1 → 1

    im1 = ax.imshow(neg_data, cmap=cmap_neg, norm=norm_neg)

    # Plot positive values with colormap B
    cmap_pos = matplotlib.colormaps.get_cmap('Reds') #cm.get_cmap('hot')  # Example for positives
    norm_pos = LogNorm(vmin=1e-15, vmax=maxval) #np.max(relu(matrix)))

    im2 = ax.imshow(pos_data, cmap=cmap_pos, norm=norm_pos)
    #fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.2, label='Negative values (-log scale)')
    #fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.06, label='Positive values (log scale)')

line_ids = [9, 19, 29, 39] #, 16]



if dotest:
    fig2, axs2 = don_code.plt.subplots(3, len(line_ids), figsize=(10,8))
    fig1, axs1 = don_code.plt.subplots(1, 3, figsize=(10,5))
else:
    fig2, axs2 = don_code.plt.subplots(2, len(line_ids), figsize=(10,6))
    fig1, axs1 = don_code.plt.subplots(1, 2, figsize=(10,5))



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

alldirecs = don_code.os.listdir(nets_dir+"/")

'''
if exponent == 0.0:
    alldirecs  = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w237_llw50_batburgers_dt0.0001_nc10_m3800_999_numd1000_lrSGD32_v0"]

else:
    alldirecs  = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-1.0_Nep4000_d5_w237_llw50_batburgers_dt0.0001_nc10_m3800_100_numd1000_lrSGD32_v0"]



if exponent == 0.0:
    alldirecs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0"]

else:
    alldirecs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp"+str(exponent)+"_Nep4000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0"]
'''

#                   whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w237_llw50_batburgers_dt0.0001_nc10_m3800_100_numd1000_lrSGD32_v0
#                   whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w237_llw50_batburgers_dt0.0001_nc10_m3800_100_numd1000_lrSGD32_v0

for w in wun_s:
    if neptag == "500":
        direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep500_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0")
    else:
        direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp"+str(exponent)+"_Nep4000_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0")
        direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp"+str(exponent)+"_Nep10000_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0")
    #direcs.append("whichT0_doStacked"+dostacked+"_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam32_v0")
    #direcs.append("whichT0_doStacked"+dostacked+"_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam32_v0")

#colors  = ["red",   "blue",      "orange",     "green","cyan","pink","brown","gray","purple","lightblue","lightgreen"]
#colors = ["darkorange", "brown", "purple", "cyan", "green"]

def get_colors():
    f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/colors.txt", "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()


_, _, llw, _, batch_name, num_data, endtag = don_code.get_dwllw(direcs[0])
print(batch_name, endtag, num_data, llw)

nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, endtag, num_data)
n_train, m_train = don_code.np.shape(utrain)
n_test,  m_test  = don_code.np.shape(utest)

uu_train, ss_train, vh_train = don_code.jnp.linalg.svd(utrain, full_matrices=False)



modeids = [0,1,2]

#if neptag == "500":

bigname = "log_diagoffdiag_new"

#else:
#    bigname = "log_diagoffdiag_big1e-08"
#fac = 100 #00_000

#1e-08




offdiags_x = []
offdiags_y = []
offdiags_label = []

min_loss_red = 1
max_loss_red = 0

k = 0



#ax3.legend()
num_epochs_loss = 80
k = 0



#figspec, axspec = don_code.plt.subplots(1, figsize=(6,6))


maxval_tr = -10.0
minval_tr = +10.0
maxval_te = -10.0
minval_te = +10.0

maxval_ratio_diag = 1.0
minval_ratio_diag = 0.0 #+1.0

for dir in direcs:
    if dir in alldirecs:
        print("YES found", dir)
        lossdata      = don_code.np.loadtxt(nets_dir+"/"+dir+"/log.txt")
        epochs0       = lossdata[:num_epochs_loss:2,0]
        tr_loss       = 10**(lossdata[:num_epochs_loss:2,1]/2)
        te_loss       = 10**(lossdata[:num_epochs_loss:2,4]/2)
        bigdata       = don_code.np.loadtxt(nets_dir+"/"+dir+"/"+bigname+".txt")
        epochs        = bigdata[:,0]
        tr_diags      = bigdata[:,1]
        tr_offdiags   = bigdata[:,2]
        tr_taylorgrad = bigdata[:,3]
        tr_actualdiff = bigdata[:,4]
        te_diags      = bigdata[:,5]
        te_offdiags   = bigdata[:,6]
        te_taylorgrad = bigdata[:,7]
        te_actualdiff = bigdata[:,8]

        print("epochs     ", epochs)
        print("tr diags   ", tr_diags)
        print("tr offdiags", tr_offdiags)
        print("tr taylor  ", tr_taylorgrad)
        print("tr true    ", tr_actualdiff)
        print("te diags   ", te_diags)
        print("te offdiags", te_offdiags)
        print("te taylor  ", te_taylorgrad)
        print("te true    ", te_actualdiff)

        for ij,j in enumerate(line_ids):
            base_loss_train          = bigdata[j,startind               :startind+llw]
            base_loss_test           = bigdata[j,startind+llw           :startind+2*llw]
            #update_matrix_train_flat = bigdata[j,startind+2*llw+0*llw**2:startind+2*llw+1*llw**2] 
            #update_matrix_test_flat  = bigdata[j,startind+2*llw+1*llw**2:startind+2*llw+2*llw**2]
            VTV_train_flat           = bigdata[j,startind+2*llw+0*llw**2:startind+2*llw+1*llw**2] 
            VTV_test_flat            = bigdata[j,startind+2*llw+1*llw**2:startind+2*llw+2*llw**2]

            print(don_code.np.sum(VTV_train_flat), don_code.np.sum(VTV_test_flat))

            #update_matrix_train = don_code.np.reshape(update_matrix_train_flat, (llw,llw)) -don_code.np.outer(base_loss_train,don_code.np.ones(llw))
            #update_matrix_test  = don_code.np.reshape(update_matrix_test_flat, (llw,llw)) -don_code.np.outer(base_loss_test, don_code.np.ones(llw))
            VTV_train           = don_code.np.reshape(VTV_train_flat, (llw,llw))
            VTV_test            = don_code.np.reshape(VTV_test_flat, (llw,llw))

            #print(ij, j, don_code.np.max(VTV_train_flat), don_code.np.min(VTV_train_flat))
            #idspos = don_code.np.where(VTV_train > 0)
            #print(don_code.np.sum(VTV_train[idspos] / len(idspos[0])))
            maxval_tr = max(maxval_tr, don_code.np.max(VTV_train_flat))
            minval_tr = min(minval_tr, don_code.np.min(VTV_train_flat))
            maxval_te = max(maxval_te, don_code.np.max(VTV_test_flat))
            minval_te = min(minval_te, don_code.np.min(VTV_test_flat))

            maxval_ratio_diag = max(maxval_ratio_diag, don_code.np.max(don_code.np.diag(VTV_test)/don_code.np.diag(VTV_train)))
            minval_ratio_diag = min(minval_ratio_diag, don_code.np.min(don_code.np.diag(VTV_test)/don_code.np.diag(VTV_train)))
                


# plot

for dir in direcs:
    if dir in alldirecs:
        print("YES found", dir)
        modeloss_data = don_code.np.loadtxt(nets_dir+"/"+dir+"/log_modes.txt")
        #loss_data = don_code.np.loadtxt(nets_dir+"/"+direc+"/log.txt")
        #epochs = loss_data[:,0]
        #train = loss_data[:,1]/2
        #test  = loss_data[:,4]/2
        
        train_modes = modeloss_data[:,1      :1+  llw]
        test_modes  = modeloss_data[:,1+2*llw:1+3*llw]

        lossdata      = don_code.np.loadtxt(nets_dir+"/"+dir+"/log.txt")
        epochs0       = lossdata[:num_epochs_loss:2,0]
        tr_loss       = 10**(lossdata[:num_epochs_loss:2,1]/2)
        te_loss       = 10**(lossdata[:num_epochs_loss:2,4]/2)
        bigdata       = don_code.np.loadtxt(nets_dir+"/"+dir+"/"+bigname+".txt")
        epochs        = bigdata[:,0]
        tr_diags      = bigdata[:,1]
        tr_offdiags   = bigdata[:,2]
        tr_taylorgrad = bigdata[:,3]
        tr_actualdiff = bigdata[:,4]
        te_diags      = bigdata[:,5]
        te_offdiags   = bigdata[:,6]
        te_taylorgrad = bigdata[:,7]
        te_actualdiff = bigdata[:,8]

        #if ratioplot:
        #    axs3.plot(epochs[1:], tr_offdiags[1:]/tr_diags[1:], '--', color=colors[k])
        #    axs3.plot(epochs[1:], te_offdiags[1:]/te_diags[1:], '.-', color=colors[k])


        print(epochs0[:10])
        print(epochs[:10])

        numsigndiff_tr = len(don_code.np.where(don_code.np.sign(tr_actualdiff[1:]) != don_code.np.sign(tr_taylorgrad[1:]))[0]) / len(epochs[1:])
        numsigndiff_te = len(don_code.np.where(don_code.np.sign(te_actualdiff[1:]) != don_code.np.sign(te_taylorgrad[1:]))[0]) / len(epochs[1:])


        if dotest:
            axs1[0].plot(epochs0, te_loss, '.-', color=colors[0], label="Test") 
        axs1[0].plot(epochs0, tr_loss, '--', color=colors[1], label="Training") #, label=d.split("Nep")[1].split("bat")[0])
        axs1[0].set_yscale("log")
        axs1[0].set_ylabel(r"Relative Error $\delta$", fontsize=12)
        axs1[0].legend(fontsize=12, loc='upper right')
        

        axs1[1].plot(epochs[1:], 0*tr_actualdiff[1:], '-.', color="gray")
        axs1[1].plot(epochs[1:], tr_actualdiff[1:], '.-', color=colors[0], label=r"Loss Change $\Delta \mathcal{L}$")
        axs1[1].plot(epochs[1:], tr_diags[1:] + tr_offdiags[1:], '--', color=colors[4], label=r"Taylor Ex. $d+\Omega$")
        axs1[1].plot(epochs[1:], tr_diags[1:], '.-', color=colors[1], label=r"Diag $d$")
        axs1[1].plot(epochs[1:], tr_offdiags[1:], '.-', color=colors[2], label=r"Off-Diag $\Omega$")

        axs1[1].set_ylim((-0.012, 0.012))
        axs1[1].set_ylabel(r"Training Loss Change Contributions", fontsize=12)
        axs1[1].legend(fontsize=12, loc='upper right')
        
        if dotest:
            axs1[2].plot(epochs[1:], 0*te_actualdiff[1:], '-.', color="gray")
            axs1[2].plot(epochs[1:], te_actualdiff[1:], '.-', color=colors[0], label=r"Loss Change $\Delta \mathcal{L}$")
            axs1[2].plot(epochs[1:], te_diags[1:] + te_offdiags[1:], '--', color=colors[4], label=r"Taylor Ex. $d+\Omega$")
            axs1[2].plot(epochs[1:], te_diags[1:], '.-', color=colors[1], label=r"Diag $d$")
            axs1[2].plot(epochs[1:], te_offdiags[1:], '.-', color=colors[2], label=r"Off-Diag $\Omega$")

            axs1[2].set_ylim((-0.012, 0.012))
            axs1[2].set_ylabel(r"Test Loss Change Contributions", fontsize=12)
            axs1[2].legend(fontsize=12, loc='upper right')

        #axs1[2].plot(epochs[1:], tr_offdiags[1:] / (tr_diags[1:] + tr_offdiags[1:]), '.-', color=colors[0])
        #axs1[2].set_ylabel(r"Relative Coupling Strength $\gamma = \Omega/(d+\Omega)$", fontsize=12)
        #axs1[2].legend(fontsize=12, loc='upper right')

        '''

        axs1[0].plot(epochs[1:], )

        ax1.plot(epochs[1:], 0*epochs[1:], '-.', color="gray")
        ax2.plot(epochs[1:], 0*epochs[1:], '-.', color="gray")
        if k==0:
            ax1.plot(epochs[1:], tr_actualdiff[1:], '*-', color=colors[k], label="Loss change")
            if "SGD" not in dir or numsigndiff_tr > 1/len(epochs[1:]):
                ax1.plot(epochs[1:], tr_taylorgrad[1:], 'o', color=colors[k], label="TaylorGD")
            ax1.plot(epochs[1:], tr_diags[1:], ':', color=colors[k], label=r"Diag $d$")
            ax1.plot(epochs[1:], tr_offdiags[1:], '--', color=colors[k], label=r"Off-Diag $\omega$")
        else:
            ax1.plot(epochs[1:], tr_actualdiff[1:], '*-', color=colors[k])
            if "SGD" not in dir or numsigndiff_tr > 1/len(epochs[1:]):
                ax1.plot(epochs[1:], tr_taylorgrad[1:], 'o', color=colors[k])
            ax1.plot(epochs[1:], tr_diags[1:], ':', color=colors[k])
            ax1.plot(epochs[1:], tr_offdiags[1:], '--', color=colors[k])

        
        ax1.set_title("Train : Diff sign taylor v act change "+str(round(numsigndiff_tr*100,2))+" %")


        if plotmode != 3:
            ax2.plot(epochs[1:], te_actualdiff[1:], '*-', color=colors[k])
            if "SGD" not in dir or numsigndiff_te > 1/len(epochs[1:]):
                ax2.plot(epochs[1:], te_taylorgrad[1:], 'o', color=colors[k])
            ax2.plot(epochs[1:], te_diags[1:], ':', color=colors[k])
            ax2.plot(epochs[1:], te_offdiags[1:], '--', color=colors[k])
            ax2.set_title("Test  : Diff sign taylor v act change "+str(round(numsigndiff_te*100,2))+" %")
    
        print("Width", wun_s)
        print("Train : Diff sign taylor v act change "+str(round(numsigndiff_tr*100,2))+" %")
        print("Test  : Diff sign taylor v act change "+str(round(numsigndiff_te*100,2))+" %")
        ax1.legend()
        tmp = dir.split("_bat")
        fig.suptitle(tmp[0]+"\n"+tmp[1]+"\n"+bigname)

        if plotmode == 2:
            ax4.plot(tr_actualdiff[1:], tr_taylorgrad[1:], '.-', color='red')
            ax4.plot(te_actualdiff[1:], te_taylorgrad[1:], '.-', color='blue')

        if plotmode == 1:
            ax4.plot(-tr_taylorgrad[1:], +tr_taylorgrad[1:], '--', color="gray", alpha=0.5)
            ax4.plot(-tr_taylorgrad[1:], -tr_taylorgrad[1:], '--', color="gray", alpha=0.5)
            ax4.plot(-tr_taylorgrad[1:], +0.1*tr_taylorgrad[1:], '--', color="k", alpha=0.5)
            ax4.plot(-tr_taylorgrad[1:], -0.1*tr_taylorgrad[1:], '--', color="k", alpha=0.5)
            ax4.plot(-tr_taylorgrad[1:], tr_offdiags[1:], '.-', color=colors[k])
            
            ax5.plot(-te_taylorgrad[1:], +te_taylorgrad[1:], '--', color="gray", alpha=0.5)
            ax5.plot(-te_taylorgrad[1:], -te_taylorgrad[1:], '--', color="gray", alpha=0.5)
            ax5.plot(-te_taylorgrad[1:], +0.1*te_taylorgrad[1:], '--', color="k", alpha=0.5)
            ax5.plot(-te_taylorgrad[1:], -0.1*te_taylorgrad[1:], '--', color="k", alpha=0.5)
            ax5.plot(-te_taylorgrad[1:], te_offdiags[1:], '.-', color=colors[k])

            if offdiagplot:
                min_loss_red = min(min_loss_red, don_code.np.min(-tr_taylorgrad[1:]))
                max_loss_red = max(max_loss_red, don_code.np.max(-tr_taylorgrad[1:]))
                offdiags_x.append(-tr_taylorgrad[1:])
                offdiags_y.append(tr_offdiags[1:])
                offdiags_label.append(dir.split("Nep")[1].split("bat")[0].split("_")[2])

            #ax4.plot(tr_loss[1:], +tr_loss[1:], '--', color="gray", alpha=0.5)
            #ax4.plot(tr_loss[1:], -tr_taylorgrad[1:], '--', color="gray", alpha=0.5)
            #ax4.plot(-tr_taylorgrad[1:], +0.1*tr_taylorgrad[1:], '--', color="k", alpha=0.5)
            #ax4.plot(-tr_taylorgrad[1:], -0.1*tr_taylorgrad[1:], '--', color="k", alpha=0.5)
            ax6.plot(tr_loss[1:], tr_offdiags[1:], '.-', color=colors[k])
            
            #ax5.plot(-te_taylorgrad[1:], +te_taylorgrad[1:], '--', color="gray", alpha=0.5)
            #ax5.plot(-te_taylorgrad[1:], -te_taylorgrad[1:], '--', color="gray", alpha=0.5)
            #ax5.plot(-te_taylorgrad[1:], +0.1*te_taylorgrad[1:], '--', color="k", alpha=0.5)
            #ax5.plot(-te_taylorgrad[1:], -0.1*te_taylorgrad[1:], '--', color="k", alpha=0.5)
            ax7.plot(te_loss[1:], te_offdiags[1:], '.-', color=colors[k])


            ax4.set_xscale("log")
            ax5.set_xscale("log")
            ax6.set_xscale("log")
            ax7.set_xscale("log")
        '''
        k += 1

        for ij,j in enumerate(line_ids):
            base_loss_train          = bigdata[j,startind               :startind+llw]
            base_loss_test           = bigdata[j,startind+llw           :startind+2*llw]
            #update_matrix_train_flat = bigdata[j,startind+2*llw+0*llw**2:startind+2*llw+1*llw**2] 
            #update_matrix_test_flat  = bigdata[j,startind+2*llw+1*llw**2:startind+2*llw+2*llw**2]
            VTV_train_flat           = bigdata[j,startind+2*llw+0*llw**2:startind+2*llw+1*llw**2] 
            VTV_test_flat            = bigdata[j,startind+2*llw+1*llw**2:startind+2*llw+2*llw**2]

            #update_matrix_train = don_code.np.reshape(update_matrix_train_flat, (llw,llw)) -don_code.np.outer(base_loss_train,don_code.np.ones(llw))
            #update_matrix_test  = don_code.np.reshape(update_matrix_test_flat, (llw,llw)) -don_code.np.outer(base_loss_test, don_code.np.ones(llw))
            VTV_train           = don_code.np.reshape(VTV_train_flat, (llw,llw))
            VTV_test            = don_code.np.reshape(VTV_test_flat, (llw,llw))

            print(j, don_code.np.sum(ss_train[:llw]**2 * base_loss_train))
            
            #plt.figure()
            
            #fig, axs = don_code.plt.subplots(2,1)
            #fig.suptitle(dir+"\n"+bigname+"\nEpoch: "+str(epochs[j])+", taylor tr: "+str(don_code.np.sum(VTV_train))+", taylor te: "+str(don_code.np.sum(VTV_test)))
            '''
            for modeid in modeids:
                upd_tr = ss_train[:llw]**2 * update_matrix_train[:,modeid] / (n_train * m_train)
                upd_te = ss_train[:llw]**2 * update_matrix_test[:,modeid] / (n_test * m_test)
                fac = don_code.np.max(upd_tr) / don_code.np.max(VTV_train[:,modeid])
                scaled_gr_tr = fac*VTV_train[:,modeid]
                scaled_gr_te = fac*VTV_test[:,modeid]
                #axs[0,0].plot(upd_tr, color='red')
                #axs[0,0].plot(upd_te, color='blue')
                #axs[0,1].plot(scaled_gr_tr, color='red')
                #axs[0,1].plot(scaled_gr_te, color='blue')

                print(don_code.np.sum( (upd_tr-scaled_gr_tr)**2 ) / don_code.np.sum( (scaled_gr_tr)**2 ))
                print(don_code.np.sum( (upd_te-scaled_gr_te)**2 ) / don_code.np.sum( (scaled_gr_te)**2 ))
            '''
            #A = don_code.np.matmul(update_matrix_train, don_code.np.diag(ss_train[:llw]**2)) / (n_train * m_train)
            #B = don_code.np.matmul(update_matrix_test, don_code.np.diag(ss_train[:llw]**2))/ (n_test * m_test)
            #megamatrix = don_code.np.concatenate((A,B), axis=1)
            #fancyshow(megamatrix, axs[0], fig)
            ##axs[1,0].imshow(megamatrix)
            #megamatrix = don_code.np.concatenate((VTV_train, VTV_test), axis=1)
            #fancyshow(megamatrix, axs[1], fig)
            
            axs2[0, ij].set_title("Epoch "+str(int(epochs[j]-1)))
            axs2[0, ij].plot(ss_train[:llw]**2/m_train, '.-', color="k")
            axs2[0, ij].plot(ss_train[:llw]**2 * train_modes[2*j, :]/m_train, color="red")
            if dotest:
                axs2[0, ij].plot(ss_train[:llw]**2 * test_modes[2*j, :]/m_test, color="blue")
            axs2[0, ij].set_yscale("log")
            axs2[0, ij].set_xlim((0, 49))
            axs2[0, ij].set_xticks([]) #[0, 24, 49], ["1", "25", "50"])
            ymin = 0.5*ss_train[llw-1]**2 / m_train
            ymax = 2*ss_train[0]**2 * train_modes[0, 0]/m_train
            axs2[0, ij].set_ylim((ymin, ymax))

            if do_mio:
                ax_tmp = axs2[0, ij].twinx() #axs[1+ij].twinx()
                ax_tmp.spines["right"].set_position(("axes", 1.0))
                ax_tmp.set_yticks([])

                ax_tmp.plot(don_code.np.zeros(llw), '--', color="lightgray")
                ax_tmp.plot(0.5*don_code.np.ones(llw), '--', color="darkgray")
                ax_tmp.plot(don_code.np.ones(llw), '--', color="black")

                #ax_tmp.plot(np.diag(VTV_train)/ss_train[:llw]**2, '--', color="orange")
                
                minv = min(minval_ratio_diag, -4)
                maxv = min(maxval_ratio_diag, 5)
                pad = 0.1*(maxv-minv)
                #ax_tmp.plot(np.diag(VTV_test)/ss_train[:llw]**2, '--', color="lightblue")
                
                ax_tmp.plot(don_code.np.diag(VTV_test)/don_code.np.diag(VTV_train), '--', color="fuchsia")
                ax_tmp.set_ylim((minv-pad, maxv+pad))

            fancyshow(VTV_train, axs2[1, ij], fig2, minval_tr, maxval_tr)
            if dotest:
                fancyshow(VTV_test, axs2[2, ij], fig2, minval_te, maxval_te)

            #fancyshow(VTV_train, axspec, figspec, minval_tr, maxval_tr)


            if don_code.np.min(train_modes[2*j,:]) < 1.0:
                a = 0
                #maxid = don_code.np.max(don_code.np.where( train_modes[2*j, :] < 1.0 )[0]) + 1
                #maxid = don_code.np.argmax(ss_train[:llw]**2 * train_modes[2*j, :]/m_train)+1 #np.where( train_modes[2*j, :] < 1.0 )[0]) + 1

                #np.argmax(ss_train[:llw]**2 * train_modes[2*j, :]/m_train)
                #axs2[0, ij].plot([maxid, maxid], [ymin, ymax], '--', color='lightgreen')
                #axs2[1, ij].plot([maxid, maxid], [0, 49], '--', color='lightgreen')
                #axs2[0, ij].text(maxid+2, 0.01, r"$h="+str(maxid+1)+"$", fontsize=14)
            
            if j != 0:
                axs2[0, ij].set_yticks([])
                axs2[1, ij].set_yticks([])
            else:
                axs2[1, ij].set_yticks([0, 24, 49], ["1", "25", "50"])


            axs2[1, ij].set_xticks([0, 19, 39], ["1", "20", "40"])
            #axs2[1, ij].set_xticks([0, 24, 49], ["1", "25", "50"])


            #axs[1,1].imshow(megamatrix)
    else:
        print("not found", dir)

axs2[0,0].set_ylabel("Weighted Training\nMode Losses", fontsize=12)
axs2[1,0].set_ylabel(r"Entries of $S_{tr}$ Matrix", fontsize=12)

axs1[0].legend()
axs1[1].legend()
if dotest:
    axs1[2].legend()

fig1.text(0.45, 0.04, r"Epochs", fontsize=12)
fig2.text(0.45, 0.04, r"Mode index $i$", fontsize=12)

fig1.subplots_adjust(wspace=0.4, hspace=0.0, right=0.95, left=0.1, top=0.95, bottom=0.15)
fig2.subplots_adjust(wspace=0.0, hspace=0.0, right=0.95, left=0.1, top=0.95, bottom=0.1)

#axspec.set_xticks([0, 24, 49], ["1", "25", "50"])
#axspec.set_yticks([0, 24, 49], ["1", "25", "50"])


name = "bid"+str(bid)+"_size"+str(size_id)+"_GD"

if not do_mio:
    name += "_nomio"

fig1.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/paper_coupl/"+name+"_exp"+str(exponent)+"_1.pdf")
#if not dotest:
fig2.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/paper_coupl/"+name+"_exp"+str(exponent)+"_2.pdf")

#figspec.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/slides_text/defense/MA_Defense/imgs/coupling/"+name+"_GD.pdf")


if showplot:
    don_code.plt.show()
