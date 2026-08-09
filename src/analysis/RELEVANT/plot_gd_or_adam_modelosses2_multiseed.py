## 1708thesisrelevant
##defenserelevant
## plot and save weighted and unweighted mode losses for one file and multiple training stages/epochs
#tag:10_02_2026


from ... import don_code

import os
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec


bid = int(sys.argv[1])

whichones = int(sys.argv[2])
sizeid    = int(sys.argv[3])
dotrain   = int(sys.argv[4])

do_keys = {}
fontsize=10
k = 0
if dotrain > 0:
    do_keys[k] = "train"
    k += 1

if dotrain%2 == 0:
    do_keys[k] = "test"
    k += 1

num_columns = k

lrtag     = 32 #default is 32
#wfix      = 50

#wun = int(sys.argv[3])

nets_dir = don_code.nets_dir

## ---------------------------------------------------------------------------
## MULTISEED VARIANT of plot_gd_or_adam_modelosses2.py
## The plotted mode losses are the ARITHMETIC MEAN over seeds v0..v9.  Mode
## losses are stored raw in log_modes.txt (not log10), so averaging the file
## directly already gives the arithmetic mean of the plotted quantity.
## log.txt is averaged too for completeness, but neither figure plots it.
## Set SHOW_BAND = True to shade the min..max spread across seeds.
## ---------------------------------------------------------------------------
from . import multiseed
# Bands are driven by the MULTISEED_BAND env var (see multiseed.py); set this to
# True to force them on for this script alone.
SHOW_BAND = multiseed.SHOW_BAND


        
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

def get_colors():
    f = open(don_code.colors_path, "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()


if bid < 2:
    wun_s = [50, 100, 222, 332, 495]
    wst_s = [-1,  -1, -1]
    llw = 20

elif bid < 6:
    wun_s = [220, 335, 495]
    wst_s = [13,  24, 42]
    llw = 50
    
else:
    wun_s = [237, 337, 494]
    wst_s = [29,  43,  65]
    llw = 50

wst = wst_s[sizeid]
wun = wun_s[sizeid]

batch_name, uendtag, _ = don_code.dic(bid)

if whichones == 0:
    direcs = [
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
            #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              ]
    #num_columns = 1
elif whichones == 1:
    direcs = [
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            ]
    #num_columns = 1

print(f"{direcs=}")

_, _, llw, _, batch_name, num_data, endtag = don_code.get_dwllw(direcs[0])
print(batch_name, endtag)

xmin = -0.1*llw
xmax = 1.1*llw

#bid = 5
#batch_name, endtag, _ = dic(bid)
nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, endtag, 1000)

mtrain = np.shape(utrain)[1]
mtest  = np.shape(utest)[1]
tags = [] #"llw"+str(llw), batch_name, endtag+"_numdata", "whichT0_doSigma1_sisc1.0", "exp0.0", "_w300"] #, "numdata"+str(numdata)]

uu_train, ss_train, vvhh_train = np.linalg.svd(utrain, full_matrices=False)

T = uu_train[:,:llw]
VT_test = np.matmul(T.T, utest)
base_loss_test = np.zeros(llw)
for i in range(llw):
    base_loss_test[i] = np.sqrt(np.sum( VT_test[i,:]**2 ))


utrain_norm = np.sum( utrain**2 )
utest_norm  = np.sum( utest**2 )

#direcs = os.listdir("nets")

k = 0
kmax_apriori = 16
alldirecs = os.listdir(nets_dir)



for direc in direcs:
    print("try", direc)
    if don_code.contains_all(direc, tags) and direc in alldirecs:
        tmp_direcs = os.listdir(nets_dir+"/"+direc)
        if "log_modes.txt" in tmp_direcs:
            print(k, direc)
            k += 1
        print("yes")

kmax = k+1


#epochs0 = [19, 39, 59, 79]
plot_epoch_ids = [1] #, 19, 39, 59, 79]
dk = 10
dashes = (5,2)
alphaval = 0.8
k = 1+dk
while k < 80:
    plot_epoch_ids.append(k)
    k += dk

max_epoch = max(plot_epoch_ids)
#epoch0 = 19

#fig, axs = plt.subplots(1, 4, figsize=(7,3))
fig = plt.figure(figsize=(6, 2.0))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15)  # gap between the two pairs

gs_left  = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.0)
gs_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.0)

axs = [None] * 4
axs[0] = fig.add_subplot(gs_left[0])
axs[1] = fig.add_subplot(gs_left[1],  sharey=axs[0])
axs[2] = fig.add_subplot(gs_right[0])
axs[3] = fig.add_subplot(gs_right[1], sharey=axs[2])

# Hide y-tick labels on the right subplot of each pair
axs[1].tick_params(labelleft=False)
axs[3].tick_params(labelleft=False)

k = 0


standard_value_adam_tr = 0
standard_value_adam_te = 0
standard_value_gd_tr   = 0
standard_value_gd_te   = 0

#axs[0].plot([], [], '.-', color="gray", label="Test")
#axs[0].plot([], [], '--', color="gray", label="Train")

fac = ss_train[:llw]**2

direcs_used = []

ymin_tr = ss_train[llw-1]**2/mtrain
ymax_tr = ss_train[0]**2/mtrain

ymin_te = np.min(base_loss_test)**2/mtest
ymax_te = np.max(base_loss_test)**2/mtest

last_trains = []
last_tests  = []

does_sgd = []


shifted_xax = np.arange(1, llw+1)



min1 = 10
min2 = ss_train[llw-1]**2
max1 = 1
max2 = np.max(ss_train[:llw]**2)





#names = [r"Normal loss ($\sigma_j^2$)", r"Less weight ($\sigma_j$)", r"Equal weights ($1$)"]
#names = [r"Normal loss ($\sigma_j^2$)", r"Equal weights ($1$)"]

for direc in direcs:
    if don_code.contains_all(direc, tags) and direc in alldirecs:
        tmp_direcs = os.listdir(nets_dir+"/"+direc)
        if "log_modes.txt" in tmp_direcs and k < kmax:
            direcs_used.append(direc+"\n")

            modeloss_data, ml_lo, ml_hi, nseed = multiseed.load_mean(direc, "log_modes.txt")
            loss_data, _, _, _ = multiseed.load_mean(direc, "log.txt")
            if modeloss_data is None:
                continue
            print("multiseed: %s -> mean over %d seeds" % (direc, nseed))
            epochs = loss_data[:,0]
            train = loss_data[:,1]/2
            test  = loss_data[:,4]/2
            #test_loss  = loss_data[:,4]
            #imin_test  = np.argmin(test_loss)
            print(k, direc, np.shape(modeloss_data))



            #tr(i)+" "+(mode_losses_train)+(min_mode_losses_train)+(mode_losses_test)+stringiy(min_mode_losses_test)

            epoch_modes = modeloss_data[:,0]
            train_modes = modeloss_data[:,1      :1+  llw]
            test_modes  = modeloss_data[:,1+2*llw:1+3*llw]
            # matching slices of the across-seed band, keyed like tmp_mode_losses
            _band_err = {"train": (ml_lo[:,1      :1+  llw], ml_hi[:,1      :1+  llw]),
                         "test":  (ml_lo[:,1+2*llw:1+3*llw], ml_hi[:,1+2*llw:1+3*llw])}

            tmp_mode_losses = {}
            tmp_mode_losses["train"] = {"err": train_modes, 
                                        "col": np.array([1.0,0,0]), 
                                        "sf": 1.0, 
                                        "base": ss_train[:llw]**2}
    
            tmp_mode_losses["test"]  = {"err": test_modes,  
                                        "col": np.array([0,0,1.0]), 
                                        "sf": mtrain/mtest, 
                                        "base": base_loss_test**2 / mtest * mtrain}

            #print(k, direc, names[k])

            for ik in do_keys.keys():
                unweighted_base_loss = tmp_mode_losses[do_keys[ik]]["base"] / ss_train[:llw]**2
                # or approx np.ones(llw)
                if ik == 0:
                    axs[ik].plot(shifted_xax, unweighted_base_loss, '.-', color="k", label="base")
                    print("ik == 0, base")
                    axs[2+ik].plot(shifted_xax, tmp_mode_losses[do_keys[ik]]["base"], '.-', color="k") #, label="base")
                else:
                    axs[ik].plot(shifted_xax, unweighted_base_loss, '.-', color="k") #, label="base")
                    axs[2+ik].plot(shifted_xax, tmp_mode_losses[do_keys[ik]]["base"], '.-', color="k", label="base")
                    print("ik != 0, base")
                
                    
            
            for epoch in plot_epoch_ids:
                lab = ""
                if epoch == min(plot_epoch_ids):
                    lab = "initial"
                elif epoch == max(plot_epoch_ids):
                    lab = "final"

                    
                print(epoch, ":", epoch_modes[epoch])
                for ik in do_keys.keys():
                    key = do_keys[ik]
                    mode_losses = tmp_mode_losses[key]["err"]
                    base_col    = tmp_mode_losses[key]["col"]
                    ## unweighted
                    ytmp = mode_losses[epoch, :]*tmp_mode_losses[key]["sf"]
                    _blo, _bhi = _band_err[key]
                    if multiseed.is_endpoint_stage(epoch, plot_epoch_ids):
                      multiseed.band(axs[ik], shifted_xax,
                                   _blo[epoch, :]*tmp_mode_losses[key]["sf"],
                                   _bhi[epoch, :]*tmp_mode_losses[key]["sf"],
                                   base_col * (0.2+0.7*epoch/max_epoch))
                    min1 = min(min1, np.min(ytmp))
                    max1 = max(max1, np.max(ytmp))
                    # if len(lab) == 0:
                    #     axs[ik].plot(shifted_xax, ytmp, '--', dashes=dashes, color=base_col * (0.2+0.7*epoch/max_epoch), alpha=alphaval)
                    # else:
                    axs[ik].plot(shifted_xax, ytmp, '--', dashes=dashes, color=base_col * (0.2+0.7*epoch/max_epoch), alpha=alphaval)
                    if ik == 0:
                        axs[ik].plot([], [], '--', color=base_col * (0.2+0.7*epoch/max_epoch), alpha=alphaval, label=lab)
                        print("i == 0, ", lab)
                            


                    ## weighted
                    ytmp = ss_train[:llw]**2 * mode_losses[epoch, :]*tmp_mode_losses[key]["sf"] #/mtrain
                    # weighted panels are axs[2+ik] (C/D), not axs[ik] (A/B)
                    if multiseed.is_endpoint_stage(epoch, plot_epoch_ids):
                      multiseed.band(axs[2+ik], shifted_xax,
                                   ss_train[:llw]**2 * _blo[epoch, :]*tmp_mode_losses[key]["sf"],
                                   ss_train[:llw]**2 * _bhi[epoch, :]*tmp_mode_losses[key]["sf"],
                                   base_col * (0.2+0.7*epoch/max_epoch))
                    min2 = min(min2, np.min(ytmp))
                    max2 = max(max2, np.max(ytmp))
                    # if len(lab) == 0:
                    #     axs[2+ik].plot(shifted_xax, ytmp, '--', dashes=dashes, color=base_col * (0.2+0.7*epoch/max_epoch), alpha=alphaval)
                    # else:
                    axs[2+ik].plot(shifted_xax, ytmp, '--', dashes=dashes, color=base_col * (0.2+0.7*epoch/max_epoch), alpha=alphaval)
                    if ik != 0:
                        axs[2+ik].plot([] ,[], '--', color=base_col * (0.2+0.7*epoch/max_epoch), alpha=alphaval, label=lab)
                        print("i != 0, ", lab)
                            


            for ik in do_keys.keys():
                axs[ik].set_yscale("log")
                axs[2+ik].set_yscale("log")
                axs[ik].set_title(do_keys[ik])
                axs[2+ik].set_title(do_keys[ik])
            



for ik in do_keys.keys():
    axs[ik].set_ylim((0.8*min1, max1*1.2))
    axs[2+ik].set_ylim((0.8*min2, max2*1.2))
    axs[ik].set_xticks([0, 19, 39], [1, 20, 40]) 
    axs[2+ik].set_xticks([0, 19, 39], [1, 20, 40]) 

axs[0].legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.05, -0.03),                        
    labelspacing=0.2,      # vertical space between entries (default: 0.5)
    handlelength=1.0,      # length of the legend handle/line (default: 2.0)
    handletextpad=0.4,     # space between handle and label (default: 0.8)
    borderpad=0.2,         # padding inside the legend box (default: 0.4)
    borderaxespad=0.5,     # padding between legend and axes (default: 0.5)
    columnspacing=1.0,     # space between columns (default: 2.0)
    framealpha=1.0).set_zorder(10)
axs[3].legend(fontsize=fontsize, loc='upper right', bbox_to_anchor=(1.05, 1.03),                        
    labelspacing=0.2,      # vertical space between entries (default: 0.5)
    handlelength=1.0,      # length of the legend handle/line (default: 2.0)
    handletextpad=0.4,     # space between handle and label (default: 0.8)
    borderpad=0.2,         # padding inside the legend box (default: 0.4)
    borderaxespad=0.5,     # padding between legend and axes (default: 0.5)
    columnspacing=1.0,     # space between columns (default: 2.0)
    framealpha=1.0).set_zorder(10)



axs[0].set_zorder(10)
axs[0].set_frame_on(True)          # make sure the axes frame is drawn
axs[0].patch.set_visible(True)
axs[3].set_zorder(10)
axs[3].set_frame_on(True)          # make sure the axes frame is drawn
axs[3].patch.set_visible(True)

fig.subplots_adjust(left=0.1, right=0.99, bottom=0.19, top=0.81) #wspace=0.31


axs[0].set_title(r"A) unweighted"+"\n"+r"train loss $L_j$", fontsize=fontsize)
axs[1].set_title(r"B) unweighted"+"\n"+r"test loss $L_j$", fontsize=fontsize)
axs[2].set_title(r"C) weighted"+"\n"+r"train loss $\sigma_j^2 L_j$", fontsize=fontsize)
axs[3].set_title(r"D) weighted"+"\n"+r"test loss $\sigma_j^2 L_j$", fontsize=fontsize)

axs[0].set_ylabel("mode loss", fontsize=fontsize)
fig.text(0.49, 0.02, r"mode index $j$", fontsize=fontsize)

#if len(do_keys.keys()) > 1:
#    axs1[1].set_yticks([])
#    axs2[1].set_yticks([])


# fig1.subplots_adjust(left=0.14, right=0.97, bottom=0.14, wspace=0.0)
# fig1.text(0.47, 0.02, r"mode index $j$", fontsize=fontsize)
# axs1[0].set_ylabel(r"unweighted mode loss $L_j$", fontsize=fontsize)
# axs1[0].set_xticks([0, 19, 39], [1, 20, 40], fontsize=fontsize) #label(r"unweighted mode loss $L_j$", fontsize=fontsize)
# axs1[1].set_xticks([0, 19, 39], [1, 20, 40], fontsize=fontsize) #label(r"unweighted mode loss $L_j$", fontsize=fontsize)

# fig2.subplots_adjust(left=0.14, right=0.97, bottom=0.14, wspace=0.0)
# fig2.text(0.47, 0.02, r"mode index $j$", fontsize=fontsize)
# axs2[0].set_ylabel(r"weighted mode loss $\sigma_j^2 L_j$", fontsize=fontsize)
# axs2[0].set_xticks([0, 19, 39], [1, 20, 40], fontsize=fontsize) #label(r"unweighted mode loss $L_j$", fontsize=fontsize)
# axs2[1].set_xticks([0, 19, 39], [1, 20, 40], fontsize=fontsize) #label(r"unweighted mode loss $L_j$", fontsize=fontsize)

if whichones == 1:
    tmp = "Fig3_multiseed" + multiseed.suffix()
else:
    tmp = "Fig5_multiseed" + multiseed.suffix()

#fig.savefig(path+tmp+".pdf")
# fig2.savefig(path+"wo"+str(whichones)+"_size"+str(sizeid)+"_wei_train"+str(int(dotrain))+".pdf")


path = don_code.figures_dir
#tmp = "Fig2"
fig.savefig(path+"/pdfs/"+tmp+".pdf")
fig.savefig(path+"/pngs/"+tmp+".png")

#plt.show()