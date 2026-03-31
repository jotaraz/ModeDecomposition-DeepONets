## 1708thesisrelevant
##defenserelevant
## plot and save weighted and unweighted mode losses for one file and multiple training stages/epochs

from ... import don_code

import os
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec


bid = int(sys.argv[1])

#whichones = int(sys.argv[2])
#sizeid    = int(sys.argv[3])
dotrain   = 2 #int(sys.argv[4])


do_keys = {}
k = 0
if dotrain > 0:
    do_keys[k] = "train"
    k += 1

if dotrain%2 == 0:
    do_keys[k] = "test"
    k += 1



lrtag     = 32 #default is 32
#wfix      = 50

#wun = int(sys.argv[3])

nets_dir = don_code.nets_dir
        
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

def get_colors():
    f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/colors.txt", "r")
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

#wst = wst_s[sizeid]
#wun = wun_s[sizeid]

batch_name, uendtag, _ = don_code.dic(bid)



llw = 100
num_data = 1000
endtag = uendtag
tags = [batch_name+"_"+endtag, "whichT0", "doSigma1_sisc1.0", "Adam", "exp0.0", "doStackedFalse"] 
print(tags)
#"llw"+str(llw), , ,, "exp0.0", "_w300"] #, "numdata"+str(numdata)]

#_, _, llw, _, batch_name, num_data, endtag = don_code.get_dwllw(direcs[0])
print(batch_name, endtag)

xmin = -0.1*llw
xmax = 1.1*llw

#bid = 5
#batch_name, endtag, _ = dic(bid)
nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, endtag, 1000)

mtrain = np.shape(utrain)[1]
mtest  = np.shape(utest)[1]


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
alldirecs = sorted(os.listdir(nets_dir))

direcs = alldirecs

for direc in direcs:
    #print("try", direc)
    if don_code.contains_all(direc, tags) and direc in alldirecs:
        tmp_direcs = os.listdir(nets_dir+"/"+direc)
        if "log_modes.txt" in tmp_direcs:
            print(k, direc)
            k += 1
            print("yes")

kmax = k+1
num_columns = k

#epochs0 = [19, 39, 59, 79]
plot_epoch_ids = [1] #, 19, 39, 59, 79]
dk = 7
alphaval = 0.8
k = 1+dk
while k < 80:
    plot_epoch_ids.append(k)
    k += dk

max_epoch = max(plot_epoch_ids)
#epoch0 = 19

if num_columns == 1:
    num_columns = 2

#fig1, axs1_tmp = plt.subplots(2, num_columns, figsize=(8,4), sharey=True)
fig2, axs2_tmp = plt.subplots(2, num_columns, figsize=(8,4), sharey=True)
print("num col", num_columns)

#if num_columns == 1:
#    axs1 = [axs1_tmp]
#    axs2 = [axs2_tmp]
#else:
#axs1 = axs1_tmp
axs2 = axs2_tmp




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


min1 = 10
min2 = ss_train[llw-1]**2
max1 = 1
max2 = np.max(ss_train[:llw]**2)



for direc in direcs:
    d, w, llw_loc, _, batch_name, num_data, endtag = don_code.get_dwllw(direc)
    shifted_xax = np.arange(1, llw_loc+1)
    if don_code.contains_all(direc, tags) and direc in alldirecs:
        tmp_direcs = os.listdir(nets_dir+"/"+direc)
        if "log_modes.txt" in tmp_direcs and k < kmax:
            direcs_used.append(direc+"\n")

            modeloss_data = np.loadtxt(nets_dir+"/"+direc+"/log_modes.txt")
            loss_data = np.loadtxt(nets_dir+"/"+direc+"/log.txt")
            epochs = loss_data[:,0]
            train = loss_data[:,1]/2
            test  = loss_data[:,4]/2
            #test_loss  = loss_data[:,4]
            #imin_test  = np.argmin(test_loss)
            print(k, direc, np.shape(modeloss_data), d, w)

            #tr(i)+" "+(mode_losses_train)+(min_mode_losses_train)+(mode_losses_test)+stringiy(min_mode_losses_test)

            epoch_modes = modeloss_data[:,0]
            train_modes = modeloss_data[:,1      :1+  llw_loc]
            test_modes  = modeloss_data[:,1+2*llw_loc:1+3*llw_loc]

            tmp_mode_losses = {}
            tmp_mode_losses["train"] = {"err": train_modes, 
                                        "col": np.array([1.0,0,0]), 
                                        "sf": 1.0, 
                                        "base": ss_train[:llw_loc]**2}
    
            tmp_mode_losses["test"]  = {"err": test_modes,  
                                        "col": np.array([0,0,1.0]), 
                                        "sf": mtrain/mtest, 
                                        "base": base_loss_test[:llw_loc]**2 / mtest * mtrain}

            #print(k, direc, names[k])

            axs2[0,k].set_title(f"d{d} w{w}") #plot(shifted_xax, tmp_mode_losses[do_keys[ik]]["base"], '.-', color="k")

            for ik in do_keys.keys():
                unweighted_base_loss = tmp_mode_losses[do_keys[ik]]["base"] / ss_train[:llw_loc]**2
                # or approx np.ones(llw)
                #axs1[ik,k].plot(shifted_xax, unweighted_base_loss, '.-', color="k")
                axs2[ik,k].plot(shifted_xax, tmp_mode_losses[do_keys[ik]]["base"], '.-', color="k")
            
            for epoch in plot_epoch_ids:
                if len(epoch_modes) > epoch:
                    print(epoch, ":", epoch_modes[epoch])
                    for ik in do_keys.keys():
                        key = do_keys[ik]
                        mode_losses = tmp_mode_losses[key]["err"]
                        base_col    = tmp_mode_losses[key]["col"]
                        ## unweighted
                        ytmp = mode_losses[epoch, :]*tmp_mode_losses[key]["sf"]
                        min1 = min(min1, np.min(ytmp))
                        max1 = max(max1, np.max(ytmp))
                        #axs1[ik,k].plot(shifted_xax, ytmp, '--', color=base_col * (0.2+0.7*epoch/max_epoch), alpha=alphaval)

                        ## weighted
                        ytmp = ss_train[:llw_loc]**2 * mode_losses[epoch, :]*tmp_mode_losses[key]["sf"] #/mtrain
                        min2 = min(min2, np.min(ytmp))
                        max2 = max(max2, np.max(ytmp))
                        axs2[ik,k].plot(shifted_xax, ytmp, '--', color=base_col * (0.2+0.7*epoch/max_epoch), alpha=alphaval)
                
            for ik in do_keys.keys():
                #axs1[ik,k].set_yscale("log")
                axs2[ik,k].set_yscale("log")
            k += 1


#for k in range(num_columns):
#    for ik in do_keys.keys():
#        axs1[ik,k].set_ylim((0.8*min1, max1*1.2))
#        axs2[ik,k].set_ylim((0.8*min2, max2*1.2))

'''

#if len(do_keys.keys()) > 1:
#    axs1[1].set_yticks([])
#    axs2[1].set_yticks([])

fig1.subplots_adjust(left=0.14, right=0.97, bottom=0.14, wspace=0.0)
fig1.text(0.4, 0.02, r"Mode index $j$", fontsize=20)
axs1[0].set_ylabel(r"Unweighted Mode Loss $L_j$", fontsize=20)

fig2.subplots_adjust(left=0.14, right=0.97, bottom=0.14, wspace=0.0)
fig2.text(0.4, 0.02, r"Mode index $j$", fontsize=20)
axs2[0].set_ylabel(r"Weighted Mode Loss $\sigma_j^2 L_j$", fontsize=20)

path = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/slides_text/defense/MA_Defense/imgs/losses/"

fig1.savefig(path+"wo"+str(whichones)+"_size"+str(sizeid)+"_unw_train"+str(int(dotrain))+".pdf")
fig2.savefig(path+"wo"+str(whichones)+"_size"+str(sizeid)+"_wei_train"+str(int(dotrain))+".pdf")
'''

plt.show()