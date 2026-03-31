## 1708thesisrelevant
##defenserelevant
## save weighted and unweighted mode losses during training of SVDONets trained with different exponents and plot them

#from ana_fct import *
#from don_code import *
from ... import don_code

import os
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

#bid = int(sys.argv[1])
#nep = int(sys.argv[2])
#llw = int(sys.argv[3])
#numdata = int(sys.argv[4])

bid = int(sys.argv[1])
whichones = int(sys.argv[2])
sizeid    = int(sys.argv[3])
dotrain = bool(int(sys.argv[4]))

lrtag     = 32 #default is 32
#wfix      = 50

#wun = int(sys.argv[3])


            

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

wst = wst_s[sizeid]
wun = wun_s[sizeid]

batch_name, uendtag, _ = don_code.dic(bid)

if whichones == 0:
    direcs = [#"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              #"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              ]
    num_columns = 2
elif whichones == 1:
    direcs = [#"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
             #"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
             "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
             "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            ]
    num_columns = 2
elif whichones == 2:
    direcs = ["whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
            ]
    num_columns = 4

elif whichones == 20:
    direcs = ["whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
            ]
    num_columns = 2

elif whichones == 21:
    direcs = [
              # SGD
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wfix)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wfix)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            ]
    num_columns = 3

elif whichones == 22:
    direcs = [
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0_upto5kepochs",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wfix)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wfix)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
            ]
    num_columns = 3

elif whichones == 3:
    direcs = ["whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp-1.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp-0.5_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
              #"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.5_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              #"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp1.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            ]
    num_columns = 3


elif whichones == 4:
    direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            ]
    num_columns = 3

elif whichones == 5:
    direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
            ]
    num_columns = 2

elif whichones == 10:
    direcs = []
    for wun in wun_s:
        direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0")
        direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0")
    num_columns = len(wun_s)

#direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep20000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD"+str(lrtag)+"_v0",
#          #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam"+str(lrtag)+"_v0"
#          ]
#num_columns = 1
#num_epochs = 400

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
alldirecs = os.listdir(don_code.nets_dir)

for direc in direcs:
    print("try", direc)
    if don_code.contains_all(direc, tags) and direc in alldirecs:
        tmp_direcs = os.listdir(don_code.nets_dir+"/"+direc)
        if "log_modes.txt" in tmp_direcs:
            print(k, direc)
            k += 1
        print("yes")

kmax = k+1


epochs0 = [19, 39, 59, 79]

epoch0 = 19

fig1, axs1_tmp = plt.subplots(1, num_columns, figsize=(10,6))
fig2, axs2_tmp = plt.subplots(1, num_columns, figsize=(10,6))

if num_columns == 1:
    axs1 = [axs1_tmp]
    axs2 = [axs2_tmp]
else:
    axs1 = axs1_tmp
    axs2 = axs2_tmp



if num_columns == 3:
    exp = [0, -0.5, -1]
else:
    exp = [0, -1]

xx = [plt.subplots(1, figsize=(8,6)) for _ in range(num_columns)]
figsi = []
axssi = []
for x in xx:
    f,a = x
    figsi.append(f)
    axssi.append(a)




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



if dotrain:
    scaled_base_loss = ss_train[:llw]**2
else:
    scaled_base_loss = base_loss_test**2 / mtest * mtrain

sizefac_test = mtrain/mtest

if dotrain:
    sizefac = 1.0
else:
    sizefac = sizefac_test

min1 = 10
min2 = 0.025
max1 = 1
max2 = np.max(ss_train[:llw]**2)





#names = [r"Normal loss ($\sigma_j^2$)", r"Less weight ($\sigma_j$)", r"Equal weights ($1$)"]
names = [r"Normal loss ($\sigma_j^2$)", r"Equal weights ($1$)"]

for direc in direcs:
    if don_code.contains_all(direc, tags) and direc in alldirecs:
        tmp_direcs = os.listdir(don_code.nets_dir+"/"+direc)
        if "log_modes.txt" in tmp_direcs and k < kmax:
            direcs_used.append(direc+"\n")

            modeloss_data = np.loadtxt(don_code.nets_dir+"/"+direc+"/log_modes.txt")
            loss_data = np.loadtxt(don_code.nets_dir+"/"+direc+"/log.txt")
            epochs = loss_data[:,0]
            train = loss_data[:,1]/2
            test  = loss_data[:,4]/2
            #test_loss  = loss_data[:,4]
            #imin_test  = np.argmin(test_loss)
            print(k, direc, np.shape(modeloss_data))



            #tr(i)+" "+(mode_losses_train)+(min_mode_losses_train)+(mode_losses_test)+stringiy(min_mode_losses_test)

            train_modes = modeloss_data[:,1      :1+  llw]
            test_modes  = modeloss_data[:,1+2*llw:1+3*llw]

            if dotrain:
                mode_losses = train_modes   
                col = "red"
            else:
                mode_losses = test_modes
                col = "blue"

            print(k, direc, names[k])

            ytmp = mode_losses[epoch0, :]*sizefac
            min1 = min(min1, np.min(ytmp))
            max1 = max(max1, np.max(ytmp))
            axs1[k].plot(np.ones(llw), '.-', color="gray")
            axs1[k].plot(ytmp, '.-', label=names[k], color=col)
            axs1[k].set_yscale("log")
            #axs1[k].legend(loc="lower right", fontsize=20)
            axs1[k].set_title(names[k], fontsize=20)

            ytmp = ss_train[:llw]**2 * mode_losses[epoch0, :]*sizefac #/mtrain
            min2 = min(min2, np.min(ytmp))
            max2 = max(max2, np.max(ytmp))
            axs2[k].plot(ss_train[:llw]**2, '.-', color="gray")
            #axs2[k].plot(scaled_base_loss, '.-', color="gray")
            axs2[k].plot(ytmp, '.-', label=names[k], color=col) #(0.2+0.8*e/80, 0, 0))
            toterr = np.sqrt(np.sum(ytmp) / np.sum(scaled_base_loss))
            axs2[k].text(llw, max2, "Error="+str(round(toterr*100,1))+"\%", horizontalalignment='right', verticalalignment='top', fontsize=20)
            axs2[k].set_yscale("log")
            axs2[k].set_title(names[k], fontsize=20)
            #axs2[k].legend(loc="lower left", fontsize=20)


            axssi[k].plot(ss_train[:llw]**2, '.-', color="gray")
            axssi[k].plot(ss_train[:llw]**2 * train_modes[epoch0, :], '.-', color="red", label="Training Data") #(0.2+0.8*e/80, 0, 0))
            axssi[k].plot(ss_train[:llw]**2 * test_modes[epoch0, :]*sizefac_test, '.-', color="blue", label="Test Data") #(0.2+0.8*e/80, 0, 0))
            axssi[k].set_title(names[k], fontsize=20)
            axssi[k].legend(loc="upper right", fontsize=20)
            axssi[k].set_yscale("log")
            axssi[k].set_ylim((0.9*0.025, 1.1*ss_train[0]**2))


            '''

            ytmp = ss_train[:llw]**2/mtrain
            ymin_tr = min(ymin_tr, np.min(ytmp))
            ymax_tr = max(ymax_tr, np.max(ytmp))


            axs[1+2*k].plot(ytmp, '.-', color="k") #(0.2+0.7*j/num_epochs, 0, 0))

            ytmp = base_loss_test**2/mtest
            ymin_te = min(ymin_te, np.min(ytmp))
            ymax_te = max(ymax_te, np.max(ytmp))
            axs[2+2*k].plot(ytmp, '.-', color="k") #, color=(0, 0, 0.2+0.7*j/num_epochs))

            last_ie = 0
            for j in range(0, num_epochs, 4):
                #for j in range(0, np.shape(train_modes)[0], 10):
                ytmp = ss_train[:llw]**2 * train_modes[j, :]/mtrain
                ymin_tr = min(ymin_tr, np.min(ytmp))
                ymax_tr = max(ymax_tr, np.max(ytmp))
                axs[1+2*k].plot(ytmp, '--', color=(0.2+0.7*j/num_epochs, 0, 0), alpha=0.5)

                ytmp = ss_train[:llw]**2 * test_modes[j, :]/mtest                
                ymin_te = min(ymin_te, np.min(ytmp))
                ymax_te = max(ymax_te, np.max(ytmp))
                axs[2+2*k].plot(ytmp, '--', color=(0, 0, 0.2+0.7*j/num_epochs), alpha=0.5)
                last_ie = j

            tmp = axs[1+2*k].text(1.05*llw, 1.0*ymax_tr, title, color="k", fontsize=14, horizontalalignment='right', verticalalignment='top')
            tmp.set_bbox(dict(facecolor=colors[k], alpha=0.4, edgecolor=colors[k]))

            tmp = axs[2+2*k].text(1.05*llw, 1.0*ymax_te, title, color="k", fontsize=14, horizontalalignment='right', verticalalignment='top')
            tmp.set_bbox(dict(facecolor=colors[k], alpha=0.4, edgecolor=colors[k]))


            axs[1+2*k].set_xlim((xmin, xmax)) #-0.1*llw, 1.1*llw))
            axs[2+2*k].set_xlim((xmin, xmax)) #())

            #if k != num_columns-1:
            '''

            '''
            axs[1+2*k].set_xticks([])
            if llw == 50:
                axs[2+2*k].set_xticks([0, 24, 49], ["1", "25", "50"])
                #if num_columns == 5:
                #    if k == 2:
                #        axs[2+2*k].set_xlabel(r"Mode index $i$", fontsize=12)
                #else:
                #    axs[2+2*k].set_xlabel(r"Mode index $i$", fontsize=12)
            '''
            #axs[2+2*k].set_xticks([])
                    



            #    #print(modeloss_data[j,0], "Train", np.log10(np.sum(ss_train[:llw]**2 * train_modes[j, :]) / utrain_norm))
            #    #print(modeloss_data[j,0], "Test ", np.log10(np.sum(ss_train[:llw]**2 * test_modes[j, :]) / utest_norm))
            #    #print(int(loss_data[j, 0]), loss_data[j, 8], loss_data[j, 9])

            #j = num_epochs-3
            #last_ie = j
            #axs[1+2*k].plot(ss_train[:llw]**2 * train_modes[j, :]/mtrain, '--', color=(0.2+0.7*j/num_epochs, 0, 0), alpha=0.5)
            #axs[2+2*k].plot(ss_train[:llw]**2 * test_modes[j, :]/mtest, '--', color=(0, 0, 0.2+0.7*j/num_epochs), alpha=0.5)
            

            #if "Adam" in direc and "doStackedFalse" in direc:
            #    y = ss_train[:llw]**2 * test_modes[last_ie, :]/mtest
            #    print(np.max(y), np.argmax(y))
            #    standard_value_adam = np.max(y)
            '''
            last_trains.append(train_modes[last_ie, :])
            last_tests.append(test_modes[last_ie, :])
            
            if "doStackedFalse" in direc and "exp0.0" in direc and "_w50" not in direc:
                if "SGD" in direc:
                    y = ss_train[:llw]**2 * test_modes[last_ie-4, :]/mtest
                    standard_value_gd_te = np.max(y)
                    
                    y = ss_train[:llw]**2 * train_modes[last_ie-4, :]/mtrain
                    standard_value_gd_tr = np.max(y)
                else:
                    y = ss_train[:llw]**2 * test_modes[last_ie-4, :]/mtest
                    standard_value_adam_te = np.max(y)
                    
                    y = ss_train[:llw]**2 * train_modes[last_ie-4, :]/mtrain
                    standard_value_adam_tr = np.max(y)


            axs[1+2*k].set_yscale("log")
            axs[2+2*k].set_yscale("log")
            '''

            k += 1

            '''
            minv = modeloss_data[imin_test, 1+2*llw:1+3*llw]
            last = modeloss_data[-1,        1+2*llw:1+3*llw]
            best = modeloss_data[-1,        1+3*llw:1+4*llw]


            sorted_ids = np.argsort(-fac*minv)
            cumsum     = np.cumsum(fac[sorted_ids]*minv[sorted_ids])
            top90ids   = np.where(cumsum < 0.9*cumsum[-1])[0]
            low10ids   = np.where(cumsum >= 0.9*cumsum[-1])[0]
            #print(top90ids)
            #for i in top90ids:
            #    j = sorted_ids[i]
            #    print(j, fac[j]*last[j], cumsum[i], cumsum[-1])
            
            top90_diff_sum  = np.sum(last[top90ids] - minv[top90ids])
            low10_diff_sum  = np.sum(last[low10ids] - minv[low10ids])
            top90_sdiff_sum  = np.sum(fac[top90ids]*last[top90ids] - fac[top90ids]*minv[top90ids])
            low10_sdiff_sum  = np.sum(fac[low10ids]*last[low10ids] - fac[low10ids]*minv[low10ids])
            top90_ratio_sum = np.sum(last[top90ids] / minv[top90ids]) - len(top90ids)
            low10_ratio_sum = np.sum(last[low10ids] / minv[low10ids]) - len(low10ids)
            
            print("min top", np.min(sorted_ids[top90ids]), "max top", np.max(sorted_ids[top90ids]), "min low", np.min(sorted_ids[low10ids]))
            print("diff", top90_diff_sum, low10_diff_sum, "sdiff", top90_sdiff_sum, low10_sdiff_sum, "ratio", top90_ratio_sum, low10_ratio_sum)
            print(" ")
            '''


for kk in range(k):
    axs1[kk].set_ylim((0.8*min1, max1*1.2))
    axs2[kk].set_ylim((0.8*min2, max2*1.2))

fig1.text(0.4, 0.02, r"Mode index $j$", fontsize=20)
axs1[0].set_ylabel(r"Unweighted Mode Loss $L_j$", fontsize=20)


fig2.text(0.4, 0.02, r"Mode index $j$", fontsize=20)
axs2[0].set_ylabel(r"Weighted Mode Loss $\sigma_j^2 L_j$", fontsize=20)

path = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/slides_text/defense/MA_Defense/imgs/losses/"



for i in range(num_columns):
    axssi[i].set_ylabel(r"Weighted Mode Loss $\sigma_j^2 L_j$", fontsize=20)
    axssi[i].set_xlabel(r"Mode index $j$", fontsize=20)
    figsi[i].savefig(path+"wo"+str(whichones)+"_e"+str(i)+"_size"+str(sizeid)+"_wei_exp"+str(exp[i])+".pdf")


##fig1.savefig(path+"wo"+str(whichones)+"_e"+str(i)+"_size"+str(sizeid)+"_unw_train"+str(int(dotrain))+".pdf")
##fig2.savefig(path+"wo"+str(whichones)+"_e"+str(i)+"_size"+str(sizeid)+"_wei_train"+str(int(dotrain))+".pdf")

#for i in range(len(epochs0)):
#    figs1[i].savefig(path+"wo"+str(whichones)+"_e"+str(i)+"_size"+str(sizeid)+"_unw.pdf")
#    figs2[i].savefig(path+"wo"+str(whichones)+"_e"+str(i)+"_size"+str(sizeid)+"_wei.pdf")
#    #            figs2[i].suptitle("Epoch "+str((epochs0[i]+1)*50), fontsize=20)

plt.show()