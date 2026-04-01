## 1708thesisrelevant
##defenserelevant
## save weighted and unweighted mode losses during training and plot them

#from ana_fct import *
#from don_code import *
from ... import don_code

import os
import numpy as np 
import sys
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

#bid = int(sys.argv[1])
#nep = int(sys.argv[2])
#llw = int(sys.argv[3])
#numdata = int(sys.argv[4])

bid       = int(sys.argv[1])
whichones = int(sys.argv[2])
sizeid    = int(sys.argv[3])
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
    wun_s = [220, 335, 495, 50, 100]
    wst_s = [13,  24, 42, -1, -1]
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
    num_columns = 1
elif whichones == 1:
    direcs = [#"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
             #"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
             "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
             "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            ]
    num_columns = 1
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
    direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            ]
    num_columns = 3

elif whichones == 5:
    direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
            ]
    num_columns = 5

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
    if don_code.contains_all(direc, tags) and direc in alldirecs:
        tmp_direcs = os.listdir(don_code.nets_dir+"/"+direc)
        if "log_modes.txt" in tmp_direcs:
            print(k, direc)
            k += 1

kmax = k+1


epochs0 = [19, 39, 59, 79]

figaxs = [plt.subplots(1, figsize=(8, 6)) for e in epochs0] #i,c in enumerate(listcoeffs)]
figs1 = []
axss1 = []

for figax in figaxs:
    tmpf, tmpa = figax
    figs1.append(tmpf)
    axss1.append(tmpa)

figaxs = [plt.subplots(1, figsize=(8, 6)) for e in epochs0] #i,c in enumerate(listcoeffs)]
figs2 = []
axss2 = []

for figax in figaxs:
    tmpf, tmpa = figax
    figs2.append(tmpf)
    axss2.append(tmpa)


#fig1, axs1 = plt.subplots(1, figsize=(8,6))
#fig2, axs2 = plt.subplots(1, figsize=(8,6))


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
            #num_epochs = np.shape(train_modes)[0]
            #print(k, "num epochs", num_epochs)
            tmp = direc.split("_doSigma1_sisc1.0_aT0.0_aB0.0_")
            tmp2 = tmp[1].split("bat")
            title = tmp[0]+"\n"+tmp2[0]+"\n"+tmp2[1]
            #axs[1+2*k].set_title(title)
            #axs[2+2*k].set_title("max epoch: "+str(modeloss_data[num_epochs-1,0])+" train "+str(round(train[-1],3))+" test "+str(round(test[-1],3)))
            #axs[2+2*k].set_title("train "+str(round(train[num_epochs-1],3))+" test "+str(round(test[num_epochs-1],3)))
            

            if "SGD" in direc:
                does_sgd.append(True)
            else:
                does_sgd.append(False)


            title = ""
            if whichones <= 1:
                if "StackedTrue" in direc:
                    title = "Stacked"
                else:
                    title = "Unstacked"              
            elif whichones == 4 or whichones == 5:
                title = tmp2[0].split("_Nep")[0]
                title = r"$e="+title[3:]+"$"
            elif whichones == 2 or whichones == 20:
                #title = ""
                #title = tmp2[0]

                if "Adam" in direc:
                    title = "Adam, "
                else:
                    title = "GD, "

                if "doStackedTrue" in direc:
                    title += "Sta."
                else:
                    title += "Unst."  
            elif whichones == 10:
                tmp   = direc.split("_Nep")[1]
                title = tmp.split("_llw")[0] 
            elif whichones == 21 or whichones == 22:
                tmp   = direc.split("_d5_w")[1]
                tmp2  = tmp.split("_llw")[0]
                width = int(tmp2)
                if "doStackedTrue" in direc:
                    title = r"Sta., $w_{sta}="+str(width)+"$"
                else:
                    title = r"Unst., $w_{unsta}="+str(width)+"$"


            else:
                title = tmp2[0]                  


            num_epochs = np.shape(train_modes)[0] # 100 #max(num_epochs, 80)




            #axs[0].plot(epochs[:num_epochs], 10**train[:num_epochs], '--', color=colors[k])
            #axs[0].plot(epochs[:num_epochs], 10**test[:num_epochs],  '.-', color=colors[k], label=title)
            #print(np.min(10**test[:num_epochs]))


            min1 = 10
            min2 = 10
            max1 = 0
            max2 = ss_train[0]**2
            for i,_ in enumerate(epochs0): #range()
                figs1[i].suptitle(str((epochs0[i]+1)*50)+" Training Steps", fontsize=20)
                figs2[i].suptitle(str((epochs0[i]+1)*50)+" Training Steps", fontsize=20)
                axss1[i].plot(np.ones(50), ".-", color="gray")
                axss2[i].plot(ss_train[:llw]**2, ".-", color="gray")
                for e in epochs0[:i+1]:
                    ytmp = train_modes[e, :] #ss_train[:llw]**2 * train_modes[e, :]/mtrain
                    axss1[i].plot(ytmp, '.-', color=(0.2+0.8*e/80, 0, 0))
                    min1 = min(min1, np.min(ytmp))
                    max1 = max(max1, np.max(ytmp))
                    

                    ytmp = ss_train[:llw]**2 * train_modes[e, :] #/mtrain
                    min2 = min(min2, np.min(ytmp))
                    max2 = max(max2, np.max(ytmp))
                    axss2[i].plot(ytmp, '.-', color=(0.2+0.8*e/80, 0, 0))

            for i in range(len(epochs0)):
                axss1[i].set_ylim((0.8*min1, 1.2*max1))
                axss2[i].set_ylim((0.8*min2, 1.2*max2))

                axss1[i].set_yscale("log") #plot(ytmp)
                axss2[i].set_yscale("log") #plot(ytmp)
                
                axss1[i].set_xlabel(r"Mode index $i$", fontsize=20) #plot(ytmp)
                axss2[i].set_xlabel(r"Mode index $i$", fontsize=20) #plot(ytmp)
                
                axss1[i].set_ylabel(r"Unweighted Mode Loss $L_i$", fontsize=20) #plot(ytmp)
                axss2[i].set_ylabel(r"Weighted Mode Loss $\sigma_i^2 L_i$", fontsize=20) #plot(ytmp)


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


path = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/slides_text/defense/MA_Defense/imgs/losses/"

for i in range(len(epochs0)):
    figs1[i].tight_layout()
    figs2[i].tight_layout()
    print(path+"bid"+str(bid)+"_wo"+str(whichones)+"_e"+str(i)+"_size"+str(sizeid)+"_unw.pdf")
    #figs1[i].savefig(path+"bid"+str(bid)+"_wo"+str(whichones)+"_e"+str(i)+"_size"+str(sizeid)+"_unw.pdf")
    #figs2[i].savefig(path+"bid"+str(bid)+"_wo"+str(whichones)+"_e"+str(i)+"_size"+str(sizeid)+"_wei.pdf")
    #            figs2[i].suptitle("Epoch "+str((epochs0[i]+1)*50), fontsize=20)

plt.show()