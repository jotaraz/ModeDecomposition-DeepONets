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
dotrain   = False
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


epochs0 = [79]

figaxs = [plt.subplots(1, figsize=(6, 5)) for e in epochs0] #i,c in enumerate(listcoeffs)]
figs1 = []
axss1 = []

for figax in figaxs:
    tmpf, tmpa = figax
    figs1.append(tmpf)
    axss1.append(tmpa)

figaxs = [plt.subplots(1, figsize=(6, 5)) for e in epochs0] #i,c in enumerate(listcoeffs)]
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



            min1 = 10
            min2 = 10
            max1 = 0
            max2 = ss_train[0]**2
            for i,_ in enumerate(epochs0): #range()
                #figs1[i].suptitle(str((epochs0[i]+1)*50)+" Training Steps", fontsize=20)
                #figs2[i].suptitle(str((epochs0[i]+1)*50)+" Training Steps", fontsize=20)
                axss1[i].plot(np.ones(50), "--", color="k")
                axss2[i].plot(ss_train[:llw]**2, ".-", color="gray", label=r"Relevancies $r_i$")
                axss2[i].legend(fontsize=20)
                for e in epochs0[:i+1]:
                    ytmp_unscaled = train_modes[e, :]
                    ytmp_scaled   = ss_train[:llw]**2 * train_modes[e, :]
                    if not dotrain:
                        ytmp_unscaled = test_modes[e, :] * mtrain/mtest
                        ytmp_scaled   = ss_train[:llw]**2 * test_modes[e, :] * mtrain/mtest
                    
                        
                    axss1[i].plot(ytmp_unscaled, '.-', color=(0.2+0.8*e/80, 0, 0))
                    min1 = min(min1, np.min(ytmp_unscaled))
                    max1 = max(max1, np.max(ytmp_unscaled))
                    

                    #/mtrain
                    min2 = min(min2, np.min(ytmp_scaled))
                    max2 = max(max2, np.max(ytmp_scaled))
                    axss2[i].plot(ytmp_scaled, '.-', color=(0.2+0.8*e/80, 0, 0))

            for i in range(len(epochs0)):
                axss1[i].set_ylim((0.8*min1, 2)) #1.2*max1))
                axss2[i].set_ylim((0.8*min2, 1.2*max2))

                axss1[i].set_yscale("log") #plot(ytmp)
                axss2[i].set_yscale("log") #plot(ytmp)
                
                axss1[i].set_xlabel(r"Mode index $i$", fontsize=20) #plot(ytmp)
                axss2[i].set_xlabel(r"Mode index $i$", fontsize=20) #plot(ytmp)
                
                axss1[i].set_ylabel(r"Unscaled Mode Loss $\varepsilon_i$", fontsize=20) #plot(ytmp)
                axss2[i].set_ylabel(r"Scaled Mode Loss $r_i \varepsilon_i$", fontsize=20) #plot(ytmp)

            k += 1


path = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/slides_text/defense/MA_Defense/imgs/losses/"

for i in range(len(epochs0)):
    figs1[i].tight_layout()
    figs2[i].tight_layout()
    figs1[i].savefig(path+"bid"+str(bid)+"_wo"+str(whichones)+"_e"+str(i)+"_size"+str(sizeid)+"_unw_nl.pdf")
    figs2[i].savefig(path+"bid"+str(bid)+"_wo"+str(whichones)+"_e"+str(i)+"_size"+str(sizeid)+"_wei_nl.pdf")
    #            figs2[i].suptitle("Epoch "+str((epochs0[i]+1)*50), fontsize=20)

plt.show()