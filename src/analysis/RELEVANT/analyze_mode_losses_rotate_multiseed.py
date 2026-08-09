

#tag:10_02_2026
## 1708thesisrelevant
# in the final thesis version this code was used to generate 3 figures (from the main part):
# 1. GD with different e's (whichones = 4)
# 2. Adam with different e's (whichones = 5)
# 3. GD and Adam for Stacked and Unstacked (whichones = 2)
# The stacked/unstacked comparison from the appendix is also done with this code 

#from ana_fct import *
from ... import don_code

import os
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

#fix_num_epochs = True
#num_epochs = 200
fontsize = 10

#bid = int(sys.argv[1])
#nep = int(sys.argv[2])
#llw = int(sys.argv[3])
#numdata = int(sys.argv[4])

bid = int(sys.argv[1])
whichones = int(sys.argv[2])
sizeid = int(sys.argv[3])
lrtag   = int(sys.argv[4]) #default is 32
if bid >= 6:
    wfix   = 43
else:
    wfix = 42
#wun = int(sys.argv[3])

if whichones == 0:
    fixed_num_epochs = 200
else:
    fixed_num_epochs = 80



# whichT0_doStackedTrue_doSigma1_sisc1_aT0.0_aB0.0_exp0.0_Nep10000_d5_w42_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam32_v0
# whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w42_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam32_v0
            
nets_dir = don_code.nets_dir

## ---------------------------------------------------------------------------
## MULTISEED VARIANT of analyze_mode_losses_rotate.py
## Every plotted point is the ARITHMETIC MEAN over seeds v0..v9.
## Note the transform order: log.txt stores log10 values and the figure plots
## 10**(col/2), so we convert each seed FIRST and average afterwards -- averaging
## the raw column would give a geometric mean.  `train`/`test` below are set to
## log10 of the averaged error so the existing 10**train plotting still works.
## Mode losses in log_modes.txt are stored raw, so those are averaged directly.
## Set SHOW_BAND = True to shade the min..max spread across seeds.
## ---------------------------------------------------------------------------
from . import multiseed
SHOW_BAND = False
multiseed.SHOW_BAND = SHOW_BAND





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

#colors = ["darkorange", "brown", "purple", "cyan", "green"]

#direcs = ["whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep500_d5_w13_llw"+str(llw)+"_batkdvnx401_dt0.0001_nc5_m5000_9999_numd1000_lrSGD"+str(lrtag)+"_v0",
#          "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep500_d5_w220_llw"+str(llw)+"_batkdvnx401_dt0.0001_nc5_m5000_9999_numd1000_lrSGD"+str(lrtag)+"_v0"]

#direcs = ["whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w43_llw"+str(llw)+"_batburgers_dt0.0001_nc10_m3800_100_numd1000_lrSGD"+str(lrtag)+"_v0_try",
#          "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w337_llw"+str(llw)+"_batburgers_dt0.0001_nc10_m3800_100_numd1000_lrSGD"+str(lrtag)+"_v0"]

#direcs = [#"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w24_llw"+str(llw)+"_batkdvnx401_dt0.0001_nc5_m5000_9999_numd1000_lrSGD"+str(lrtag)+"_v0",
#          "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w43_llw"+str(llw)+"_batburgers_dt0.0001_nc10_m3800_100_numd1000_lrAdam"+str(lrtag)+"_v0",
#          #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w335_llw"+str(llw)+"_batkdvnx401_dt0.0001_nc5_m5000_9999_numd1000_lrSGD"+str(lrtag)+"_v0",
#          "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w337_llw"+str(llw)+"_batburgers_dt0.0001_nc10_m3800_100_numd1000_lrAdam"+str(lrtag)+"_v0"]
#      nets/whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w337_llw"+str(llw)+"_batburgers_dt0.0001_nc10_m3800_100_numd1000_lrAdam"+str(lrtag)+"_v0/


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
    direcs = ["whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              #"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              #"whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              ]
    num_columns = 2
elif whichones == 1:
    direcs = ["whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
             "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
             "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0", #_try",
             "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
             "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
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
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wfix)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w"+str(wfix)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wfix)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            ]
    num_columns = 3

elif whichones == 22:
    direcs = [
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0_upto5kepochs",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
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
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.5_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp1.0_Nep4000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            ]
    num_columns = 5


elif whichones == 4:
    direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD"+str(lrtag)+"_v0",
            ]
    num_columns = 5

elif whichones == 5:
    direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-1.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp-0.5_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
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

elif whichones == 11:
    direcs = [
            #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAda32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0",
            #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam32_v0",
            #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAda32_v0",
            #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0",
            #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam32_v0",
            #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAda32_v0",
        ]
    num_columns = 3


#direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep20000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD"+str(lrtag)+"_v0",
#          #"whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrAdam"+str(lrtag)+"_v0"
#          ]
#num_columns = 1
#num_epochs = 400

_, _, llw, _, batch_name, num_data, endtag = don_code.get_dwllw(direcs[0])
print(batch_name, endtag)

xmin = -0.05*llw
xmax = 1.05*llw

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

print

print("x")
for direc in direcs:
    #if don_code.contains_all(direc, tags) and direc in alldirecs:
    if direc in alldirecs:
        tmp_direcs = os.listdir(nets_dir+"/"+direc)
        if "log_modes.txt" in tmp_direcs:
            print(k, "       ", direc)
            k += 1
        else:
            print("no modes ", direc)
    else:
        print("not there", direc)
print("x")
print("x")
print("x")
kmax = k+1


# hochkant
'''
fig = plt.figure(figsize=(8, 11))  # Width x Height in inches
gs  = gridspec.GridSpec(num_columns+2, 2, figure=fig)

axs = []
axs.append(fig.add_subplot(gs[:2, :]))
axs.append(fig.add_subplot(gs[2, 0]))
axs.append(fig.add_subplot(gs[2, 1]))
for i in range(1,num_columns):
    axs.append(fig.add_subplot(gs[2+i, 0])) #, sharey=axs[1]))
    axs.append(fig.add_subplot(gs[2+i, 1])) #, sharey=axs[1]))
'''


if whichones == 0:
    fig = plt.figure(figsize=(3.2, 4))
elif whichones == 1:
    fig = plt.figure(figsize=(6, 4))
else:
    fig = plt.figure(figsize=(6, 4.5))

# Outer grid: two big blocks
outer = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)

# First block: just one row (row 1)
gs_top = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])

# Second block: two rows (row 2 and row 3) with hspace=0 between them
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    2, num_columns, subplot_spec=outer[1], hspace=0
)

#fig = plt.figure(figsize=(8, 6))  # Width x Height in inches
#gs  = gridspec.GridSpec(3, num_columns, figure=fig)

axs = []
axs.append(fig.add_subplot(gs_top[0]))
for i in range(num_columns):
    axs.append(fig.add_subplot(gs_bottom[0, i]))
    axs.append(fig.add_subplot(gs_bottom[1, i]))

#for i in range(1,num_columns):
#    axs.append(fig.add_subplot(gs[1, i])) #, sharey=axs[1]))
#    axs.append(fig.add_subplot(gs[2, i])) #, sharey=axs[1]))


#fig,  axs  = plt.subplots(2, num_columns, sharey=True)
k = 0


standard_value_adam_tr = 0
standard_value_adam_te = 0
standard_value_gd_tr   = 0
standard_value_gd_te   = 0

axs[0].plot([], [], marker='o', linestyle='none', fillstyle='none', color="gray", label="test")
axs[0].plot([], [], '--', color="gray", label="train")

fac = ss_train[:llw]**2

direcs_used = []

ymin_tr = ss_train[llw-1]**2/mtrain
ymax_tr = 2*ss_train[0]**2/mtrain

ymin_te = np.min(base_loss_test)**2/mtest
ymax_te = np.max(base_loss_test)**2/mtest

last_trains = []
last_tests  = []

does_sgd = []

for direc in direcs:
    if don_code.contains_all(direc, tags) and direc in alldirecs:
        tmp_direcs = os.listdir(nets_dir+"/"+direc)
        if "log_modes.txt" in tmp_direcs and k < kmax:
            direcs_used.append(direc+"\n")

            modeloss_data, ml_lo, ml_hi, nseed = multiseed.load_mean(direc, "log_modes.txt")
            loss_data, _, _, _ = multiseed.load_mean(direc, "log.txt")
            if modeloss_data is None or loss_data is None:
                continue
            print("multiseed: %s -> mean over %d seeds" % (direc, nseed))
            epochs = loss_data[:,0]
            # arithmetic mean of the PLOTTED error, i.e. convert per seed then average
            _etr, _etr_lo, _etr_hi, _ = multiseed.load_mean(
                direc, "log.txt", transform=lambda a: 10**(a[:,1]/2))
            _ete, _ete_lo, _ete_hi, _ = multiseed.load_mean(
                direc, "log.txt", transform=lambda a: 10**(a[:,4]/2))
            # stored back as log10 so the existing 10**train / 10**test still work
            train = np.log10(_etr)
            test  = np.log10(_ete)
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
                    title = "stacked"
                else:
                    title = "unstacked"              
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
                    title = r"Unst., $w_{unst}="+str(width)+"$"

            elif whichones == 11:
                if "SGD" in direc:
                    title = "GD"
                elif "Adam" in direc:
                    title = "Adam"
                elif "Ada" in direc:
                    title = "AdaGrad"
                else:
                    title = tmp2[0]

            else:
                title = tmp2[0]                  


            #num_epochs = 80
            #if not fix_num_epochs:
            num_epochs = fixed_num_epochs # min(fixed_num_epochs, np.shape(train_modes)[0]) # 100 #max(num_epochs, 80)

            if whichones == 0:
                de = 6
            else:
                de = 3
            axs[0].plot(epochs[:num_epochs:de], 10**train[:num_epochs:de], '--', color=colors[k])
            axs[0].plot(epochs[:num_epochs:de], 10**test[:num_epochs:de],  marker='o', linestyle='none', fillstyle='none', color=colors[k])
            axs[0].plot([], [],  marker='s', linestyle='none', color=colors[k], label=title)
            print(title, ":", np.min(10**test[:num_epochs]), "(min test loss log)")

            
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

            print(title)
            # use this for GD (whichones = 21, or 1)
            text_y = 1.5e-3 
            if whichones == 22 or whichones == 0:
                # use this for Adam (whichones = 22, or 0)
                text_y = 1e-6
            elif whichones == 5:
                text_y = 2e-5
            # after the thesis, I changed the y-position, it used to be 1.0*ymax_tr (but that looks weird)
            
            tmp = axs[1+2*k].text(1.0, text_y, title, color="k", fontsize=fontsize, horizontalalignment='left', verticalalignment='bottom')
            # tmp = axs[1+2*k].text(0.0, 1e-6, title, color="k", fontsize=fontsize, horizontalalignment='left', verticalalignment='bottom')
            #tmp = axs[1+2*k].text(1.05*llw, 500, title, color="k", fontsize=fontsize, horizontalalignment='right', verticalalignment='top')
            tmp.set_bbox(dict(facecolor=colors[k], alpha=0.4, edgecolor=colors[k]))


            # after the thesis, I think the box in the lower row is stupid. in the thesis its included.
            #tmp = axs[2+2*k].text(1.05*llw, 1.0*ymax_te, title, color="k", fontsize=fontsize, horizontalalignment='right', verticalalignment='top')
            #tmp.set_bbox(dict(facecolor=colors[k], alpha=0.4, edgecolor=colors[k]))


            axs[1+2*k].set_xlim((xmin, xmax)) #-0.1*llw, 1.1*llw))
            axs[2+2*k].set_xlim((xmin, xmax)) #())

            #if k != num_columns-1:
            '''
            axs[1+2*k].set_xticks([])
            if llw == 50:
                axs[2+2*k].set_xticks([0, 24, 49], ["1", "25", "50"])
                #if num_columns == 5:
                #    if k == 2:
                #        axs[2+2*k].set_xlabel(r"Mode index $i$", fontsize=fontsize)
                #else:
                #    axs[2+2*k].set_xlabel(r"Mode index $i$", fontsize=fontsize)
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

if whichones == 4:
    axs[2+2*2].set_xlabel(r"mode index $j$", fontsize=fontsize)
else:
    fig.text(0.47, 0.02, r"mode index $j$", fontsize=fontsize)
lower = 0.5
upper = 2.0

if num_columns == 2 and len(last_trains) > 1:
    ids_0_lower_train = np.where(last_trains[0] < 1.0 * last_trains[1])
    ids_0_lower_test  = np.where(last_tests[0]  < 1.0 * last_tests[1])
    ids_0_lower_train_but_not_test = np.setdiff1d(ids_0_lower_train, ids_0_lower_test)

    ids_1_lower_train = np.where(last_trains[1] < 1.0 * last_trains[0])
    ids_1_lower_test  = np.where(last_tests[1]  < 1.0 * last_tests[0])
    ids_1_lower_train_but_not_test = np.setdiff1d(ids_1_lower_train, ids_1_lower_test)
    
    ratio_train = last_trains[0]/last_trains[1]
    ratio_test  = last_tests[0] /last_tests[1]
    ids_0_bad = []
    ids_1_bad = []

    for i in range(llw):
        if ratio_train[i] < lower and ratio_test[i] > 1.1:
            ids_0_bad.append(i)
            #print(i, ratio_train[i], ratio_test[i], "stacked generalizes poorly")

        if ratio_train[i] > upper and ratio_test[i] < 0.9:
            ids_1_bad.append(i)
            #print(i, ratio_train[i], ratio_test[i], "unstacked generalizes poorly")

    print("ids 0 lower train but not test")
    print(ids_0_lower_train_but_not_test)
    print(np.array(ids_0_bad, dtype=np.int64))
    print("ids 1 lower train but not test")
    print(ids_1_lower_train_but_not_test)
    print(np.array(ids_1_bad, dtype=np.int64))


print("len(does_sgd)", len(does_sgd), "num_columns", num_columns)
for k in range(num_columns):
    #axs[1,k].plot(standard_value_adam*np.ones(llw), '-.', color="gold", alpha=0.6)
    if does_sgd[k] or whichones == 11:
        axs[1+2*k].plot([xmin, xmax], [standard_value_gd_tr, standard_value_gd_tr], '--', color="fuchsia", alpha=0.8, linewidth=1.5)
        axs[2+2*k].plot([xmin, xmax], [standard_value_gd_te, standard_value_gd_te], '--', color="fuchsia", alpha=0.8, linewidth=1.5)
    else:
        axs[1+2*k].plot([xmin, xmax], [standard_value_adam_tr, standard_value_adam_tr], '--', color="fuchsia", alpha=0.8, linewidth=1.5)
        axs[2+2*k].plot([xmin, xmax], [standard_value_adam_te, standard_value_adam_te], '--', color="fuchsia", alpha=0.8, linewidth=1.5)

    if k != 0:
        axs[1+2*k].set_yticks([])
        axs[2+2*k].set_yticks([])
    
    axs[1+2*k].set_ylim((ymin_tr/2, 2*ymax_tr))
    axs[1+2*k].set_xticks([]) #0, 19, 39], [1, 20, 40], fontsize=fontsize) #label(r"unweighted mode loss $L_j$", fontsize=fontsize)

    axs[2+2*k].set_ylim((ymin_te/2, 2*ymax_te))
    axs[2+2*k].set_xticks([0, 19, 39], [1, 20, 40], fontsize=fontsize) #label(r"unweighted mode loss $L_j$", fontsize=fontsize)




axs[0].set_yscale("log")
axs[0].set_xlabel("epochs", fontsize=fontsize)
axs[0].set_ylabel(r"relative error $\delta$", fontsize=fontsize)
#axs[0].set_ylabel(r"$\delta = ||A-\hat{A}||_F/||A||_F$", fontsize=fontsize)
if whichones == 4 or whichones == 5:
    axs[0].legend(ncol=4, fontsize=fontsize, loc='upper right')
elif whichones in [0,1]:
    axs[0].legend(ncol=2, fontsize=fontsize, loc='upper right')
else:
    axs[0].legend(ncol=3, fontsize=fontsize, loc='upper right')

if whichones == 4:
    axs[0].set_yticks(ticks=[0.3, 1.0], labels=[r"$3 \times 10^{-1}$", r"$10^0$"])#(0.09, 4.0))

#axs[0].set_xlim((-10, 4010))

if whichones == 2:
    axs[0].set_ylim((3e-2, 5.0))
elif whichones == 22:
    axs[0].set_ylim((6e-3, 5.0))
    axs[0].set_xlim((-300, 10_000+300))

elif whichones == 11:
    axs[0].set_ylim((0.03, 4.0))

axs[1].set_ylabel("weighted training\nmode losses", fontsize=fontsize, multialignment='center')
axs[2].set_ylabel("weighted test\nmode losses", fontsize=fontsize, multialignment='center')


#plt.tight_layout()



if whichones == 0:
    name = "un_stacked_adam/wo"+str(whichones)+"_bid"+str(bid)
elif whichones == 1:
    name = "un_stacked_sgd/wo"+str(whichones)+"_bid"+str(bid)
elif whichones == 2:
    name = "un_stacked_adam_sgd/wo"+str(whichones)+"_bid"+str(bid)
elif whichones == 4:
    name = "weighting_sgd/wo"+str(whichones)+"_bid"+str(bid)
elif whichones == 5:
    name = "weighting_adam/wo"+str(whichones)+"_bid"+str(bid)
else:
    name = "rest/wo"+str(whichones)+"_bid"+str(bid)

if lrtag != 32:
    name += "_lr"+str(lrtag)

if fixed_num_epochs != 80:
    name += "_neAll"+str(fixed_num_epochs)

if whichones == 0:
    plt.subplots_adjust(bottom=0.11, top=0.99, right=0.97, wspace=0.0, hspace=0.2, left=0.23)
else:
    plt.subplots_adjust(bottom=0.11, top=0.99, right=0.98, wspace=0.0, hspace=0.2, left=0.15)


if whichones == 4:
    tmp = "Fig4_multiseed"
elif whichones == 5:
    tmp = "Fig10_multiseed"
else:
    tmp = "Fig11_multiseed"

#plt.savefig(path + tmp + ".pdf")
path = don_code.figures_dir
#tmp = "Fig2"
plt.savefig(path+"/pdfs/"+tmp+".pdf")
plt.savefig(path+"/pngs/"+tmp+".png")
#f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/"+name+"_2.txt", "w")
#f.writelines(direcs_used)
#f.close()

#plt.show()