

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
## MULTISEED VARIANT of analyze_mode_losses_rotate2.py
## Every plotted point is the ARITHMETIC MEAN over seeds v0..v9.
## Note the transform order: log.txt stores log10 values and the figure plots
## 10**(col/2), so we convert each seed FIRST and average afterwards -- averaging
## the raw column would give a geometric mean.  `train`/`test` below are set to
## log10 of the averaged error so the existing 10**train plotting still works.
## Mode losses in log_modes.txt are stored raw, so those are averaged directly.
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
    direcs = ["whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wst)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam"+str(lrtag)+"_v0",
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

elif whichones == 11:
    direcs = [
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAda32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep5000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAda32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrSGD32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAdam32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w335_llw50_bat"+batch_name+"_"+uendtag+"_numd1000_lrAda32_v0",
        ]
    num_columns = 3


_, _, llw, _, batch_name, num_data, endtag = don_code.get_dwllw(direcs[0])
print(batch_name, endtag)

xmin = -0.05*llw
xmax = 1.05*llw

nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, endtag, 1000)

mtrain = np.shape(utrain)[1]
mtest  = np.shape(utest)[1]
tags = []

uu_train, ss_train, vvhh_train = np.linalg.svd(utrain, full_matrices=False)

T = uu_train[:,:llw]
VT_test = np.matmul(T.T, utest)
base_loss_test = np.zeros(llw)
for i in range(llw):
    base_loss_test[i] = np.sqrt(np.sum( VT_test[i,:]**2 ))


utrain_norm = np.sum( utrain**2 )
utest_norm  = np.sum( utest**2 )

k = 0
kmax_apriori = 16
alldirecs = os.listdir(nets_dir)

print("x")
for direc in direcs:
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
kmax = k+1


# -------------------------------------------------------------------------
# Figure 1: loss curves (top panel from the original combined figure)
# -------------------------------------------------------------------------
if whichones == 0:
    fig1 = plt.figure(figsize=(3.2, 1.6))
else:
    fig1 = plt.figure(figsize=(6, 2.0))

ax_loss = fig1.add_subplot(1, 1, 1)
ax_loss.plot([], [], marker='o', linestyle='none', fillstyle='none', color="gray", label="test")
ax_loss.plot([], [], '--', color="gray", label="train")

# -------------------------------------------------------------------------
# Figure 2: mode loss panels (bottom panel from the original combined figure)
# -------------------------------------------------------------------------
if whichones == 0:
    fig2 = plt.figure(figsize=(3.2, 2.8))
else:
    fig2 = plt.figure(figsize=(6, 2.8))

gs_modes = gridspec.GridSpec(2, num_columns, figure=fig2, hspace=0)

axs_tr = []  # top row: weighted training mode losses
axs_te = []  # bottom row: weighted test mode losses
for i in range(num_columns):
    axs_tr.append(fig2.add_subplot(gs_modes[0, i]))
    axs_te.append(fig2.add_subplot(gs_modes[1, i]))

# -------------------------------------------------------------------------
# Convenience wrapper so the rest of the plotting code can keep using
# axs[0], axs[1+2*k], axs[2+2*k] indexing unchanged.
# -------------------------------------------------------------------------
class _AxsProxy:
    """Mimics the original flat axs list via index."""
    def __init__(self, ax0, tr_list, te_list):
        self._ax0 = ax0
        self._tr  = tr_list
        self._te  = te_list

    def __getitem__(self, idx):
        if idx == 0:
            return self._ax0
        col = (idx - 1) // 2
        row = (idx - 1) %  2   # 0 → train, 1 → test
        if row == 0:
            return self._tr[col]
        else:
            return self._te[col]

axs = _AxsProxy(ax_loss, axs_tr, axs_te)

# -------------------------------------------------------------------------
# (Everything below is identical to the original plotting loop)
# -------------------------------------------------------------------------
fac = ss_train[:llw]**2

direcs_used = []

ymin_tr = ss_train[llw-1]**2/mtrain
ymax_tr = 2*ss_train[0]**2/mtrain

ymin_te = np.min(base_loss_test)**2/mtest
ymax_te = np.max(base_loss_test)**2/mtest

last_trains = []
last_tests  = []

does_sgd = []

k = 0
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
            print(k, direc, np.shape(modeloss_data))

            train_modes = modeloss_data[:,1      :1+  llw]
            test_modes  = modeloss_data[:,1+2*llw:1+3*llw]
            # same slices of the across-seed band
            tr_lo = ml_lo[:,1      :1+  llw]; tr_hi = ml_hi[:,1      :1+  llw]
            te_lo = ml_lo[:,1+2*llw:1+3*llw]; te_hi = ml_hi[:,1+2*llw:1+3*llw]

            tmp = direc.split("_doSigma1_sisc1.0_aT0.0_aB0.0_")
            tmp2 = tmp[1].split("bat")

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

            num_epochs = fixed_num_epochs

            if whichones == 0:
                de = 6
            else:
                de = 3
            multiseed.band(axs[0], epochs[:num_epochs:de],
                           _etr_lo[:num_epochs:de], _etr_hi[:num_epochs:de], colors[k])
            multiseed.band(axs[0], epochs[:num_epochs:de],
                           _ete_lo[:num_epochs:de], _ete_hi[:num_epochs:de], colors[k])
            axs[0].plot(epochs[:num_epochs:de], 10**train[:num_epochs:de], '--', color=colors[k])
            axs[0].plot(epochs[:num_epochs:de], 10**test[:num_epochs:de],  marker='o', linestyle='none', fillstyle='none', color=colors[k])
            axs[0].plot([], [],  marker='s', linestyle='none', color=colors[k], label=title)
            print(title, ":", np.min(10**test[:num_epochs]), "(min test loss log)")

            ytmp = ss_train[:llw]**2/mtrain
            ymin_tr = min(ymin_tr, np.min(ytmp))
            ymax_tr = max(ymax_tr, np.max(ytmp))
            axs[1+2*k].plot(ytmp, '.-', color="k")

            ytmp = base_loss_test**2/mtest
            ymin_te = min(ymin_te, np.min(ytmp))
            ymax_te = max(ymax_te, np.max(ytmp))
            axs[2+2*k].plot(ytmp, '.-', color="k")

            last_ie = 0
            _stage_ids = list(range(0, num_epochs, 4))
            for j in range(0, num_epochs, 4):
                ytmp = ss_train[:llw]**2 * train_modes[j, :]/mtrain
                ymin_tr = min(ymin_tr, np.min(ytmp))
                ymax_tr = max(ymax_tr, np.max(ytmp))
                if multiseed.is_endpoint_stage(j, _stage_ids):
                  multiseed.band(axs[1+2*k], np.arange(llw),
                               ss_train[:llw]**2 * tr_lo[j, :]/mtrain,
                               ss_train[:llw]**2 * tr_hi[j, :]/mtrain,
                               (0.2+0.7*j/num_epochs, 0, 0))
                axs[1+2*k].plot(ytmp, '--', color=(0.2+0.7*j/num_epochs, 0, 0), alpha=0.5)

                ytmp = ss_train[:llw]**2 * test_modes[j, :]/mtest                
                ymin_te = min(ymin_te, np.min(ytmp))
                ymax_te = max(ymax_te, np.max(ytmp))
                if multiseed.is_endpoint_stage(j, _stage_ids):
                  multiseed.band(axs[2+2*k], np.arange(llw),
                               ss_train[:llw]**2 * te_lo[j, :]/mtest,
                               ss_train[:llw]**2 * te_hi[j, :]/mtest,
                               (0, 0, 0.2+0.7*j/num_epochs))
                axs[2+2*k].plot(ytmp, '--', color=(0, 0, 0.2+0.7*j/num_epochs), alpha=0.5)
                last_ie = j

            print(title)
            text_y = 1.5e-3 
            if whichones == 22 or whichones == 0:
                text_y = 1e-6
            elif whichones == 5:
                text_y = 2e-5

            tmp = axs[1+2*k].text(1.0, text_y, title, color="k", fontsize=fontsize, horizontalalignment='left', verticalalignment='bottom')
            tmp.set_bbox(dict(facecolor=colors[k], alpha=0.4, edgecolor=colors[k]))

            axs[1+2*k].set_xlim((xmin, xmax))
            axs[2+2*k].set_xlim((xmin, xmax))

            last_trains.append(train_modes[last_ie, :])
            last_tests.append(test_modes[last_ie, :])
            
            standard_value_adam_tr = 0
            standard_value_adam_te = 0
            standard_value_gd_tr   = 0
            standard_value_gd_te   = 0

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

if whichones == 4:
    axs_te[2].set_xlabel(r"mode index $j$", fontsize=fontsize)
else:
    fig2.text(0.47, 0.02, r"mode index $j$", fontsize=fontsize)

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
        if ratio_train[i] > upper and ratio_test[i] < 0.9:
            ids_1_bad.append(i)

    print("ids 0 lower train but not test")
    print(ids_0_lower_train_but_not_test)
    print(np.array(ids_0_bad, dtype=np.int64))
    print("ids 1 lower train but not test")
    print(ids_1_lower_train_but_not_test)
    print(np.array(ids_1_bad, dtype=np.int64))


print("len(does_sgd)", len(does_sgd), "num_columns", num_columns)
for k in range(num_columns):
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
    axs[1+2*k].set_xticks([])

    axs[2+2*k].set_ylim((ymin_te/2, 2*ymax_te))
    axs[2+2*k].set_xticks([0, 19, 39], [1, 20, 40], fontsize=fontsize)


ax_loss.set_yscale("log")
ax_loss.set_xlabel("epochs", fontsize=fontsize)
ax_loss.set_ylabel(r"relative error $\delta$", fontsize=fontsize)
if whichones == 4 or whichones == 5:
    ax_loss.legend(ncol=4, fontsize=fontsize, loc='upper right',
                labelspacing=0.2,      
                handlelength=1.0,      # length of the legend handle/line (default: 2.0)
                handletextpad=0.4,     # space between handle and label (default: 0.8)
                borderpad=0.2,         # padding inside the legend box (default: 0.4)
                borderaxespad=0.5,     # padding between legend and axes (default: 0.5)
                columnspacing=1.0)
elif whichones in [0, 1]:
    ax_loss.legend(ncol=2, fontsize=fontsize, loc='upper right',
                labelspacing=0.2,      
                handlelength=1.0,      # length of the legend handle/line (default: 2.0)
                handletextpad=0.4,     # space between handle and label (default: 0.8)
                borderpad=0.2,         # padding inside the legend box (default: 0.4)
                borderaxespad=0.5,     # padding between legend and axes (default: 0.5)
                columnspacing=1.0)
else:
    ax_loss.legend(ncol=3, fontsize=fontsize, loc='upper right',
                labelspacing=0.2,      
                handlelength=1.0,      # length of the legend handle/line (default: 2.0)
                handletextpad=0.4,     # space between handle and label (default: 0.8)
                borderpad=0.2,         # padding inside the legend box (default: 0.4)
                borderaxespad=0.5,     # padding between legend and axes (default: 0.5)
                columnspacing=1.0)

if whichones == 4:
    ax_loss.set_yticks(ticks=[0.3, 1.0], labels=[r"$3 \times 10^{-1}$", r"$10^0$"])

if whichones == 2:
    ax_loss.set_ylim((3e-2, 5.0))
elif whichones == 22:
    ax_loss.set_ylim((6e-3, 5.0))
    ax_loss.set_xlim((-300, 10_000+300))
elif whichones == 11:
    ax_loss.set_ylim((0.03, 4.0))

axs_tr[0].set_ylabel("weighted training\nmode losses", fontsize=fontsize, multialignment='center')
axs_te[0].set_ylabel("weighted test\nmode losses", fontsize=fontsize, multialignment='center')


# -------------------------------------------------------------------------
# Build save-name (same logic as before)
# -------------------------------------------------------------------------
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

path = don_code.figures_dir + "/"
# panel letter goes before the suffix -> Fig8b_multiseed[_band]
_fig8 = lambda panel: "Fig8" + panel + "_multiseed" + multiseed.suffix()

# -------------------------------------------------------------------------
# Save figure 1 (loss curves)
# -------------------------------------------------------------------------
if whichones == 0:
    fig1.subplots_adjust(bottom=0.25, top=0.97, right=0.97, left=0.23)
else:
    fig1.subplots_adjust(bottom=0.25, top=0.97, right=0.98, left=0.15)
fig1.savefig(path+"/pdfs/"+_fig8("b")+".pdf")
fig1.savefig(path+"/pngs/"+_fig8("b")+".png")


# -------------------------------------------------------------------------
# Save figure 2 (mode losses)
# -------------------------------------------------------------------------
if whichones == 0:
    fig2.subplots_adjust(bottom=0.14, top=0.99, right=0.97, wspace=0.0, hspace=0.0, left=0.23)
else:
    fig2.subplots_adjust(bottom=0.14, top=0.99, right=0.98, wspace=0.0, hspace=0.0, left=0.15)

fig2.savefig(path+"/pdfs/"+_fig8("c")+".pdf")
fig2.savefig(path+"/pngs/"+_fig8("c")+".png")


# -------------------------------------------------------------------------
# Write direcs log (unchanged)
# -------------------------------------------------------------------------
#f = open(imgs_dir + name + "_2.txt", "w")
#f.writelines(direcs_used)
#f.close()

#plt.show()