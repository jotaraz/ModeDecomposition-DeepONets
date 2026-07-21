import os
import numpy as np 
import sys 

Nepochs     = 4000
vtag        = 0
depth       = 5
width       = 335
llw         = 50
doplot      = 0
batch_name  = 'kdvnx401_dt0.0001_nc5_m5000'
lrstag      = '32'
init_lr     = 1e-4
decay_rate  = 0.95
num_data    = 1000
which_T     = 0
dotruesigma = 0
uendtag     = '1999'
sigmascale  = 'First'
exponent    = 1.0
doadam      = 1
dostacked   = 0 

def dic(tag):
    if tag == 0:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1000", 20
    if tag == 1:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1999", 20
    if tag == 2:
        return "kdvnx401_dt0.0001_nc5_m5000", "10", 50
    if tag == 3:
        return "kdvnx401_dt0.0001_nc5_m5000", "1999", 50
    if tag == 4:
        return "kdvnx401_dt0.0001_nc5_m5000", "5999", 50
    if tag == 5:
        return "kdvnx401_dt0.0001_nc5_m5000", "9999", 50    
    if tag == 6:
        return "burgers_dt0.0001_nc10_m3800", "100", 50
    if tag == 7:
        return "burgers_dt0.0001_nc10_m3800", "999", 50
    if tag == 8:
        return "NormalHO50_.2to.9_exTo1.1_m1000", "1", 20
    if tag == 9:
        return "waveeq1_N10_m1000", "-3.0", 10 
    if tag == 10:
        return "waveeq1_N10_m1000", "-1.0", 10 
    if tag == 11:
        return "waveeq1_N10_m1000", "-0.1", 10 
    if tag == 12:
        return "waveeq1_N10_m1000", "0.1", 10 
    if tag == 13:
        return "waveeq1_N10_m1000", "1.0", 10 
    if tag == 14:
        return "waveeq1_N10_m1000", "3.0", 10 
    if tag == 15:
        return "waveeq1f_N10_m1000", "-3.0", 10 
    if tag == 16:
        return "waveeq1f_N10_m1000", "-1.0", 10 
    if tag == 17:
        return "waveeq1f_N10_m1000", "-0.1", 10 
    if tag == 18:
        return "waveeq1f_N10_m1000", "0.1", 10 
    if tag == 19:
        return "waveeq1f_N10_m1000", "1.0", 10 
    if tag == 20:
        return "waveeq1f_N10_m1000", "3.0", 10 

def lrs(lrtag):
    if lrtag == 32:
        return 1e-4, 0.95
    elif lrtag == 34:
        return 5e-4, 0.95
    elif lrtag == 38:
        return 1e-3, 0.95
    elif lrtag == 40:
        return 2e-3, 0.95
    elif lrtag == 42:
        return 4e-3, 0.95
    elif lrtag == 43:
        return 8e-3, 0.95

# for vtag in range(0,1):
#     #for lrstag, lr, decay in zip(lrstags, lrs, decays):
#     for lrstag in lrstags:
#         lr, decay = lrs(lrstag)
#         for iw in range(len(gen_widths)):
#             for depth in depths:
#                 for tag in btags:
#                     for (which_T, dotruesigma, sigmascale) in zip(whTs, dtss, sisc):
#                         if tag < 2:
#                             llw = 20
#                         else:
#                             llw = 50
#                         batch_name, endtag, _ = dic(tag)

#                         doadam    = 0
#                         dostacked = gen_stacked[iw]
#                         width = gen_widths[iw]

dwllw = str(depth)+" "+str(width)+" "+str(int(llw))
tmp  = str(Nepochs)+" "+str(vtag)+" "+dwllw+" 0 "+batch_name+" "
tmp += str(lrstag)+" "+str(init_lr)+" "+str(decay_rate)+" "+str(num_data)+" "
tmp += str(which_T)+" "+str(dotruesigma)+" "+uendtag+" "+sigmascale+" "+str(exponent)+" "+str(doadam)+" "+str(dostacked)
os.system("python execute_don.py "+tmp)
                        