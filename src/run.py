import os
import numpy as np 
import sys 

mode = int(sys.argv[1])

mode0 = False
mode1 = False
mode2 = False
mode3 = False
mode4 = False

if mode < 0:
    mode0 = True
    mode1 = True
    mode2 = True
    mode3 = True
    mode4 = True
elif mode == 0:
    mode0 = True
elif mode == 1:
    mode1 = True
elif mode == 2:
    mode2 = True
elif mode == 3:
    mode3 = True
elif mode == 4:
    mode4 = True

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


#lrstags     = [32]
#lrs         = [1e-4]

lrstags     = [34] #32, 40, 43]
#lrs         = [2e-3]
#decays      = [0.95]
num_data    = 1000
#which_T     = -1
#dotruesigma = 0
#sigmascale  = "1.0"
exp         = 0.0

#llw         = 20
#btags       = [0, 1]
#w_unstacked = [50, 100, 222, 332, 495]
#w_stacked   = [-1]

btags       = [3] #, 4, 5]
#w_unstacked = [-1] #, 335, 495] #50, 100, 220, 495] #[50, 100, 220, 335, 495]
#w_stacked   = [42]
Nepochs     = 5_000
depths      = [5]

gen_widths  = [42, 495, 42]
gen_stacked = [0, 0, 1]

#llw         = 50
#btags       = [9, 10, 11] #, 4, 5]
#btags       = [19, 18, 20] #, 4, 5]
#w_unstacked = [150] #50, 100, 220, 495] #[50, 100, 220, 335, 495]
#w_stacked   = [24, 42, 13]

#llw         = 50
#btags       = [6, 7] #, 7]
#w_unstacked = [50, 100, 237, 494] #337] #, 494, 237] #[50, 100, 237, 337, 494]
#w_stacked   = [43] #, 65, 29]

dtss = []
sisc = []
whTs = []


if mode0:
    dtss.append(0)
    sisc.append("First")
    whTs.append(-1)
if mode1:
    dtss.append(0)
    sisc.append("First")
    whTs.append(0)
if mode2:
    dtss.append(1)
    sisc.append("1.0")
    whTs.append(0)
if mode3:
    dtss.append(0)
    sisc.append("1.0")
    whTs.append(0)
if mode4:
    dtss.append(1)
    sisc.append("First")
    whTs.append(0)

print("dotruesigmas", dtss)
print("sigmascales ", sisc)
print("whichTs     ", whTs)


for vtag in range(0,1):
    #for lrstag, lr, decay in zip(lrstags, lrs, decays):
    for lrstag in lrstags:
        lr, decay = lrs(lrstag)
        for iw in range(len(gen_widths)):
            for depth in depths:
                for tag in btags:
                    for (which_T, dotruesigma, sigmascale) in zip(whTs, dtss, sisc):
                        if tag < 2:
                            llw = 20
                        else:
                            llw = 50
                        batch_name, endtag, _ = dic(tag)

                        doadam    = 0
                        dostacked = gen_stacked[iw]
                        #if dostacked:
                        #    width     = w_stacked[iw]
                        #else:
                        #    width     = w_unstacked[iw]
                        width = gen_widths[iw]

                        for exp in [0.0]: #-1.0, -0.5, 0.5, 1.0]:
                            dwllw = str(depth)+" "+str(width)+" "+str(int(llw))
                            tmp  = str(Nepochs)+" "+str(vtag)+" "+dwllw+" 0 "+batch_name+" "
                            tmp += str(lrstag)+" "+str(lr)+" "+str(decay)+" "+str(num_data)+" "
                            tmp += str(which_T)+" "+str(dotruesigma)+" "+endtag+" "+sigmascale+" "+str(exp)+" "+str(doadam)+" "+str(dostacked)+" 0 1"
                            os.system("python execute_don.py "+tmp)
                        

                        #    #width = 25
                        #    dostacked = 0
                        #    #for width in w_stacked:
                        #    width = w_stacked[iw]
                        #    dwllw = str(depth)+" "+str(width)+" "+str(int(llw))
                        #    tmp  = str(Nepochs)+" "+str(vtag)+" "+dwllw+" 0 "+batch_name+" "
                        #    tmp += str(lrstag)+" "+str(lr)+" "+str(decay)+" "+str(num_data)+" "
                        #    tmp += str(which_T)+" "+str(dotruesigma)+" "+endtag+" "+sigmascale+" "+str(exp)+" "+str(doadam)+" "+str(dostacked)
                        #    os.system("python execute_don.py "+tmp)
                        
                        
                        #doadam    = 0
                        #dostacked = 0
                        #width = w_unstacked[iw]
                        #dwllw = str(depth)+" "+str(width)+" "+str(int(llw))
                        #tmp  = str(Nepochs)+" "+str(vtag)+" "+dwllw+" 0 "+batch_name+" "
                        #tmp += str(lrstag)+" "+str(lr)+" "+str(decay)+" "+str(num_data)+" "
                        #tmp += str(which_T)+" "+str(dotruesigma)+" "+endtag+" "+sigmascale+" "+str(exp)+" "+str(doadam)+" "+str(dostacked)
                        #os.system("python execute_don.py "+tmp)

                        '''
                        for exp in [-1.0, -0.5, 0.5, 1.0]:
                            #exp = 0.0
                            dostacked = 1
                            width = w_stacked[iw]
                            dwllw = str(depth)+" "+str(width)+" "+str(int(llw))
                            tmp  = str(Nepochs)+" "+str(vtag)+" "+dwllw+" 0 "+batch_name+" "
                            tmp += str(lrstag)+" "+str(lr)+" "+str(decay)+" "+str(num_data)+" "
                            tmp += str(which_T)+" "+str(dotruesigma)+" "+endtag+" "+sigmascale+" "+str(exp)+" "+str(doadam)+" "+str(dostacked)+" 0 1"
                            os.system("python execute_don.py "+tmp)
                        '''