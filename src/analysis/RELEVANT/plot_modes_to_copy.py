#1708thesisrelevant

#from don_code import *

import numpy as np 
import matplotlib.pyplot as plt 


alpha = 1.0 #0.75

def get_colors():
    f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/colors.txt", "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()


base = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/data"

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

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

    tag0 = 21
    for sign in [+1.0, -1.0]:
        for sigm in [-1.0,-0.5,-0.01]:
            for fs in [0.05, 0.1, 0.2, 0.4]:
                bs = sign*fs
                ss = sigm
                if tag0 == tag:
                    return "synthv2_n100_N20_m5000", f"fs{bs}ss{ss}", 20

                tag0 += 1
    
    tag0 = 51
    for sign in [+1.0]:
        for sigm in [-1.0,-0.5]:
            for fs in [0.05, 0.1, 0.2, 0.4, 1.0]:
                bs = sign*fs
                ss = sigm
                if tag0 == tag:
                    return "synthv3_n100_N5_m5000", f"fs{bs}ss{ss}", 5

                tag0 += 1

    tag0 = 61 
    freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
    sigmaexpss    = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
    norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv4_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    
    tag0 = 71        
    freqincscales = [ -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0, -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0]
    sigmaexpss    = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [ 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0), 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0)]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            if tag0 >= 81:
                tmp = "v6"
            else:
                tmp = "v5"
            return "synth"+tmp+"_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(round(norm0scales[i], 4)), 5
        tag0 += 1
    
    tag0 = 91
    freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
    sigmaexpss    = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv7_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    
    tag0 = 101
    freqincscales = [-0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0] #, -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0]
    sigmaexpss    = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1] #, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [0.2*np.exp(4*0.2), 0.4*np.exp(4*0.2), 1.0*np.exp(4*0.2), 2.0*np.exp(4*0.2), 0.2*np.exp(4*0.4), 0.2*np.exp(4*1.0)] #, 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0)]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv8_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(round(norm0scales[i], 4)), 5
        tag0 += 1
    
    tag0 = 111
    freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
    sigmaexpss    = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
    norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv9_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    
    tag0 = 121
    freqincscales = [-0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0] #, -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0]
    sigmaexpss    = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01] #, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [0.2*np.exp(4*0.2), 0.4*np.exp(4*0.2), 1.0*np.exp(4*0.2), 2.0*np.exp(4*0.2), 0.2*np.exp(4*0.4), 0.2*np.exp(4*1.0)] #, 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0)]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv10_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(round(norm0scales[i], 4)), 5
        tag0 += 1
    
    return None

sigmabase = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/slides_text/defense/MA_Defense/imgs/sigmas"

for bid in [3]: #0, 1, 3, 4, 5, 6, 7]:

    name, uend, _ = dic(bid)

    #fig.suptitle(name)

    yy = np.loadtxt(base+"/"+name+"_"+uend+"_U.txt")
    uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
    r = len(ss)

    # fig = plt.figure()
    # plt.plot(np.arange(1, r), ss[:-1], '.-', color="gray")
    # plt.yscale("log")
    # plt.ylabel(r"Singular Values $\sigma_i$", fontsize=20)
    # plt.xlabel(r"Index $i$", fontsize=20)
    # plt.ylim((0.8*ss[-2], 1.2*ss[0]))
    # plt.xlim((0, r))
    # fig.savefig(sigmabase+"/sigma400.pdf")


    # fig = plt.figure()
    # plt.plot(np.arange(1, r), ss[:-1], '.-', color="gray")
    # plt.plot([0, 60], [ss[60], ss[60]], '-', color="k")
    # plt.plot([60, 60], [ss[60], 1.2*ss[0]], '-', color="k")
    # plt.yscale("log")
    # plt.ylabel(r"Singular Values $\sigma_i$", fontsize=20)
    # plt.xlabel(r"Index $i$", fontsize=20)
    # plt.ylim((0.8*ss[-2], 1.2*ss[0]))
    # plt.xlim((0, r))
    # fig.savefig(sigmabase+"/sigma400_window.pdf")

    # fig = plt.figure()
    # r = 61
    # plt.plot(np.arange(1, r), ss[:60], '.-', color="gray")
    # #plt.plot([1, 60], [ss[60], ss[60]], '-', color="k")
    # #plt.plot([60, 60], [ss[60], 1.2*ss[0]], '-', color="k")
    # plt.yscale("log")
    # plt.ylabel(r"Singular Values $\sigma_i$", fontsize=20)
    # plt.xlabel(r"Index $i$", fontsize=20)
    # plt.ylim((0.8*ss[60], 1.2*ss[0]))
    # plt.xlim((0, 60))
    # fig.savefig(sigmabase+"/sigma60.pdf")

    # fig = plt.figure()
    # plt.plot([0, 60], [ss[49], ss[49]], '--', color="fuchsia")
    # plt.plot([0, 60], [ss[0], ss[0]], '--', color="fuchsia")
    # plt.plot(np.arange(1, r), ss[:60], '.-', color="gray")
    # plt.text(30, np.sqrt(ss[0]*ss[49]), r"$\sigma_1 / \sigma_{50} > 100$", fontsize=20)
    # plt.yscale("log")
    # plt.ylabel(r"Singular Values $\sigma_i$", fontsize=20)
    # plt.xlabel(r"Index $i$", fontsize=20)
    # plt.ylim((0.8*ss[60], 1.2*ss[0]))
    # plt.xlim((0, 60))
    # fig.savefig(sigmabase+"/sigma60_lines.pdf")

    print("s1/s50", ss[0]/ss[49])

    background_color = "#FF964F"

    N = min(20, len(np.where(ss > 1e-10)[0]))

    fig, axs = plt.subplots(1,1,figsize=(4,2.5))
    #fig.patch.set_facecolor(background_color)

    n = np.shape(uu)[0]

    N = 4

    len0 = len(uu[:,0])
    skips = 6
    len1 = len0 // skips

    for i in range(N):
        axs.plot(np.zeros(len1)-i, '-.', color='gray', alpha=0.5)
        #axs[1].plot(ks, np.zeros(len(ks))-i, '--', color='gray')
        axs.plot(-0.4*uu[::skips,i]/np.max(np.abs(uu[:,i]))-i, '.-', color="k", linewidth=1, alpha=alpha)
        axs.text(1.05*len1, -i*1.05, r"$\mathbf{\phi_{"+str(i+1)+"}}$", fontsize=24)
        #axs.text(105, -i, r"$\sigma_{"+str(i+1)+"} \phi_{"+str(i+1)+"}$", fontsize=24)
    
    axs.text(len1/2, -N, r"$\vdots$", fontsize=24)
    axs.text(1.1*len1, -N, r"$\vdots$", fontsize=24)

    i = N+1
    axs.plot(np.zeros(len1)-i, '-.', color='gray', alpha=0.5)
    axs.plot(-0.4*uu[::skips,i+3]/np.max(np.abs(uu[:,i+3]))-i, '.-', color="k", linewidth=1, alpha=alpha)
    axs.text(1.05*len1, -i*1.05, r"$\mathbf{\phi_N}$", fontsize=24)
    
    axs.set_xlim((-0.05*len1, 1.05*len1))
    #axs.set_facecolor(background_color)

    #plt.xticks([])
    #plt.yticks([])
    axs.set_axis_off()

plt.tight_layout()

plt.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/GA_modes.svg", facecolor=fig.get_facecolor())
plt.show()