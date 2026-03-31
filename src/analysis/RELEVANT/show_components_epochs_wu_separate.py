## 1708thesisrelevant
## adam mode coupling
## generates (a) the loss contributions for w=50 and w=335 and (b)/(c) the S matrices and mode losses for w=50 and w=335

from don_code import * 
#import numpy as np 
#import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.gridspec as gridspec
import matplotlib #.cm as cm

bid = int(sys.argv[1])
lrtag = int(sys.argv[2])
size = int(sys.argv[3])
opttag = sys.argv[4]
doplot = bool(int(sys.argv[5]))
do_test = True
bigname = "log_diagoffdiag_bigwu1e-08"
num_data = 1000

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

print("bid=",bid)

colors  = ["red",   "blue",      "orange",     "green","cyan","pink","brown","gray","purple","lightblue","lightgreen"]

line_ids = [0, 5, 10, 15] #, 16] # 1, 5, 10] #0, 10, 20, 30]
#line_ids = [0, 3, 6, 10, 20, 30, 39] # 1, 5, 10] #0, 10, 20, 30]

def get_similarity(VTV_train, VTV_test):
    n = np.shape(VTV_train)[0]
    mask_diag = np.eye(n)
    mask_offdiag = np.ones((n,n))-mask_diag

    diag_tr = mask_diag * VTV_train
    diag_te = mask_diag * VTV_test
    offdiag_tr = mask_offdiag * VTV_train
    offdiag_te = mask_offdiag * VTV_test

    sim_diag    = 1.0 - np.sum( (diag_tr - diag_te)**2 ) / np.sum( diag_tr**2 )
    sim_offdiag = 1.0 - np.sum( (offdiag_tr - offdiag_te)**2 ) / np.sum( offdiag_tr**2 )

    print("sim diag", sim_diag, "sim offdiag", sim_offdiag)

def relu(x):
    return x*(np.sign(x)+1.0)/2

def fancyshow(matrix, ax, fig, maxval, minval):
    neg_mask = matrix < 0
    pos_mask = matrix >= 0

    # Prepare masked arrays
    neg_data = np.ma.masked_where(~neg_mask, -matrix)  # Flip sign for log scale
    pos_data = np.ma.masked_where(~pos_mask, matrix)

    # Create figure and axes
    #fig, ax = plt.subplots()

    # Plot negative values with custom normalization and colormap A
    #print(np.max(relu(-matrix)))
    cmap_neg = matplotlib.colormaps.get_cmap('Blues') #cm.get_cmap('Blues')  # Example for negatives
    norm_neg = LogNorm(vmin=1e-15, vmax=-minval) #np.max(relu(-matrix)))  # Because we flipped -1 → 1

    im1 = ax.imshow(neg_data, cmap=cmap_neg, norm=norm_neg)

    # Plot positive values with colormap B
    cmap_pos = matplotlib.colormaps.get_cmap('Reds') #cm.get_cmap('hot')  # Example for positives
    norm_pos = LogNorm(vmin=1e-15, vmax=maxval) #np.max(relu(matrix)))
    #norm_pos = LogNorm(vmin=1e-10, vmax=np.max(relu(matrix)))

    im2 = ax.imshow(pos_data, cmap=cmap_pos, norm=norm_pos)
    #fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label='Negative values (-log scale)')
    #fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Positive values (log scale)')


direcs  = []
direcsB = []

#direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_5999_numd1000_lrSGD32_v0"]
#direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w335_llw50_batkdvnx401_dt0.0001_nc5_m5000_5999_numd1000_lrAdam32_v0"]
#direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w337_llw50_batburgers_dt0.0001_nc10_m3800_100_numd1000_lrAdam32_v0"]
if bid < 2:
    wun_s = [50, 100, 222, 332, 495]
    llw = 20
elif bid < 6:
    wun_s = [50, 100, 220, 335, 495]
    llw = 50
else:
    wun_s = [50, 100, 237, 337, 494]
    llw = 50



batch_name, uendtag, _ = dic(bid)
alldirecs = os.listdir("nets")

nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = load_dataset(batch_name, uendtag, num_data)
ntrain, mtrain = np.shape(utrain)
ntest,  mtest  = np.shape(utest)

uu_train, ss_train, vh_train = jnp.linalg.svd(utrain, full_matrices=False)

if size < 0:
    sizes = range(len(wun_s))
else:
    sizes = [size]
sizes = [0, 3]

maxval_tr = -10.0
minval_tr = +10.0
maxval_te = -10.0
minval_te = +10.0

maxval_ratio_diag_sizes = []
minval_ratio_diag_sizes = []

for size in sizes:

    wun = wun_s[size]

    direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lr"+opttag+"32_v0",
            "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lr"+opttag+"32_v0",
            ]

    maxval_ratio_diag = 1.0
    minval_ratio_diag = 0.0 #+1.0
    for dir in direcs:
        if dir in alldirecs:
            bigdata       = np.loadtxt("nets/"+dir+"/"+bigname+".txt")
            for ij, j in enumerate(line_ids):
                VTV_train_flat           = bigdata[j,11+2*llw+2*llw**2:11+2*llw+3*llw**2] 
                maxval_tr = max(maxval_tr, np.max(VTV_train_flat))
                minval_tr = min(minval_tr, np.min(VTV_train_flat))
                
                VTV_test_flat            = bigdata[j,11+2*llw+3*llw**2:11+2*llw+4*llw**2]
                maxval_te = max(maxval_te, np.max(VTV_test_flat))
                minval_te = min(minval_te, np.min(VTV_test_flat))

                VTV_train_diag = np.diag(np.reshape(VTV_train_flat, (llw,llw)))
                VTV_test_diag  = np.diag(np.reshape(VTV_test_flat, (llw,llw)))
        
                maxval_ratio_diag = max(maxval_ratio_diag, np.max(VTV_test_diag/VTV_train_diag))
                minval_ratio_diag = min(minval_ratio_diag, np.min(VTV_test_diag/VTV_train_diag))
                #print(maxval_ratio_diag, minval_ratio_diag)

    maxval_ratio_diag_sizes.append(min(maxval_ratio_diag, 5)) 
    minval_ratio_diag_sizes.append(max(minval_ratio_diag, -4)) 

pad = 0.1*(maxval_ratio_diag-minval_ratio_diag)
print(maxval_tr, minval_tr)
print(maxval_te, minval_te)

fig1, axs1 = plt.subplots(2, 2, figsize=(10,5), sharey="row")
figa, axsa = plt.subplots(3, 4, figsize=(10,8))
figb, axsb = plt.subplots(3, 4, figsize=(10,8))

figspec, axspec = plt.subplots(1, figsize=(6,6))

figs = [figa, figb]
axss = [axsa, axsb]

def get_colors():
    f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/colors.txt", "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()

for sizeid,size in enumerate(sizes):
    wun = wun_s[size]

    direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lr"+opttag+"32_v0",
              "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lr"+opttag+"32_v0",
            ]


    _, _, llw, _, batch_name, num_data, endtag = get_dwllw(direcs[0])
    print("w=",wun)


    truedirecs = []
    k = 0
    for d in direcs:
        if d in alldirecs:
            truedirecs.append(d)
    #        data = np.loadtxt("nets/"+d+"/log.txt")
    #        axs[0].plot(data[:,0], 10**(data[:,1]/2), '--', color=colors[k], label=d[10:35])
    #        axs[0].plot(data[:,0], 10**(data[:,4]/2), '.-', color=colors[k])
    #        k += 1 
    #axs1[0].set_yscale("log")
    #axs1[0].set_xticks([])


    tmp = truedirecs[0].split("_Nep")
    #fig.suptitle(tmp[0]+"\n"+tmp[1])


    for dir in truedirecs:
        bigdata       = np.loadtxt("nets/"+dir+"/"+bigname+".txt")
        modeloss_data = np.loadtxt("nets/"+dir+"/log_modes.txt")
        epoch_modes   = modeloss_data[:,0]
        train_modes   = modeloss_data[:,1      :1+  llw]
        test_modes    = modeloss_data[:,1+2*llw:1+3*llw]


        epochs        = bigdata[:,0]
        tr_diags      = bigdata[:,1]
        tr_offdiags   = bigdata[:,2]
        tr_taylorgrad = bigdata[:,3]
        tr_actualdiff = bigdata[:,4]
        te_diags      = bigdata[:,5]
        te_offdiags   = bigdata[:,6]
        te_taylorgrad = bigdata[:,7]
        te_actualdiff = bigdata[:,8]
        tr_taylorupd  = bigdata[:,9]
        te_taylorupd  = bigdata[:,10]


        axs1[sizeid,0].plot(epochs[1:], 0*tr_actualdiff[1:], '-.', color="gray")
        axs1[sizeid,0].plot(epochs[1:], tr_actualdiff[1:], '.-', color=colors[0], label=r"Loss Change $\Delta \mathcal{L}$")
        #axs1[2*sizeid].plot(epochs[1:], tr_diags[1:] + tr_offdiags[1:], '--', color=colors[4], label=r"Taylor Ex. $d+\Omega$")
        axs1[sizeid,0].plot(epochs[1:], tr_taylorgrad[1:], '--', color=colors[4], label=r"Taylor Ex. $d+\Omega$")
        axs1[sizeid,0].plot(epochs[1:], tr_diags[1:], '.-', color=colors[1], label=r"Diag $d$")
        axs1[sizeid,0].plot(epochs[1:], tr_offdiags[1:], '.-', color=colors[2], label=r"Off-Diag $\Omega$")
        axs1[sizeid,0].plot(epochs[1:], tr_taylorupd[1:],  '-.', color=colors[3], label=r"Taylor Ex. Adam")

        axs1[sizeid,1].plot(epochs[1:], 0*te_actualdiff[1:], '-.', color="gray")
        axs1[sizeid,1].plot(epochs[1:], te_actualdiff[1:], '.-', color=colors[0], label=r"Loss Change $\Delta \mathcal{L}$")
        #axs1[2*sizeid+1].plot(epochs[1:], te_diags[1:] + tr_offdiags[1:], '--', color=colors[4], label=r"Taylor Ex. $d+\Omega$")
        axs1[sizeid,1].plot(epochs[1:], te_taylorgrad[1:], '--', color=colors[4], label=r"Taylor Ex. $d+\Omega$")
        axs1[sizeid,1].plot(epochs[1:], te_diags[1:], '.-', color=colors[1], label=r"Diag $d$")
        axs1[sizeid,1].plot(epochs[1:], te_offdiags[1:], '.-', color=colors[2], label=r"Off-Diag $\Omega$")
        axs1[sizeid,1].plot(epochs[1:], te_taylorupd[1:],  '-.', color=colors[3], label=r"Taylor Ex. Adam")

        if sizeid == 0:
            axs1[sizeid,1].legend(loc='lower right', ncol=2)


        '''
        axs[2*sizeid].plot()
        axs[len(line_ids)+1].plot(epochs[1:], 0*epochs[1:],      '-.', color="gray")
        axs[len(line_ids)+1].plot(epochs[1:], tr_actualdiff[1:], '*-', color="red", label="Actual")
        axs[len(line_ids)+1].plot(epochs[1:], tr_taylorupd[1:],  '-.', color="orange", label="TaylorUpd")
        axs[len(line_ids)+1].plot(epochs[1:], tr_taylorgrad[1:], '.-', color="blue", label="TaylorGD")
        axs[len(line_ids)+1].plot(epochs[1:], tr_diags[1:],      '--', color="cyan", label="Diags")
        axs[len(line_ids)+1].plot(epochs[1:], tr_offdiags[1:],   '--', color="purple", label="Off-Diags")
        axs[len(line_ids)+1].text(500, min(np.min(tr_actualdiff[1:]), np.min(tr_taylorgrad[1:]), np.min(tr_diags[1:])), r"Training", fontsize=16)
        axs[len(line_ids)+1].set_box_aspect(1)

        axs[len(line_ids)+1].set_xticks([]) 
        '''

            
        
        for ij, j in enumerate(line_ids):
            base_loss_train          = bigdata[j,11               :11+llw]
            base_loss_test           = bigdata[j,11+llw           :11+2*llw]
            update_matrix_train_flat = bigdata[j,11+2*llw+0*llw**2:11+2*llw+1*llw**2] 
            update_matrix_test_flat  = bigdata[j,11+2*llw+1*llw**2:11+2*llw+2*llw**2]
            VTV_train_flat           = bigdata[j,11+2*llw+2*llw**2:11+2*llw+3*llw**2] 
            VTV_test_flat            = bigdata[j,11+2*llw+3*llw**2:11+2*llw+4*llw**2]

            update_matrix_train = np.reshape(update_matrix_train_flat, (llw,llw)) -np.outer(base_loss_train,np.ones(llw))
            update_matrix_test  = np.reshape(update_matrix_test_flat, (llw,llw)) -np.outer(base_loss_test, np.ones(llw))
            VTV_train           = np.reshape(VTV_train_flat, (llw,llw))
            VTV_test            = np.reshape(VTV_test_flat, (llw,llw))

            print(j, epochs[j], epoch_modes[2*j], np.sum(ss_train[:llw]**2 * base_loss_train))
            diag_tr = np.diag(VTV_train)
            diag_te = np.diag(VTV_test)


            count_low  = 0
            count_high = 0
            nothigh    = []
            for i in range(llw):
                if diag_tr[i] < 0:
                    if diag_te[i] < 0:
                        count_high += 1
                    else:
                        nothigh.append(i)
                    if -diag_te[i] > -0.5*diag_tr[i]:
                        count_low += 1
            print("count high", llw-count_high, "count low", llw-count_low, "nothigh", nothigh)

            axss[sizeid][0,ij].set_title("Epoch "+str(int(epochs[j])-1))
            axss[sizeid][0,ij].plot(ss_train[:llw]**2/mtrain, '.-', color="k")
            axss[sizeid][0,ij].plot(ss_train[:llw]**2 * train_modes[2*j, :]/mtrain, color="red")
            axss[sizeid][0,ij].plot(ss_train[:llw]**2 * test_modes[2*j, :]/mtest, color="blue")
            

            ymin = 0.5*ss_train[llw-1]**2 / mtrain
            ymax = 2*ss_train[0]**2 * train_modes[0, 0]/mtrain

            ax_tmp = axss[sizeid][0,ij].twinx() #axs[1+ij].twinx()
            ax_tmp.spines["right"].set_position(("axes", 1.0))
            ax_tmp.set_yticks([])

            ax_tmp.plot(np.zeros(llw), '--', color="lightgray")
            ax_tmp.plot(0.5*np.ones(llw), '--', color="darkgray")
            ax_tmp.plot(np.ones(llw), '--', color="black")

            #ax_tmp.plot(np.diag(VTV_train)/ss_train[:llw]**2, '--', color="orange")
            
            
            pad = 0.1*(maxval_ratio_diag_sizes[sizeid]-minval_ratio_diag_sizes[sizeid])
            #ax_tmp.plot(np.diag(VTV_test)/ss_train[:llw]**2, '--', color="lightblue")
            
            ax_tmp.plot(np.diag(VTV_test)/np.diag(VTV_train), '--', color="fuchsia")
            ax_tmp.set_ylim((minval_ratio_diag_sizes[sizeid]-pad, maxval_ratio_diag_sizes[sizeid]+pad))
                

            axss[sizeid][0,ij].set_yscale("log")            
            axss[sizeid][0,ij].set_xlim((0,49)) #yscale("log")
            axss[sizeid][0,ij].set_xticks([])
            axss[sizeid][0,ij].set_ylim((ymin, ymax))


            
            fancyshow(VTV_train, axss[sizeid][1,ij], figs[sizeid], maxval_tr, minval_tr)

            if ij == 3 and sizeid == 1: #xxx
                fancyshow(VTV_train, axspec, figspec, maxval_tr, minval_tr)
                axspec.set_xticks([0, 24, 49], ["1", "25", "50"])
                axspec.set_yticks([0, 24, 49], ["1", "25", "50"])



            #axss[sizeid][1,ij].set_xticks([])
            #axss[sizeid][1,ij].set_yticks([])
            #if do_test:
            fancyshow(VTV_test, axss[sizeid][2,ij], figs[sizeid], maxval_te, minval_te)
                #axss[sizeid][2,ij].set_yticks([])

            if ij > 0:
                a = 0
                #maxid = np.argmax(ss_train[:llw]**2 * train_modes[2*j, :]/mtrain)+1 #np.where( train_modes[2*j, :] < 1.0 )[0]) + 1
                #axss[sizeid][0,ij].plot([maxid, maxid], [ymin, ymax], '--', color='lightgreen')
                #axss[sizeid][0,ij].text(2, 0.005, r"$h="+str(maxid+1)+"$", fontsize=14)

                #for kk in [1,2]:
                #    axss[sizeid][kk,ij].plot([maxid, maxid], [0, 49], '--', color='lightgreen')
            else:
                axss[sizeid][1,ij].set_yticks([0, 19, 39], ["1", "20", "40"])
                axss[sizeid][2,ij].set_yticks([0, 19, 39], ["1", "20", "40"])
                axss[sizeid][0,ij].set_ylabel("Weighted Training\nMode Losses", fontsize=12)
                axss[sizeid][1,ij].set_ylabel(r"Entries of $S_{tr}$ Matrix", fontsize=12) 
                axss[sizeid][2,ij].set_ylabel(r"Entries of $S_{te}$ Matrix", fontsize=12) 
                #ticks([0, 19, 39], ["1", "20", "40"])
                


            axss[sizeid][1,ij].set_xticks([]) #0, 19, 39], ["1", "20", "40"])
            axss[sizeid][2,ij].set_xticks([0, 19, 39], ["1", "20", "40"])

            if ij != 0:
                axss[sizeid][0,ij].set_yticks([])
                axss[sizeid][1,ij].set_yticks([])
                axss[sizeid][2,ij].set_yticks([])


            #axs[1,1].imshow(megamatrix)

            #get_similarity(VTV_train, VTV_test)


    #plt.tight_layout(h_pad=0)

    #fig1.subplots_adjust(wspace=0.2, hspace=0.0, left=0.06, bottom=0.04, right=0.96, top=0.9)

axs1[0,0].set_xticks([])
axs1[0,1].set_xticks([])
axs1[1,0].set_xticks([0, 1000, 2000, 3000], ["0", "1000", "2000", "3000"])
axs1[1,1].set_xticks([0, 1000, 2000, 3000], ["0", "1000", "2000", "3000"])

axs1[0,0].set_title("Training", fontsize=12)
axs1[0,1].set_title("Test", fontsize=12)


axs1[0,0].set_ylabel(r"$w=50$", fontsize=12)
axs1[1,0].set_ylabel(r"$w=335$", fontsize=12)


fig1.subplots_adjust(wspace=0.0, hspace=0.0, right=0.95, left=0.15, top=0.9, bottom=0.15)
figa.subplots_adjust(wspace=0.0, hspace=0.0, right=0.95, left=0.1, top=0.95, bottom=0.1)
figb.subplots_adjust(wspace=0.0, hspace=0.0, right=0.95, left=0.1, top=0.95, bottom=0.1)

fig1.text(0.45, 0.04, r"Epochs", fontsize=12)
figa.text(0.45, 0.04, r"Mode index $i$", fontsize=12)
figb.text(0.45, 0.04, r"Mode index $i$", fontsize=12)
fig1.text(0.04, 0.4, r"Loss Change Contributions", fontsize=12, rotation=90)

name = "bid"+str(bid)+"_sizes_Adam"

fig1.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/final_coupl/"+name+"_1.pdf")
figa.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/final_coupl/"+name+"_w50.pdf")
figb.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/final_coupl/"+name+"_w335.pdf")

figspec.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/slides_text/defense/MA_Defense/imgs/coupling/"+name+"_Adam.pdf")

if doplot:
    plt.show()
