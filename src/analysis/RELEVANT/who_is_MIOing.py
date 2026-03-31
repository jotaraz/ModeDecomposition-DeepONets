## 1708 maybe thesisrelevant
## adam mode coupling
## generates (a) the loss contributions for w=50 and w=335 and (b)/(c) the S matrices and mode losses for w=50 and w=335

from don_code import * 
#import numpy as np 
#import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.gridspec as gridspec
import matplotlib #.cm as cm

#bid = int(sys.argv[1])
lrtag = 32 # int(sys.argv[2])
#size = int(sys.argv[3])
opttag = "Adam" #sys.argv[4]
#doplot = bool(int(sys.argv[5]))
#do_test = True
bigname = "log_diagoffdiag_bigwu1e-08"
num_data = 1000

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


colors  = ["red",   "blue",      "orange",     "green","cyan","pink","brown","gray","purple","lightblue","lightgreen"]

line_ids = [1,2,3] #, 5, 7, 9] # 10, 20, 30] #0, 5, 10, 15, 20] #, 16] # 1, 5, 10] #0, 10, 20, 30]
maxlineid = np.max(np.array(line_ids))
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

bids = [0, 3, 6] # 1, 3, 4, 5, 6, 7]
sizeids = [1, 3]

for bid in bids:
    print(bid)
    
    #fig = plt.figure()
    fig, axs = plt.subplots(len(sizeids), 2)
    fig.suptitle(str(bid))
    

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

    for sij, sizeid in enumerate(sizeids): #wun in wun_s[1:2]:
        wun = wun_s[sizeid]
        direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep4000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lr"+opttag+"32_v0",
                "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(wun)+"_llw"+str(llw)+"_bat"+batch_name+"_"+uendtag+"_numd1000_lr"+opttag+"32_v0",
                ]

        for dir in direcs:
            if dir in alldirecs:
                bigdata       = np.loadtxt("nets/"+dir+"/"+bigname+".txt")
                epochs        = bigdata[:,0]

                data = np.loadtxt("nets/"+dir+"/log.txt")
                axs[sij, 0].plot(data[::10,0], 10**(data[::10,1]/2), '--', color=colors[0])
                axs[sij, 0].plot(data[::10,0], 10**(data[::10,4]/2), '.-', color=colors[1])
                axs[sij, 0].set_yscale("log")
                


                for ij, j in enumerate(line_ids):

                    VTV_train_flat           = bigdata[j,11+2*llw+2*llw**2:11+2*llw+3*llw**2] 
                    #maxval_tr = max(maxval_tr, np.max(VTV_train_flat))
                    #minval_tr = min(minval_tr, np.min(VTV_train_flat))
                    
                    VTV_test_flat            = bigdata[j,11+2*llw+3*llw**2:11+2*llw+4*llw**2]
                    #maxval_te = max(maxval_te, np.max(VTV_test_flat))
                    #minval_te = min(minval_te, np.min(VTV_test_flat))

                    VTV_train_diag = np.diag(np.reshape(VTV_train_flat, (llw,llw)))
                    VTV_test_diag  = np.diag(np.reshape(VTV_test_flat, (llw,llw)))
            
                    #maxval_ratio_diag = max(maxval_ratio_diag, np.max(VTV_test_diag/VTV_train_diag))
                    #minval_ratio_diag = min(minval_ratio_diag, np.min(VTV_test_diag/VTV_train_diag))
                    #print(maxval_ratio_diag, minval_ratio_diag)

                    axs[sij, 1].plot(np.tanh(VTV_test_diag / VTV_train_diag), label=str(wun)+" "+str(epochs[j]), color=( j/maxlineid, 1.0-j/maxlineid, 0 ))

                    print(wun)
        
        plt.legend()


plt.show()
