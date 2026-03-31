## make a 2x2 plot with two widths and each has the mode losses (at different stages) and the S matrix (at some stage)


from ... import don_code
#import numpy as np 
#import matplotlib.pyplot as don_code.plt.
from matplotlib.colors import Normalize, LogNorm
import matplotlib.gridspec as gridspec
import matplotlib #.cm as cm

bid     = int(don_code.sys.argv[1])
sizeid1 = int(don_code.sys.argv[2])
sizeid2 = int(don_code.sys.argv[3])
neptag1 = don_code.sys.argv[4]
neptag2 = don_code.sys.argv[5]

showplot = True #bool(int(don_code.sys.argv[4]))
exponent = 0.0 #float(don_code.sys.argv[5])
do_mio   = False #bool(int(don_code.sys.argv[6]))

dostacked = "False"
dotest  = False
#lrtag = int(don_code.sys.argv[2])
#size = 
plotmode = 0
ratioplot = True
offdiagplot = True

startind = 11

fontsize = 10

nets_dir = don_code.nets_dir

don_code.plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

if bid < 2:
    wun_s = [50, 100, 222, 332, 495] #20, 50, 100, 222, 332, 495]
    llw = 20
elif bid < 6:
    wun_s = [50, 100, 220, 335, 495] #50, 100, 220, 335, 495]
    llw = 50    
else:
    wun_s = [50, 100, 237, 337, 494] #50, 100, 237, 337, 494]
    llw = 50

w1 = wun_s[sizeid1]
w2 = wun_s[sizeid2]

direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep"+neptag1+"_d5_w"+str(w1)+"_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0",
          "whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep"+neptag2+"_d5_w"+str(w2)+"_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v0"]


print(direcs)

num_points = 80

def relu(x):
    return x*(don_code.np.sign(x)+1.0)/2

def fancyshow(matrix, ax, fig, minval, maxval):
    neg_mask = matrix < 0
    pos_mask = matrix >= 0

    # Prepare masked arrays
    neg_data = don_code.np.ma.masked_where(~neg_mask, -matrix)  # Flip sign for log scale
    pos_data = don_code.np.ma.masked_where(~pos_mask, matrix)

    # Create figure and axes
    #fig, ax = don_code.plt.subplots()

    # Plot negative values with custom normalization and colormap A
    #print(don_code.np.max(relu(-matrix)))
    cmap_neg = matplotlib.colormaps.get_cmap('Blues') #cm.get_cmap('Blues')  # Example for negatives
    norm_neg = LogNorm(vmin=1e-15, vmax=-minval) #np.max(relu(-matrix)))  # Because we flipped -1 → 1

    im1 = ax.imshow(neg_data, cmap=cmap_neg, norm=norm_neg)

    # Plot positive values with colormap B
    cmap_pos = matplotlib.colormaps.get_cmap('Reds') #cm.get_cmap('hot')  # Example for positives
    norm_pos = LogNorm(vmin=1e-15, vmax=maxval) #np.max(relu(matrix)))

    im2 = ax.imshow(pos_data, cmap=cmap_pos, norm=norm_pos)
    #fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.2, label='Negative values (-log scale)')
    #fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.06, label='Positive values (log scale)')

line_id = 39 #, 16]


fig, axs = don_code.plt.subplots(2, 2, figsize=(3.2,3), dpi=300)




def get_colors():
    f = open(don_code.colors_path, "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()


_, _, llw, _, batch_name, num_data, endtag = don_code.get_dwllw(direcs[0])
print(batch_name, endtag, num_data, llw)

nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, endtag, num_data)
n_train, m_train = don_code.np.shape(utrain)
n_test,  m_test  = don_code.np.shape(utest)

uu_train, ss_train, vh_train = don_code.jnp.linalg.svd(utrain, full_matrices=False)



modeids = [0,1,2]

#if neptag == "500":

bigname = "log_diagoffdiag_new"

#else:
#    bigname = "log_diagoffdiag_big1e-08"
#fac = 100 #00_000

#1e-08




offdiags_x = []
offdiags_y = []
offdiags_label = []

min_loss_red = 1
max_loss_red = 0




#ax3.legend()
num_epochs_loss = 80


alldirecs = don_code.os.listdir(nets_dir)


#figspec, axspec = don_code.plt.subplots(1, figsize=(6,6))


maxval_tr = -10.0
minval_tr = +10.0
maxval_te = -10.0
minval_te = +10.0

maxval_ratio_diag = 1.0
minval_ratio_diag = 0.0 #+1.0

for dir in direcs:
    if dir in alldirecs:
        print("YES found", dir)
        lossdata      = don_code.np.loadtxt(nets_dir+"/"+dir+"/log.txt")
        epochs0       = lossdata[:num_epochs_loss:2,0]
        tr_loss       = 10**(lossdata[:num_epochs_loss:2,1]/2)
        te_loss       = 10**(lossdata[:num_epochs_loss:2,4]/2)
        bigdata       = don_code.np.loadtxt(nets_dir+"/"+dir+"/"+bigname+".txt")
        epochs        = bigdata[:,0]
        tr_diags      = bigdata[:,1]
        tr_offdiags   = bigdata[:,2]
        tr_taylorgrad = bigdata[:,3]
        tr_actualdiff = bigdata[:,4]
        te_diags      = bigdata[:,5]
        te_offdiags   = bigdata[:,6]
        te_taylorgrad = bigdata[:,7]
        te_actualdiff = bigdata[:,8]

        #print("epochs     ", epochs)
        #print("tr diags   ", tr_diags)
        #print("tr offdiags", tr_offdiags)
        #print("tr taylor  ", tr_taylorgrad)
        #print("tr true    ", tr_actualdiff)
        #print("te diags   ", te_diags)
        #print("te offdiags", te_offdiags)
        #print("te taylor  ", te_taylorgrad)
        #print("te true    ", te_actualdiff)

        j = line_id

        base_loss_train          = bigdata[j,startind               :startind+llw]
        base_loss_test           = bigdata[j,startind+llw           :startind+2*llw]
        #update_matrix_train_flat = bigdata[j,startind+2*llw+0*llw**2:startind+2*llw+1*llw**2] 
        #update_matrix_test_flat  = bigdata[j,startind+2*llw+1*llw**2:startind+2*llw+2*llw**2]
        VTV_train_flat           = bigdata[j,startind+2*llw+0*llw**2:startind+2*llw+1*llw**2] 
        VTV_test_flat            = bigdata[j,startind+2*llw+1*llw**2:startind+2*llw+2*llw**2]

        #print(don_code.np.sum(VTV_train_flat), don_code.np.sum(VTV_test_flat))

        #update_matrix_train = don_code.np.reshape(update_matrix_train_flat, (llw,llw)) -don_code.np.outer(base_loss_train,don_code.np.ones(llw))
        #update_matrix_test  = don_code.np.reshape(update_matrix_test_flat, (llw,llw)) -don_code.np.outer(base_loss_test, don_code.np.ones(llw))
        VTV_train           = don_code.np.reshape(VTV_train_flat, (llw,llw))
        VTV_test            = don_code.np.reshape(VTV_test_flat, (llw,llw))

        #print(ij, j, don_code.np.max(VTV_train_flat), don_code.np.min(VTV_train_flat))
        #idspos = don_code.np.where(VTV_train > 0)
        #print(don_code.np.sum(VTV_train[idspos] / len(idspos[0])))
        maxval_tr = max(maxval_tr, don_code.np.max(VTV_train_flat))
        minval_tr = min(minval_tr, don_code.np.min(VTV_train_flat))
        maxval_te = max(maxval_te, don_code.np.max(VTV_test_flat))
        minval_te = min(minval_te, don_code.np.min(VTV_test_flat))

        maxval_ratio_diag = max(maxval_ratio_diag, don_code.np.max(don_code.np.diag(VTV_test)/don_code.np.diag(VTV_train)))
        minval_ratio_diag = min(minval_ratio_diag, don_code.np.min(don_code.np.diag(VTV_test)/don_code.np.diag(VTV_train)))
    else:
        print("not found", dir) 


# plot

plot_epoch_ids = [1] #, 19, 39, 59, 79]
dk = 10
alphaval = 0.8
k = 1+dk
while k < 80:
    plot_epoch_ids.append(k)
    k += dk

k = 0

for dir in direcs:
    if dir in alldirecs:
        print("YES found", dir)
        modeloss_data = don_code.np.loadtxt(nets_dir+"/"+dir+"/log_modes.txt")
        #loss_data = don_code.np.loadtxt(nets_dir+"/"+direc+"/log.txt")
        #epochs = loss_data[:,0]
        #train = loss_data[:,1]/2
        #test  = loss_data[:,4]/2
        
        train_modes = modeloss_data[:,1      :1+  llw]
        test_modes  = modeloss_data[:,1+2*llw:1+3*llw]

        lossdata      = don_code.np.loadtxt(nets_dir+"/"+dir+"/log.txt")
        epochs0       = lossdata[:num_epochs_loss:2,0]
        tr_loss       = 10**(lossdata[:num_epochs_loss:2,1]/2)
        te_loss       = 10**(lossdata[:num_epochs_loss:2,4]/2)
        bigdata       = don_code.np.loadtxt(nets_dir+"/"+dir+"/"+bigname+".txt")
        epochs        = bigdata[:,0]
        tr_diags      = bigdata[:,1]
        tr_offdiags   = bigdata[:,2]
        tr_taylorgrad = bigdata[:,3]
        tr_actualdiff = bigdata[:,4]
        te_diags      = bigdata[:,5]
        te_offdiags   = bigdata[:,6]
        te_taylorgrad = bigdata[:,7]
        te_actualdiff = bigdata[:,8]

        print(train_modes.shape)


        axs[0,k].plot(ss_train[:llw]**2 / m_train, '.-', color="k", label="base") #(0.2+0.7*epoch/max(plot_epoch_ids), 0, 0), alpha=alphaval)
        for epoch in plot_epoch_ids:
            label = ""
            if epoch == plot_epoch_ids[0]:
                label = "initial"
            elif epoch == plot_epoch_ids[-1]:
                label = "final"
            axs[0,k].plot(ss_train[:llw]**2*train_modes[epoch,:] / m_train, '--', color=(0.2+0.7*epoch/max(plot_epoch_ids), 0, 0), alpha=alphaval, label=label)
            
        axs[0,k].set_yscale("log")
        axs[0,k].set_xlim((-1,50)) #yscale("log")

        if k == 1:
            axs[0,k].legend(
                fontsize=10, 
                loc='upper right', 
                bbox_to_anchor=(1.04, 1.04),                        
                labelspacing=0.2,      # vertical space between entries (default: 0.5)
                handlelength=1.0,      # length of the legend handle/line (default: 2.0)
                handletextpad=0.4,     # space between handle and label (default: 0.8)
                borderpad=0.2,         # padding inside the legend box (default: 0.4)
                borderaxespad=0.5,     # padding between legend and axes (default: 0.5)
                columnspacing=1.0,     # space between columns (default: 2.0)
            )

        j = line_id
        base_loss_train          = bigdata[j,startind               :startind+llw]
        base_loss_test           = bigdata[j,startind+llw           :startind+2*llw]
        VTV_train_flat           = bigdata[j,startind+2*llw+0*llw**2:startind+2*llw+1*llw**2] 
        VTV_test_flat            = bigdata[j,startind+2*llw+1*llw**2:startind+2*llw+2*llw**2]

        VTV_train           = don_code.np.reshape(VTV_train_flat, (llw,llw))
        VTV_test            = don_code.np.reshape(VTV_test_flat, (llw,llw))


        fancyshow(VTV_train, axs[1, k], fig, minval_tr, maxval_tr)
        
        k += 1

    else:
        print("not found", dir)

axs[0, 0].set_xticks([]) #0, 19, 39], ["1", "20", "40"])
axs[0, 1].set_xticks([]) #0, 19, 39], ["1", "20", "40"])
axs[0, 1].set_yticks([]) #0, 19, 39], ["1", "20", "40"])

axs[1, 0].set_yticks([0, 19, 39], ["1", "20", "40"])
axs[1, 1].set_yticks([]) #0, 19, 39], ["1", "20", "40"])
axs[1, 0].set_xticks([0, 19, 39], ["1", "20", "40"])
axs[1, 1].set_xticks([0, 19, 39], ["1", "20", "40"])


fig.text(0.5, 0.01, r"mode index $j$", fontsize=fontsize)
axs[0,0].set_ylabel(r"weighted \\ mode losses $\sigma_j^2 L_j$", fontsize=fontsize)
axs[1,0].set_ylabel(r"coupling matrix $S_{ij}$", fontsize=fontsize)

xvals,yvals = axs[0,0].axes.get_xlim(),axs[0,0].axes.get_ylim()
yrange = don_code.np.log10(yvals[1]) - don_code.np.log10(yvals[0])

axs[0,0].set_aspect(50.0/yrange, adjustable='box')
axs[0,1].set_aspect(50.0/yrange, adjustable='box')

axs[0,0].set_title(r"$w=$ "+str(w1), fontsize=fontsize)
axs[0,1].set_title(r"$w=$ "+str(w2), fontsize=fontsize)



fig.subplots_adjust(wspace=0.0, hspace=0.0, right=0.99, left=0.25, top=0.92, bottom=0.11)

path = don_code.figures_dir
tmp = "Fig9b"
fig.savefig(path+"/pdfs/"+tmp+".pdf")
fig.savefig(path+"/pngs/"+tmp+".png")
#don_code.plt.show()
