#1708thesisrelevant

from don_code import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

def getname(direc):
    if "whichT-1_doStackedFalse_doSigma0_siscFirst" in direc:
        return r"DeepONet", 0
    elif "whichT0_doStackedFalse_doSigma0_siscFirst" in direc:
        return r"$C=I$", 1
        #return r"POD-DON, $C=I$", 1
    elif "whichT0_doStackedFalse_doSigma1_sisc1.0" in direc:
        return r"$C=\Sigma_1$", 2
        #return r"SVDONet, $C=\Sigma_1$", 2
    elif "whichT0_doStackedFalse_doSigma0_sisc1.0" in direc:
        #return r"hs-POD-DON, $C=\sigma_1 I$", 3
        return r"$C=\sigma_1 I$", 3
    elif "whichT0_doStackedFalse_doSigma1_siscFirst" in direc:
        #return r"hs-SVDONet, $C=\Sigma_1/\sigma_1$", 4
        return r"$C=\Sigma_1/\sigma_1$", 4
        
        
def get_colors():
    f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/colors.txt", "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()


bid = int(sys.argv[1])
#lrt = int(sys.argv[2])
#if lrt == 0:
#    lrt = ""
#else:
#    lrt = str(lrt)

batch_name, uend, _ = dic(bid)

alldirecs = sorted(os.listdir("nets"))


fig1, axs1 = plt.subplots(1, 3, sharey=True, figsize=(9,5))
axs1[0].plot([], [], ".-", color="gray", label="Test")
axs1[0].plot([], [], "--", color="gray", label="Training")
if bid == 3:
    fig2, axs2 = plt.subplots(1, 4, sharey=True, figsize=(10,5))
    axs2[0].plot([], [], ".-", color="gray", label="Test")
    axs2[0].plot([], [], "--", color="gray", label="Training")

k = 0

archtags = ["_aT0.0_aB0.0_exp0.0_Nep10000_d5_w200_llw",
            "_aT0.0_aB0.0_exp0.0_Nep5000_d3_w50_llw",
            "_aT0.0_aB0.0_exp0.0_Nep5000_d8_w50_llw",
            "_aT0.0_aB0.0_exp0.0_Nep5000_d3_w100_llw",
            "_aT0.0_aB0.0_exp0.0_Nep5000_d8_w100_llw"]


for direc in alldirecs:
    if batch_name+"_"+uend in direc and "lrAdam" in direc and archtags[0] in direc: 
        name, modelid = getname(direc)
        lrtag = int(direc.split("_lrAdam")[1][:2])
        if lrtag == 32:
            ilr = 0
        elif lrtag == 40:
            ilr = 1
        else:
            ilr = 2
        #print(direc)
        loss_data = np.loadtxt("nets/"+direc+"/log.txt")
        if len(np.shape(loss_data)) == 2:
            epochs = loss_data[:,0]
            train = loss_data[:,1]/2
            test  = loss_data[:,4]/2

            axs1[ilr].plot(epochs, 10**test,  '.-', color=colors[modelid], label=name) #+" "+str(lrtag))
            axs1[ilr].plot(epochs, 10**train, '--', color=colors[modelid])

            #print(k, direc)

            k += 1
        #else:
        #    print("-", direc)

if bid == 3:
    for direc in alldirecs:
        if batch_name+"_"+uend in direc and "lrAdam40" in direc: # and archtags[0] in direc: 
            name, modelid = getname(direc)
            lrtag = int(direc.split("_lrAdam")[1][:2])
            j = -1
            for ij,a in enumerate(archtags[1:]):
                if a in direc:
                    j = ij 
                    break

            if j >= 0:
                loss_data = np.loadtxt("nets/"+direc+"/log.txt")
                if len(np.shape(loss_data)) == 2:
                    epochs = loss_data[:,0]
                    train = loss_data[:,1]/2
                    test  = loss_data[:,4]/2

                    axs2[j].plot(epochs, 10**test,  '.-', color=colors[modelid], label=name) #+" "+str(lrtag))
                    axs2[j].plot(epochs, 10**train, '--', color=colors[modelid], alpha=0.5)

                    #print(k, direc)

                    print(len(epochs), direc)

                k += 1
            #else:
            #    #print("-", direc)


axs1[0].legend(fontsize=12)

axs1[0].set_title(r"$\alpha_1 = 10^{-4}$", fontsize=12)
axs1[1].set_title(r"$\alpha_1 = 2 \times 10^{-3}$", fontsize=12)
axs1[2].set_title(r"$\alpha_1 = 8 \times 10^{-3}$",fontsize=12)

axs1[0].set_ylabel(r"Relative error $\delta$", fontsize=12)

for i in range(3):
    axs1[i].set_yscale("log")
    axs1[i].set_xlabel("Epochs", fontsize=12)
fig1.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/poddon-svdon/bid"+str(bid)+".pdf")


if bid == 3:
    axs2[0].set_ylabel(r"Relative error $\delta$", fontsize=12)
    axs2[0].legend(fontsize=12)
    axs2[0].set_title(r"$w=50, D=3$", fontsize=12)
    axs2[1].set_title(r"$w=50, D=8$", fontsize=12)
    axs2[2].set_title(r"$w=100, D=3$", fontsize=12)
    axs2[3].set_title(r"$w=100, D=8$", fontsize=12)

    for i in range(4):
        axs2[i].set_yscale("log")
        axs2[i].set_xlabel("Epochs", fontsize=12)
    fig2.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/poddon-svdon/bid"+str(bid)+"_2.pdf")

#plt.show()






