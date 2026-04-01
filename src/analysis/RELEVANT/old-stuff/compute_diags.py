## Script to compute gradient components and Taylor approximations for trained SVDONets (saves under "log_diagoffdiag.txt")

from .. import don_code
#import optax
import sys 

#together = bool(int(sys.argv[1]))
#bid       = int(sys.argv[1])
dostacked = bool(int(sys.argv[1]))
nepstr    = sys.argv[2]
#w         = sys.argv[3]
#num_vs    = int(sys.argv[4])
#opttag   = #sys.argv[3]

if dostacked:
    stacked_str = "doStackedTrue"
else:
    stacked_str = "doStackedFalse"



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


def norm2(x):
    return don_code.np.sum(x**2)

def flatten_pars(params):
    #vals, _ = don_code.jax.tree.flatten(params) #np.hstack([np.concatenate((params['params']['branch_net'][key]['bias'].flatten(), params['params']['branch_net'][key]['kernel'].flatten())) for key in params['params']['branch_net'].keys()])
    leaves, _ = don_code.jax.tree_util.tree_flatten(params)

    # Step 2: Convert to NumPy and flatten each leaf
    flat_leaves = [don_code.np.ravel(don_code.np.asarray(leaf)) for leaf in leaves]

    # Step 3: Concatenate into one array
    return don_code.np.concatenate(flat_leaves)
    #return don_code.np.array(vals)

def get_num_pars(state):
    tmp = flatten_pars(state.params)
    return don_code.np.shape(tmp)[0]

@don_code.jax.jit
def all_grads(par_las, par_cur, state_las, ptrain, T, 
              truesigma, VT_train, exponent, ScaledSigma, scaleB):
    def loss_fn(params):
        L_data, _ = don_code.TEall_losses(params, state_las, ptrain, T, 
                                truesigma, VT_train, exponent, 
                                ScaledSigma, scaleB)
        return L_data #+ alphaB * L_orthoB
    
    def loss_fn_test(params):
        L_data, _ = don_code.TEall_losses(params, state_las, ptest, T, 
                                truesigma, VT_test, exponent, 
                                ScaledSigma, scaleB)
        return L_data

    #loss_bcur, grads_bcur = don_code.jax.value_and_grad(loss_fn)(params_bcur)
    loss_train_las, grads_las     = don_code.jax.value_and_grad(loss_fn)(par_las)
    loss_test_las, grads_las_test = don_code.jax.value_and_grad(loss_fn_test)(par_las)
    loss_train_cur, _             = don_code.jax.value_and_grad(loss_fn)(par_cur)
    loss_test_cur, _              = don_code.jax.value_and_grad(loss_fn_test)(par_cur)
            
    return loss_train_las, grads_las, loss_test_las, grads_las_test, loss_train_cur, loss_test_cur

@don_code.jax.jit
def TE_component(state, p, T, truesigma, VT, exponent, ScaledSigma, i, params_tmp):
    n, m = don_code.np.shape(VT)
    def loss_fn(params):
        G_pred, _, B = state.apply_fn(params, p, T, ScaledSigma)
        return don_code.jnp.mean((truesigma[i]**(1+exponent) * VT[i,:] - truesigma[i]**exponent * ScaledSigma[i,i] * B[:,i])**2)/n

    loss, grads = don_code.jax.value_and_grad(loss_fn)(params_tmp)
    return loss, grads

def get_component_grads(state, p, T, truesigma, VT, exponent, ScaledSigma, params, llw):
    grads = don_code.np.zeros((get_num_pars(state), llw))
    for i in range(llw):
        loss_i, grad_i = TE_component(state, p, T, truesigma, VT, exponent, ScaledSigma, i, params)
        grads[:,i] = flatten_pars(grad_i)
    return grads

def get_dwllw(name):
    #tmp = name.split("/")[1]
    tmp = name.split("_")
    whichT = int(tmp[0][6:])
    d = int(tmp[8][1:])
    w = int(tmp[9][1:])
    llw = int(tmp[10][3:])
    #where_data = 0
    batch_name = tmp[11][3:]
    num_data = 0
    for j,x in enumerate(tmp[12:]):
        if "numd" in x:
            where_data = j
            num_data = x[4:]
            break
        else:
            batch_name += "_"+x 

    tmp = batch_name.split("_")
    batch_name = tmp[0]
    #print(batch_name)
    for t in tmp[1:-1]:
        #print("t", t)
        batch_name += "_"+t 
    
    return d, w, llw, whichT, batch_name, int(num_data), tmp[-1]

# all networks in nets_dir
direcs = sorted(don_code.os.listdir(don_code.nets_dir))

init_lr = 1e-4

epochs = []
#for i in range(0,50,1):
#    epochs.append(str(10*i+1))

for i in range(0,40,1):
    epochs.append(str(100*i+1))

lr_schedule = don_code.optax.exponential_decay(
    init_value=init_lr,  
    transition_steps=500,  
    decay_rate=0.95,
    staircase=True  # Set to True if you want discrete decay steps
)
optimizer = don_code.optax.sgd(learning_rate=lr_schedule)

stuff = []

todo_counter = 0

for jdd in range(len(direcs)):
    dir = direcs[jdd]
    doneyet = True
    tmp_dir = don_code.os.listdir(don_code.nets_dir+"/"+dir)
    doneyet = ("log_diagoffdiag.txt" in tmp_dir)
    complete = True 
    for epoch in epochs:
        if epoch+"cur_chp" not in tmp_dir:
            complete = False

    if stacked_str in dir and not doneyet and complete and nepstr in dir:
        #print(dir)
        todo_counter += 1

print("To do counter", todo_counter)
counter = 0

for jdd in range(len(direcs)):
    dir = direcs[jdd]
    doneyet = True
    tmp_dir = don_code.os.listdir(don_code.nets_dir+"/"+dir)
    doneyet = ("log_diagoffdiag.txt" in tmp_dir)
    complete = True 
    for epoch in epochs:
        if epoch+"cur_chp" not in tmp_dir:
            complete = False

    if stacked_str in dir and not doneyet and complete and nepstr in dir:
        print(counter, "/", todo_counter)
        print(dir)


        _, _, llw, _, batch_name, num_data, endtag = get_dwllw(dir)
        print(batch_name, endtag, num_data, llw)


        nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, endtag, num_data)

        uu_train, ss_train, vh_train = don_code.jnp.linalg.svd(utrain, full_matrices=False)
        n_train, m_train = don_code.np.shape(utrain)

        VT_train = vh_train[:llw, :]

        ts = ss_train[:llw] 
        VT_test  = don_code.jnp.matmul(don_code.jnp.diag(1/ts), don_code.jnp.matmul(uu_train[:, :llw].T, utest))
        #print(don_code.np.shape(VT_train), don_code.np.shape(VT_test))

        T = uu_train[:,:llw]
        truesigma = ss_train[:llw]
        ScaledSigma = don_code.jnp.diag(truesigma)
        scaleB = m_train
        exponent = 0.0

        utrain_norm  = don_code.jnp.mean(utrain**2)
        utest_norm   = don_code.jnp.mean(utest**2)
        utrainE_norm = don_code.jnp.mean(don_code.jnp.matmul(don_code.jnp.diag(truesigma**(1+exponent)), VT_train)** 2)
        utestE_norm  = don_code.jnp.mean(don_code.jnp.matmul(don_code.jnp.diag(truesigma**(1+exponent)), VT_test)** 2)

        epoch_nums    = []
        tr_diags      = []
        tr_offdiags   = []
        tr_taylorgrad = []
        tr_actualdiff = []
        te_diags      = []
        te_offdiags   = []
        te_taylorgrad = []
        te_actualdiff = []

        d, w, llw, _, _, _, _ = get_dwllw(dir)
        if "doStackedFalse" in dir:
            model = don_code.TDeepONet(nb, d, w, llw) 
        else:
            model = don_code.StackedTDeepONet(nb, d, w, llw)
        init_params = model.init(don_code.jax.random.PRNGKey(0), ptrain, T, ScaledSigma)


        for epoch in epochs:
            epoch_nums.append(int(epoch))
            tmp_lr = init_lr * 0.95**int((int(epoch)-1)/500)
            tmp = don_code.nets_dir+"/"+dir+"/"+epoch

            par_cur, upd_cur = don_code.load_checkpoint(init_params, init_params, path=tmp+"cur_chp")
            par_las, upd_las = don_code.load_checkpoint(init_params, init_params, path=tmp+"las_chp")
            
            #last_layer = "layers_"+str((d-1)*2)
            #print("d", d, "w", w, "llw", llw, "ll", last_layer)

            #model_cur, params_cur = get_Tmodel(tmp+"cur", last_layer, d, w, llw, nb)
            #model_las, params_las = get_Tmodel(tmp+"las", last_layer, d, w, llw, nb)
            #update_cur            = get_Tparams(tmp+"upd_cur", w, llw, last_layer)

            #params_bcur = optax.apply_updates(params_cur, don_code.jax.tree.map(lambda x: -x, update_cur))
            state_las  = don_code.TrainState.create(
                apply_fn=model.apply,
                params=par_las,
                tx=optimizer  
            )

            flat_par_cur = flatten_pars(par_cur)
            flat_par_las = flatten_pars(par_las)
            flat_update  = flatten_pars(upd_cur)
            
            print("grads")
            loss_train_las, grads_las, loss_test_las, grads_las_test, loss_train_cur, loss_test_cur = all_grads(par_las, par_cur, state_las, ptrain, T, 
                                                                                                                truesigma, VT_train, exponent, ScaledSigma, scaleB)

            flat_puregrads_las_train = flatten_pars(grads_las)
            flat_puregrads_las_test  = flatten_pars(grads_las_test)

            scaling = don_code.np.sqrt(norm2(flat_update) / norm2(flat_puregrads_las_train))
            flat_grads_las           = -flatten_pars(grads_las) * scaling #*(-5e-4)

            taylor_grad_train = don_code.np.sum(flat_puregrads_las_train * flat_grads_las)
            taylor_grad_test  = don_code.np.sum(flat_puregrads_las_test  * flat_grads_las)
            taylor_upd_train  = don_code.np.sum(flat_puregrads_las_train * flat_update)
            taylor_upd_test   = don_code.np.sum(flat_puregrads_las_test  * flat_update)
            loss_change_train = loss_train_cur - loss_train_las
            loss_change_test  = loss_test_cur  - loss_test_las

            check_str = ""
            if "doStackedFalse" in dir:
                grads_tr = get_component_grads(state_las, ptrain, T, truesigma, 
                                            VT_train, 0.0, ScaledSigma, par_las, llw)

                grads_te = get_component_grads(state_las, ptest, T, truesigma, 
                                            VT_test, 0.0, ScaledSigma, par_las, llw)
                VTV_train     = -scaling*don_code.np.matmul(grads_tr.T, grads_tr)
                VTV_test      = -scaling*don_code.np.matmul(grads_te.T, grads_tr)
                mask_diag     = don_code.np.eye(llw)
                mask_offdiag  = don_code.np.ones((llw,llw))-mask_diag
                diag_train    = don_code.np.sum(mask_diag * VTV_train)
                diag_test     = don_code.np.sum(mask_diag * VTV_test)
                offdiag_train = don_code.np.sum(mask_offdiag * VTV_train)
                offdiag_test  = don_code.np.sum(mask_offdiag * VTV_test)

                ones                     = don_code.np.ones(llw)
                grad_train_reconstructed = don_code.np.matmul(grads_tr, ones)
                grad_test_reconstructed  = don_code.np.matmul(grads_te, ones)
            
                reconstruct_diff_train = norm2(grad_train_reconstructed - flat_puregrads_las_train) / norm2(flat_puregrads_las_train)
                reconstruct_diff_test  = norm2(grad_test_reconstructed  - flat_puregrads_las_test)  / norm2(flat_puregrads_las_test)
                if reconstruct_diff_train > 1e-10 or reconstruct_diff_test > 1e-10:
                    print("grad - grads*1 : train : ", reconstruct_diff_train, "test : ", reconstruct_diff_test)
                else:
                    check_str += "grad from grads reconstruction worked, "
                
            else:
                diag_train = taylor_grad_train
                diag_test  = taylor_grad_test 
                offdiag_train = 0
                offdiag_test  = 0

            
            print("done, scaling", scaling)
            print(epoch, " lr:", tmp_lr)
            if epoch == epochs[0]:
                print("num pars", get_num_pars(state_las))

            print("taylor_grad_train", round(taylor_grad_train,6), "taylor_upd_train", round(taylor_upd_train,6), "loss_change_train", round(loss_change_train,6))
            print("taylor_grad_test ", round(taylor_grad_test,6),  "taylor_upd_test ", round(taylor_upd_test,6), "loss_change_test ", round(loss_change_test,6))
            print("train : diag ", diag_train, "offdiag", offdiag_train, "sum", diag_train+offdiag_train)
            print("test  : diag ", diag_test,  "offdiag", offdiag_test,  "sum", diag_test +offdiag_test)
            
            tr_diags.append(diag_train)
            te_diags.append(diag_test)
            tr_offdiags.append(offdiag_train)
            te_offdiags.append(offdiag_test)
            tr_taylorgrad.append(taylor_grad_train)
            te_taylorgrad.append(taylor_grad_test)
            tr_actualdiff.append(loss_change_train)
            te_actualdiff.append(loss_change_test)

            update_grad_diff       = norm2(flat_grads_las-flat_update) / norm2(flat_update)
            las_upd_vs_cur         = norm2(flat_par_cur - flat_par_las - flat_update) / norm2(flat_par_cur)
            if update_grad_diff > 1e-10 or las_upd_vs_cur > 1e-10:
                print("update-(-lr)grad : ", update_grad_diff, "las+upd-cur : ", las_upd_vs_cur)

                maxentry = don_code.np.max(don_code.np.abs(flat_update))
                k = 0
                wrongindices = []
                for i in range(len(flat_grads_las)):
                    if don_code.np.sign(flat_grads_las[i]) != don_code.np.sign(flat_update[i]):
                        wrongindices.append(i)
                if len(wrongindices) > 0:
                    print(don_code.np.min(don_code.np.array(wrongindices)), don_code.np.max(don_code.np.array(wrongindices)), len(wrongindices), len(flat_grads_las))
                else:
                    print("no wrongindices")
            else:
                check_str += "update = grad and true update"
            print(check_str)
            print(" ")
        

        diag_data = don_code.np.zeros((len(epoch_nums), 9))
        diag_data[:,0] = don_code.np.array(epoch_nums)
        diag_data[:,1] = don_code.np.array(tr_diags)
        diag_data[:,2] = don_code.np.array(tr_offdiags)
        diag_data[:,3] = don_code.np.array(tr_taylorgrad)
        diag_data[:,4] = don_code.np.array(tr_actualdiff)
        diag_data[:,5] = don_code.np.array(te_diags)
        diag_data[:,6] = don_code.np.array(te_offdiags)
        diag_data[:,7] = don_code.np.array(te_taylorgrad)
        diag_data[:,8] = don_code.np.array(te_actualdiff)

        don_code.np.savetxt(don_code.nets_dir+"/"+dir+"/log_diagoffdiag.txt", diag_data)
        counter += 1
        #lossdata = don_code.np.loadtxt(don_code.nets_dir+"/"+dir+"/log.txt")
        #stuff.append( (epoch_nums, 
        #               don_code.np.array(tr_diags), don_code.np.array(tr_offdiags), don_code.np.array(tr_taylorgrad), don_code.np.array(tr_actualdiff), 
        #               don_code.np.array(te_diags), don_code.np.array(te_offdiags), don_code.np.array(te_taylorgrad), don_code.np.array(te_actualdiff),
        #               lossdata[:,0], lossdata[:,1]/2, lossdata[:,4]/2) )

    
'''
maxloss = -10
minloss = 10
maxdiff = -1
mindiff = 1

for jdd in range(len(direcs)):
    epoch_nums    = stuff[jdd][0]
    tr_diags      = stuff[jdd][1]
    tr_offdiags   = stuff[jdd][2]
    tr_taylorgrad = stuff[jdd][3]
    tr_actualdiff = stuff[jdd][4]
    te_diags      = stuff[jdd][5]
    te_offdiags   = stuff[jdd][6]
    te_taylorgrad = stuff[jdd][7]
    te_actualdiff = stuff[jdd][8]
    ep2           = stuff[jdd][9]
    trainloss     = stuff[jdd][10]
    testloss      = stuff[jdd][11]

    maxdiff = max(maxdiff, 
                  don_code.np.max(tr_diags[1:]), don_code.np.max(tr_offdiags[1:]), don_code.np.max(tr_actualdiff[1:]), don_code.np.max(tr_taylorgrad[1:]), 
                  don_code.np.max(te_diags[1:]), don_code.np.max(te_offdiags[1:]), don_code.np.max(te_actualdiff[1:]), don_code.np.max(te_taylorgrad[1:]))
    maxloss = max(maxloss, don_code.np.max(trainloss), don_code.np.max(testloss))

    mindiff = min(mindiff, 
                  don_code.np.min(tr_diags[1:]), don_code.np.min(tr_offdiags[1:]), don_code.np.min(tr_actualdiff[1:]), don_code.np.min(tr_taylorgrad[1:]), 
                  don_code.np.min(te_diags[1:]), don_code.np.min(te_offdiags[1:]), don_code.np.min(te_actualdiff[1:]), don_code.np.min(te_taylorgrad[1:]))
    minloss = min(minloss, don_code.np.min(trainloss), don_code.np.min(testloss))


if not together:
    fig, axs = plt.subplots(len(direcs), 3)
    fig.suptitle(batch_name)

    for jdd in range(len(direcs)):
        axs[jdd,1].set_title(direcs[jdd])
        
        epoch_nums    = stuff[jdd][0]
        tr_diags      = stuff[jdd][1]
        tr_offdiags   = stuff[jdd][2]
        tr_taylorgrad = stuff[jdd][3]
        tr_actualdiff = stuff[jdd][4]
        te_diags      = stuff[jdd][5]
        te_offdiags   = stuff[jdd][6]
        te_taylorgrad = stuff[jdd][7]
        te_actualdiff = stuff[jdd][8]
        ep2           = stuff[jdd][9]
        trainloss     = stuff[jdd][10]
        testloss      = stuff[jdd][11]
        
        axs[jdd,0].plot(epoch_nums, don_code.np.zeros(len(epochs)), '--', color='gray')
        axs[jdd,1].plot(epoch_nums, don_code.np.zeros(len(epochs)), '--', color='gray')


        axs[jdd,0].plot(epoch_nums, tr_diags, '-.', color='green')
        axs[jdd,0].plot(epoch_nums, tr_offdiags, '-.', color='red')
        axs[jdd,0].plot(epoch_nums, tr_taylorgrad, color='black')
        axs[jdd,0].plot(epoch_nums, tr_actualdiff, '*-', color='brown', alpha=0.5)
        
        axs[jdd,1].plot(epoch_nums, te_diags, '-.', color='green')
        axs[jdd,1].plot(epoch_nums, te_offdiags, '-.', color='red')
        axs[jdd,1].plot(epoch_nums, te_taylorgrad, color='black')
        axs[jdd,1].plot(epoch_nums, te_actualdiff, '*-', color='brown', alpha=0.5)

        axs[jdd,2].plot(ep2, trainloss, color="red")
        axs[jdd,2].plot(ep2, testloss, color="blue")

        axs[jdd,0].set_ylim((mindiff, maxdiff))
        axs[jdd,1].set_ylim((mindiff, maxdiff))
        axs[jdd,2].set_ylim((minloss, maxloss))
else:
    fig, axs = plt.subplots(1, 4)
    fig.suptitle(batch_name)

    markers = ["-", "--", "-.", "*-"]

    for jdd in range(len(direcs)):
        axs[1].set_title(direcs[jdd])
        
        epoch_nums    = stuff[jdd][0]
        tr_diags      = stuff[jdd][1]
        tr_offdiags   = stuff[jdd][2]
        tr_taylorgrad = stuff[jdd][3]
        tr_actualdiff = stuff[jdd][4]
        te_diags      = stuff[jdd][5]
        te_offdiags   = stuff[jdd][6]
        te_taylorgrad = stuff[jdd][7]
        te_actualdiff = stuff[jdd][8]
        ep2           = stuff[jdd][9]
        trainloss     = stuff[jdd][10]
        testloss      = stuff[jdd][11]
        
        axs[0].plot(epoch_nums, don_code.np.zeros(len(epochs)), '--', color='gray')
        axs[1].plot(epoch_nums, don_code.np.zeros(len(epochs)), '--', color='gray')


        axs[0].plot(epoch_nums, tr_diags, markers[jdd], color='green')
        axs[0].plot(epoch_nums, tr_offdiags, markers[jdd], color='red')
        axs[0].plot(epoch_nums, tr_taylorgrad, markers[jdd], color='black')
        axs[0].plot(epoch_nums, tr_actualdiff, markers[jdd], color='brown', alpha=0.5)
        
        axs[1].plot(epoch_nums, te_diags, markers[jdd], color='green')
        axs[1].plot(epoch_nums, te_offdiags, markers[jdd], color='red')
        axs[1].plot(epoch_nums, te_taylorgrad, markers[jdd], color='black')
        axs[1].plot(epoch_nums, te_actualdiff, markers[jdd], color='brown', alpha=0.5)

        axs[2].plot(ep2, trainloss, markers[jdd], color="red")
        axs[2].plot(ep2, testloss, markers[jdd], color="blue")

        axs[3].plot(ep2[:-1], 10**(trainloss[1:]) - 10**(trainloss[:-1]), markers[jdd], color="red")
        axs[3].plot(ep2[:-1], 10**(testloss[1:])  - 10**(testloss[:-1]),  markers[jdd], color="blue")


        axs[0].set_ylim((mindiff, maxdiff))
        axs[1].set_ylim((mindiff, maxdiff))
        axs[2].set_ylim((minloss, maxloss))

'''


don_code.plt.show()