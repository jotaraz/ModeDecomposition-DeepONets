## Script to compute gradient components and Taylor approximations for trained SVDONets (saves under bigname)
### currently:
### bigname = "log_diagoffdiag_new"

## previous current version:
### optstepsize = 1e-8
### stepsizetag = str(optstepsize)
### bigname = "log_diagoffdiag_bigwu"+stepsizetag

from ... import don_code
#import optax
import sys 

#together = bool(int(sys.argv[1]))
bid       = int(sys.argv[1])
dostacked = False #bool(int(sys.argv[1]))
nepstr    = sys.argv[2] #"Nep" #usually Nep4000 (as of 19. sept) #sys.argv[2]
#num_vs    = int(sys.argv[4])
optstr   = "SGD" #sys.argv[3]
exponent = float(sys.argv[3])
w        = sys.argv[4]
## seed of the net to process, i.e. the "_v<vtag>" at the end of its directory name.
## optional 5th argument; defaults to 0 so old invocations keep working.
vtag     = int(sys.argv[5]) if len(sys.argv) > 5 else 0
vstr     = "_v"+str(vtag)

nets_dir = don_code.nets_dir

if bid < 2:
    llw = 20
else:
    llw = 50

if nepstr == "exp":
    do_exp_stuff = True
else:
    do_exp_stuff = False

if do_exp_stuff:
    if bid <= 5:
        w = 335
    else:
        w = 337

if dostacked:
    stacked_str = "doStackedTrue"
else:
    stacked_str = "doStackedFalse"


def norm2(x):
    return don_code.np.sum(x**2)

def flatten_pars(params):
    #vals, _ = don_code.jax.tree.flatten(params) #np.hstack([don_code.np.concatenate((params['params']['branch_net'][key]['bias'].flatten(), params['params']['branch_net'][key]['kernel'].flatten())) for key in params['params']['branch_net'].keys()])
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
    ## sigma_i^(2+2e)*L_i
    def loss_fn(params):
        G_pred, _, B = state.apply_fn(params, p, T, ScaledSigma)
        return don_code.jnp.mean((truesigma[i]**(1+exponent) * VT[i,:] - truesigma[i]**exponent * ScaledSigma[i,i] * B[:,i])**2)/n

    loss, grads = don_code.jax.value_and_grad(loss_fn)(params_tmp)
    return loss, grads

## returns (grad, not_flattened)
## grads = matrix s.t. the i-th column is the gradient wrt sigma_i^(2+2e)*L_i
## not_flattend = list of gradients wrt sigma_i^(2+2e)*L_i, in pytree form
def get_component_grads(state, p, T, truesigma, VT, exponent, ScaledSigma, params, llw):
    grads = don_code.np.zeros((get_num_pars(state), llw))
    not_flattened = []
    for i in range(llw):
        loss_i, grad_i = TE_component(state, p, T, truesigma, VT, exponent, ScaledSigma, i, params)
        grads[:,i] = flatten_pars(grad_i)
        not_flattened.append(grad_i)
    return grads, not_flattened

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

#batch_name, endtag, _ = dic(bid)
#direcs = []
#if dostacked:
#    for vtag in range(num_vs):
#        direcs.append("whichT0_doStackedTrue_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep500_d5_w"+w+"_llw50_bat"+batch_name+"_"+endtag+"_numd1000_lrSGD32_v"+str(vtag))
#
#else:
#    for vtag in range(num_vs):
#        direcs.append("whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep500_d5_w"+w+"_llw50_bat"+batch_name+"_"+endtag+"_numd1000_lrSGD32_v"+str(vtag))

#direcs = sorted(don_code.os.listdir(nets_dir+""))
## only for debugging of the exp != 0 case


batch_name, endtag, _ = don_code.dic(bid)

#if bid <= 5:
#    w = 335
#else:
#    w = 337

if do_exp_stuff:
    if exponent == 0.0 and (bid in [3, 5, 6]):
        direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp0.0_Nep10000_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lrSGD32"+vstr]

    else:
        direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp"+str(exponent)+"_Nep4000_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lrSGD32"+vstr]
else:
    direcs = ["whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp"+str(exponent)+"_Nep"+nepstr+"_d5_w"+str(w)+"_llw"+str(llw)+"_bat"+batch_name+"_"+endtag+"_numd1000_lrSGD32"+vstr]

init_lr = 1e-4

#epochs = ["101", "201", "301"]

#for i in range(0,50,1):
#    epochs.append(str(10*i+1))

epochs = []
for i in range(0,40,1):
    epochs.append(str(100*i+1))

lr_schedule = don_code.optax.exponential_decay(
    init_value=init_lr,  
    transition_steps=500,  
    decay_rate=0.95,
    staircase=True  # Set to True if you want discrete decay steps
)
optimizer = don_code.optax.sgd(learning_rate=lr_schedule)

#d = 5
#llw = 50
#last_layer = "layers_"+str((d-1)*2)



stuff = []

todo_counter = 0

bigname = "log_diagoffdiag_new"

for jdd in range(len(direcs)):
    dir = direcs[jdd]
    doneyet = True
    tmp_dir = don_code.os.listdir(nets_dir+"/"+dir)
    doneyet = (bigname+".txt" in tmp_dir)
    complete = True 
    for epoch in epochs:
        if epoch+"cur_chp" not in tmp_dir:
            complete = False
    
    #doneyet = False

    if stacked_str in dir and not doneyet and complete and nepstr in dir and optstr in dir:
        print("Will do", dir, "\n")
        todo_counter += 1
    elif stacked_str in dir and nepstr in dir and optstr in dir:
        print("Wont do", dir)
        print(stacked_str in dir, not doneyet, complete, nepstr in dir, optstr in dir)
    #if stacked_str in dir and doneyet and complete and nepstr in dir and optstr in dir:
    #    print("Already done", dir)
    #if stacked_str in dir and doneyet and not complete and nepstr in dir and optstr in dir:
    #    print("Not complete", dir)
    

print("To do counter", todo_counter)
counter = 0

## loop through direcs (i.e. all networks) and compute gradient stuff
for jdd in range(len(direcs)):
    dir = direcs[jdd]
    ## doneyet = True if the gradient stuff has already been computed
    doneyet = True
    tmp_dir = don_code.os.listdir(nets_dir+"/"+dir)
    doneyet = (bigname+".txt" in tmp_dir)
    ## has the network been trained for all relevant epochs
    complete = True 
    for epoch in epochs:
        if epoch+"cur_chp" not in tmp_dir:
            complete = False
    ## if not done yet, but finished and containing the right strings (stacked?, #epochs?, right optimizer?):
    ## compute it!
    if not doneyet and complete and ( stacked_str in dir and nepstr in dir and optstr in dir ) :
        ## get depth, width and llw (last layer width, or N), batch_name etc
        d, w, llw, _, batch_name, num_data, endtag = get_dwllw(dir)
        ## print stuff
        print(counter, "/", todo_counter)
        print(dir)
        print(batch_name, endtag, num_data, llw)

        ## load dataset
        nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, endtag, num_data)
        ## do SVD of training data matrix: utrain = uu_train * diag(ss_train) * vh_train
        uu_train, ss_train, vh_train = don_code.jnp.linalg.svd(utrain, full_matrices=False)
        ## get shapes
        n_train, m_train = don_code.np.shape(utrain)

        ## truncate to llw
        VT_train = vh_train[:llw, :]
        ts = ss_train[:llw] 
        T = uu_train[:,:llw]
        truesigma = ss_train[:llw]
        
        ## compute optimal coefficients for test data using training data modes
        VT_test  = don_code.jnp.matmul(don_code.jnp.diag(1/ts), don_code.jnp.matmul(uu_train[:, :llw].T, utest))

        ScaledSigma = don_code.jnp.diag(truesigma)
        scaleB = m_train

        ## compute norms of training and test data matrices (both classic Frobenius and re-weighted with exponent)
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
        tr_taylorupd  = []
        te_taylorupd  = []
        appendices    = []

        ## initialize (stacked or unstacked) model
        if "doStackedFalse" in dir:
            model = don_code.TDeepONet(nb, d, w, llw) 
        else:
            model = don_code.StackedTDeepONet(nb, d, w, llw)
        init_params = model.init(don_code.jax.random.PRNGKey(0), ptrain, T, ScaledSigma)

        
        ## loop through epochs
        for epoch in epochs:
            epoch_nums.append(int(epoch))
            ## learning rate in this epoch
            tmp_lr  = init_lr * 0.95**int((int(epoch)-1)/500) / truesigma[0]**(2*exponent)
            tmp_str = nets_dir+"/"+dir+"/"+epoch

            ## load parameters
            par_cur, upd_cur = don_code.load_checkpoint(init_params, init_params, path=tmp_str+"cur_chp")
            par_las, upd_las = don_code.load_checkpoint(init_params, init_params, path=tmp_str+"las_chp")
            
            ## create the TrainState corresponding to the 'last epoch' (i.e. the one before the current one)
            state_las  = don_code.TrainState.create(
                apply_fn=model.apply,
                params=par_las,
                tx=optimizer  
            )

            ## flatten parameters and updates to vectors
            flat_par_cur = flatten_pars(par_cur)
            flat_par_las = flatten_pars(par_las)
            flat_update  = flatten_pars(upd_cur)
            
            print("grads")
        
            # here L^e is the loss function, i.e., grads_las (_test) is the gradient of the training (test) loss wrt L^e
            loss_train_las_rew, grads_las_train_rew, loss_test_las_rew, grads_las_test_rew, loss_train_cur_rew, loss_test_cur_rew = all_grads(par_las, par_cur, 
                                                                                                                                        state_las, ptrain, T, 
                                                                                                                                        truesigma, VT_train, 
                                                                                                                                        exponent, ScaledSigma, scaleB)

            loss_train_las_wei, grads_las_train_wei, loss_test_las_wei, grads_las_test_wei, loss_train_cur_wei, loss_test_cur_wei = all_grads(par_las, par_cur, 
                                                                                                                                        state_las, ptrain, T, 
                                                                                                                                        truesigma, VT_train, 
                                                                                                                                        0.0, ScaledSigma, scaleB)

            ## flatten gradients to vectors
            flat_puregrads_las_train_rew = flatten_pars(grads_las_train_rew)
            flat_puregrads_las_test_rew  = flatten_pars(grads_las_test_rew)
            flat_puregrads_las_train_wei = flatten_pars(grads_las_train_wei)
            flat_puregrads_las_test_wei  = flatten_pars(grads_las_test_wei)

            ## scale the gradient to have same norm as the actual update
            ## why not use tmp_lr???
            # flat_grads_las-flat_update
            scaling                         = don_code.np.sqrt(norm2(flat_update) / norm2(flat_puregrads_las_train_rew))
            flat_grads_las_train_rew_scaled = - scaling * flat_puregrads_las_train_rew #flatten_pars(grads_las_train_rew) #*(-5e-4) #used to be flat_grads_las


            ## compute Taylor approximations to (normally) weighted test and train loss change, using either the reweighted gradient or the actual update
            taylor_gradtrrew_train = don_code.np.sum(flat_puregrads_las_train_wei * flat_grads_las_train_rew_scaled)
            taylor_gradtrrew_test  = don_code.np.sum(flat_puregrads_las_test_wei * flat_grads_las_train_rew_scaled)
            taylor_upd_train       = don_code.np.sum(flat_puregrads_las_train_wei * flat_update)
            taylor_upd_test        = don_code.np.sum(flat_puregrads_las_test_wei * flat_update)
            ## true/observed reweighted loss changes (test and train)
            loss_wei_change_train  = loss_train_cur_wei - loss_train_las_wei
            loss_wei_change_test   = loss_test_cur_wei  - loss_test_las_wei

            check_str = ""
            ## if SVDONet is not stacked, compute gradient components and S (coupling) matrix
            if "doStackedFalse" in dir:
                ## grads_tr_rew (and grads_te_rew) are matrices s.t. the i-th column is the gradient wrt sigma_i^(2+2e)*L_i
                grads_tr_rew, not_flattened = get_component_grads(state_las, ptrain, T, truesigma, 
                                                VT_train, exponent, ScaledSigma, par_las, llw)

                grads_te_rew, _             = get_component_grads(state_las, ptest, T, truesigma, 
                                                VT_test, exponent, ScaledSigma, par_las, llw)
                
                ## grads_tr_wei (and grads_te_wei) are matrices s.t. the i-th column is the gradient wrt sigma_i^2*L_i
                grads_tr_wei = don_code.np.matmul(grads_tr_rew, don_code.jnp.diag(truesigma**(-2*exponent)))
                grads_te_wei = don_code.np.matmul(grads_te_rew, don_code.jnp.diag(truesigma**(-2*exponent)))

                #print()
                
                ## all unweighted loss components at las parameters
                base_loss_train     = don_code.Tall_loss_components(par_las, state_las, ptrain, T, VT_train, ScaledSigma)
                base_loss_test      = don_code.Tall_loss_components(par_las, state_las, ptest, T, VT_test, ScaledSigma)
                

                # if they are both _rew: you get the Taylor expansion of the loss change (\mathca{L}^e) based on the gradient wrt sigma_i^(2+2e)*L_i
                # if you use _rew and _wei: you get the Taylor expansion of the loss change (\mathcal{L}^e) based on the gradient wrt sigma_i^(2+2e)*L_i 
                VTV_train     = -scaling*don_code.np.matmul(grads_tr_rew.T, grads_tr_wei)
                VTV_test      = -scaling*don_code.np.matmul(grads_tr_rew.T, grads_te_wei)

                appendix = don_code.np.zeros(2*llw*llw + 2*llw)
                appendix[:llw]      = base_loss_train
                appendix[llw:2*llw] = base_loss_test
                appendix[2*llw+0*llw**2:2*llw+1*llw**2] = VTV_train.flatten()
                appendix[2*llw+1*llw**2:2*llw+2*llw**2] = VTV_test.flatten()

                mask_diag     = don_code.np.eye(llw)
                mask_offdiag  = don_code.np.ones((llw,llw))-mask_diag
                diag_train    = don_code.np.sum(mask_diag * VTV_train)
                diag_test     = don_code.np.sum(mask_diag * VTV_test)
                offdiag_train = don_code.np.sum(mask_offdiag * VTV_train)
                offdiag_test  = don_code.np.sum(mask_offdiag * VTV_test)

                ones                     = don_code.np.ones(llw)
                #grad_train_reconstructed = don_code.np.matmul(grads_tr_rew, ones)
                #grad_test_reconstructed  = don_code.np.matmul(grads_te_rew, ones)
            
                reconstruct_diff = don_code.np.zeros(4)
                reconstruct_diff[0] = norm2(don_code.np.matmul(grads_tr_rew, ones) - flat_puregrads_las_train_rew) / norm2(flat_puregrads_las_train_rew)
                reconstruct_diff[1] = norm2(don_code.np.matmul(grads_te_rew, ones) - flat_puregrads_las_test_rew)  / norm2(flat_puregrads_las_test_rew)
                reconstruct_diff[2] = norm2(don_code.np.matmul(grads_tr_wei, ones) - flat_puregrads_las_train_wei) / norm2(flat_puregrads_las_train_wei)
                reconstruct_diff[3] = norm2(don_code.np.matmul(grads_te_wei, ones) - flat_puregrads_las_test_wei)  / norm2(flat_puregrads_las_test_wei)

                if don_code.np.abs(don_code.np.max(reconstruct_diff)) > 1e-6: #reconstruct_diff_train > 1e-10 or reconstruct_diff_test > 1e-10:
                    print("grad - grads*1 : train : ", reconstruct_diff) #, "test : ", reconstruct_diff_test)
                else:
                    check_str += "grad from grads reconstruction worked, "
                
            else:
                diag_train = taylor_gradtrrew_train
                diag_test  = taylor_gradtrrew_test 
                offdiag_train = 0
                offdiag_test  = 0
                appendix = 0

            
            if abs(scaling-tmp_lr) > 1e-6:
                check_str += "scaling != lr "+str(scaling)+" vs "+str(tmp_lr)+", "
            else:
                check_str += "scaling = lr, "                
            #print("done, scaling", scaling)
            #print(epoch, " lr:", tmp_lr)
            if epoch == epochs[0]:
                print("num pars", get_num_pars(state_las))

            print("taylor_grad_train", round(taylor_gradtrrew_train,9), "taylor_upd_train", round(taylor_upd_train,9), "loss_change_train", round(loss_wei_change_train,9))
            print("taylor_grad_test ", round(taylor_gradtrrew_test,9),  "taylor_upd_test ", round(taylor_upd_test,9), "loss_change_test ", round(loss_wei_change_test,9))
            print("train : diag ", diag_train, "offdiag", offdiag_train, "sum", diag_train+offdiag_train)
            print("test  : diag ", diag_test,  "offdiag", offdiag_test,  "sum", diag_test +offdiag_test)

            ##QQQ: is diag_train+offdiag_train == taylor_grad_train?

            diag_off_diag_diff = don_code.np.zeros(2)
            #train difference between sum of diag and offdiag and 1st order taylor expansion of loss change based on gradient (checks whether weighting and summing is done right)
            diag_off_diag_diff[0] = (diag_train+offdiag_train - taylor_gradtrrew_train)/(diag_train+offdiag_train)
            #test difference between sum of diag and offdiag and 1st order taylor expansion of loss change based on gradient (checks whether weighting and summing is done right)
            diag_off_diag_diff[1] = (diag_test +offdiag_test  - taylor_gradtrrew_test)/(diag_test+offdiag_test)
            #train difference between sum of diag and offdiag and true loss change
            #diag_off_diag_diff[2] = (diag_train+offdiag_train - loss_change_train)/(diag_train+offdiag_train)
            #test difference between sum of diag and offdiag and true loss change
            #diag_off_diag_diff[3] = (diag_test +offdiag_test  - loss_change_test)/(diag_test+offdiag_test)
            if don_code.np.abs(don_code.np.max(diag_off_diag_diff)) < 1e-6:
                check_str += "summing/splitting works, "
            else:
                check_str += "difference btw diag+offdiag and taylor (tr/te): "+str(diag_off_diag_diff)+", "
            
            
            tr_diags.append(diag_train)
            te_diags.append(diag_test)
            tr_offdiags.append(offdiag_train)
            te_offdiags.append(offdiag_test)
            tr_taylorgrad.append(taylor_gradtrrew_train)
            te_taylorgrad.append(taylor_gradtrrew_test)
            tr_actualdiff.append(loss_wei_change_train)
            te_actualdiff.append(loss_wei_change_test)
            tr_taylorupd.append(taylor_upd_train)
            te_taylorupd.append(taylor_upd_test)
            appendices.append(appendix)


            update_grad_diff       = norm2(flat_grads_las_train_rew_scaled - flat_update) / norm2(flat_update)
            las_upd_vs_cur         = norm2(flat_par_cur - flat_par_las     - flat_update) / norm2(flat_update)
            if update_grad_diff > 1e-6 or las_upd_vs_cur > 1e-6:
                print("update-(-lr)grad : ", update_grad_diff, "las+upd-cur : ", las_upd_vs_cur)

                maxentry = don_code.np.max(don_code.np.abs(flat_update))
                k = 0
                wrongindices = []
                for i in range(len(flat_grads_las_train_rew_scaled)):
                    if don_code.np.sign(flat_grads_las_train_rew_scaled[i]) != don_code.np.sign(flat_update[i]):
                        wrongindices.append(i)
                if len(wrongindices) > 0:
                    print("wi", don_code.np.min(don_code.np.array(wrongindices)), don_code.np.max(don_code.np.array(wrongindices)), len(wrongindices), len(flat_grads_las_train_rew_scaled))
                else:
                    print("no wrongindices")
            else:
                check_str += "update = grad and true update"

            diagoffdiag_vs_true = don_code.np.zeros(2)
            diagoffdiag_vs_true[0] = (diag_train+offdiag_train - loss_wei_change_train)/loss_wei_change_train
            diagoffdiag_vs_true[1] = (diag_test +offdiag_test  - loss_wei_change_test )/loss_wei_change_test
            if don_code.np.abs(don_code.np.max(diagoffdiag_vs_true)) > 0.01:   
                check_str += "train loss change != diag+offdiag "+str(diagoffdiag_vs_true) #loss_wei_change_train - diag_train - offdiag_train)
            else:
                check_str += "diag+offdiag ~= true loss change, "
            print("epoch", epoch, ":", check_str)
            print(" ")
        

        
        if "doStackedFalse" in dir:
            diag_data = don_code.np.zeros((len(epochs), 11+2*llw+2*llw*llw))
            print("lens", len(appendices), len(epochs), don_code.np.shape(diag_data))
            for ie in range(len(epochs)):
                diag_data[ie,11:] = don_code.np.array(appendices[ie])
        else:
            diag_data = don_code.np.zeros((len(epoch_nums), 11))

        diag_data[:,0]  = don_code.np.array(epoch_nums)
        diag_data[:,1]  = don_code.np.array(tr_diags)
        diag_data[:,2]  = don_code.np.array(tr_offdiags)
        diag_data[:,3]  = don_code.np.array(tr_taylorgrad)
        diag_data[:,4]  = don_code.np.array(tr_actualdiff)
        diag_data[:,5]  = don_code.np.array(te_diags)
        diag_data[:,6]  = don_code.np.array(te_offdiags)
        diag_data[:,7]  = don_code.np.array(te_taylorgrad)
        diag_data[:,8]  = don_code.np.array(te_actualdiff)
        diag_data[:,9]  = don_code.np.array(tr_taylorupd)
        diag_data[:,10] = don_code.np.array(te_taylorupd)


        print("saving data (", exponent, ")\n")
        don_code.np.savetxt(nets_dir+"/"+dir+"/"+bigname+".txt", diag_data)

        counter += 1
        #lossdata = don_code.np.loadtxt(nets_dir+"/"+dir+"/log.txt")
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