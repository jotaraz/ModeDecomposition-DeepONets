## 1708thesisrelevant
## trunk error for test data
## this produces table 3 (appendix a.7) in the final thesis version

from .. import don_code

def niceformat(x):
    s = str(x)
    if "-" in s:
        return s[:5]
    else:
        return s[:4]

bids = [0, 1, 3, 4, 5, 6, 7]
llws = [20, 20, 50, 50, 50, 50, 50]

Ns20 = don_code.np.array([10, 15, 20], dtype=don_code.np.int64)
Ns50 = don_code.np.array([30, 50, 70], dtype=don_code.np.int64)

for i in range(len(bids)):
    bid = bids[i]
    llw = llws[i]
    batch_name, endtag, _ = don_code.dic(bid)
    nt, nb, rtrain, rtest, ptrain, ptest, utrain, utest = don_code.load_dataset(batch_name, endtag, 1000)
    m_train = don_code.np.shape(utrain)[1]
    m_test  = don_code.np.shape(utest)[1]
    n = don_code.np.shape(utest)[0]

    Phi, Sigma, VT = don_code.np.linalg.svd(utrain, full_matrices=True)


    WT = don_code.np.matmul(don_code.np.linalg.pinv(don_code.np.diag(Sigma)), don_code.np.matmul(Phi.T, utest))
    rec_Ate = don_code.np.matmul(Phi, don_code.np.matmul(don_code.np.diag(Sigma), WT))
    rec_max_err = don_code.np.max(don_code.np.abs( rec_Ate - utest )) 

    frob_utr = don_code.np.sum(utrain**2)
    frob_ute = don_code.np.sum(utest**2)


    #print(bid, m_train, m_test, "rec max err", rec_max_err)

    #print()

    if llw == 20:
        Ns = Ns20
    else:
        Ns = Ns50

    string = "bid"

    for N in Ns: #[int(llw*0.7), int(llw*0.85), llw]:
        string += " & "
        Phi1 = Phi[:,N]
        Sigma2 = Sigma[N:]


        W = WT.T 
        W2 = W[:,N:]

        Sig2W2T = don_code.np.matmul(don_code.np.diag(Sigma2), W2.T)

        #print(N,  / m_train,  / m_test)

        frob_tr = don_code.np.sum(Sigma2**2)
        frob_te = don_code.np.sum(Sig2W2T**2)
        string += niceformat(0.5*don_code.np.log10(frob_tr / frob_utr)) + " & " + niceformat(0.5*don_code.np.log10(frob_te / frob_ute))

    print(string+" \\\\ ")


    print(" ")





