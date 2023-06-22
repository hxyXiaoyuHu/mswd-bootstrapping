"""
    use k-fold cross-validation to select the sparisty parameter
"""

from scipy.stats import wasserstein_distance
import torch
import numpy as np
from pyCode import maxSlicedWD
from pyCode import myProj
from pyCode.global_var import device

def tune(X, Y, lams, reps=1, k=5):
    X = X.to(device)
    Y = Y.to(device)
    n_lam = lams.size(0)
    n1 = X.size(0)
    n2 = Y.size(0)
    dim = X.size(1)
    n1_test = n1 // k
    n2_test = n2 // k
    n1_train = n1 - n1_test
    n2_train = n2 - n2_test
    rec_mswd = torch.zeros(k, n_lam)
    print('cross validation: sparsity parameter')
    for i in range(k):
        if n1==n2:
            loc = np.linspace(n1_test*i, n1_test*(i+1), num=n1_test, endpoint=False, dtype=int)
            X_te = X[loc,:]
            Y_te = Y[loc,:]
            loc2 = np.delete(np.arange(n1), loc)
            X_tr = X[loc2,:]
            Y_tr = Y[loc2,:]
        else:
            loc = np.linspace(n1_test*i, n1_test*(i+1), num=n1_test, endpoint=False, dtype=int)
            X_te = X[loc,:]
            loc2 = np.delete(np.arange(n1), loc)
            X_tr = X[loc2,:]
            loc = np.linspace(n2_test*i, n2_test*(i+1), num=n2_test, endpoint=False, dtype=int)
            Y_te = Y[loc,:]                
            loc2 = np.delete(np.arange(n2), loc)
            Y_tr = Y[loc2,:]
        for i_lam in range(n_lam):
            p1 = torch.ones(n1_train)/n1_train
            p2 = torch.ones(n2_train)/n2_train
            p1_te = torch.ones(n1_test)/n1_test
            p2_te = torch.ones(n2_test)/n2_test            
            lam = lams[i_lam]
            mswd_max = 0
            for i_rep in range(reps):
                # V0 = torch.randn(dim, 1)
                V0 = np.random.normal(0,1,(dim,1))
                V0 = torch.from_numpy(V0).float()
                V0 = myProj.myProj(V0, lam)
                mswd, V = maxSlicedWD.maxSlicedWD(X_tr, Y_tr, V0, p1, p2, lam)
                if mswd > mswd_max:
                    V_opt = V.clone()
                    mswd_max = mswd.clone()
            data_proj1 = torch.matmul(X_te, V_opt).squeeze().to('cpu')
            data_proj2 = torch.matmul(Y_te, V_opt).squeeze().to('cpu')
            rec_mswd[i, i_lam] = wasserstein_distance(data_proj1, data_proj2, p1_te, p2_te)
    lam = lams[torch.argmax(torch.mean(rec_mswd, dim=0))]
    return lam
