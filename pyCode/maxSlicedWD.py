
from scipy.stats import wasserstein_distance
import torch
import numpy as np
import ot ## need to pip install POT=='0.8.2'; weights need to be at least nonnegative
# the marginal weights p1 and p2 can be on cpu (when dist_mat is on gpu)
# for general case, p1 and p2 should be on the same device as dist_mat
from pyCode import myProj
from pyCode.global_var import device

def pdist(X, Y, V):
    """ 
        compute the distance matrix of projected varaibles
    """
    x_proj = torch.matmul(X, V)
    y_proj = torch.matmul(Y, V)
    dist_mat = torch.cdist(x_proj, y_proj, p=2)
    return dist_mat

def update_OT(X, Y, V, p1, p2):
    """
        compute the optimal coupling (transport plan)
        linear programming
    """
    dist_mat = pdist(X, Y, V)
    T_coupling = ot.emd(p1, p2, dist_mat)
    T_coupling = T_coupling.to(device)
    return T_coupling

def update_V(X, Y, T_coupling, old_V, lam, learn_rate):
    """
        update the projection V
        lam: sparsity parameter which is greater than or equal to 1.
    """
    n1 = X.size(0)
    n2 = Y.size(0)
    dist_mat = pdist(X, Y, old_V)
    dist_mat = torch.sqrt(dist_mat**2 + 1e-6) # smooth approximation of the Euclidean norm
    PP = T_coupling/dist_mat
    G = 0
    for i in range(n1):
        x_temp = X[i,:].repeat(n2, 1).T #p*n2
        PP_temp = torch.diag(PP[i,:])
        G = G + torch.matmul(torch.matmul(x_temp-Y.T, PP_temp), (x_temp-Y.T).T)
    grad = torch.matmul(G, old_V)
    V = old_V + learn_rate*grad
    V = myProj.myProj(V, t=lam)
    return V

def maxSlicedWD(X, Y, V0, p1, p2, lam, learn_rate=100, thresh=1e-6, max_iter=100):
    """
        optimize the max-sliced Wasserstein distance
        output distance and the optimal projection V
        V0: initial value of V
        p1, p2: marginal weights
        lam: sparsity parameter, which is greater than or equal to 1.
    """
    X = X.to(device)
    Y = Y.to(device)
    V0 = V0.to(device)
    p1 = p1.to(device)
    p2 = p2.to(device)
    old_V = V0.clone()
    iter_id = 0
    error = 1
    while error > thresh and iter_id <= max_iter:
        iter_id = iter_id+1   
        learn_rate = learn_rate / iter_id**0.5
        # learn_rate = 0.5 * learn_rate
        T_coupling = update_OT(X, Y, old_V, p1, p2)
        V = update_V(X, Y, T_coupling, old_V, lam=lam, learn_rate=learn_rate) 
        error = torch.linalg.norm(V-old_V) / (1+torch.linalg.norm(old_V))
        old_V = V.clone()
    x_proj = torch.matmul(X, V).squeeze()
    y_proj = torch.matmul(Y, V).squeeze()
    max_mswd = wasserstein_distance(x_proj.to('cpu'), y_proj.to('cpu'), p1.to('cpu'), p2.to('cpu'))
    max_mswd = torch.tensor(max_mswd).float()
    # dist_mat = pdist(X, Y, V)
    # max_mswd = ot.emd2(p1, p2, dist_mat).to('cpu') # ot.emd2() sometimes wrong, use ot.emd() instead to compute distance
    # max_mswd = torch.sum(ot.emd(p1, p2, dist_mat) * dist_mat).to('cpu')
    return max_mswd, V
