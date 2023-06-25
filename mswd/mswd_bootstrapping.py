import torch
import numpy as np
import ot
from scipy.stats import wasserstein_distance
from numpy.random import multinomial

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Two-Sample Distribution Tests for High-Dimensional Data
# @description Test whether the two populations have equal distribution or not
# @param X (Y) tensor n1*p (n2*p) matrix, with each row representing an observation
# @param lam tensor, a candidate set of sparsity parameter(s)
# @param B the number of bootstrap replicates; default value: 500
# @param alpha the significance level; default value: 0.05
# @param reps the number of different initial directions used; default value: 10
# @param istune whether to use k-fold cross validation to tune the sparsity parameter; default value: True
# @param tune_k the number of folds for cross validation; default value: 2
# @param tune_reps the number of different initial directions used for cross validation; default value: 5
# @param opt_lr the learning rate for the projected gradient descent algorithm; default value: 100
# @param opt_thresh the tolerance level for the projected gradient descent algorithm; default value: 1e-6
# @param opt_maxIter the maximum number of iterations for the projected gradient descent algorithm; default value: 100
# @param proj_thresh the tolerance level for the quadratic approximation secant bisection method; default value: 1e-6
# @param proj_maxIter the maximum number of iterations for the quadratic approximation secant bisection method; default value: 100
# @return a dictionary with
#               \key{test statistic}: tensor test statistic
#               \key{pvalue}: tensor p-value
#               \key{reject}: 1 (True) or 0 (False), whether the null hypothesis is rejected
#               \key{accept}: 1 (True) or 0 (False), whether the null hypothesis is not rejected
#               \key{optimal direction}: tensor, the optimal projection direction that maximizes the Wasserstein distance between the resulting univariate distributions
#               \key{alpha}: float, the significance level
#               \key{bootstrap sample}: tensor, the bootstrapped statistic
#               \key{lam}: tensor, the sparsity parameter(s)
#               \key{selected lam}: the selected sparsity parameter with k-fold cross validation
#               \key{scale}: (n1*n2/(n1+n2))**0.5
#               \key{sci bound optproj}: the bound of the SCI for the max-sliced Wasserstein distance (under the optimal projection direction)
#               \key{selected initial direction}: tensor, the initial direction used for obtaining the final optimal direction

def mswdtest(X, Y, lam, istune=True, reps=10, tune_k=2, tune_reps=5, alpha=0.05, B=500, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
    X = X.to(device)
    Y = Y.to(device)
    n1, sample_dim = X.size()
    n2 = Y.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    scale = (n1*n2/(n1+n2))**0.5

    if istune == True:
        if lam.size() == torch.Size([]):
            lam = torch.tensor([lam])
        if lam.size(0) > 1:
            tmp_tune = tune(X, Y, lam, tune_k, tune_reps)
            lam_selected = tmp_tune['selected lam']
        else:
            lam_selected = lam[0]
        
        tmp = maxSlicedWD(X, Y, p1, p2, lam_selected, reps, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
        Tn = scale * tmp['distance']
        V = tmp['optimal direction']
        V0 = tmp['selected initial direction']
        mswd_bsample = maxSlicedWD_bootstrap(X, Y, lam_selected, V0, B, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
        pval = torch.mean((mswd_bsample>Tn).float())
        reject = (pval < alpha)
        accept = (pval >= alpha)
        sci_bound_optproj = (Tn - torch.quantile(mswd_bsample, 1-alpha)) / scale
        return {'test statistic': Tn, 'reject': reject, 'accept': accept, 'pvalue': pval, 'optimal direction': V, 'alpha': alpha, \
                'bootstrap sample': mswd_bsample, 'lam': lam, 'selected lam': lam_selected, 'sci bound optproj': sci_bound_optproj, \
                'selected initial direction': V0, 'scale': scale}

    else:
        nlam = lam.size(0)
        Tn = torch.zeros(nlam)
        pval = torch.zeros(nlam)
        reject = torch.zeros(nlam)
        accept = torch.zeros(nlam)
        sci_bound_optproj = torch.zeros(nlam)
        V_all = torch.zeros(sample_dim, nlam)
        V0_all = torch.zeros(sample_dim, nlam)
        mswd_bsample_all = torch.zeros(B, nlam)

        for i in range(nlam):
            tmp = maxSlicedWD(X, Y, p1, p2, lam[i], reps, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
            Tn[i] = scale * tmp['distance']
            V_all[:, i:(i+1)] = tmp['optimal direction']
            V0 = tmp['selected initial direction']
            V0_all[:, i:(i+1)] = V0
            mswd_bsample = maxSlicedWD_bootstrap(X, Y, lam[i], V0, B, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
            mswd_bsample_all[:, i] = mswd_bsample
            pval[i] = torch.mean((mswd_bsample>Tn[i]).float())
            reject[i] = (pval[i] < alpha)
            accept[i] = (pval[i] >= alpha)
            sci_bound_optproj[i] = (Tn[i] - torch.quantile(mswd_bsample, 1-alpha)) / scale
        return {'test statistic': Tn, 'reject': reject, 'accept': accept, 'pvalue': pval, 'optimal direction': V_all, \
                    'bootstrap sample': mswd_bsample_all, 'alpha': alpha, 'lam': lam, 'sci bound optproj': sci_bound_optproj, \
                    'selected initial direction': V0_all, 'scale': scale}

# @description Construct the one-sided simultaneous confidence interval (SCI) for projection directions of interest
# @param V tensor, the projection directions of interest
# @param bsample tensor, the bootstrapped statistic
# @details check if the projection direction satisfy \|v\|_1 <= lam, if so, return the corresponding results,
#          if not, return nan
def mswd_sci_direction(X, Y, V, lam, bsample=None, B=500, reps=10, tune_k=2, tune_reps=5, alpha=0.05, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
    
    if lam.size() == torch.Size([]):
        lam = torch.tensor([lam])
    if lam.size(0) > 1:
        tmp_tune = tune(X, Y, lam, tune_k, tune_reps)
        lam_selected = tmp_tune['selected lam']
        # if the selected sparsity parameter is not provided
        # a candidate set of sparsity parameters is given, then we should obtain the bootstrapped
        # sample with the selected sparsity parameter by cross validation
        bsample = None
    else:
        lam_selected = lam[0]

    X = X.to(device)
    Y = Y.to(device)
    V = V.to(device)
    n1, sample_dim = X.size()
    n2 = Y.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    scale = (n1*n2/(n1+n2))**0.5
    if bsample == None:
        tmp = maxSlicedWD(X, Y, p1, p2, lam_selected, reps, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
        V0 = tmp['selected initial direction']
        bsample = maxSlicedWD_bootstrap(X, Y, lam_selected, V0, B, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
    
    thresh = torch.quantile(bsample, 1-alpha)
    num_directions = V.size(1)
    X_proj = torch.matmul(X, V).to('cpu')
    Y_proj = torch.matmul(Y, V).to('cpu')
    num_directions = X_proj.size(1)
    distance = torch.zeros(num_directions)
    sci_bounds = torch.zeros(num_directions)
    for i in range(num_directions):
        if torch.norm(V[:,i:(i+1)], 1) > lam_selected:
            distance[i] = float('nan')
            sci_bounds[i] = float('nan')
        else:
            distance[i] = wasserstein_distance(X_proj[:,i], Y_proj[:,i], p1, p2)
            sci_bounds[i] = distance[i] - thresh / scale
    V = V.to('cpu')
    return {'sci bounds': sci_bounds, 'distance': distance, 'lam': lam, 'selected lam': lam_selected, \
            'bootstrap sample': bsample, 'alpha': alpha, 'direction': V, 'scale': scale}

# @description Construct the one-sided simultaneous confidence interval (SCI) for marginal distributions of interest 
# @param idx list, the indices of marginal coordinates of interest
# @detail given a subset of coordinates, compute the max-sliced Wasserstein distance on this subset
def mswd_sci_marginal(X, Y, idx, lam, bsample=None, B=500, reps=10, tune_k=2, tune_reps=5, alpha=0.05, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
    
    if lam.size() == torch.Size([]):
        lam = torch.tensor([lam])
    if lam.size(0) > 1:
        tmp_tune = tune(X, Y, lam, tune_k, tune_reps)
        lam_selected = tmp_tune['selected lam']
        # if the selected sparsity parameter is not provided
        # a candidate set of sparsity parameters is given, then we should obtain the bootstrapped
        # sample with the selected sparsity parameter by cross validation
        bsample = None 
    else:
        lam_selected = lam[0]

    X = X.to(device)
    Y = Y.to(device)
    n1, sample_dim = X.size()
    n2 = Y.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    scale = (n1*n2/(n1+n2))**0.5
    if bsample == None:
        tmp = maxSlicedWD(X, Y, p1, p2, lam_selected, reps, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
        V0 = tmp['selected initial direction']
        bsample = maxSlicedWD_bootstrap(X, Y, lam_selected, V0, B, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
    
    thresh = torch.quantile(bsample, 1-alpha)
    num_marginals = len(idx)
    distance = torch.zeros(num_marginals)
    sci_bounds = torch.zeros(num_marginals)
    # directions = [None] * num_marginals # list
    directions = torch.zeros(sample_dim, num_marginals)
    for i in range(num_marginals):
        X_tmp = X[:, idx[i]]
        Y_tmp = Y[:, idx[i]]
        out = maxSlicedWD(X_tmp, Y_tmp, p1, p2, lam_selected, reps, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
        distance[i] = out['distance']
        # V = torch.zeros(sample_dim, 1)
        # V[idx[i],:] = out['optimal direction']
        # directions[i] = V # list
        directions[idx[i],i:(i+1)] = out['optimal direction']
        sci_bounds[i] = distance[i] - thresh / scale
    return {'sci bounds': sci_bounds, 'distance': distance, 'optimal direction': directions, 'bootstrap sample': bsample, \
            'alpha': alpha, 'lam': lam, 'selected lam': lam_selected, 'index set': idx, 'scale': scale}

 
def maxSlicedWD(X, Y, p1, p2, lam, reps=10, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):

    sample_dim = X.size(1)
    max_mswd = 0
    mswd = torch.zeros(reps)
    for i in range(reps): # different initial values to alleviate the non-convexity issue
        V0 = torch.randn(sample_dim, 1)
        V0 = V0 / torch.norm(V0)
        V0 = myProj(V0, lam, proj_thresh, proj_maxIter)
        mswd[i], V = maxSlicedWD_PGD(X, Y, V0, p1, p2, lam, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
        if mswd[i] >= max_mswd:
            V_opt = V.clone()
            V0_opt = V0.clone()
            max_mswd = mswd[i]
    V_opt = V_opt.to('cpu')
    V0_opt = V0_opt.to('cpu')    
    return {'distance': max_mswd, 'all distance': mswd, 'optimal direction': V_opt, 'selected initial direction': V0_opt}

    
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

def update_V(X, Y, T_coupling, old_V, lam, opt_lr, proj_thresh=1e-6, proj_maxIter=100):
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
    V = old_V + opt_lr*grad
    V = myProj(V, lam, proj_thresh, proj_maxIter)
    return V

def maxSlicedWD_PGD(X, Y, V0, p1, p2, lam, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
    """
        projected gradient descent
        output distance and the optimal projection V
        V0: initial projection direction
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
    while error > opt_thresh and iter_id <= opt_maxIter:
        iter_id = iter_id+1   
        opt_lr = opt_lr / iter_id**0.5
        # opt_lr = 0.8 * opt_lr
        T_coupling = update_OT(X, Y, old_V, p1, p2)
        V = update_V(X, Y, T_coupling, old_V, lam, opt_lr, proj_thresh, proj_maxIter) 
        error = torch.linalg.norm(V-old_V) / (1+torch.linalg.norm(old_V))
        old_V = V.clone()
    x_proj = torch.matmul(X, V).squeeze()
    y_proj = torch.matmul(Y, V).squeeze()
    mswd = wasserstein_distance(x_proj.to('cpu'), y_proj.to('cpu'), p1.to('cpu'), p2.to('cpu'))
    mswd = torch.tensor([mswd]).float()
    V = V.to('cpu')
    return mswd, V

def tune(X, Y, lam, k=5, reps=10, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
    
    if lam.size() == torch.Size([]): 
        lam = torch.tensor([lam])
    nlam = lam.size(0)
    n1, sample_dim = X.size()
    n2 = Y.size(0)
    n1_test = n1 // k
    n2_test = n2 // k
    n1_train = n1 - n1_test
    n2_train = n2 - n2_test
    rec_mswd = torch.zeros(k, nlam)
    # print('cross validation: sparsity parameter')
    for i in range(k):
        loc = np.linspace(n1_test*i, n1_test*(i+1), num=n1_test, endpoint=False, dtype=int)
        X_te = X[loc,:]
        loc2 = np.delete(np.arange(n1), loc)
        X_tr = X[loc2,:]
        loc = np.linspace(n2_test*i, n2_test*(i+1), num=n2_test, endpoint=False, dtype=int)
        Y_te = Y[loc,:]                
        loc2 = np.delete(np.arange(n2), loc)
        Y_tr = Y[loc2,:]
        X_te = X_te.to('cpu')
        Y_te = Y_te.to('cpu')
        for i_lam in range(nlam):
            p1 = torch.ones(n1_train)/n1_train
            p2 = torch.ones(n2_train)/n2_train
            p1_te = torch.ones(n1_test)/n1_test
            p2_te = torch.ones(n2_test)/n2_test            
            tmp = maxSlicedWD(X_tr, Y_tr, p1, p2, lam[i_lam], reps, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
            V = tmp['optimal direction']
            data_proj1 = torch.matmul(X_te, V).squeeze()
            data_proj2 = torch.matmul(Y_te, V).squeeze()
            rec_mswd[i, i_lam] = wasserstein_distance(data_proj1, data_proj2, p1_te, p2_te)
    lam_selected = lam[torch.argmax(torch.mean(rec_mswd, dim=0))]
    return {'selected lam': lam_selected, 'cv result': rec_mswd, 'lam': lam, 'nfold': k, 'reps': reps}

def maxSlicedWD_bootstrap(X, Y, lam, V0=None, B=500, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):

    n1, sample_dim = X.size()
    n2 = Y.size(0)
    if V0 == None:
        V0 = torch.randn(sample_dim, 1)
        V0 = V0 / torch.norm(V0)
        V0 = myProj(V0, lam, proj_thresh, proj_maxIter)
    X = X.to(device)
    Y = Y.to(device)
    V0 = V0.to(device)
    data = torch.cat((X, Y), axis=0)
    n = n1 + n2
    mswd_bsample = torch.zeros(B)
    for i_bootstrap in range(B):
        epsilon1 = multinomial(n1, np.ones(n1)/n1)
        epsilon2 = multinomial(n2, np.ones(n2)/n2)
        epsilon1 = torch.from_numpy(epsilon1)
        epsilon2 = torch.from_numpy(epsilon2)
        p1 = torch.cat((epsilon1/(2*n1),torch.ones(n2)/(2*n2)))
        p2 = torch.cat((torch.ones(n1)/(2*n1), epsilon2/(2*n2)))
        mswd_bsample[i_bootstrap],_ = maxSlicedWD_PGD(data, data, V0, p1, p2, lam, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)   
    mswd_bsample = 2* mswd_bsample * (n1*n2/(n1+n2))**0.5
    return mswd_bsample


def phi_rho(rho, vals, lam):
    """
        compute the value of phi(rho) (\|v\|_1 <= lam)
    """
    abs_v_sorted = vals['abs_v_sorted']
    idx = torch.where(abs_v_sorted > rho)
    phi = torch.norm(abs_v_sorted[idx] - rho, 1)**2 - lam**2*torch.norm(abs_v_sorted[idx]-rho, 2)**2 # better calculation precision
    return phi

def get_values(V):
    """ output some useful quantities """
    V = V.to(device)  
    abs_v = torch.abs(V)
    abs_v_sorted, indices= abs_v.sort(descending=True, dim=0) # V[indices] = values
    s_l1 = torch.cumsum(abs_v_sorted, dim=0)
    w_l2 = torch.cumsum(abs_v_sorted**2, dim=0)
    return {'abs_v_sorted': abs_v_sorted, 's_l1': s_l1, 'w_l2': w_l2, 'indices': indices}

def get_V(V, lam):
    V = V.to(device)
    dim = V.size(0)
    ones = torch.ones(dim,1,device=device)
    abs_V = torch.abs(V)
    V1 = abs_V - lam*ones
    indices = (V1>0)
    VV = torch.zeros(dim,1,device=device)
    VV[indices] = V1[indices]
    VV = VV / torch.norm(VV, p=2)
    return VV

def myProj(V, lam, thresh=1e-6, maxIter=100):
    V = V.to(device)
    if torch.norm(V, p=1) <= (lam*torch.norm(V, p=2)):
        VV = V / torch.norm(V, p=2)
    else:
        sign_v = torch.sign(V)
        dim = V.size(0)       
        vals = get_values(V)
        abs_v_sorted = vals['abs_v_sorted']
        I1 = torch.sum(abs_v_sorted >= abs_v_sorted[0])   
        if I1 > (lam**2):
            if I1 == 1:
                print('UserWarning: there exists no solution!')
                return None
            else:              
                VV = torch.zeros(dim, 1, device=device)
                inds = vals['indices'][0:I1]
                VV[inds[1:I1]] = (lam*(I1-1) - ((I1-1)*(I1-lam**2))**0.5) / (I1*(I1-1))
                VV[inds[0]] = lam - (I1-1)*VV[inds[1]]
                VV = VV * sign_v
        elif I1 == (lam**2):
            VV = torch.zeros(dim, 1, device=device)
            VV[vals['indices'][0:I1],0] = torch.ones(I1,1,device=device)/I1**0.5
            VV = VV * sign_v
        else:
            l = 0 # phi(l)>0
            temp = torch.topk(torch.unique(abs_v_sorted), k=2).values
            r = temp[1] # phi(r) < 0
            phi_l = phi_rho(l, vals, lam)
            phi_r = phi_rho(r, vals, lam)
            if torch.abs(phi_l) <= torch.abs(phi_r):
                rho = l
                phi_rho_val = phi_l
            else:
                rho = r
                phi_rho_val = phi_r
            error = torch.abs(phi_rho_val)
            count = 0
            while (r-l)>thresh and error>thresh and (phi_l-phi_r)>thresh and count<=maxIter:
                count += 1
                rho_S = r - (l-r)/(phi_l - phi_r)*phi_r # l < rho_S < r; phi(rho_S) < 0
                Il = torch.sum(abs_v_sorted >= l)
                s = vals['s_l1'][Il-1]
                w = vals['w_l2'][Il-1]
                if (Il*w - s**2) <= 0: # to avoid numerical issues caused by calculation precision; this quantity should be nonnegative
                    rho_Q = s / Il
                else:
                    rho_Q = (s - lam*((Il*w - s**2)/(Il-lam**2))**0.5) / Il # phi(rho_Q) >= 0
                rho = (rho_S+rho_Q) / 2
                phi_rho_val = phi_rho(rho, vals, lam)
                if torch.sum(torch.logical_and(abs_v_sorted>l, abs_v_sorted<rho_Q))==0 and rho_Q < abs_v_sorted[0]:
                    rho = rho_Q # phi(rho) = 0
                    VV = get_V(V, rho)
                    VV = VV * sign_v
                    return VV
                elif phi_rho_val >= 0:
                    l = rho
                    r = rho_S
                else:
                    l = rho_Q
                    r = rho
                error = torch.abs(phi_rho_val)
                phi_l = phi_rho(l, vals, lam)
                phi_r = phi_rho(r, vals, lam)
            if rho < abs_v_sorted[0]:
                VV = get_V(V, rho)
                VV = VV * sign_v
            else:
                VV = torch.where(torch.abs(V) == abs_v_sorted[0], 1, 0)
                VV = VV / torch.norm(VV)
                VV = VV * sign_v
    return VV
