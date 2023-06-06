import torch
import numpy as np
# from torch.distributions.multinomial import Multinomial
from numpy.random import multinomial
from pyCode import maxSlicedWD
from pyCode.global_var import device

def maxSlicedWD_bootstrap(X, Y, V0, lam, learn_rate=100, thresh=1e-6, B=500, alpha=0.05):
    """
        approximate quantiles by the empirical bootstrap 
        sampling multinomial variables
    """
    X = X.to(device)
    Y = Y.to(device)
    V0 = V0.to(device)
    n1 = X.size(0)
    n2 = Y.size(0)
    data = torch.cat((X, Y), axis=0)
    n = n1 + n2
    mswd = torch.zeros(B)
    for i_bootstrap in range(B):
        # gen1 = Multinomial(n1, torch.ones(n1, device=device)/n1) # generate independent multinomial samples (n1, 1/n1)
        # epsilon1 = gen1.sample()
        # gen2 = Multinomial(n2, torch.ones(n2, device=device)/n2) # generate independent multinomial samples (n2, 1/n2)
        # epsilon2 = gen2.sample()
        # p1 = torch.cat((epsilon1/(2*n1),torch.ones(n2,device=device)/(2*n2)))
        # p2 = torch.cat((torch.ones(n1,device=device)/(2*n1), epsilon2/(2*n2)))
        epsilon1 = multinomial(n1, np.ones(n1)/n1)
        epsilon2 = multinomial(n2, np.ones(n2)/n2)
        epsilon1 = torch.from_numpy(epsilon1)
        epsilon2 = torch.from_numpy(epsilon2)
        p1 = torch.cat((epsilon1/(2*n1),torch.ones(n2)/(2*n2)))
        p2 = torch.cat((torch.ones(n1)/(2*n1), epsilon2/(2*n2)))
        mswd[i_bootstrap],_ = maxSlicedWD.maxSlicedWD(data, data, V0, p1, p2, lam=lam, learn_rate=learn_rate, thresh=thresh)
        if (i_bootstrap+1)%100==0:
            print('max sliced wd empirical bootstrap {} thresh {}'.format(i_bootstrap+1, 2*(n1*n2/(n1+n2))**0.5*torch.quantile(mswd[0:(i_bootstrap+1)], q=1-alpha)))    
    mswd = 2* mswd * (n1*n2/(n1+n2))**0.5
    mswd_bootstrap_thresh = torch.quantile(mswd, q=1-alpha)
    return mswd_bootstrap_thresh, mswd













