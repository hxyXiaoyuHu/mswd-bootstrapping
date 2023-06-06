import torch
import numpy as np
from pyCode.global_var import device

def phi_lambda(la, vals, t):
    """
        compute the phi function
        la: compute the value of phi(la) at la, t: \|v\|_1 <= t
    """
    abs_v_sorted = vals['abs_v_sorted']
    num = torch.sum(abs_v_sorted >= la)
    if num<=0:
        s=w=0
    else:
        s = vals['s_l1'][num-1]
        w = vals['w_l2'][num-1]
    phi = (num-t**2)*(num*la-2*s)*la + s**2 - t**2*w
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

def myProj(V, t, thresh=1e-8):
    V = V.to(device)
    if torch.norm(V, p=1) <= (t*torch.norm(V, p=2)):
        VV = V / torch.norm(V, p=2)
    else:
        sign_v = torch.sign(V)
        dim = V.size(0)       
        vals = get_values(V)
        abs_v_sorted = vals['abs_v_sorted']
        I1 = torch.sum(abs_v_sorted >= abs_v_sorted[0])   
        if I1 > (t**2):
            if I1 == 1:
                print('UserWarning: there exists no solution!')
                return None
            else:              
                VV = torch.zeros(dim, 1, device=device)
                inds = vals['indices'][0:I1]
                VV[inds[1:I1]] = (t*(I1-1) - ((I1-1)*(I1-t**2))**0.5) / (I1*(I1-1))
                VV[inds[0]] = t - (I1-1)*VV[inds[1]]
                VV = VV * sign_v
        elif I1 == (t**2):
                VV = torch.zeros(dim, 1, device=device)
                VV[vals['indices'][0:I1],0] = torch.ones(I1,1,device=device)/I1**0.5
                VV = VV * sign_v
        else:
            l = 0 # phi(0)>0
            temp = torch.topk(torch.unique(abs_v_sorted), k=2).values
            r = temp[1]
            phi_l = phi_lambda(l, vals, t)
            phi_r = phi_lambda(r, vals, t)
            error=1
            count = 0
            while (r-l)>thresh and error>thresh and (phi_l-phi_r)>thresh and count<=100:
                count += 1
                if phi_r>=0:
                    lam_S = r 
                # phi_r should be smaller than zero
                # if it is larger than or equal to 0, it means that it is close to 0.
                else:
                    lam_S = r - (l-r)/(phi_l - phi_r)*phi_r
                Il = torch.sum(abs_v_sorted >= l) 
                if Il<=0:
                    s=w=0
                else:
                    s = vals['s_l1'][Il-1]
                    w = vals['w_l2'][Il-1]
                if (Il*w - s**2) <= 0:
                    lam_Q = s / Il
                else:
                    lam_Q = (s - t*((Il*w - s**2)/(Il-t**2))**0.5) / Il
                lam = (lam_S+lam_Q) / 2
                phi_lam = phi_lambda(lam, vals, t)
                if torch.sum(torch.logical_and(abs_v_sorted>l, abs_v_sorted<lam_Q))==0:
                    lam = lam_Q
                    VV = get_V(V, lam)
                    VV = VV * sign_v
                elif phi_lam > 0:
                    l = lam
                    r = lam_S
                else:
                    l = lam_Q
                    r = lam
                error = torch.abs(phi_lam)
                phi_l = phi_lambda(l, vals, t)
                phi_r = phi_lambda(r, vals, t)
            VV = get_V(V, lam)
            VV = VV * sign_v
    return VV


