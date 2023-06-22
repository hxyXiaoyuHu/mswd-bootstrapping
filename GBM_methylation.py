# implement the data analysis
import torch
import numpy as np
import math
from scipy.stats import wasserstein_distance
from pyCode import maxSlicedWD
from pyCode.maxSlicedWD_bootstrap import maxSlicedWD_bootstrap
from pyCode import myProj
from pyCode.tune import tune
from pyCode.global_var import device

alpha = 0.05
data1 = np.loadtxt('GBM/GBM_prognostic_low.txt')
data2 = np.loadtxt('GBM/GBM_prognostic_high.txt')
n1, sample_dim = data1.shape
n2 = data2.shape[0]
n = n1 + n2

nB = 500
lams = torch.exp(torch.linspace(math.log(2), math.log(sample_dim**0.5), steps=20))
reps = 10
data1 = torch.from_numpy(data1).float().to(device)
data2 = torch.from_numpy(data2).float().to(device)
p1 = torch.ones(n1)/n1
p2 = torch.ones(n2)/n2
# tune the sparsity parameter with CV
lam = tune(data1, data2, lams, k=2, reps=10)
scale = (n1*n2/(n1+n2))**0.5
mswd = torch.zeros(reps)
max_mswd = 0
for i in range(reps):
    V0 = torch.randn(sample_dim, 1)
    V0 = myProj.myProj(V0, lam)
    mswd[i], V = maxSlicedWD.maxSlicedWD(data1, data2, V0, p1, p2, lam=lam, learn_rate=100, thresh=1e-6)
    if mswd[i] > max_mswd:
        max_mswd = mswd[i]
        V0_opt = V0.clone()
        V_opt = V.clone()
statistic = scale * max_mswd 
V_opt = V_opt.to('cpu')
mswd_quantile, mswd_bs = maxSlicedWD_bootstrap(data1, data2, V0_opt, lam=lam, B=nB, learn_rate=100, thresh=1e-6)
mswd_pval = torch.mean((mswd_bs > statistic).float())

# investigate marginal difference
data1 = data1.to('cpu')
data2 = data2.to('cpu')
stat_rec = torch.zeros(sample_dim)
is_significant = torch.zeros(sample_dim)
pvals = torch.zeros(sample_dim)
for j in range(sample_dim):
    stat_rec[j] = scale * wasserstein_distance(data1[:,j], data2[:,j], p1, p2)
    pvals[j] = torch.mean((mswd_bs>stat_rec[j]).float())
    is_significant[j] = stat_rec[j] > (torch.quantile(mswd_bs, 1-alpha))
# np.savetxt('GBM/GBM_grade_marginal_pvals.txt', pvals)

# genes in both prognostic genes and some GO terms (BP)
gene_sets = ['mitotic_cell_cycle', 'mitotic_cell_cycle_process', 'cell_cycle_process', \
            'cell_cycle', 'DNA_metabolic_process', 'cell_division', 'mitotic_nuclear_division']
data1 = data1.to(device)
data2 = data2.to(device)
statistic = torch.zeros(len(gene_sets))
for l in range(len(gene_sets)):
    gene_set_name = gene_sets[l]
    idx = np.loadtxt('GBM/idx_genes_{}.txt'.format(gene_set_name))
    idx = idx - 1
    idx = idx.astype(int)
    reps = 10
    wd_val = torch.zeros(reps)
    max_wd = 0
    V_opt = torch.ones(np.size(idx))
    for i in range(reps):
        V0 = torch.randn(np.size(idx), 1)
        V0, _ = torch.linalg.qr(V0)
        wd_val[i], V = maxSlicedWD.maxSlicedWD(data1[:,idx], data2[:,idx], V0, p1, p2, lam=lam, learn_rate=100, thresh=1e-6)
        if wd_val[i]>max_wd:
            max_wd = wd_val[i]
            V_opt = V      
    statistic[l] = scale * max_wd
    V_opt = V_opt.to('cpu')
    # np.savetxt('GBM/direction_{}.txt'.format(gene_set_name), V_opt)
