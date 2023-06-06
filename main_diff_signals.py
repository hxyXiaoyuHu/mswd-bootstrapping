""" Usage example: python main_diff_signals.py --model mean-decay --sample_dim 500 --sig sig5 --signal 0.8 """
"""
    implement the simulations under different signals
    mswd: the proposed bootstrapping method
    pwdPerm: projected Wasserstein distance with k=3, permutation
"""
import torch
import numpy as np
import argparse
from pyCode.sim import sim
from pyCode.global_var import device

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default='mean-decay', type=str, help='')
        parser.add_argument('--nrun', default=500, type=int, help='num of runs')
        parser.add_argument('--n1', default=250, type=int, help='')
        parser.add_argument('--n2', default=250, type=int, help='')
        parser.add_argument('--sample_dim', default=500, type=int, help='')
        parser.add_argument('--alpha', default=0.05, type=float, help='')
        parser.add_argument('--mswd', default=1, type=int, help='')
        parser.add_argument('--perm', default=1, type=int, help='')
        parser.add_argument('--smmd', default=1, type=int, help='')
        parser.add_argument('--pwdPerm', default=1, type=int, help='')
        parser.add_argument('--wd_NNapprox', default=-1, type=int, help='')
        parser.add_argument("--signal", type=float, help="signal", default=[1.0], nargs='+')
        parser.add_argument('--sig', type=str, default='', help='')

        self.args = parser.parse_args()
        self.process_args()
    
    def process_args(self):
        self.args.signal = torch.tensor(self.args.signal).float()
        self.args.n = self.args.n1 + self.args.n2
        self.args.methods = {'mswd': self.args.mswd>0, 'perm': self.args.perm>0, 'smmd': self.args.smmd>0, \
                            'pwdPerm': self.args.pwdPerm>0, 'wd_NNapprox': self.args.wd_NNapprox>0}

if __name__ == '__main__':

    opts = Options()
    opts = opts.args
    print(opts.methods)

    opts.nB = 500
    n1, n2, sample_dim, nrun = opts.n1, opts.n2, opts.sample_dim, opts.nrun
    sig = opts.sig
    signal = opts.signal
    model = opts.model
    opts.tune = True
    opts.lams = torch.exp(torch.linspace(np.log(1.5), np.log(5), steps=5))
    opts.reps = 10

    hyper_params = {'learn_rate': 0.001, 'nepoch': 1000, \
        'in_features': sample_dim, 'hidden_layers': 3, 'node': 200}

    opts.hyper_params = hyper_params

    if model == 'joint':
        if sig == 'sig5':
            signal = torch.ones(100)
        elif sig == 'sig4':
            signal = torch.ones(80)
        elif sig == 'sig3':
            signal = torch.ones(60)
        elif sig == 'sig2':
            signal = torch.ones(40)
        else:
            signal = torch.ones(20)

    mmd_decision = np.zeros(nrun)
    edl1_decision = np.zeros(nrun)
    edl2_decision = np.zeros(nrun)
    bg_decision = np.zeros(nrun)
    mswd_decision = np.zeros(nrun)
    kPWD_perm_decision = np.zeros(nrun)
    smmd_decision = np.zeros(nrun)
    wd_nnapprox_decision = np.zeros(nrun)
    for mcrun in range(nrun):
        print('run {}'.format(mcrun))
        result = sim(opts, signal, model=model) # correlated 
        if opts.methods['perm'] == True:
            mmd_decision[mcrun] = result['perm']['mmd_decision']
            edl1_decision[mcrun] = result['perm']['edl1_decision']
            edl2_decision[mcrun] = result['perm']['edl2_decision']
            bg_decision[mcrun] = result['perm']['bg_decision']

            mmd_rejPerc = np.mean(mmd_decision[0:(mcrun+1)])
            edl1_rejPerc = np.mean(edl1_decision[0:(mcrun + 1)])
            edl2_rejPerc = np.mean(edl2_decision[0:(mcrun + 1)])
            bg_rejPerc = np.mean(bg_decision[0:(mcrun + 1)])

            print('run {}, percentage of rejection: mmd perm {}\n, edl1 {}, edl2 {}, bg {} \
                '.format(mcrun, mmd_rejPerc, edl1_rejPerc, edl2_rejPerc, bg_rejPerc))

        if opts.methods['mswd'] == True:
            mswd_decision[mcrun] = result['mswd_decision']
            mswd_rejPerc = np.mean(mswd_decision[0:(mcrun + 1)])

            print('run {}, percentage of rejection: max sliced wd emp {}'.format(mcrun, mswd_rejPerc))

        if opts.methods['smmd'] == True:
            smmd_decision[mcrun] = result['smmd_decision']
            smmd_rejPerc = np.mean(smmd_decision[0:(mcrun + 1)])
            print('run {}, percentage of rejection: smmd {}'.format(mcrun, smmd_rejPerc))

        if opts.methods['pwdPerm'] == True:
            kPWD_perm_decision[mcrun] = result['kPWD_perm_decision']
            pwd_perm_rejPerc = np.mean(kPWD_perm_decision[0:(mcrun+1)])
            print('run {}, percentage of rejection: pwd perm k3 {}'.format(mcrun, pwd_perm_rejPerc))           

        if opts.methods['wd_NNapprox'] == True:
            wd_nnapprox_decision[mcrun] = result['wd_nnapprox_decision']
            wd_nnapprox_rejPerc = np.mean(wd_nnapprox_decision[0:(mcrun + 1)])
            print('run {}, percentage of rejection: wd nnapprox {}'.format(mcrun, wd_nnapprox_rejPerc))


