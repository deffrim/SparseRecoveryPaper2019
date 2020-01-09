from sparse_recovery.structured_sparsity.objective_function.l1 import l1
from sparse_recovery.structured_sparsity.sparsity_pattern.s_sparse import s_sparse
import numpy as np

class l1_rec_exp(object):
    def __init__(self, matr, sparsity_range=[1], nr_sampled_supports=1, recovery_sensitivity = 1e-6):
        if len(sparsity_range) == 0 or min(sparsity_range) < 1:
            raise Exception('l1_rec_exp::__init__: Faulty sparsity range.')

        if matr.shape[1] < max(sparsity_range):
            raise Exception('The maximum sparsity {} is larger than the number of columns of the matrix {}.'.format(max(sparsity_range), matr.shape[1]))

        if nr_sampled_supports < 1:
            raise Exception('l1_rec_exp::__init__: Number of sampled supports cannot be less than 1. It is currently {}'.format(nr_sampled_supports))

        self.matr = matr
        self.sig_dim = matr.shape[1]
        self.sparsity_range = sparsity_range
        self.nr_sampled_supports = nr_sampled_supports
        self.recovery_sensitivity = recovery_sensitivity

        self.obj_func = l1(A=self.matr)
        self.all_exp_results = {s:[] for s in self.sparsity_range}
        self.mean_rec_probs = {}


    def run(self):
        for s in self.sparsity_range:
            sp = s_sparse(sig_dim=self.sig_dim, s=s)
            for count in range(self.nr_sampled_supports):
                xbar, _ = sp.random_sample()
                b = np.matmul(self.matr, xbar)
                soln_dict = self.obj_func.solve(b=b)
                xhat = soln_dict['x']
                xbar = np.reshape(xbar, newshape=xhat.shape)
                diff_norm = np.linalg.norm(xbar-xhat)
                self.all_exp_results[s].append(diff_norm)

            self.mean_rec_probs[s] = np.mean([val <= self.recovery_sensitivity for val in self.all_exp_results[s]])


    def get_mean_rec_probs(self):
        if len(self.mean_rec_probs) == 0:
            raise Exception('l1_rec_exp::__init__: No mean recovery probability recorded.')

        return self.mean_rec_probs