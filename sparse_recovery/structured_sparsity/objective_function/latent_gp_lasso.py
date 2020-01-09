from sparse_recovery.structured_sparsity.objective_function.obj_function import obj_function
from sparse_recovery.structured_sparsity.objective_function.group_lasso import group_lasso
import numpy as np


class latent_gp_lasso(obj_function):
    def __init__(self, A, groups):
        super(latent_gp_lasso, self).__init__()
        self.A = np.asarray(A)
        self.sig_dim = self.A.shape[1]
        self.groups = [list(g) for g in groups]
        self.latent_A = np.concatenate([self.A[...,g] for g in self.groups], axis=1)
        self.latent_groups = []
        offset = 0
        for g in self.groups:
            self.latent_groups.append([offset+i for i in range(len(g))])
            offset += len(g)
        self.latent_obj_func = group_lasso(A=self.latent_A, groups=self.latent_groups)
        self.latent_solution = None
        self.solution = None

        self.class_name = 'latent_gp_lasso'


    def solve(self,b):
        self.latent_solution = self.latent_obj_func.solve(b)
        self.solution = {key: self.latent_solution[key] for key in self.latent_solution if key not in ['x', 'fun', 'jac', 'hess', 'hess_inv']}
        est_x = self.latent_to_original(self.latent_solution['x'])
        self.solution['x'] = est_x
        return self.solution


    def latent_to_original(self, latent_var):
        orig_var = np.zeros(self.sig_dim)
        for gg in zip(self.groups, self.latent_groups):
            dummy = np.zeros_like(orig_var)
            for i in zip(gg[0], gg[1]):
                dummy[i[0]] = latent_var[i[1]]
            orig_var += dummy
        return orig_var






if __name__ == '__main__':
    from structured_sparsity.sparsity_pattern.chain_sparse import chain_sparse
    from structured_sparsity.objective_function.l1 import l1
    from scipy.sparse import random as sp_random

    m = 10
    n = 16
    gr_size = 3
    nr_groups = 8
    sparsity = 2

    A = sp_random(m=m,n=n,density=0.5).A
    sp = chain_sparse(sig_dim=n, group_size=gr_size, nr_groups=nr_groups, s=sparsity)
    obj_func_0 = l1(A)
    obj_func_1 = latent_gp_lasso(A,[g for g in sp.groups()])
    x_bar = sp.random_sample()
    b = np.matmul(A,x_bar)
    b.reshape([A.shape[0],1])

    print('Original signal:')
    print(x_bar)

    print('L1 solution:')
    x_hat_0 = obj_func_0.solve(b=b)
    print(x_hat_0)

    print('Latent Group Lasso solution:')
    x_hat = obj_func_1.solve(b = b)
    print(x_hat)
