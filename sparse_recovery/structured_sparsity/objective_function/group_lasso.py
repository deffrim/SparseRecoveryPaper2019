from sparse_recovery.structured_sparsity.objective_function.obj_function import obj_function
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import random as sp_random
from utils import signature_from_grps, bottom_clip

class group_lasso(obj_function):
    def __init__(self, A, groups):
        super(group_lasso,self).__init__()

        self.A = A
        self.sig_dim = A.shape[1]
        self.groups = [list(g) for g in groups]
        self.signs_dict = signature_from_grps(sig_dim=self.sig_dim, groups=self.groups)
        self.func = lambda x : sum([np.linalg.norm(x[g]) for g in self.groups])
        self.grad_vec = lambda x : np.array([sum([bottom_clip(x[i], 1.0e-9)/np.linalg.norm(x[list(g)]) for g in self.signs_dict[i]]) for i in range(self.sig_dim)])
        self.class_name = 'group_lasso'


    def solve(self, b):
        lin_const = lambda x: self.A @ x - b
        cons = [{"type": "eq", "fun": lin_const}]
        x0 = sp_random(1, self.sig_dim, density=1).A
        norm = np.linalg.norm(x0)
        x0 /= norm
        # x0 = np.zeros(shape=[1,self.sig_dim])
        res = minimize(fun=self.func, x0=x0, jac=self.grad_vec, constraints=cons, method='SLSQP', options={'maxiter':1000})
        self.solution = {key: res[key] for key in res if key not in ['jac', 'hess', 'hess_inv']}
        # res = minimize(fun=self.func, x0=x0, constraints=cons)
        # if not res.success:
        #     # raise Exception('l1 : Optimization did not converge. Quitting.')
        #     print('{}::solve : Optimization did not converge.'.format(self.class_name))
        #     return None
        # self.solution = res.x
        return self.solution


    def hessian(self): # Incomplete!
        matr = np.zeros(shape=[self.sig_dim, self.sig_dim])
        for i in range(self.sig_dim):
            for g in self.signs_dict[i]:
                for j in g:
                    pass





if __name__ == '__main__':
    from structured_sparsity.sparsity_pattern.chain_sparse import chain_sparse
    from structured_sparsity.objective_function.l1 import l1
    m = 10
    n = 16
    gr_size = 3
    nr_groups = 8
    sparsity = 2

    A = sp_random(m=m,n=n,density=0.5).A
    sp = chain_sparse(sig_dim=n, group_size=gr_size, nr_groups=nr_groups, s=sparsity)
    obj_func_0 = l1(A)
    obj_func_1 = group_lasso(A,[g for g in sp.groups()])
    x_bar = sp.random_sample()
    b = np.matmul(A,x_bar)
    b.reshape([A.shape[0],1])

    print('Original signal:')
    print(x_bar)

    print('L1 solution:')
    x_hat_0 = obj_func_0.solve(b=b)
    print(x_hat_0)

    print('Group Lasso solution:')
    x_hat = obj_func_1.solve(b = b)
    print(x_hat)
