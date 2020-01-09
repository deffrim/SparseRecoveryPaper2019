from sparse_recovery.structured_sparsity.objective_function.obj_function import obj_function
import numpy as np
from scipy.optimize import linprog
from cvxopt import solvers, matrix

class l1(obj_function):
    def __init__(self, A, method='interior-point', solver_opts={'maxiter':1000}):
        super(l1,self).__init__()
        self.method = method
        self.solver_opts = solver_opts

        self.sig_dim = A.shape[1]
        self.A_eq = np.concatenate([A, -A], axis=1)

        self.c = np.concatenate([np.ones_like(A[0]), np.ones_like(A[0])])

        self.class_name = 'l1'


    def solve(self, b):
        res = linprog(self.c, A_eq=self.A_eq, b_eq=b, method=self.method, options=self.solver_opts)
        self.solution = {key: res[key] for key in res if key not in ['x', 'slack', 'con', 'jac', 'hess', 'hess_inv']}
        self.solution['x'] = np.reshape(res.x[:self.sig_dim] - res.x[self.sig_dim:], newshape=[self.sig_dim,1])
        return self.solution


if __name__ == '__main__':
    nr_cols = 101
    nr_rows = 31
    s=2
    matr = np.random.sample(size=[nr_rows,nr_cols])
    x_bar = np.zeros(shape=[nr_cols,1])
    S= np.random.choice(range(nr_cols), s, replace=False)
    for k in S:
        x_bar[k] = np.random.normal(0.0,1.0)

    x_bar /= np.linalg.norm(x_bar)
    b = np.matmul(matr, x_bar)

    obj_f = l1(matr)
    x_hat_dict = obj_f.solve(b=b)
    x_hat = np.reshape(x_hat_dict['x'], newshape=[nr_cols,1])
    print(np.linalg.norm(x_bar-x_hat))
    pass