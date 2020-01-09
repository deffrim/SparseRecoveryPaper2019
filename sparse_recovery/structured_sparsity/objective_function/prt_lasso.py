from sparse_recovery.structured_sparsity.objective_function.group_lasso import group_lasso
from sparse_recovery.structured_sparsity.objective_function.obj_function import obj_function
from numpy import asarray
from utils import partition_from_grps


class prt_lasso(obj_function): ## Derive from group_lasso. Makes more sense!!
    def __init__(self, A, groups):
        super(prt_lasso, self).__init__()
        self.A = asarray(A)
        self.sig_dim = A.shape[1]
        self.prtion_dict = partition_from_grps(sig_dim=A.shape[1], groups=groups)
        self.groups = [list(g) for g in self.prtion_dict.keys()]
        self.aux_obj_func = group_lasso(A=self.A, groups=self.groups)

        self.class_name = 'prt_lasso'


    def solve(self,b):
        self.solution = self.aux_obj_func.solve(b)
        return self.solution
