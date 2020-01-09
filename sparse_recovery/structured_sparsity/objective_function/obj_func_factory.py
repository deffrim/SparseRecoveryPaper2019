import importlib
import numpy as np

class obj_func_factory(object):
    def __init__(self, obj_str, A, groups=None):
        self.A = np.asarray(A)
        # self.sig_dim = self.A.shape[1]
        self.obj_ext = obj_str
        module_name = 'sparse_recovery.structured_sparsity.objective_function.' + self.obj_ext
        class_name = self.obj_ext
        module = importlib.import_module(module_name)
        self.class_ = getattr(module, class_name)

        self.groups = [list(g) for g in groups]


    def run(self):
        method_name = 'obj_func_' + self.obj_ext
        method = getattr(self, method_name, lambda: 'obj_func_factory::run : {} is NOT a valid objective function name.'.format(self.obj_ext))
        return method()


    def obj_func_l1(self):
        return self.class_(A=self.A)

    def obj_func_group_lasso(self):
        return self.class_(A=self.A, groups=self.groups)

    def obj_func_prt_lasso(self):
        return self.class_(A=self.A, groups=self.groups)

    def obj_func_latent_gp_lasso(self):
        return self.class_(A=self.A, groups=self.groups)