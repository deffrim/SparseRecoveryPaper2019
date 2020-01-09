import importlib

class sp_factory(object):
    def __init__(self, obj_str, sig_dim, config=None, s=1):
        self.s = s
        self.sig_dim = sig_dim
        self.obj_ext = obj_str
        module_name = 'sparse_recovery.structured_sparsity.sparsity_pattern.' + self.obj_ext
        class_name = self.obj_ext
        module = importlib.import_module(module_name)
        self.class_ = getattr(module, class_name)

        self.config = config


    def run(self):
        method_name = 'obj_func_' + self.obj_ext
        method = getattr(self, method_name, lambda: 'sp_factory::run : {} is NOT a valid sparsity pattern name.'.format(self.obj_ext))
        return method()

    def obj_func_s_sparse(self):
        return self.class_(sig_dim=self.sig_dim, s=self.s)

    def obj_func_block_sparse(self):
        block_size = self.config['block_sparsity'].getint('block_size')
        return self.class_(sig_dim=self.sig_dim, block_size=block_size, s=self.s)

    def obj_func_chain_sparse(self):
        group_size = self.config['group_sparsity'].getint('gr_size')
        nr_groups = self.config['group_sparsity'].getint('nr_groups')
        return self.class_(sig_dim=self.sig_dim, group_size=group_size, nr_groups=nr_groups, s=self.s)