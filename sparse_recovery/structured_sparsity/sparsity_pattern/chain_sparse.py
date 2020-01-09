from sparse_recovery.structured_sparsity.sparsity_pattern.group_sparse import group_sparse

class chain_sparse(group_sparse):
    def __init__(self, sig_dim=2, group_size=1, nr_groups=2, s=1):
        super(chain_sparse,self).__init__(sig_dim=sig_dim, s=s)
        if nr_groups*group_size < sig_dim:
            raise Exception('Group size times the number of groups has to be larger than signal dimension.')
        if sig_dim % nr_groups != 0:
            raise Exception('Even though overlap is allowed, we require that signal dimension is divisible by number of groups for simplicity.')
        self.g_size = group_size
        self.nr_groups = nr_groups
        self.step = self.sig_dim//self.nr_groups


    def partition(self):
        pass


    def gr(self,j):
        if j >= self.nr_groups:
            return None
        return [i%self.sig_dim for i in range(j*self.step, j*self.step+self.g_size)]


if __name__ == '__main__':
    sp = chain_sparse(sig_dim=16, group_size=3, nr_groups=8, s=2)
    a = sp.random_sample()
    f = sp.facets()
    for i in f:
        print(i)