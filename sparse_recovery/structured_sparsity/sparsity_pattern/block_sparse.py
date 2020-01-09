from sparse_recovery.structured_sparsity.sparsity_pattern.group_sparse import group_sparse

class block_sparse(group_sparse):
    def __init__(self, sig_dim=2, block_size=1, s=1):
        if sig_dim % block_size != 0:
            raise Exception('block_sparse : Signal dimension has to be divisible by block size. Quitting...')
        super(block_sparse,self).__init__(sig_dim=sig_dim, s=s)
        self.block_size = block_size
        self.nr_groups = sig_dim//block_size
        self.nr_blocks = self.nr_groups

    def blocks(self):
        yield from self.groups()

    def gr(self,j):
        if j>= self.nr_groups:
            return None
        return [i for i in range(j*self.block_size, (j+1)*self.block_size)]


if __name__ == '__main__':
    sp = block_sparse(sig_dim=16, block_size=2, s=3)
    a = sp.random_sample()
    f = sp.facets()
    for i in f:
        print(i)