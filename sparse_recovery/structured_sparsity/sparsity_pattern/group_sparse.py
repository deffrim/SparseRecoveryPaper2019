from sparse_recovery.structured_sparsity.sparsity_pattern.sparsity_pattern import sparsity_pattern
from random import sample, random
import numpy as np
from utils import combs, join_lists

class group_sparse(sparsity_pattern):
    def __init__(self, sig_dim=2, s=1):
        super(group_sparse,self).__init__(sig_dim=sig_dim)
        self.s = s
        self.nr_groups = None

    def groups(self):
        yield from (self.gr(j) for j in range(self.nr_groups))

    def gr(self,j):
        raise Exception('group_sparse::gr : Subclasses must implement this function which returns the jth group.')

    def facets(self):
        yield from (self.join_groups(i) for i in combs(self.nr_groups, self.s))

    def join_groups(self, gr_indices):
        l = [self.gr(i) for i in gr_indices]
        b = join_lists(l)
        return b

    def random_sample(self):
        gr_indices = sample(range(self.nr_groups), self.s)
        support = self.join_groups(gr_indices)
        vec = np.zeros(shape=[1,self.sig_dim])
        for i in support:
            vec[0][i] = random()
        norm = np.linalg.norm(vec)
        return [vec[0]/norm, support]