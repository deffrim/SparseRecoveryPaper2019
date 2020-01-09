from sparse_recovery.structured_sparsity.sparsity_pattern.sparsity_pattern import sparsity_pattern
from scipy.sparse import random as sp_random
from scipy import stats
import numpy as np

from utils import combs


class s_sparse(sparsity_pattern):
    def __init__(self, sig_dim=2, s=1):
        super(s_sparse,self).__init__(sig_dim=sig_dim)
        self.s = s


    def facets(self):
        yield from combs(self.sig_dim, self.s)


    def random_sample(self):
        vec = sp_random(1, self.sig_dim, density=self.s/self.sig_dim, data_rvs=stats.norm(loc=0.0, scale=1.0).rvs).A
        norm = np.linalg.norm(vec)
        support = list(np.nonzero(vec[0]))
        return [vec[0]/norm, support]

    def groups(self):
        yield from ([i] for i in range(self.sig_dim))



if __name__ == '__main__':
    sp = s_sparse(sig_dim=7, s=3)
    a = sp.random_sample()
    for i in sp.facets():
        print(i)