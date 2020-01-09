import cmath
import numpy as np
import matplotlib.pyplot as plt
import partialDFT.pDFTutils

class chordal_prod(object):
    def __init__(self, N):
        self.N = N
        self.angle = -2*cmath.pi/self.N
        self.xi = complex(cmath.cos(self.angle), cmath.sin(self.angle))
        self.polyn = None
        self.zeros = None
        self.zero_indices = None

    def set_polyn(self, zero_indices):
        self.zeros = None
        self.zero_indices = None
        self.zero_indices = zero_indices
        self.zeros = [np.power(self.xi, k) for k in self.zero_indices]

        self.polyn = None
        self.polyn = lambda z : np.prod([z - z0 for z0 in self.zeros])


    def eval_polyn(self, z):
        if self.polyn is None:
            raise Exception("chordal_prod :: eval_polyn : You have to set the polynomial first.")

        return self.polyn(z)

    def get_polyn_vals(self):
        if self.polyn is None or self.zeros is None:
            raise Exception("chordal_prod :: get_polyn_vals : You have to set the polynomial first.")
        vals = []
        for k in range(self.N):
            if k in self.zero_indices:
                vals.append(complex(0.0,0.0))
            else:
                pt = np.power(self.xi, k)
                vals.append(self.eval_polyn(pt))
        return vals

    def get_polyn_val_norms_q(self, zero_indices):
        vals = []
        for k in range(self.N):
            if k in zero_indices:
                vals.append(0.0)
            else:
                pt = np.power(self.xi, k)
                val = 1.0
                for t in zero_indices:
                    z_pt = np.power(self.xi, t)
                    val *= np.linalg.norm(pt - z_pt)
                vals.append(val)
        return vals


if __name__ == "__main__":
    N = 1009

    nr_zeros = 23
    # nr_zeros = 3

    # zero_indices = [i for i in range(53) if i not in [0, 1, 7, 13, 19, 25, 28, 34, 40, 46]]
    # zero_indices = range(0,N,2)
    zero_indices = range(nr_zeros)
    # zero_indices = [0, 3, 5, 10, 12, 15] + list(range(16, N))
    # zero_indices = [0, 2, 5, 14, 17, 19] + list(range(20, N))
    zero_indices = list(np.random.choice(range(N), nr_zeros, replace=False))

    ch_p = chordal_prod(N=N)
    ch_p.set_polyn(zero_indices=zero_indices)
    vals = ch_p.get_polyn_vals()

    norms = [np.linalg.norm(z) for z in vals]
    # max_norm = max(norms)
    one_norm = sum(norms)
    # norms = [k/max_norm for k in norms]
    norms = [k/one_norm for k in norms]
    sorted_norms = norms.copy()
    sorted_norms.sort()
    sorted_norms.reverse()
    the_sum = sorted_norms[0]
    s = 0
    while the_sum < 0.5:
        s += 1
        the_sum += sorted_norms[s]


    # max_vals = []
    # for k in range(31,102):
    #     ch_p = chordal_prod(k)
    #     ch_p.set_polyn(zero_indices=zero_indices)
    #     vals = ch_p.get_polyn_vals()
    #     norms = [np.linalg.norm(z) for z in vals]
    #     # max_norm = max(norms)
    #     sum_norms = sum(norms)
    #     # norms = [k/max_norm for k in norms]
    #     norms = [k / sum_norms for k in norms]
    #     max_vals.append(max(norms))

    # max_vals = []
    # for k in range(31,102):
    #     ch_p = chordal_prod(k)
    #     ch_p.set_polyn(zero_indices=zero_indices)
    #     vals = ch_p.get_polyn_vals()
    #     norms = [np.linalg.norm(z) for z in vals]
    #     # max_norm = max(norms)
    #     sum_norms = sum(norms)
    #     # norms = [k/max_norm for k in norms]
    #     norms = [k / sum_norms for k in norms]
    #     max_vals.append(max(norms))

    plt.figure()
    plt.plot(norms, 'x-')
    plt.grid()
    plt.title('1-norm: {:.2f} -- Max sparsity: {} -- Est.Bound: {:.2f} -- Integral {}'.format(one_norm, s, N/(2*(nr_zeros+1)), np.log(2*np.pi*one_norm/N)))
    plt.show()
