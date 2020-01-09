import sympy
import numpy as np
from numpy.linalg import svd
from scipy.special import comb
import math
import matplotlib.pyplot as plt

def combs(n,k):
    if k > n or k<0:
        raise Exception('You cannot choose {} from {}'.format(k,n))

    if k == 0:
        return [[]]

    if k == 1:
        return [[i] for i in range(n)]
    else:
        allc = []
        for f in range(n-k+1):
            allc = allc + [[f] + [1+f+i for i in j] for j in combs(n - f -1, k -1)]

    return allc


def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def gcd_list(l):
    if len(l) < 2:
        raise Exception("List must have at least 2 elements.")

    if len(l) == 2:
        return math.gcd(l[0], l[1])
    else:
        return math.gcd(l[0], gcd_list(l[1:]))

def prime_nr_list(n, start_from=2):
    m = start_from
    if n<2:
        raise Exception('utils :: prime_nr_generator : The smallest prime is 2')

    if n < m:
        return []

    if n == 2:
        return [2]

    if n % 2 == 0:
        return prime_nr_list(n-1,m)

    sqt = int( 1 + n**0.5)

    for k in range(3,sqt,2):
        if n % k == 0:
            return prime_nr_list(n-1,m)

    return prime_nr_list(n - 1,m) + [n]

def complex_to_real(A):
    r_A = np.real(A)
    i_A = np.imag(A)
    row1 = np.concatenate([r_A, (-1)*i_A], axis=1)
    row2 = np.concatenate([i_A, r_A], axis=1)
    B= np.concatenate([row1, row2], axis=0)
    return B


def check_NUP_pDFT_minors(minor_dets_norms , nr_meas, N, s=2):
    if s < 2:
        raise Exception('utils::check_NUP_pDFT_minors : Sparsity has to be strictly larger than 1.')
    if s >= nr_meas:
        raise Exception('utils::check_NUP_pDFT_minors : Sparsity has to be strictly smaller than the number of measurements.')
    if nr_meas < 1:
        raise Exception('utils::check_NUP_pDFT_minors : Number of measurements cannot be less than 1.')
    if nr_meas >= N-1:
        raise Exception('utils::check_NUP_pDFT_minors : Number of measurements is {}, whereas signal length is {}'.format(nr_meas, N))

    all_combs = combs(N, nr_meas+1)
    sub_combs = combs(nr_meas + 1, nr_meas)

    nup_constants_dict = {}
    inv_nup_constants_dict = {}
    global_nup_const = 0.0

    for comb in all_combs:
        vals = []
        for sc in sub_combs:
            c = tuple([comb[i] for i in sc])
            vals.append(minor_dets_norms[c][1])

        vals = sorted(vals)
        # top_sum = sum(vals[nr_meas+1-s:])
        # bottom_sum = sum(vals[:nr_meas+1-s])
        the_sum = sum(vals)
        nup_cons = sum(vals[nr_meas+1-s:])/the_sum
        nup_constants_dict[tuple(comb)] = [nup_cons, the_sum]
        if nup_cons in inv_nup_constants_dict:
            inv_nup_constants_dict[nup_cons].append(comb)
        else:
            inv_nup_constants_dict[nup_cons] = [comb]
        if nup_cons > global_nup_const:
            global_nup_const = nup_cons

    return [global_nup_const, nup_constants_dict, inv_nup_constants_dict]

def plot_worst_configs(p, worst = {}):
    circle = plt.Circle((0, 0), 1.0, color='silver')
    worst_config_angles = [2*np.pi*k/p for k in worst['worst_config']]
    worst_loc_angles = [2*np.pi*k/p for k in worst['worst_locs']]

    fig, ax = plt.subplots()

    ax.add_artist(circle)
    ax.plot([np.cos(2*np.pi*k/p) for k in range(p)], [np.sin(2*np.pi*k/p) for k in  range(p)],'.',  color='k')
    ax.plot([np.cos(k) for k in worst_config_angles], [np.sin(k) for k in worst_config_angles],'*', color='b', label='worst_config_of_zeros')
    ax.plot([np.cos(k) for k in worst_loc_angles], [np.sin(k) for k in worst_loc_angles], '*', color='r', label='worst_locs')

    plt.legend()
    plt.title('p={} -- |\u03A9|={} -- Max Sparsity={}'.format(p, p-len(worst_config_angles)-1, len(worst['worst_locs'])))
    ax.set_aspect('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # com = combs(11,7)
    # vals = [sum([comb(c[i]+1, i+1) for i in range(len(c))]) for c in com]
    # vals = sorted(vals)

    vals = prime_nr_list(50, start_from=13)

    if len(vals) != 0:
        plt.figure()
        plt.xticks(range(1+len(vals)))
        plt.yticks(range(2+max(vals)))
        plt.plot(range(1,1+len(vals)), vals, 'b+')
        plt.grid()
        plt.show()
    else:
        print('No prime found!')