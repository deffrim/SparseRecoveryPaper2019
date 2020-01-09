import utils
from partialDFT.pDFT import pDFT
from partialDFT.pDFTutils import complex_to_real
import numpy as np
from partialDFT.chordal_prod import chordal_prod
import matplotlib
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
from sparse_recovery.structured_sparsity.objective_function.l1 import l1
# from sparse_recovery.structured_sparsity.sparsity_pattern.s_sparse import s_sparse
import utils
import itertools
from sparse_recovery.structured_sparsity.sparsity_pattern.s_sparse import s_sparse
import time


def run_exp(p, nr_exps = 2000, recovery_sensitivity = 1e-6, calc_l1_exp_bound=False, calc_ext_pts_bound=False, save_results=True):
    if p//4 < 2 :
        raise Exception('TrIT_paper_pDFT_Exp_2 :: run_exp : p = {} is too small.'.format(p))

    # card_omega_range = set([p-(p//(2*k)) for k in range(2,1+p//4)])
    down_step = 2*max(p//100, 1)
    card_omega_range = list(range(p - 2, p//4 - down_step, (-1)*down_step ))
    results_dict = {c:{} for c in card_omega_range}
    prev_l1_max_sparsity = p

    for card_omega in card_omega_range:
        m_bar = card_omega//2
        meas_ind = list(range(m_bar+1)) + list(range(p-m_bar, p))
        pdft = pDFT(N=p, meas_ind=meas_ind)
        pdft_matr = np.asarray(pdft.get_pdft_matrix())
        meas_matr = complex_to_real(pdft_matr)[:, :p]

        ## Our bound ##
        results_dict[card_omega]['coh_bound'] = p/(2*(p-card_omega))

        # ## Mutual coherence bound ##
        # mc_val = utils.cal_mutual_coherence(meas_matr)
        # results_dict[card_omega]['mc_bound'] = (1.0+1.0/mc_val)/2.0
        # # results_dict[card_omega]['mc_our_bound'] = (1.0+np.pi/(p*np.sin(np.pi*card_omega/p)))/2.0


        # Using chordal prod
        if calc_ext_pts_bound:
            N = p
            nr_zeros = N - card_omega - 1
            all_max_sparsity = []
            for _ in range(nr_exps):
                zero_indices = list(np.random.choice(range(N), nr_zeros, replace=False))

                ch_p = chordal_prod(N=N)
                ch_p.set_polyn(zero_indices=zero_indices)
                vals = ch_p.get_polyn_vals()

                norms = [np.linalg.norm(z) for z in vals]
                # max_norm = max(norms)
                one_norm = sum(norms)
                # norms = [k/max_norm for k in norms]
                norms = [k / one_norm for k in norms]
                sorted_norms = norms.copy()
                sorted_norms.sort()
                sorted_norms.reverse()
                the_sum = sorted_norms[0]
                s = 0
                while the_sum < 0.5:
                    s += 1
                    the_sum += sorted_norms[s]
                all_max_sparsity.append(s)

            results_dict[card_omega]['ext_pts_bound'] = min(all_max_sparsity)

        ## Experimental bound ##
        if calc_l1_exp_bound:
            sparsity_range = range(min(p//2,prev_l1_max_sparsity+5), 1, -1)
            # exp_bound = max(sparsity_range)
            results_dict[card_omega]['l1_exp_bound'] = 1
            prev_l1_max_sparsity = 1

            for s in sparsity_range:
                sp = s_sparse(sig_dim=p, s=s)
                fail_flag = False
                for exp_count in range(nr_exps):
                    xbar, _ = sp.random_sample()
                    # xbar = np.abs(xbar) # Attention here!! This is needed.
                    b = np.matmul(meas_matr, xbar)
                    obj_func = l1(A=meas_matr)
                    try:
                        soln_dict = obj_func.solve(b=b)
                    except Exception:
                        continue
                    xhat = soln_dict['x']
                    xbar = np.reshape(xbar, newshape=xhat.shape)
                    diff_norm = np.linalg.norm(xbar-xhat)
                    if diff_norm > recovery_sensitivity:
                        fail_flag = True
                        break

                if not fail_flag:
                    results_dict[card_omega]['l1_exp_bound'] = s
                    prev_l1_max_sparsity = s
                    print('l1 experiment bound for |\u03A9| = {} is {}.'.format(card_omega, s))
                    break

    if save_results:
        utils.save_nparray_with_date(data=results_dict,
                                     file_prefix='TrIT_paper_pDFT_Exp_2_p{}_NrExp{}'.format(p, nr_exps),
                                     subfolder_name='output')

    plot_results(res_file=results_dict)

def plot_results(res_file_path = '', res_file = None):
    if res_file is None:
        y = np.load(res_file_path, allow_pickle=True)
        results = y.item()
    else:
        results = res_file

    ## Plot array ##
    x_vals = list(results.keys())
    y_series = list(results[x_vals[0]].keys())
    # all_y_vals = {k:[results[x][k] for x in x_vals] for k in y_series}
    all_y_vals = {k:[] for k in y_series}

    for x in x_vals:
        y_series = list(results[x].keys())
        for y in y_series:
            all_y_vals[y].append(results[x][y])

    fig, ax = plt.subplots()
    plot_symbs_list = itertools.cycle(['o-', '*-','s-','d-','x-', '+-'])
    # plt.rc('text', usetex=True)

    for y_key in all_y_vals:
        y_vals = all_y_vals[y_key]
        if y_key == 'mc_bound':
            # y_key = 'coh_bound'
            y_key = r'Lower bound $\frac{n}{2(n-|\Omega_j|)}$'
        if y_key == 'ext_pts_bound':
            y_key = r'Upper bound $\widehat{s_{\max}}(\Omega)$'
        if y_key == 'l1_exp_bound':
            y_key = r'Upper bound $\widetilde{s_{\max}}(\Omega_j)$'
        ax.plot(x_vals[:len(y_vals)], y_vals, next(plot_symbs_list), label=y_key)

    legend = ax.legend()
    ax.set_yscale('log')
    plt.grid()
    plt.xlabel(r'$|\Omega|$')
    plt.ylabel('Max Sparsity')
    # plt.title('p={}'.format(p))
    plt.show()


if __name__ == '__main__':
    p = 1009
    # p = 503
    nr_exps = 2

    start_time = time.time()
    # run_exp(p=p, nr_exps=nr_exps)
    # run_exp(p=p, nr_exps=nr_exps, calc_l1_exp_bound=True, calc_ext_pts_bound=False, save_results=False)
    # run_exp(p=p, nr_exps=nr_exps, save_results=False)
    end_time = time.time()
    print('Elapsed time is {}'.format(time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))))

    # res_file_path = 'output/TrIT_paper_pDFT_Exp_2_p1009_NrExp1000--2019-6-20-3-34.npy'
    res_file_path ='output/TrIT_paper_pDFT_Exp_2_p61_NrExp1000--2019-6-21-13-45.npy'

    plot_results(res_file_path=res_file_path)
