from partialDFT.pDFT import pDFT
from partialDFT.pDFTutils import complex_to_real
import numpy as np
import scipy
from partialDFT.chordal_prod import chordal_prod
import matplotlib.pyplot as plt
from sparse_recovery.structured_sparsity.objective_function.l1 import l1
# from sparse_recovery.structured_sparsity.sparsity_pattern.s_sparse import s_sparse
import utils
import itertools
import os

def run_experiment(p, m_bar, nr_sampled_supp_sets, sparsity_interval = [1,1], rec_sensitivity = 1e-6):
    meas_indices = list(range(m_bar + 1)) + list(range(p - m_bar, p))
    sparsity_range = list(range(sparsity_interval[0], sparsity_interval[1]+1))
    pdft = pDFT(N=p, meas_ind=meas_indices)
    pdft_matr = np.asarray(pdft.get_pdft_matrix())

    results_dict = {}
    results_dict['l1'] = {s: [] for s in sparsity_range}
    results_dict['pdft_matr'] = pdft_matr
    results_dict['params'] = {'p':p, 'm_bar':m_bar, 'nr_sampled_supp_sets':nr_sampled_supp_sets, 'sparsity_interval':sparsity_interval}



    for s in sparsity_range:
        for _ in range(nr_sampled_supp_sets):
            S = list(np.random.choice(range(p), s, replace=False))

            ## Solve optimization problem
            # a0 = datetime.datetime.now()
            x_bar = np.zeros(shape=[p,1])
            for k in S:
                x_bar[k] = np.random.normal(0.0,1.0)
            x_bar /= np.linalg.norm(x_bar)
            meas_matr = complex_to_real(pdft_matr)[:,:p]
            obj_func = l1(meas_matr, solver_opts={'maxiter':1000, 'rr':False})
            b = np.matmul(meas_matr, x_bar)
            x_hat_dict = obj_func.solve(b=b)
            x_hat = x_hat_dict['x']
            if np.linalg.norm(x_hat-x_bar) <= rec_sensitivity:
                results_dict['l1'][s].append(1.0)
            else:
                results_dict['l1'][s].append(0.0)
            # a1 = datetime.datetime.now()
            # print('Solving l1 minimization takes {} microseconds'.format((a1-a0).microseconds))
            # return


    utils.save_nparray_with_date(data=results_dict, file_prefix='TrIT_paper_pDFT_Exp_1_p{}_mbar{}_nrsamples{}'.format(p, m_bar, nr_sampled_supp_sets), subfolder_name='output')

def plot_exp_results(results_file_path, max_asc_file_path = '', s_in_max_asc_file_path = ''):
    results_dict = np.load(results_file_path, allow_pickle=True)
    results_dict = results_dict.item()
    res = results_dict['l1']

    mean_exact_rec_rates = []
    sparsity_range = []

    for s in res:
        sparsity_range.append(s)
        mean_exact_rec_rates.append(np.mean(res[s]))

    ## Plot
    plot_symbs_list = itertools.cycle(['o-', '*-','s-','d-','x-', '+-'])
    fig, ax = plt.subplots()

    ax.plot(sparsity_range, mean_exact_rec_rates, next(plot_symbs_list), label='Exact rec. prob. via '+r'$\ell_1$')

    plt.xlabel('Sparsity (s)')

    if max_asc_file_path != '':
        max_asc = np.load(max_asc_file_path, allow_pickle=True)
        max_asc = max_asc.item()
        p = results_dict['params']['p']
        lbounds = []
        for s in sparsity_range:
            nr_combs = scipy.special.comb(p, s)
            if max_asc[s]['type']:
                prob = len(max_asc[s]['idx_sets'])/nr_combs
            else:
                prob = 1.0 - len(max_asc[s]['idx_sets'])/nr_combs
            lbounds.append(prob)


        ax.plot(sparsity_range, lbounds, next(plot_symbs_list), label='Ratio of index sets in maxASC')

        legend = ax.legend()

    if s_in_max_asc_file_path != '':
        p = results_dict['params']['p']
        mbar = results_dict['params']['m_bar']
        y = np.load(s_in_max_asc_file_path, allow_pickle=True)
        y = y.item()
        vals = [y[ky]['success_rate'] for ky in y if 'success_rate' in y[ky].keys()]
        if vals != []:
            ax.plot(range(1, 1+len(vals)), vals, next(plot_symbs_list), label='Proportion of sets in MASC')
        legend = ax.legend()
        plt.xlabel('Sparsity/Cardinality (s)')

    plt.ylabel('Probability')
    plt.xticks(range(min(sparsity_range),max(sparsity_range)+1,1))
    # plt.title('Nr Supp Sets = {}, Max Nr Ext pts = {}'.format(nr_sampled_supp_sets, nr_sampled_ext_pts))
    plt.grid()
    plt.show()

def yield_extreme_pts(p, m_bar):
    m = 2*m_bar+1
    ext_pts = []
    for zero_loc in utils.combs(n=p, k=p - m - 1):
        cp = chordal_prod(N=p)
        cp.set_polyn(zero_indices=zero_loc)
        ext_pts.append(cp.get_polyn_vals())

    yield from ext_pts

def cal_max_asc(p, m_bar, memory_efficient = False):
    sparsity_range = range(1,p+1)
    max_asc = {s: {} for s in sparsity_range}

    if memory_efficient:
        ext_pts = yield_extreme_pts(p=p, m_bar=m_bar)
    else:
        ext_pts = list(yield_extreme_pts(p=p, m_bar=m_bar))


    for s in sparsity_range:
        fail = []
        success = []
        for supp_set in utils.combs(n=p, k=s):
            fail_flag = False
            for extp in ext_pts:
                supp_set_sum = sum([np.linalg.norm(extp[k]) for k in supp_set])
                supp_set_comp_sum = sum([np.linalg.norm(extp[k]) for k in range(p) if k not in supp_set])
                if supp_set_sum >= supp_set_comp_sum:
                    fail_flag = True
                    break
            if fail_flag:
                fail.append(supp_set)
            else:
                success.append(supp_set)

        if len(success) <= len(fail) :
            max_asc[s]['type'] = True
            max_asc[s]['idx_sets'] = success
        else:
            max_asc[s]['type'] = False
            max_asc[s]['idx_sets'] = fail

    return max_asc

def cal_s_in_max_asc_prob(p, m_bar):
    sparsity_range = range(1, p) ## ATTENTION HERE!!
    all_probs = {s: {} for s in sparsity_range}

    if scipy.special.comb(p, 2*m_bar+2) >= 5000:
        raise Exception('cal_s_in_max_asc_prob : Probably there are too many extreme points. Comment out this warning to continue.' )

    ext_pts = list(yield_extreme_pts(p=p, m_bar=m_bar))

    succ_rate_zero = False

    for s in sparsity_range:
        print('Working with sparsity {}...'.format(s))
        fail = 0.0
        success = 0.0
        for supp_set in utils.combs(n=p, k=s):
            fail_flag = False
            for extp in ext_pts:
                supp_set_sum = sum([np.linalg.norm(extp[k]) for k in supp_set])
                supp_set_comp_sum = sum([np.linalg.norm(extp[k]) for k in range(p) if k not in supp_set])
                if supp_set_sum >= supp_set_comp_sum:
                    fail_flag = True
                    break
            if fail_flag:
                fail += 1.0
            else:
                success += 1.0

        success_rate = success/(success+fail)
        if not succ_rate_zero:
            all_probs[s]['success_rate'] = success_rate
        else:
            break
        succ_rate_zero = (success_rate == 0.0)


    return all_probs

def plot_s_in_maxASC(subfolder = 'output', file_prefix = 'TrIT_paper_pDFT_Exp_1_s_in_maxASC_'):
    path = subfolder
    files = []
    # r=root, d=directories, f = files
    plot_symbs_list = itertools.cycle(['o-', '*-','s-','d-','x-', '+-'])
    fig, ax = plt.subplots()

    for r, d, f in os.walk(path):
        for file in f:
            if file[:len(file_prefix)] == file_prefix and file[-4:] == '.npy':
                l = len(file_prefix+'p')
                s0 = file.find(file_prefix + 'p') + l
                s1 = file.find('_', s0)
                p = int(file[s0:s1])

                str = file_prefix + 'p{}_mbar'.format(p)
                l = len(str)
                s0 = file.find(str) + l
                s1 = file.find('--', s0)
                mbar = int(file[s0:s1])

                y = np.load(path+'/'+file)
                y = y.item()

                vals = [y[ky]['success_rate'] for ky in y if 'success_rate' in y[ky].keys()]
                if vals != []:
                    ax.plot(range(1, 1+len(vals)), vals, next(plot_symbs_list), label='p={} and mbar = {}'.format(p,mbar))

    plt.grid()
    plt.xticks(range(1, 11))
    plt.xlabel('Cardinality (s)')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    p = 101
    m_bar = 48

    nr_sampled_supp_sets = 2000
    sparsity_interval = [1, int(3*p/4)]
    rec_sensitivity = 1e-6


    # run_experiment(p=p, m_bar=m_bar, nr_sampled_supp_sets=nr_sampled_supp_sets, sparsity_interval=sparsity_interval, rec_sensitivity=rec_sensitivity)


    # max_asc = cal_max_asc(p=p, m_bar=m_bar)
    # utils.save_nparray_with_date(data=max_asc, file_prefix='TrIT_paper_pDFT_Exp_1_maxASC_p{}_mbar{}'.format(p,m_bar), subfolder_name='output')


    # all_probs = cal_s_in_max_asc_prob(p=p, m_bar=m_bar)
    # utils.save_nparray_with_date(data=all_probs, file_prefix='TrIT_paper_pDFT_Exp_1_s_in_maxASC_p{}_mbar{}'.format(p,m_bar), subfolder_name='output')


    results_file_path = 'output/TrIT_paper_pDFT_Exp_1_p19_mbar7_nrsamples1000--2019-4-29-11-22.npy'
    s_in_max_asc_file_path = 'output/TrIT_paper_pDFT_Exp_1_s_in_maxASC_p19_mbar7--2019-4-22-22-2.npy'
    plot_exp_results(results_file_path=results_file_path, s_in_max_asc_file_path=s_in_max_asc_file_path)
