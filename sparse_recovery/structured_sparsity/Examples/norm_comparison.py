from sparse_recovery.structured_sparsity.sparsity_pattern.sp_factory import sp_factory
from sparse_recovery.structured_sparsity.objective_function.obj_func_factory import obj_func_factory
import numpy as np
from scipy.sparse import random as sp_random
import time
import configparser
import utils
import itertools
import matplotlib.pyplot as plt



def run_exp(config_file_path = ''):
    ###################
    ## Configuration ##
    ###################
    config = configparser.ConfigParser()
    config.read(config_file_path)

    mat_is_rand = config['meas_matrix'].getboolean('random')
    A = None
    matr_density = None

    if mat_is_rand:
        m = config['meas_matrix'].getint('nr_rows')
        n = config['meas_matrix'].getint('nr_cols')
        nr_matrices = config['meas_matrix'].getint('nr_matrices')
        matr_density = config['meas_matrix'].getfloat('density') # Should crash here!
    else:
        A = config['meas_matrix']['A']
        A = utils.config_str_to_np_array(A)
        m = A.shape[0]
        n = A.shape[1]
        nr_matrices = 1

    max_s = config['sparsity'].getint('max_sparsity')
    nr_trials = config['experiment'].getint('nr_trials')
    obj_funcs = [x.strip() for x in config['optimization']['obj_funcs'].split(",")]
    sparsity_type = config['sparsity']['type']


    ####################
    ## The Experiment ##
    ####################
    test_sp_range = range(1, max_s+1)
    # obj_func_res = {obj_f:[[] for i in test_sp_range] for obj_f in obj_funcs}
    res_dict = {}

    for mat_nr in range(nr_matrices):
        if mat_is_rand:
            A = sp_random(m=m, n=n, density=matr_density).A
            A = np.asarray(A)
        matr_str= 'meas_matr_{}'.format(mat_nr)
        res_dict[matr_str] = [A, {}]
        for sparsity in test_sp_range:
            print('Matrix nr: {:>2}. {}: {:>2}.'.format(mat_nr,sparsity_type,sparsity))
            sp_str = sparsity_type+'={}'.format(sparsity)
            res_dict[matr_str][1][sp_str] = {}
            sp = sp_factory(obj_str=sparsity_type, sig_dim=n, config=config, s = sparsity).run()
            obj_func_classes = {obj_f:obj_func_factory(obj_str=obj_f, A=A, groups=sp.groups()).run() for obj_f in obj_funcs}
            # sp_res = {obj_f:[] for obj_f in obj_funcs}
            for trial in range(nr_trials):
                print('.', sep=' ', end='', flush=True)
                [x_bar, _] = sp.random_sample()
                b = np.matmul(A, x_bar)
                x_hat_dict = {obj_f:obj_func_classes[obj_f].solve(b=b) for obj_f in obj_funcs}
                res_dict[matr_str][1][sp_str][tuple(x_bar)] = x_hat_dict
            # for obj_f in obj_funcs:
            #     obj_func_res[obj_f][sparsity-1] += sp_res[obj_f]
            print('\n')
    utils.save_nparray_with_date(data=res_dict, file_prefix='Norm_comp_exp',subfolder_name='output')
    # np.save('output/res_dict_{}'.format(matr_str), res_dict)




def plot_results(res_file_path=''):
    y = np.load(res_file_path, allow_pickle=True)
    results = y.item()

    obj_funcs = set()
    obj_func_res = dict()
    max_s = 1
    sparsity_type = ''
    for matr_key in results:
        for sparsity_key in results[matr_key][1]:
            sparsity_type, sparsity = sparsity_key.split('=')
            sparsity = int(sparsity)
            if sparsity > max_s:
                max_s = sparsity
            for xb in results[matr_key][1][sparsity_key]:
                x_bar = np.asarray(xb)
                for obj_func in results[matr_key][1][sparsity_key][xb]:
                    obj_funcs.add(obj_func)
                    x_hat = results[matr_key][1][sparsity_key][xb][obj_func]['x']
                    x_hat = np.reshape(x_hat, newshape=x_bar.shape)
                    diff_norm = np.linalg.norm(x_bar - x_hat)
                    if obj_func not in obj_func_res:
                        obj_func_res[obj_func] = {}
                    if sparsity in obj_func_res[obj_func]:
                        obj_func_res[obj_func][sparsity].append(diff_norm)
                    else:
                        obj_func_res[obj_func][sparsity] = [diff_norm]


    minmax = {obj_f: np.asarray([[min(obj_func_res[obj_f][s]) for s in obj_func_res[obj_f]], [max(obj_func_res[obj_f][s]) for s in obj_func_res[obj_f]]]) for obj_f in obj_funcs}
    medians = {obj_f: [np.median(obj_func_res[obj_f][s]) for s in obj_func_res[obj_f]] for obj_f in obj_funcs}

    x_ticks = 1+np.arange(0, max_s)
    marker = itertools.cycle(('*', 'o','d', 's', 'p', 'h', 'x'))

    for obj_f in obj_funcs:
        plt.errorbar(x_ticks, y=medians[obj_f], yerr=minmax[obj_f], linestyle='None', marker=next(marker), capsize=5)

    plt.grid()
    plt.legend(obj_funcs, loc='upper left')
    plt.title(sparsity_type)
    plt.xlabel('Nr of nonzero groups')
    plt.ylabel('Median/min/max recovery error')
    plt.xticks(range(1, max_s + 1))

    plt.show()


if __name__ == '__main__':
    config_file_path = 'norm_comparison_config.ini'

    ##
    start_time = time.time()
    ##
    # run_exp(config_file_path=config_file_path)
    plot_results(res_file_path='output/Norm_comp_exp--2020-1-9-13-48.npy')
    ##
    end_time = time.time()
    print('Execution time is {:.2f} seconds'.format(end_time - start_time))
    ##

