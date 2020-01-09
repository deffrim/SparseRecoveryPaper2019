from graphs.erdos_renyi_graph import erdos_renyi_graph
import numpy as np
from sparse_recovery.sparse_recovery_experiments.l1_rec_exp import l1_rec_exp
import utils
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time


def run_exp(par_dict, output_dir='output', save_results=True):
    all_rec_probs = {p:{s:[] for s in par_dict['sparsity_range']} for p in par_dict['edge_probs_list']}
    edge_probs_list = par_dict['edge_probs_list']
    nr_vertices = par_dict['nr_vertices']
    nr_gr_samples = par_dict['nr_gr_samples']
    sparsity_range = par_dict['sparsity_range']
    nr_sampled_supports = par_dict['nr_sampled_supports']

    for edge_prob in edge_probs_list:
        rand_gr = erdos_renyi_graph(nr_vertices=nr_vertices, edge_probability=edge_prob)

        for _ in range(nr_gr_samples):
            gr = rand_gr.sample()
            inc_matr = gr.get_incidence_matr()
            exp = l1_rec_exp(matr=inc_matr, sparsity_range=sparsity_range, nr_sampled_supports=nr_sampled_supports)
            exp.run()
            rec_probs = exp.get_mean_rec_probs()
            for s in rec_probs:
                all_rec_probs[edge_prob][s].append(rec_probs[s])
    if save_results:
        utils.save_nparray_with_date(data=all_rec_probs,
                                     file_prefix='TrIT_paper_Inc_Matr_Exp_2_NrV{}_NrS{}'.format(nr_vertices, nr_sampled_supports),
                                     subfolder_name=output_dir)

def run_par_exp(par_dict, output_dir='output', save_results=True):
    edge_probs_list = par_dict['edge_probs_list']
    results = Parallel(n_jobs=-1)(delayed(the_exp)(edge_prob, par_dict) for edge_prob in edge_probs_list)
    all_rec_probs = {}

    for k in zip(range(len(edge_probs_list)), edge_probs_list):
        all_rec_probs[k[1]] = results[k[0]][k[1]]

    nr_vertices = par_dict['nr_vertices']
    nr_sampled_supports = par_dict['nr_sampled_supports']

    if save_results:
        utils.save_nparray_with_date(data=all_rec_probs,
                                     file_prefix='TrIT_paper_Inc_Matr_Exp_2_NrV{}_NrS{}'.format(nr_vertices, nr_sampled_supports),
                                     subfolder_name=output_dir)

def the_exp(edge_prob, par_dict):
    nr_vertices = par_dict['nr_vertices']
    nr_gr_samples = par_dict['nr_gr_samples']
    sparsity_range = par_dict['sparsity_range']
    nr_sampled_supports = par_dict['nr_sampled_supports']

    p_rec_probs = {edge_prob: {s: [] for s in sparsity_range}}

    rand_gr = erdos_renyi_graph(nr_vertices=nr_vertices, edge_probability=edge_prob)

    for _ in range(nr_gr_samples):
        gr = rand_gr.sample()
        inc_matr = gr.get_incidence_matr()
        exp = l1_rec_exp(matr=inc_matr, sparsity_range=sparsity_range, nr_sampled_supports=nr_sampled_supports)
        exp.run()
        rec_probs = exp.get_mean_rec_probs()
        for s in rec_probs:
            p_rec_probs[edge_prob][s].append(rec_probs[s])

    return p_rec_probs

def plot_results(results_file_path = ''):
    res_file = np.load(results_file_path, allow_pickle=True)
    res_file = res_file.item()
    edge_probs_list = list(res_file.keys())
    nr_edge_probs = len(edge_probs_list)
    sparsity_range = list(res_file[edge_probs_list[0]].keys())
    nr_sparsity = len(sparsity_range)
    res_array = np.zeros(shape=[nr_sparsity, nr_edge_probs])

    for edge_prob_count in range(nr_edge_probs):
        edge_prob = edge_probs_list[edge_prob_count]
        for s_count in range(nr_sparsity):
            s = sparsity_range[s_count]
            val = np.mean(res_file[edge_prob][s])
            res_array[nr_sparsity-1-s_count, edge_prob_count] = val

    ## Plot array ##
    fig, ax = plt.subplots()
    im = ax.imshow(res_array)

    # We want to show all ticks...
    ax.set_xticks(np.arange(nr_edge_probs))
    ax.set_yticks(np.arange(nr_sparsity))
    # ... and label them with the respective list entries
    ax.set_xticklabels([round(i,3) for i in edge_probs_list])
    ax.set_yticklabels(sparsity_range[::-1])

    for i in range(nr_sparsity):
        for j in range(nr_edge_probs):
            text = ax.text(j, i, round(res_array[i, j],3),
                           ha="center", va="center", color="w")

    # ax.set_title("Harvest of local farmers (in tons/year)")
    ax.set_aspect(0.333)
    # fig.tight_layout()
    plt.xlabel('Edge probability')
    plt.ylabel('Sparsity')
    plt.show()


if __name__ == '__main__':
    par_dict = {}
    par_dict['nr_vertices']  = 30
    par_dict['nr_gr_samples'] = 10
    par_dict['nr_sampled_supports'] = 2
    par_dict['sparsity_range'] = list(range(2,10))
    thresh = np.log(par_dict['nr_vertices'])/par_dict['nr_vertices']
    par_dict['edge_probs_list'] = [np.power(thresh, k/7) for k in np.arange(8,0,-1)]

    start_time = time.time()

    # run_exp(par_dict = par_dict, save_results=False)
    # run_par_exp(par_dict = par_dict, save_results=False)

    results_file_path = 'output/TrIT_paper_Inc_Matr_Exp_2_NrV20_NrS100--2019-5-22-13-23.npy'
    plot_results(results_file_path=results_file_path)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed time is: {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
