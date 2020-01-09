import numpy as np
from graphs.basic_graph import basic_graph
from sparse_recovery.sparse_recovery_experiments.l1_rec_exp import l1_rec_exp
import matplotlib.pyplot as plt
import utils
import itertools


def run_exp(gr0, nr_sampled_supports, nr_gr_in_family=5):
    gr = gr0
    gr.calc_incidence_matrix()
    inc_matr = gr.get_incidence_matr()

    all_rec_probs = []

    for count in range(nr_gr_in_family):
        sparsity_range = np.arange(1,1+gr.nr_edges, 1)
        exp = l1_rec_exp(matr=inc_matr,sparsity_range=sparsity_range, nr_sampled_supports=nr_sampled_supports)
        exp.run()
        rec_probs = exp.get_mean_rec_probs()
        all_rec_probs.append(rec_probs)

        ## Change graph ##
        for _ in range(2):
            gr.subdivide_edge(edge=(gr.nr_vertices-1, 0))

        gr.calc_incidence_matrix()
        inc_matr = gr.get_incidence_matr()

    # SAVE RESULTS ##
    utils.save_nparray_with_date(data=all_rec_probs, file_prefix='TrIT_paper_Inc_Matr_Exp_1_NrS{}'.format(nr_sampled_supports), subfolder_name='output')


def plot_res(res_file_path = ''):
    all_rec_probs = np.load(res_file_path, allow_pickle=True)
    # all_rec_probs = all_rec_probs.item()

    plot_symbs_list = itertools.cycle(['o-', '*-', 's-', 'd-', 'x-', '+-'])
    fig, ax = plt.subplots()

    for (res, idx) in zip(all_rec_probs, range(len(all_rec_probs))):
        vals = [res[s] for s in res]
        ax.plot(list(res.keys()), vals, next(plot_symbs_list), label='G{}'.format(2*idx+3))

    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid()
    plt.xlabel('Sparsity (s)')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    nr_vertices_0 = 4
    edges_0 = [(0, 1), (1, 2), (2, 3), (3, 0), (2, 0)]  # Vertex numbers must start with 0!!
    # nr_edges = len(edges_0)
    # sparsity_range = range(1,6)
    gr = basic_graph(nr_vertices=nr_vertices_0)
    gr.set_edges(edges_0)
    nr_sampled_supports = 2000

    # run_exp(gr0=gr, nr_sampled_supports=nr_sampled_supports)

    res_file_path = 'output/TrIT_paper_Inc_Matr_Exp_1_NrS2000--2019-5-21-18-8.npy'
    plot_res(res_file_path=res_file_path)
