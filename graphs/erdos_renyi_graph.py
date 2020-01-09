from graphs.basic_graph import basic_graph
import numpy as np
import utils


class erdos_renyi_graph(object):
    def __init__(self, nr_vertices, edge_probability, directed=True):
        if edge_probability <= 0 or edge_probability >1:
            raise Exception('erdos_renyi_graph::__init__ : The edge probabilities have to be greater than 0 and smaller than or equal to 1.')

        if nr_vertices < 2:
            raise Exception('erdos_renyi_graph::__init__ : Number of vertices cannot be less than 2.')

        self.edge_probability = edge_probability
        self.nr_vertices = nr_vertices
        self.directed = directed


    def sample(self):
        gr = basic_graph(nr_vertices=self.nr_vertices, directed=self.directed)
        edge_set = set()
        for c in utils.combs(self.nr_vertices, 2):
            if np.random.uniform(low=0.0, high=1.0) <= self.edge_probability:
                edge_set.add(tuple(c))

        gr.set_edges(edge_container=edge_set)

        return gr

if __name__ == '__main__':
    rand_gr = erdos_renyi_graph(nr_vertices=10, edge_probability=0.149)
    gr = rand_gr.sample()
    matr = gr.get_incidence_matr()
    print(matr)