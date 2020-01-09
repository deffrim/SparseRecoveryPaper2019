import numpy as np


class basic_graph(object):
    def __init__(self, nr_vertices, directed = True):
        if nr_vertices < 1:
            raise Exception('basic_graph::__init__ : Number of vertices cannot be less than 1.')

        self.nr_vertices = nr_vertices
        self.directed = directed
        self.edge_set = set()
        self.nr_edges = None
        self.inc_matr = None

    def set_edges(self, edge_container):
        self.edge_set = set()
        for k in edge_container:
            self.edge_set.add(tuple(k))
        self.nr_edges = len(self.edge_set)
        self.calc_incidence_matrix()

    def calc_incidence_matrix(self):
        if self.nr_edges == 0 or len(self.edge_set) == 0:
            raise Exception('basic__graph::calc_incidence_matrix : Graph has no edges.')

        self.inc_matr = np.zeros(shape=[self.nr_vertices, self.nr_edges])

        for (e,idx) in zip(self.edge_set, range(self.nr_edges)):
            if self.directed:
                self.inc_matr[e[0]][idx] = -1.0
            else:
                self.inc_matr[e[0]][idx] = 1.0
            self.inc_matr[e[1]][idx] = 1.0

    def get_incidence_matr(self):
        if self.inc_matr is None:
            raise Exception('basic_graph::get_incidence_matr : Incidence matrix is None.')

        return self.inc_matr


    def subdivide_edge(self, edge):
        if edge not in self.edge_set:
            raise Exception('basic_graph::subdivide_edge : Edge {} is not in the graph.'.format(edge))
        new_edge_1 = (edge[0], self.nr_vertices)
        new_edge_2 = (self.nr_vertices, edge[1])
        self.edge_set.remove(edge)
        self.edge_set.add(new_edge_1)
        self.edge_set.add(new_edge_2)

        self.nr_vertices += 1
        self.nr_edges += 1
        self.calc_incidence_matrix()
