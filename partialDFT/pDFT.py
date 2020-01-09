import random
import cmath
import numpy as np
import partialDFT.pDFTutils as utils
import matplotlib.pyplot as plt
# from sets import Set


class pDFT(object):
    def __init__(self, N=3, meas_ind=[0], rand_meas=False, nr_meas=2):
        if N < 3:
            raise Exception('N = {} has a trivial solution. Work it out with a pen and paper.'.format(N))
        if (not rand_meas and len(meas_ind) > N) or (rand_meas and (nr_meas > N)):
            raise Exception('You cannot have more than {} measurements.'.format(N))
        if not rand_meas and (min(meas_ind) < 0 or max(meas_ind) >= N ):
            raise Exception('Invalid measurement index.')

        self.N = N
        self.angle = -2*cmath.pi/self.N
        self.z = complex(cmath.cos(self.angle), cmath.sin(self.angle))
        self.nr_meas = None
        self.meas_ind = None
        self.comp_ind = None
        self.pdft_matrix= None
        self.comp_pdft_matrix= None
        self.extreme_pts = None
        self.maxASC = None
        self.minor_inds_to_dets_and_norms_dict = {}
        self.minor_det_norms_to_inds_dict = {}

        if rand_meas:
            self.nr_meas = nr_meas
            self.meas_ind = sorted(random.sample(range(self.N), self.nr_meas))
        else:
            self.meas_ind = sorted(meas_ind)
            self.nr_meas = len(self.meas_ind)

        self.comp_ind = [i for i in range(self.N) if i not in self.meas_ind]
        self.comp_ind_gcd = utils.gcd_list(self.comp_ind)
        # print('GCD of the complemantry indices is {}'.format(self.comp_ind_gcd))

        self.calc_pdft_matrix()
        # print('Measurement indices are {}'.format(self.meas_ind))

    def calc_pdft_matrix(self):
        pdft_list = []
        comp_pdft_list = []
        for i in range(self.N):
            row = []
            for j in range(self.N):
                row.append(pow(self.z, i*j)) # Consider dividing by sqrt(N)

            if i in self.meas_ind:
                pdft_list.append(row)
            else:
                comp_pdft_list.append(row)

        if len(pdft_list) == 0:
            raise Exception('pDFT::calc_pdft_matrix : pdft_list is empty.')

        self.pdft_matrix = np.matrix(pdft_list)
        self.comp_pdft_matrix = np.matrix(comp_pdft_list)



    def get_pdft_matrix(self):
        return self.pdft_matrix

    # def calc_extreme_pts(self, rnd_dig = 6):
    #     self.extreme_pts = set()
    #     all_combs = utils.combs(self.N//self.comp_ind_gcd, self.N-self.nr_meas-1)
    #     rows = range(self.N - self.nr_meas)
    #     for cols in all_combs:
    #         ixgrid = np.ix_(rows, cols)
    #         submatrix = self.comp_pdft_matrix[ixgrid]
    #         nullsp = utils.nullspace(submatrix.transpose())
    #         if nullsp.shape[1] != 1:
    #             raise Exception('pDFT::calc_maxASC() : Nullspace is of dimension {}'.format(nullsp.shape[1]))
    #         matr = nullsp.transpose() * self.comp_pdft_matrix
    #         l = tuple(np.around([matr[0,i] for i in range(matr.shape[1])], rnd_dig))
    #         self.extreme_pts.add(l)
    #     return self.extreme_pts

    def calc_extreme_pts(self, rnd_dig = 6):
        # if self.minor_inds_to_dets_and_norms_dict is None:
        #     self.calc_maximal_minors()
        self.extreme_pts = set()
        all_combs = utils.combs(self.N, self.nr_meas+1)
        rows = range(self.nr_meas)
        for cols in all_combs:
            ixgrid = np.ix_(rows, cols)
            submatrix = self.pdft_matrix[ixgrid]
            nullsp = utils.nullspace(submatrix)
            if nullsp.shape[1] != 1:
                raise Exception('pDFT::calc_maxASC() : Nullspace is of dimension {}'.format(nullsp.shape[1]))
            cn_dict = {cols[i]: nullsp[i,0] for i in range(len(cols))}
            l = []
            for i in range(self.N):
                if i in cols:
                    l.append(cn_dict[i])
                else:
                    l.append(0)
            l = tuple(l)
            # simple_l = self.calc_extreme_pt_simply2(cols=cols)
            # s_l = [simple_l[i] for i in cols]
            # norm_check = np.matmul(submatrix, s_l)
            # print('The norm is {}'.format(np.linalg.norm(norm_check)))
            self.extreme_pts.add(l)
        return self.extreme_pts

    def get_extreme_pts(self):
        return self.extreme_pts


    def calc_extreme_pt_simply(self, cols):
        m = self.nr_meas//2
        l = [0 for i in range(self.N)]
        for i in cols:
            diffs = [np.power(self.z, i) - np.power(self.z, j) for j in cols if j!=i]
            prod = np.prod(diffs)
            prod *= np.power(self.z, (self.N-m)*i)
            inv_prod = 1.0/prod
            l[i] = inv_prod

        norm_l = np.linalg.norm(l)
        l = [i/norm_l for i in l]
        l = tuple(l)
        return l


    def calc_extreme_pt_simply2(self, cols):
        m = self.nr_meas//2
        l = [0 for i in range(self.N)]
        comp_cols = [i for i in range(self.N) if i not in cols]
        for i in cols:
            diffs = [np.power(self.z, i) - np.power(self.z, j) for j in comp_cols]
            prod = np.prod(diffs)
            prod /= (self.N * np.power(self.z, (self.N-m-1)*i))
            l[i] = prod

        norm_l = np.linalg.norm(l)
        l = [i/norm_l for i in l]
        l = tuple(l)
        return l

    def calc_maxASC(self, rnd_dig = 6):
        self.calc_extreme_pts()
        norms_list = []
        # s_l = []
        for vec in self.extreme_pts:
            l = []
            # v_sum = 0.0
            for v in vec:
                val = round(np.absolute(v), rnd_dig)
                l.append(val)
                # v_sum += val
            # s_l.append(v_sum)
            norms_list.append(l)
        self.maxASC = []
        s_sum = tuple([0 for l in norms_list])
        sc_sum = tuple([sum(l) for l in norms_list])
        self.lnl_cols = [[l[i] for l in norms_list] for i in range(self.N)]

        self.maxASC = self.calc_ASC(s_sum, sc_sum)
        return self.maxASC

    def get_maxASC(self):
        return self.maxASC


    def calc_ASC(self, s_sum_list, sc_sum_list, idx=0):
        if idx == (self.N):
            return [[]]

        ASC = [[]]

        for i in range(idx, self.N):
            col = self.lnl_cols[i]
            if self.check_col(col,s_sum_list, sc_sum_list):
                ssl = tuple([j+k for k,j in zip(col, s_sum_list)])
                scl = tuple([j-k for k,j in zip(col, sc_sum_list)])
                ascmplx = self.calc_ASC(ssl, scl, i+1)
                ASC = ASC + [[i]+l for l in ascmplx]
        return ASC


    def check_col(self, col, s_sum, sc_sum):
        for i,j,k in zip(col, s_sum, sc_sum):
            if (j+i) >= (k-i):
                return False
        return True

    def calc_maximal_minors(self, rnd_dig = 6):
        all_combs = utils.combs(self.N, self.nr_meas)
        rows = range(self.nr_meas)
        for cols in all_combs:
            ixgrid = np.ix_(rows, cols)
            submatrix = self.pdft_matrix[ixgrid]
            det = np.linalg.det(submatrix)
            val = np.around(np.abs(det), rnd_dig)
            self.minor_inds_to_dets_and_norms_dict[tuple(cols)] = [det,val]
            if val in self.minor_det_norms_to_inds_dict:
                self.minor_det_norms_to_inds_dict[val].append(cols)
            else:
                self.minor_det_norms_to_inds_dict[val] = [cols]

        return [self.minor_inds_to_dets_and_norms_dict, self.minor_det_norms_to_inds_dict]



if __name__ == "__main__":
    N = 7
    meas_ind = [0, 2, 5]
    pdft = pDFT(N=N, meas_ind=meas_ind)
    ext_pts = pdft.calc_extreme_pts()
    # fig = plt.figure()
    # ax = fig.add_subplot()
    t = np.linspace(0, 2 * np.pi, 100)
    c_x = np.cos(t)
    c_y = np.sin(t)
    # unique_roots = set()
    for ext in ext_pts:
        print('Extreme pt is:')
        print(ext)
        roots = np.roots([ext[N-k-1] for k in range(N)])
        # unique_roots.add(tuple(roots))
        print('Roots are:')
        print(roots)
        fig = plt.figure()
        X = [x.real for x in roots]
        Y = [x.imag for x in roots]
        plt.plot(c_x, c_y)
        plt.scatter(X, Y, color='red')
        plt.grid()
        plt.axes().set_aspect('equal', 'datalim')
        # plt.show()
        print('#############')
    # print('Unique roots are:')
    # for k in unique_roots:
    #     print(k)
    plt.show()


