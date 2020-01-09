from sympy import Matrix
from importlib import import_module
from numpy import asarray,reshape, abs
import datetime
import numpy as np
import matplotlib.pyplot as plt


def combs(n,k):
    if k > n:
        raise Exception('utils::combs : You cannot choose {} from {}'.format(k,n))

    if k == 1:
        yield from ([i] for i in range(n))
    elif k>1:
        yield from ([f] + [1+f+i for i in j] for f in range(n-k+1) for j in combs(n - f -1, k -1))


def join_lists(list_of_list):
    a = set()
    for i in list_of_list:
        for j in i:
            a.add(j)
    b = list(a)
    b.sort()
    return b

def nullspace(A):
    return Matrix(A).nullspace()

def str_to_obj_handle(module_name, class_name):
    module = import_module(module_name)
    return getattr(module, class_name)


def config_str_to_np_array(A):
    m = A.count('[')-1
    A = A.split(",")
    A = [x.strip() for x in A]
    A = [x.strip('[') for x in A]
    A = [x.strip(']') for x in A]
    A = [float(x) for x in A]
    A = asarray(A)
    return A.reshape(m, A.shape[0]//m)

def partition_from_grps(sig_dim, groups):
    if type(groups) != list:
        gps_list = [i for i in groups]
    else:
        gps_list = groups

    signature = dict()
    for i in range(sig_dim):
        signature[i] = tuple()

    overlap_flag = False

    for g in gps_list:
        for i in g:
            if len(signature[i]) != 0:
                overlap_flag = True
            signature[i]+=(tuple(g),)

    if not overlap_flag:
        return {tuple(g):(tuple(g),) for g in gps_list}

    sign_to_idx = dict()
    for idx in signature:
        sign = signature[idx]
        if sign in sign_to_idx:
            sign_to_idx[sign] += (idx,)
        else:
            sign_to_idx[sign] = (idx,)

    part_dict = {v: k for k, v in sign_to_idx.items()}

    return part_dict


def signature_from_grps(sig_dim, groups):
    if type(groups) != list:
        gps_list = [i for i in groups]
    else:
        gps_list = groups

    signature = dict()
    for i in range(sig_dim):
        signature[i] = tuple()

    for g in gps_list:
        for i in g:
            signature[i]+=(tuple(g),)

    return signature

def bottom_clip(x, epsilon):
    if abs(x) < abs(epsilon):
        return 0.0
    else:
        return x


def save_nparray_with_date(data, file_prefix = '', subfolder_name = ''):
    dt = datetime.datetime.now()
    time_str = '{}-{}-{}-{}-{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    if subfolder_name != '':
        subfolder_name = subfolder_name + '/'
    file_path = subfolder_name + file_prefix + '--' + time_str
    np.save(file_path, data)


def cal_mutual_coherence(matr):
    np_matr = np.asarray(matr)
    sym_matr = np.matmul(np.transpose(np.conjugate(np_matr)), np_matr)
    sym_matr = np.abs(sym_matr)
    diag = np.diag(sym_matr)
    sqrt_diag = np.sqrt(diag)
    inv_sqrt_diag = np.diag(1.0/sqrt_diag)
    sym_matr = np.matmul(inv_sqrt_diag, sym_matr)
    sym_matr = np.matmul(sym_matr, inv_sqrt_diag)
    np.fill_diagonal(sym_matr, -1.0)
    mc_val = np.max(sym_matr)
    return mc_val



if __name__ == '__main__':
    a = combs(7,6)
    for i in a:
        print(i)