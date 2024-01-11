import numpy as np
import pandas as pd
import functools
import scanpy as sc

def H(x):
    output = np.sum(x*np.log(x+1e-10))
    
    return output

def pdconcat(x, y, join="outer"):
        return pd.concat([x,y], join=join)

def concatenate(x, y):
        return np.concatenate([x,y])

def vstack(x, y):
        return np.vstack([x,y])

def add(x, y):
    return x+y

def pp(dat_list, hvg = True, n_top_genes=5000):
    
    c = len(dat_list)
    
    ref_dat_pp = list()
    for i in range(c):
        print("precessing {}-th data".format(i))
        adata = sc.AnnData(dat_list[i])
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if hvg:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata)
        ref_dat_pp.append(adata.to_df())
        
    if c > 1:
        ref_dat_all = functools.reduce(pdconcat, ref_dat_pp)
        ref_dat_all[np.isnan(ref_dat_all)] = 0
        for i in range(c):
            index=list(dat_list[i].index)
            index = [str(item) for item in index]
            ref_dat_pp[i] = np.array(ref_dat_all.loc[index])
        
    return ref_dat_pp


def find_index(a, b):
    
    index = list()
    for i in sorted(np.unique(b)):
        index.append(list(a).index(i))
    
    return np.array(index)


def _mix_rbf_kernel(X, Y, sigma_list):

    m = X.shape[0]

    Z = np.concatenate([X, Y], 0)
    ZZT = Z @ Z.T
    diag_ZZT = np.diag(ZZT)
    Z_norm_sqr = np.broadcast_to(diag_ZZT, ZZT.shape)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.T

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += np.exp(-gamma*exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

def _mmd2(K_XX, K_XY, K_YY, biased=True):

    m = K_XX.shape[0]
    n = K_YY.shape[0]

    diag_X = np.diag(K_XX)
    diag_Y = np.diag(K_YY)
    sum_diag_X = np.sum(diag_X)
    sum_diag_Y = np.sum(diag_Y)

    Kt_XX_sums = np.sum(K_XX, axis = 0) - diag_X
    Kt_YY_sums = np.sum(K_YY, axis = 0) - diag_Y
    K_XY_sums_0 = np.sum(K_XY, axis = 1)

    Kt_XX_sum = np.sum(Kt_XX_sums)
    Kt_YY_sum = np.sum(Kt_YY_sums)
    K_XY_sum = np.sum(K_XY_sums_0)

    if biased:
        mmd2 = (Kt_XX_sum + sum_diag_X) / (m*m) + (Kt_YY_sum + sum_diag_Y) / (n*n) - 2.0 * K_XY_sum /(m*n)
    else:
        mmd2 = Kt_XX_sum / (m*(m-1)) + Kt_YY_sum / (n*(n-1)) - 2.0 * K_XY_sum / (m*n)

    return mmd2

def cal_mmd(X, Y, sigma_list=[1, 2, 4, 8, 16], biased=True):

    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2(K_XX, K_XY, K_YY, biased=biased)

def dist_ensemble(dist, weight):
    
    weight = np.array(weight).reshape(len(weight), 1)
    k = dist[0].shape[1]
    c = len(dist)
    
    ind = list()
    for i in range(c):
        temp = np.sum(dist[i], axis = 0)
        index = np.full((k), 1)
        index[np.where(temp == 0)[0]] = 0
        ind.append(index)
    ind = np.array(ind)
    
    temp_sum = [dist[l] * weight[l] for l in range(c)]
    temp_sum = functools.reduce(add, temp_sum)
    dist_emsenble = temp_sum.copy()
    dist_emsenble /= np.max(dist_emsenble)
    
    return dist_emsenble