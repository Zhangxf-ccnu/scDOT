import numpy as np
import pandas as pd
import functools
from collections import Counter
import scipy.spatial as sp
import math
import ot
from utils import pp, concatenate, find_index, vstack, cal_mmd, dist_ensemble, H


def cal_ct_margin(ref_dat_list, ref_label_list, label_set, query_dat):

    ct_margin = list()
    for r in range(len(ref_label_list)):
        s_f = np.zeros(len(label_set))
        sum_feat = pd.DataFrame(Counter(ref_label_list[r]).items())
        sum_feat = sum_feat.sort_values(by = 0)
        id = find_index(label_set, sum_feat.iloc[:,0])
        s_f[id] = sum_feat.iloc[:,1]/np.sum(sum_feat.iloc[:,1])
        ct_margin.append(s_f)
        
    ct_margin = functools.reduce(vstack, ct_margin)
    counts = ct_margin.astype(bool).sum(axis=0)
    ct_margin = np.sum(ct_margin, axis = 0)/counts
    ct_margin[np.isnan(ct_margin)] = 0
    ct_margin = ct_margin/np.sum(ct_margin)
    
    return ct_margin

def cal_single_dist(ref_dat, ref_label, query_dat):
    
    
    ct_count = pd.DataFrame(Counter(ref_label).items())
    ct_count = ct_count.sort_values(by = 0)
    

    dist = list()
    for i in range(ct_count.shape[0]):
        id = np.where(ref_label == ct_count.iloc[i,0])[0]
        ct_feat = np.mean(np.array(ref_dat[id,:]), axis = 0).reshape(1, ref_dat.shape[1])
        dist.append(sp.distance.cdist(query_dat, ct_feat, 'cosine').T)
            
    dist = functools.reduce(concatenate, dist).T
    
                
    return dist
    

def cal_multi_dist(ref_dat, ref_label, label_set, query_dat):
    
    
    dist = list()
    for i in range(len(ref_dat)):
        single_dist = cal_single_dist(ref_dat[i],ref_label[i], query_dat)
        
        temp_dist = np.full([query_dat.shape[0], len(label_set)], 0.)
        
        id = find_index(label_set, ref_label[i])
        temp_dist[:,id] = single_dist
        temp_dist /= temp_dist.max()
        temp_dist = pd.DataFrame(temp_dist, columns=label_set, index=np.array(np.arange(0, temp_dist.shape[0], 1, dtype='int'), dtype='object'))
        
        dist.append(temp_dist)
        
    return dist


def D_inpute_with_expr(Dist, label_set, dat_list, label_list):
    c = len(dat_list)
    intersection = set(label_list[0])    
    
    for df in label_list[1:]:
        ct = set(df)
        intersection = intersection.intersection(ct)
    
    common_ct = np.array(list(intersection))
    
    D_temp = list()
    for d in Dist:
        D_temp.append(d[common_ct])
    
    dat_temp = list()
    for d in range(len(dat_list)):
        idx = np.isin(label_list[d], common_ct)
        dat_temp.append(dat_list[d][idx])
    
    w_list = list()
    for l in range(c):
        idx_dat = ([ii for ii in range(len(dat_temp)) if ii != l])
        for ll in idx_dat:
            W = cal_mmd(dat_temp[l], dat_temp[ll])
            W = math.exp(-W)
            w_list.append(W)
        
    w_list = np.array(w_list).reshape(len(dat_list), len(dat_list)-1)
    w_list_norm = w_list / np.sum(w_list, axis=1)[:, np.newaxis]
        
    D_inpute = list()
    for l in range(c):
        ind = Dist[l].eq(0).all()
        if np.sum(ind) == 0:
            D_inpute.append(Dist[l])
        else:
            index = [ii for ii in range(c) if ii != l]
            D_unique = list()
            for i in range(np.sum(ind)):
                a = [Dist[ii][label_set[ind][i]] for ii in index]
                D_unique.append(pd.concat(a, axis=1))
            temp = list()
            for i in range(np.sum(ind)):
                id_temp = D_unique[i].ne(0).any()
                w_use = w_list_norm[l,:][id_temp]
                if w_use.all() == 0:
                    temp.append(pd.Series(np.ones(Dist[l].shape[0])))
                else:
                    w_use = w_use/ np.sum(w_use)
                    temp.append(D_unique[i].iloc[:,np.array(range(D_unique[i].shape[1]))[id_temp]] @ w_use)
            inputed = pd.concat(temp, axis = 1)
            Dist[l][label_set[ind]] = inputed
            D_inpute.append(Dist[l])
            
    return D_inpute, w_list


def init_solve(D, a, b, lambda1, w_piror=None, lambda2=0.01):
    c = len(D)
    n, t = D[0].shape
    
    if w_piror is None:
        w_piror = np.ones(c)*(1/c)
    else:
        w_piror = w_piror.copy()
        
    M = dist_ensemble(D, w_piror)
    x = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, lambda1, lambda2)
    
    tt = [(np.dot(D[l], x.T).trace()) for l in range(c)]
    beta = np.median(tt)
    
    numerator = [math.exp(-1*tt[l]/beta) for l in range(c)]
    w = np.array([numerator[l]/np.sum(numerator) for l in range(c)])
        
        
    return beta, w

def _solve(D, a, b, w_piror=None, lambda1=0.01, lambda2=0.01, beta=1, maxIter=100, tol=1e-5):

    n,t = D[0].shape
    c = len(D)
    
    if w_piror is None:
        w_prev = np.full((c, 1), 1/c)
    else:
        w_prev = w_piror.copy()
        
    loss_prev = 1e4
    for i in range(maxIter):
        
        M = dist_ensemble(D, w_prev)
        M = M/np.max(M)
        x = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, lambda1, lambda2)
        
        tt = [(np.dot(D[l], x.T).trace()) for l in range(c)]
        numerator = [math.exp(-1*np.dot(D[l], x.T).trace()/beta) for l in range(c)]
        w = np.array([numerator[l]/np.sum(numerator) for l in range(c)])
        loss1=0
        for l1, l2 in zip(tt, w):
            loss1 += l1*l2
        loss2 = lambda1*H(x) + lambda2*np.sum(a*np.log1p(a/(np.sum(x, axis=1) + 1e-10))) + lambda2*np.sum(b*np.log1p(b/(np.sum(x, axis=0) + 1e-10)))
        loss3 = beta*H(w)
        loss = loss1 + loss2 + loss3
        
        diff_loss = np.absolute(loss_prev - loss)/np.absolute(loss_prev + 1e-10)
        
        
        if diff_loss < tol:
            break
        
        w_prev = w.copy()
        loss_prev = loss.copy()
    
    return x, w


def solve(D, a, b, label_set, beta, w_piror=None, lambda1=0.01, lambda2=0.01, maxIter2=100, tol=1e-2):
    
    
    x, w = _solve(D, a, b, w_piror, maxIter=maxIter2, lambda1 = lambda1, lambda2 = lambda2, beta=beta, tol=tol)
    
    y_hat = np.argmax(x, axis = 1)
    y_hat = np.array(y_hat, dtype='object')
    ct = sorted(label_set)
    for i in range(len(ct)):
        y_hat[np.where(y_hat == i)[0]] = ct[i]

        
    return x, w, y_hat

def scDOT(loc, ref_name, query_name, lambda1=0.01, lambda2=0.01, threshold=0.9):

    # 	'''
    # 	Input:
    # 	:loc: Where reference data and query data are stored.
    # 	:ref_name: A list of names of the reference datasets.
    # 	:query_name: The name of the query datasets.
    # 	:lambda1: Numerical value, and the default value is 0.01.
    # 	:lambda2: Numerical value, and the default value is 0.01.
    # 	:threshold: The threshold for unseen cell type identification, and the default value is 0.9.
    # 	Output:
    # 	final_annotation: 1D-array, final annotation including unseen cell-type identification.
    # 	m: 1D-array, metric for unseen cell-type identification.
    # 	'''

    expression_s = list()
    label_s = list()
    for i in ref_name:
        file_name = loc + "{}_cell.csv".format(i)
        a=pd.read_csv(file_name, header=0, index_col=0)
        expression_s.append(a)
        file_name = loc + "{}_label.csv".format(i)
        a=pd.read_csv(file_name, header=0, index_col=0).iloc[:,0].values
        label_s.append(a)

    file_name = loc + "{}_cell.csv".format(query_name)
    expression_t=pd.read_csv(file_name, header=0, index_col=0)
    del file_name,a,i

    dat_list = expression_s.copy()
    dat_list.append(expression_t)
    dat_list_pp = pp(dat_list)
    query_pp = dat_list_pp[-1]
    dat_list_pp = dat_list_pp[0:-1]
    del dat_list, expression_s
    
    label_set = functools.reduce(concatenate, label_s)
    label_set = sorted(np.unique(label_set))
    a_hat = cal_ct_margin(dat_list_pp, label_s, label_set, query_pp)
    b_hat = np.array(np.sum(expression_t, axis = 1))/np.sum(np.sum(expression_t))
    
    D = cal_multi_dist(dat_list_pp, label_s, label_set, query_pp)
    D_inp, w_list = D_inpute_with_expr(Dist=D, label_set=np.array(label_set), dat_list=dat_list_pp, label_list=label_s)
    D_inp = [np.array(ll) for ll in D_inp]
    
    w = np.full((len(D_inp), 1), 1/len(D_inp))
    lambda3, w = init_solve(D_inp, b_hat, a_hat, w_piror=w, lambda1=lambda1, lambda2=lambda2)
    x, w, y_hat = solve(D_inp, b_hat, a_hat, label_set, w_piror=w, lambda1=lambda1, lambda2=lambda2, beta=lambda3)
    
    x = x / np.sum(x, axis=1, keepdims=True)
    score = np.max(x, axis=1)
    
    y_hat_with_unseen = y_hat.copy()
    y_hat_with_unseen[np.where(score<threshold)[0]]="unseen"
    
    return y_hat, score, y_hat_with_unseen
