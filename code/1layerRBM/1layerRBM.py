'''
Typical usage:
python -m code.1layerRBM.1layerRBM 1 50 500 pcd
from sdbn folder
'''
import sys
import random
import numpy as np
from ..utils import poisson_tools as pt
import matplotlib.pyplot as plt
from scipy.special import expit

def sigmoid_sampling(data, weight, bias):
    sum_data = np.dot(data, weight) + bias
    prob = expit(sum_data)
    rdm = np.random.random(prob.shape)
    index_on = rdm < prob
    samples = np.zeros(prob.shape)
    samples[index_on]=1.
    return samples

def sampling_k(a, b, w, sample_num, init_v):
    gibbs_v = np.zeros((sample_num, a.shape[0]))
    gibbs_h = np.zeros((sample_num, b.shape[0]))
    gibbs_v[0] = init_v
    for g_step in range(1, sample_num):
        gibbs_h[g_step-1] = sigmoid_sampling(gibbs_v[g_step-1], w, b)
        gibbs_v[g_step] = sigmoid_sampling(gibbs_h[g_step-1], w.transpose(), a)
    gibbs_h[-1] = sigmoid_sampling(gibbs_v[-2], w, b)
    return gibbs_v, gibbs_h

def matrix_times(m, n):
    m_matrix = np.transpose(np.tile(m,(len(n), 1)))
    n_matrix = np.tile(n,(len(m), 1))
    return m_matrix*n_matrix
    
def update_para(v0, h0, v1, h1, a, b, w, eta ):
    delta_a = np.zeros(v0.shape)
    delta_b = np.zeros(h0.shape)
    delta_w = np.zeros(w.shape)
    b_size = v0.shape[0]
    
    delta_a = eta * (v0 - v1)
    delta_b = eta * (h0 - h1)
    for k in range(b_size):
        v0_matrix = np.transpose(np.tile(v0[k],(w.shape[1],1)))
        v1_matrix = np.transpose(np.tile(v1[k],(w.shape[1],1)))
        h0_matrix = np.tile(h0[k],(w.shape[0],1))
        h1_matrix = np.tile(h1[k],(w.shape[0],1))
        delta_w += eta * (v0_matrix*h0_matrix - v1_matrix*h1_matrix)
    
    a += np.average(delta_a,0)
    b += np.average(delta_b,0)
    w += delta_w/np.float(b_size)
    return a, b, w

def update_cdk(a, b, w, data_v, cd_size):
    max_bsize = data_v.shape[0]
    ind_rdm = int(np.floor(np.random.random()*max_bsize))
    init_v = data_v[ind_rdm]
    gibbs_v, gibbs_h = sampling_k(a, b, w, cd_size, init_v)
    avg_h = []
    for vd in range(max_bsize):
        data_h = np.zeros((cd_size, b.shape[0]))
        for h in range(cd_size):
            data_h[h] = sigmoid_sampling(data_v[vd], w, b)
        avg_h.append(np.average(data_h, axis=0))
    
    
    delta_a = eta * (np.average(data_v,0) - np.average(gibbs_v,0))
    delta_b = eta * (np.average(avg_h,0) - np.average(gibbs_h,0))
    pos_delta_w = np.zeros(w.shape)
    neg_delta_w = np.zeros(w.shape)
    for vd in range(max_bsize):
        pos_delta_w += matrix_times(data_v[vd], avg_h[vd])
    for gstep in range(cd_size):
        neg_delta_w += matrix_times(gibbs_v[gstep], gibbs_h[gstep])    
    a += delta_a
    b += delta_b
    w += eta * pos_delta_w/np.float(max_bsize)
    w -= eta * neg_delta_w/np.float(cd_size)
    return a, b, w


train_num = int(sys.argv[1])
b_size = int(sys.argv[2])
epoc = int(sys.argv[3])
alg = sys.argv[4]

np.random.seed(0)

train_x, train_y = pt.get_train_data()
train_x = train_x > 50

digit = 5
label_list = np.array(train_y).astype(int)
index_digit = np.where(label_list==digit)[0]
if train_num <= 1 or train_num > len(index_digit):
    train_num = len(index_digit) - 1
else:
    train_num = train_num - 1

index_train = index_digit[0:train_num]
Data_v = np.array(train_x[index_train]).astype(float)

hiden_num = 500
W = np.random.normal(0,0.01,Data_v.shape[1]*hiden_num)
W = W.reshape((Data_v.shape[1],hiden_num))
b = np.zeros(hiden_num)
pixel_on = np.sum(Data_v,0)
a = np.log((pixel_on + 0.01)/(train_num - pixel_on + 0.01))
eta = 0.001

for iteration in range(epoc):
    if alg == 'pcd':
        # Persistant CD
        print alg
        for k in range(0,train_num,b_size):
            max_bsize = min(train_num-k, b_size)
            data_v = Data_v[k:k+max_bsize]
            data_h = sigmoid_sampling(data_v, W, b)
            gibbs_v = np.zeros(data_v.shape)
            gibbs_h = np.zeros(data_h.shape)
            gibbs_v[0] = sigmoid_sampling(data_h[0], W.transpose(), a)
            gibbs_h[0] = sigmoid_sampling(data_v[0], W, b)
            for g_step in range(1, max_bsize):
                gibbs_v[g_step] = sigmoid_sampling(gibbs_h[g_step-1], W.transpose(), a)
                gibbs_h[g_step] = sigmoid_sampling(gibbs_v[g_step-1], W, b)
            a, b, W = update_para(data_v, data_h, gibbs_v, gibbs_h, a, b, W, eta)
            print iteration+1, k+1 
    elif alg == 'kcd':
        # thinking CD_k is the sum of k CD_1.
        # It is supposed to be WRONG
        for k in range(0,train_num,b_size):
            max_bsize = min(train_num-k, b_size)
            data_v = Data_v[k:k+max_bsize]
            data_h = sigmoid_sampling(data_v, W, b)
            gibbs_v = sigmoid_sampling(data_h, W.transpose(), a)
            gibbs_h = sigmoid_sampling(gibbs_v, W, b)
            a, b, W = update_para(data_v, data_h, gibbs_v, gibbs_h, a, b, W, eta)
            print iteration+1, k+1
    
    elif alg == 'cdk':
        # thinking CD_k is the sum of k CD_1.
        # It is supposed to be WRONG
        for k in range(0,train_num,b_size):
            max_bsize = min(train_num-k, b_size)
            data_v = Data_v[k:k+max_bsize]
            a, b, W = update_cdk(a, b, W, data_v, b_size)
            print iteration+1, k+1
    np.save('/home/liuq/apt/2ndYear/sDBN/theta/%04d_b%04d_epoc%05d_%s.npy'%(train_num, b_size, iteration+1, alg),[a,b,W])
    data_v = np.array(train_x[index_digit[train_num]]).astype(float)
    data_h = sigmoid_sampling(data_v, W, b)
    recon = sigmoid_sampling(data_h, W.transpose(), a)
    pt.plot_digit(recon)
    #plt.draw()
    #plt.savefig('/home/liuq/apt/2ndYear/sDBN/results/%04d_b%04d_epoc%05d_%s.pdf'%(train_num, b_size, iteration+1, alg))
    
#plt.show()
