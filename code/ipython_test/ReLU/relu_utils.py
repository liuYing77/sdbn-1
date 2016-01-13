'''
Functions to be used in ReLU tasks.
'''

import mnist_utils as mu
import maths_utils as matu
import numpy as np

def init_para(vis_num, hid_num, eta):
    para = {}
    para['h_num'] = hid_num
    para['v_num'] = vis_num
    para['eta'] = eta
    w = np.random.normal(0,0.01, vis_num*hid_num)
    para['w'] = w.reshape((vis_num,hid_num))
    return para

def plot_recon(digit_img, para):
    data_v = np.array(digit_img).astype(float)
    data_h, gibbs_v, gibbs_h = sampling(para, data_v)
    mu.plot_digit(gibbs_v)
    
def update_batch_cd1(para, data_v):
    eta = para['eta']
    max_bsize = data_v.shape[0]
    data_h, gibbs_v, gibbs_h = sampling(para, data_v)
    
    pos_delta_w = np.zeros((para['v_num'], para['h_num']))
    neg_delta_w = np.zeros((para['v_num'], para['h_num']))
    for i in range(max_bsize):
        pos_delta_w += matu.matrix_times(data_v[i], data_h[i])
        neg_delta_w += matu.matrix_times(gibbs_v[i], gibbs_h[i])    
    delta_w_pos = eta * pos_delta_w/np.float(max_bsize)
    delta_w_neg = eta * neg_delta_w/np.float(max_bsize)
    para['w'] += delta_w_pos
    para['w'] -= delta_w_neg
    #print delta_w_pos.max(), delta_w_neg.max()
    return para
    
def ReLU(data, weight):
    sum_data = np.dot(data, weight)
    sum_data[sum_data < 0] = 0
    return sum_data

def sampling(para, data_v):
    w = para['w']
    h0 = ReLU(data_v, w)
    v1 = ReLU(h0, w.transpose())
    h1 = ReLU(v1, w)
    return h0, v1, h1
