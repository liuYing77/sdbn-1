'''
Functions to be used in Sigmoid tasks.
'''

import mnist_utils as mu
import maths_utils as matu
import numpy as np
from scipy.special import expit

def init_para(vis_num, hid_num, eta):
    para = {}
    para['h_num'] = hid_num
    para['v_num'] = vis_num
    para['eta'] = eta
    w = np.random.normal(0,0.01, vis_num*hid_num)
    para['w'] = w.reshape((vis_num,hid_num))
    para['b'] = np.zeros(hid_num)
    para['a'] = np.zeros(vis_num)
    #pixel_on = np.sum(Data_v,0)
    #para['a'] = np.log((pixel_on + 0.01)/(train_num - pixel_on + 0.01))
    return para

def plot_recon(digit_img, para):
    data_v = np.array(digit_img).astype(float)
    data_h, gibbs_v, gibbs_h = sampling_nb(para, data_v)
    mu.plot_digit(gibbs_v)
    
def update_batch_cd1(para, data_v):
    eta = para['eta']
    max_bsize = data_v.shape[0]
    data_h, gibbs_v, gibbs_h = sampling_nb(para, data_v)
       
    pos_delta_w = np.zeros((para['v_num'], para['h_num']))
    neg_delta_w = np.zeros((para['v_num'], para['h_num']))
    for i in range(max_bsize):
        pos_delta_w += matu.matrix_times(data_v[i], data_h[i])
        neg_delta_w += matu.matrix_times(gibbs_v[i], gibbs_h[i])    
    delta_w_pos = eta * pos_delta_w/np.float(max_bsize)
    delta_w_neg = eta * neg_delta_w/np.float(max_bsize)
    para['w'] += delta_w_pos
    para['w'] -= delta_w_neg
    delta_a = data_v - gibbs_v
    delta_b = data_h - gibbs_h
    delta_a = eta * np.average(delta_a,0)
    delta_b = eta * np.average(delta_b,0)
    para['a'] += delta_a
    para['b'] += delta_b
    #print delta_w_pos.max(), delta_w_neg.max()
    return para
    

    
def sigmoid(data, weight, bias):
    sum_data = np.dot(data, weight) + bias
    prob = expit(sum_data)
    return prob



def sigmoid_sampling(data, weight, bias):
    prob = sigmoid(data, weight, bias)
    rdm = np.random.random(prob.shape)
    index_on = rdm < prob
    samples = np.zeros(prob.shape)
    samples[index_on]=1.
    return samples
  
def sampling(para, data_v): #non binary
    w = para['w']
    a = para['a']
    b = para['b']
    h0 = sigmoid_sampling(data_v, w, b)
    v1 = sigmoid_sampling(data_h, w.transpose(), a)
    h1 = sigmoid_sampling(gibbs_v, w, b)

    return h0, v1, h1  

def sampling_nb(para, data_v): #non binary
    w = para['w']
    a = para['a']
    b = para['b']
    h0 = sigmoid_sampling(data_v, w, b)
    v1 = sigmoid(h0, w.transpose(), a)
    h1 = sigmoid_sampling(v1, w, b)

    return h0, v1, h1
