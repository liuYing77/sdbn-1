import numpy as np
import poisson_tools as pt
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
'''
def update_para(v0, h0, v1, h1, a, b, w, eta ):
    delta_a = np.zeros(w.shape[0])
    delta_b = np.zeros(w.shape[1])
    delta_w = np.zeros(w.shape)
    batch_size = v0.shape[0]
    for k in range(batch_size):
        for i in range(w.shape[0]):
            delta_a[i] = eta * (v0[k][i] - v1[k][i])
        for j in range(w.shape[1]):
            delta_b[j] = eta * (h0[k][j] - h1[k][j])
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                delta_w[i][j] += eta * (v0[k][i]*h0[k][j] - v1[k][i]*h1[k][j])
    a += delta_a/batch_size
    b += delta_b/batch_size
    w += delta_w/batch_size
    return a, b, w
'''
def update_para(v0, h0, v1, h1, a, b, w, eta ):
    delta_a = np.zeros(v0.shape)
    delta_b = np.zeros(h0.shape)
    delta_w = np.zeros(w.shape)
    batch_size = v0.shape[0]
    
    delta_a = eta * (v0 - v1)
    delta_b = eta * (h0 - h1)
    for k in range(batch_size):
        v0_matrix = np.transpose(np.tile(v0[k],(w.shape[1],1)))
        v1_matrix = np.transpose(np.tile(v1[k],(w.shape[1],1)))
        h0_matrix = np.tile(h0[k],(w.shape[0],1))
        h1_matrix = np.tile(h1[k],(w.shape[0],1))
        delta_w += eta * (v0_matrix*h0_matrix - v1_matrix*h1_matrix)
    
    a += np.average(delta_a,0)
    b += np.average(delta_b,0)
    w += delta_w/np.float(batch_size)
    return a, b, w

train_x, train_y = pt.get_train_data()
train_x = train_x > 50

digit = 5
label_list = np.array(train_y).astype(int)
index_digit = np.where(label_list==digit)[0]
train_num = len(index_digit)-1
#train_num = 110
index_train = index_digit[0:train_num]
Data_v = np.array(train_x[index_train]).astype(float)

hiden_num = 500
W = np.random.normal(0,0.01,Data_v.shape[1]*hiden_num)
W = W.reshape((Data_v.shape[1],hiden_num))
b = np.zeros(hiden_num)
pixel_on = np.sum(Data_v,0)
a = np.log((pixel_on + 0.01)/(train_num - pixel_on + 0.01))
eta = 0.001

batch_size = 2
for iteration in range(4):
    '''
    # Persistant CD
    for k in range(0,train_num,batch_size):
        max_bsize = min(train_num-k, batch_size)
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
        print iteration, k 
    '''
    for k in range(0,train_num,batch_size):
        max_bsize = min(train_num-k, batch_size)
        data_v = Data_v[k:k+max_bsize]
        data_h = sigmoid_sampling(data_v, W, b)
        gibbs_v = sigmoid_sampling(data_h, W.transpose(), a)
        gibbs_h = sigmoid_sampling(gibbs_v, W, b)
        a, b, W = update_para(data_v, data_h, gibbs_v, gibbs_h, a, b, W, eta)
        print iteration, k
    
    data_v = np.array(train_x[index_digit[train_num]]).astype(float)
    data_h = sigmoid_sampling(data_v, W, b)
    recon = sigmoid_sampling(data_h, W.transpose(), a)
    pt.plot_digit(recon)
    plt.draw()

np.save('theta/test.npy',[a,b,W])
plt.show()
