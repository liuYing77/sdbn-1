import numpy as np
import matplotlib.pyplot as plt
import pyNN.nest as p
import scipy.io as sio
import mnist_utils as mu
import random
import relu_utils as alg
import copy
import spiking_relu as sr
from scipy import signal

def conv2d(x, w, mode='c'):
    if len(x.shape)== 2:
        input_size = np.sqrt(x.shape[1])
        for i in range(x.shape[0]):
            xi = np.reshape(x[i],(input_size, input_size),order='F')
            y = signal.convolve2d(xi,w,mode='valid')
            y = np.nan_to_num(y)
            if mode=='s':
                w_size = w.shape[0]
                y = y[0::w_size, 0::w_size]
            y = np.reshape(y, (1, y.shape[0]*y.shape[0]),order='F')
            if i==0:
                y_list = y
            else:
                y_list = np.append(y_list, y, axis=0)
        return y_list
    else:
        print 'x has to be an array of'
        return -1

def conv_ReLU(input_num, output_num, input_list, w_list):
    out_list=[]
    for j in range(output_num):
        for i in range(input_num):
            if i==0:
                out_a = conv2d(input_list[i], w_list[i][j])
            else:
                out_a += conv2d(input_list[i], w_list[i][j])
        out_a[out_a<0] = 0
        out_list.append(out_a)
    return out_list

def pool_ReLU(input_num, input_list, w_list):
    out_list=[]
    for i in range(input_num):
        out_a = conv2d(input_list[i], w_list, mode='s')
        out_a[out_a<0] = 0
        out_list.append(out_a)
    return out_list

def out_ReLU(input_list, w_list):
    out_list=[]
    input_a = np.transpose(input_list, (1, 0, 2))
    input_a = np.reshape(input_a, (input_a.shape[0], input_a.shape[1]*input_a.shape[2]))
    out_a = np.dot(input_a, np.transpose(w_list))
    out_a[out_a<0] = 0
    out_list.append(out_a)
    return out_list

def plot_img(img_raw, width, height):
    plt.figure(figsize=(5,5))
    im = plt.imshow(np.reshape(img_raw,(width,height)), cmap=cm.gray_r,interpolation='none')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
def getLayerType(layer):
    return layer[0][0][0][0][0]
def getInputSize(layer):
    return layer[0][0][0][1][0][0].shape[0]
def getOutputSize(layer):
    return layer[0][0][0][1][0][0]
def getKernelSize(layer):
    return layer[0][0][0][2][0][0]
def getScale(layer):
    return layer[0][0][0][1][0][0]
def getOutLayer(cnn):
    return cnn['cnn'][0][0][2]
def getWeights(layer):
    w_list = []
    input_size = layer[0][0][0][3][0].size
    for i in range(input_size):
        w_list.append(layer[0][0][0][3][0][i][0])
    w_list = np.array(w_list)
    return w_list
def readmat(matfile):
    import scipy.io as sio
    cnn = sio.loadmat(matfile)
    for layer in cnn['cnn'][0][0][0]:
        if getLayerType(layer)=='i':
            W = list()
            lsize = list()
            lsize.append([1, getInputSize(layer)])
        elif getLayerType(layer)=='c':
            W.append(getWeights(layer))
            lsize.append([getOutputSize(layer), getKernelSize(layer)])
        elif getLayerType(layer)=='s':
            scale = getScale(layer)
            W.append(1./scale**2*np.ones((scale,scale)))
            lsize.append([0, scale])
    W.append(getOutLayer(cnn))
    lsize.append([-1, getOutLayer(cnn).shape[0]])
    return W, lsize
    
def test(W, L, tx):
    a = list()
    a.append(tx)
    a = np.array(a)
    a_list = list()
    for l in range(len(L)-1):
        input_num = a.shape[0]
        output_num = L[l+1][0]        
        a_list.append(a)
        if output_num == 0: #pooling layer S
            a = pool_ReLU(input_num, a_list[l], W[l])
        elif output_num > 0: #conv layer C
            a = conv_ReLU(input_num, output_num, a_list[l], W[l])
        elif output_num == -1: #output layer O
            a = out_ReLU(a_list[l], W[l])
        a = np.array(a)
    a_list.append(a)
    
    return a_list
    
def scale_weight(W, L, tx):
    a = list()
    a.append(tx)
    a = np.array(a)
    a_list = list()
    for l in range(len(L)-1):
        input_num = a.shape[0]
        output_num = L[l+1][0]        
        a_list.append(a)
        if output_num == 0: #pooling layer S
            a, W[l] = pool_ReLU_scalew(input_num, a_list[l], W[l])
        elif output_num > 0: #conv layer C
            a, W[l] = conv_ReLU_scalew(input_num, output_num, a_list[l], W[l])
        elif output_num == -1: #output layer O
            a, W[l] = out_ReLU_scalew(a_list[l], W[l])
        a = np.array(a)
    a_list.append(a)
    return W, a_list[-1][0]
def conv_ReLU_scalew(input_num, output_num, input_list, w_list):
    out_list=[]
    for j in range(output_num):
        for i in range(input_num):
            if i==0:
                out_a = conv2d(input_list[i], w_list[i][j])
            else:
                out_a += conv2d(input_list[i], w_list[i][j])
        out_a[out_a<0] = 0
        out_list.append(out_a)

    scale = get_scale(np.array(out_list))
    print scale
    for j in range(output_num):
        w_list[:,j] *= scale
        out_list[j] = predict_rate(out_list[j], scale)
    return out_list, w_list

def pool_ReLU_scalew(input_num, input_list, w_list):
    out_list=[]
    for i in range(input_num):
        out_a = conv2d(input_list[i], w_list, mode='s')
        out_a[out_a<0] = 0
        out_list.append(out_a)
    scale = get_scale(np.array(out_list))
    w_list *= scale
    
    for i in range(input_num):
        out_rate = predict_rate(out_list[i], scale)
        out_list[i] = out_rate
    return out_list, w_list

def out_ReLU_scalew(input_list, w_list):
    out_list=[]
    input_a = np.transpose(input_list, (1, 0, 2))
    input_a = np.reshape(input_a, (input_a.shape[0], input_a.shape[1]*input_a.shape[2]))
    out_a = np.dot(input_a, np.transpose(w_list))
    out_a[out_a<0] = 0
    scale = get_scale(out_a)
    w_list *= scale
    out_rate = predict_rate(out_a, scale)
    out_list.append(out_rate)
    return out_list, w_list
def get_scale(output):
    x_mid = 1./10. * np.log(np.exp(1./14.*50.)-1)
    output = output[output>0]
    if np.mean(output) > 0:
        scale = x_mid * 1000. / np.mean(output)#(np.mean(output)+3*np.std(output))#max(curr)
    else:
        scale = 1
    
    return scale

def predict_rate(output, scale):
    output *= scale/1000.
    out_rate = 14.*np.log(1.+np.exp(output*10.))
    return out_rate
'''   
def get_scale(output):
    
    k = 167.6
    x0 = 0.1
    y0 = 4.48
    #x0 = 0.
    #y0 = 0.
    x_mid = sr.rev_transf(k, x0, y0, 50.) #50 Hz as max output rate
    
    #x_mid = 1./10. * np.log(np.exp(1./14.*50.)-1)
    output = output[output>0]
    if np.mean(output) > 0:
        scale = x_mid * 1000. / np.mean(output)#(np.mean(output)+3*np.std(output))#max(curr)
    else:
        scale = 1
    
    return scale

def predict_rate(output, scale):
    
    k = 167.6
    x0 = 0.1
    y0 = 4.48
    #x0 = 0.
    #y0 = 0.
    
    output *= scale/1000.
    #out_rate = 14.*np.log(1.+np.exp(output*10.))
    out_rate = sr.transf(k, x0, y0, output)
    return out_rate
'''