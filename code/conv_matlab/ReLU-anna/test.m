clear;
addpath(genpath('../../conv_matlab'));
% [cnn, er_sigm] = cnn_relu_example('sigm');
% save('cnn_sigm', 'cnn');
% 
% 
% [cnn, er_relu] = cnn_relu_example('relu');
% save('cnn_relu', 'cnn');


[cnn, er_softplus] = cnn_relu_example('noisy_softplus'); % noisy_softplus
% save('cnn_softplus', 'cnn');