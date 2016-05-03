clear;
addpath(genpath('../../conv_matlab'));
% [cnn, er_sigm] = cnn_relu_example('sigm');
% save('cnn_sigm', 'cnn');
% 
% 
% [cnn, er_relu] = cnn_relu_example('relu');
% save('cnn_relu', 'cnn');

% cnn_scale('noisy_softplus'); % noisy_softplus
% [cnn, er_softplus] = cnn_relu_example('noisy_softplus'); % noisy_softplus

% save('scale_noisy_softplus_6', 'cnn');

% clear;
% [cnn, er_softplus] = cnn_relu_example('softplus'); % noisy_softplus
% save('scale_softplus_30', 'cnn');

load scale_noisy_softplus_30.mat
cnn = cnn_scale('noisy_softplus',cnn);
% [cnn, er_softplus] = cnn_relu_example('noisy_softplus'); % noisy_softplus