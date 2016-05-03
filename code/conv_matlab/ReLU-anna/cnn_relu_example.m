function [cnn, er] = cnn_relu_example( trans )
%CNN_RELU_EXAMPLE Summary of this function goes here
%   Detailed explanation goes here
    load mnist_uint8;
    
    train_x = double(reshape(train_x',28,28,60000))/255;
    test_x = double(reshape(test_x',28,28,10000))/255;
    train_y = double(train_y');
    test_y = double(test_y');

    % Set the derivative to be the binary derivative of a ReLU

    %% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
    %will run 1 epoch in about 200 second and get around 11% error. 
    %With 100 epochs you'll get around 1.2% error
    rand('state',1)
    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %subsampling layer
    };

    bias = false; %true;

    cnn = cnn_relu_setup(cnn, train_x, train_y, trans);
    if strcmp(trans, 'softplus') || strcmp(trans, 'noisy_softplus') 
        opts.alpha = [1 1 1 1 0.1] * 0.05;
%         opts.alpha = [1 1 1 1 1] * 0.06;
%         opts.alpha = [1 1 1 1 1] * 0.005;
    elseif strcmp(trans, 'relu')
        opts.alpha = [1 1 1 1 1];
    elseif strcmp(trans, 'sigm')
        opts.alpha = [1 1 1 1 1];
    end
    opts.batchsize = 50;
    opts.numepochs = 5;
    
    cnn = cnn_relu_train(cnn, train_x, train_y, opts, trans, bias);
    [er, bad] = cnn_relu_test(cnn, test_x, test_y, trans, bias);
    fprintf('Testing Accuracy: %2.2f%%.\n', (1-er)*100);

    % plot mean squared error
    figure; plot(cnn.rL);
    xlabel('Steps')
    ylabel('Accuracy')
    title(trans)

end

