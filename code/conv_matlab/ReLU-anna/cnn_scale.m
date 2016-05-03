function [ cnn ] = cnn_scale( trans, cnn )
%CNN_SCALE Summary of this function goes here
%   Detailed explanation goes here
    load mnist_uint8;
%     load cnn_softplus_max;
    train_x = double(reshape(train_x',28,28,60000))/255;
    test_x = double(reshape(test_x',28,28,10000))/255;
    train_y = double(train_y');
    test_y = double(test_y');

    bias = false;
    cnn = cnn_normalise(cnn, train_x(:,:,1:1000), trans); %(:,:,1:10000)
    save('scale5', 'cnn');
    [er, bad] = cnn_relu_test(cnn, test_x, test_y, trans, bias);
    fprintf('Testing Accuracy: %2.2f%%.\n', (1-er)*100);

    % plot mean squared error
    figure; plot(cnn.rL);
    xlabel('Steps')
    ylabel('Accuracy')
    title(trans)

end

