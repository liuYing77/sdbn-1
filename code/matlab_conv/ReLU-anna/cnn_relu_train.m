function net = cnn_relu_train(net, x, y, opts, trans, bias)
%CNN_RELU_TRAIN Summary of this function goes here
%   Detailed explanation goes here
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));

            net = cnnff_relu(net, batch_x, trans, bias);                       
%             net = cnnff_dan(net, batch_x);
            
            net = cnnbp_relu(net, batch_y, trans, bias);
%             net = cnnbp_dan(net, batch_y);
            
            net = cnnapplygrads_relu(net, opts, bias);
%             net = cnnapplygrads_dan(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end

end

