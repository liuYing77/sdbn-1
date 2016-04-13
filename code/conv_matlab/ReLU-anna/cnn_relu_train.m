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
%         min_o = 0;
%         max_o = 0;
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            net = cnnff_relu(net, batch_x, trans, bias);                       
%             net = cnnff(net, batch_x);
            
            net = cnnbp_relu(net, batch_y, trans, bias);
%             net = cnnbp(net, batch_y);
            
            net = cnnapplygrads_relu(net, opts, bias);
%             net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
%             net.rL(end + 1) = net.L;
            
%             if net.rL(end) < 0.1
%                 opts.alpha = [1 1 1 1 1] * 0.006;
%             end
            if net.rL(end) < 0.001
                return;
            end
            if net.rL(end) < 0.02
                net.rL(end)
            end

%             disp(['numbatch: ' num2str(l) ', loss: ' num2str(net.L)]);
            disp(['numbatch: ' num2str(l) ', loss: ' num2str(net.L)...
                ', out: ' num2str(mean(mean(net.o))),...
                ', 3a: ' num2str(mean(mean(mean(net.layers{3}.a{1})))),...
                ', 4k: ' num2str(net.layers{4}.k{1}{1}(1)),...
                ', 4a: ' num2str(mean(mean(mean(net.layers{4}.a{1})))),...
                ', 5d: ' num2str(net.layers{5}.d{1}(1)),...
                ', 5a: ' num2str(mean(mean(mean(net.layers{5}.a{1}))))
                ])
%               min_o = min_o + min(min(net.o));
%               max_o = max_o + max(max(net.o));
%               disp(['numbatch: ' num2str(l) ', out_min:' num2str(min(min(net.o))) ', out_max:' num2str(max(max(net.o)))]);          
%             ', 2d: ' num2str(mean(mean(mean(net.layers{2}.d{1})))),...
%                 ', 3d: ' num2str(mean(mean(mean(net.layers{3}.d{1})))),...
%                 ', 4d: ' num2str(mean(mean(mean(net.layers{4}.d{1})))),...
%                 ', 5d: ' num2str(mean(mean(mean(net.layers{5}.d{1})))),...
            
            
%             disp(['numbatch: ' num2str(l) ', loss: ' num2str(net.L)...
%                 ', 2d: ' num2str(net.layers{2}.d{1}(1)),...
%                 ', 2dk: ' num2str(net.layers{4}.dk{1}{1}(1)),...
%                 ', 4d: ' num2str(net.layers{4}.d{1}(1)),...
%                 ', 4k: ' num2str(net.layers{4}.k{1}{1}(1)),...
%                 ', 4dk: ' num2str(net.layers{4}.dk{1}{1}(1))...
%                 ])
        end
        toc;
%         min_o =  min_o / numbatches;
%         max_o =  max_o / numbatches;
%         disp([num2str(min_o), num2str(max_o)]);
    end

end

