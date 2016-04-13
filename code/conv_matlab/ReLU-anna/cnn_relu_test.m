function [er, bad] = cnn_relu_test(net, x, y, trans, bias)
    %  feedforward
    net = cnnff_relu(net, x,trans, bias);
    [~, h] = max(net.o);
    [~, a] = max(y);
    bad = find(h ~= a);

    er = numel(bad) / size(y, 2);
end