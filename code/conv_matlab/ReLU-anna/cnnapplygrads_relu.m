function net = cnnapplygrads_relu(net, opts, bias)
    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha(l) * net.layers{l}.dk{ii}{j};
                end
                % no bias
%                 net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
            end
        end
    end

    net.ffW = net.ffW - opts.alpha(numel(net.layers)) * net.dffW;
    if bias
        net.ffb = net.ffb - opts.alpha(numel(net.layers)) * net.dffb;
    end
end
