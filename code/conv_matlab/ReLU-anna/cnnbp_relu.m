function net = cnnbp_relu(net, y, trans, bias)
    n = numel(net.layers);

    %   error
    %   softmax layer
    exp_o = exp(net.o);
    prob_o = exp_o;
    for i = 1:size(net.o, 2)
        prob_o(:,i) = exp_o(:,i)/sum(exp_o(:,i));
    end
    
    net.e = prob_o - y;
    %  loss function
    net.L = sum(sum(y .* log(prob_o))) / size(net.e, 2) * -1;
    net.od = net.e; 

    
    %%  backprop deltas
        
%     net.e = net.o - y;
%     %  loss function
%     net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
%     if strcmp(trans, 'relu') %|| strcmp(trans, 'softplus') 
%         net.od = net.e .* double(net.o>0);
%     elseif strcmp(trans, 'softplus')
%         net.od = net.e .* logistic(net.o);
%     elseif strcmp(trans, 'sigm')
%         net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
%     elseif strcmp(trans, 'noisy_softplus') 
%         net.od = net.e .* noisy_logistic(net.o, net.noise_o);
%     end

    net.fvd = (net.ffW' * net.od);              %  feature vector delta
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        if strcmp(trans, 'relu')
            net.fvd = net.fvd .* double(net.fv>0);% (net.fv .* (1 - net.fv)); % double(net.fv>0); % 
        elseif strcmp(trans, 'softplus')
            net.fvd = net.fvd .* logistic(net.fv);
        elseif strcmp(trans, 'sigm')
            net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
        elseif strcmp(trans, 'noisy_softplus')
            net.fvd = net.fvd .* noisy_logistic(net.fv, net.noise_fv);
        end
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                if strcmp(trans, 'relu') %|| strcmp(trans, 'softplus')
                    net.layers{l}.d{j} = double(net.layers{l}.a{j}>0)...
                        .* (expand(net.layers{l + 1}.d{j}, ...
                            [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) ...
                            / net.layers{l + 1}.scale ^ 2);
                elseif strcmp(trans, 'softplus')
                    net.layers{l}.d{j} = logistic(net.layers{l}.a{j})...
                        .* (expand(net.layers{l + 1}.d{j}, ...
                            [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) ...
                            / net.layers{l + 1}.scale ^ 2);
                elseif strcmp(trans, 'noisy_softplus') 
                    net.layers{l}.d{j} = noisy_logistic(net.layers{l}.a{j}, net.layers{l}.noise{j})...
                        .* (expand(net.layers{l + 1}.d{j}, ...
                            [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) ...
                            / net.layers{l + 1}.scale ^ 2);
                elseif strcmp(trans, 'sigm')
                    net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
                end
                
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                    z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                if bias
                    net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
                end
%                 net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    if bias
        net.dffb = mean(net.od, 2);
    end
%     net.dffb = 0;%mean(net.od, 2);
    
    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end




