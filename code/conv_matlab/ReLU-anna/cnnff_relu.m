function net = cnnff_relu(net, x, trans, bias)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;

    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                noise = z;
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    if strcmp(trans, 'noisy_softplus')
                        mean_v = convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                        std_n = sqrt(convn(abs(net.layers{l - 1}.a{i}), net.layers{l}.k{i}{j}.^2, 'valid'));
                        v = noisy_softplus(mean_v, std_n);
                        z = z + v;
                        noise = noise + std_n;
                    else
                        z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                    end
                end
                if bias
                    z = z + net.layers{l}.b{j};
                end  
                
                if strcmp(trans, 'sigm')
                    net.layers{l}.a{j} = sigm(z);
                elseif strcmp(trans, 'relu')
                    net.layers{l}.a{j} = relu(z);
                elseif strcmp(trans, 'softplus')
                    net.layers{l}.a{j} = softplus(z);
                elseif strcmp(trans, 'noisy_softplus')
                    net.layers{l}.a{j} = z;
                    net.layers{l}.noise{j} = noise;
%                 %  add bias, pass through nonlinearity
%                 net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
                end

            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable

                if strcmp(trans, 'relu')
                    z = relu(z);
                elseif strcmp(trans, 'softplus')
                    z = softplus(z);
                elseif strcmp(trans, 'noisy_softplus')
                    noise = sqrt(convn(abs(net.layers{l - 1}.a{j}), ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 4), 'valid'));
                    z = noisy_softplus(z, noise);
                    net.layers{l}.noise{j} = noise(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                end
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    net.noise_fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
        if strcmp(trans, 'noisy_softplus')
            net.noise_fv = [net.noise_fv; reshape(net.layers{n}.noise{j}, sa(1) * sa(2), sa(3))];
        end
    end
    %  feedforward into output perceptrons
    if bias
        z = net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2));
    else
        z = net.ffW * net.fv;
    end
    if strcmp(trans, 'sigm')
        net.o = sigm(z);
    elseif strcmp(trans, 'relu')
        net.o = relu(z);
    elseif strcmp(trans, 'softplus')
        net.o = softplus(z);
    elseif strcmp(trans, 'noisy_softplus')
        noise = sqrt(net.ffW.^2 * abs(net.fv));
        net.o = noisy_softplus(z, noise);
        net.noise_o = noise;
    end
        
end