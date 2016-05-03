function [ y ] = noisy_logistic( x, noise )
%NOISY_LOGISTIC Summary of this function goes here
%   Detailed explanation goes here
    a = noise/3;
    b = ones(size(a));
    ind = find(a>0);
    b(ind) = 1./a(ind);
    y = ones(size(x));
    ind = find(x<30 & x>-30);
    y(ind) = a(ind) .* b(ind) .* exp(b(ind).*x(ind))./(1+exp(b(ind).*x(ind)));
    y(x<=-30) = 0;

end

