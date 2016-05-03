function [ y ] = noisy_softplus( x, noise )
%NOISY_SOFTPLUS Summary of this function goes here
%   Detailed explanation goes here
    a = noise/3;
    b = ones(size(a));
    ind = find(a>0);
    b(ind) = 1./a(ind);
    y = x;
    ind = find(x<10 & x>-10);
    y(ind) = a(ind).*log( 1 + exp(b(ind).*x(ind)) );
    y(x<=-10) = 0;

end

