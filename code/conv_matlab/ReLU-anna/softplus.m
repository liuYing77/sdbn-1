function y = softplus( x, a, b, bias )
%SOFTPLUS Summary of this function goes here
%   Detailed explanation goes here
%     y = log( 1 + exp(x) );
    switch nargin
        case 1
            a = 0.2;
            b = 5.0;
            bias = false;
        case 2
            b = 5.0;
            bias = false;
        case 3
            bias = false;
    end
    
    a = 1;%3;
    b = 1;%1/3;
    y = a * b * x;
    ind = find(x<10 & x>-10);
    y(ind) = a * log( 1 + exp(b * x(ind)) );
    y(x<=-10) = 0;
    
    if bias
        y = y - a * log(2);
    end
    

%     y = x;
%     y(y<0) = 0;
end

