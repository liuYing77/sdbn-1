function y = logistic( x )
%LOGISTIC Summary of this function goes here
%   Detailed explanation goes here
    a = 1;%3;
    b = 1;%1/3;
    %a = 0.2;
    %b = 5.0;
    y = a * b * ones(size(x));
    ind = find(x<30 & x>-30);
    y(ind) = a * b * exp(b*x(ind))./(1+exp(b*x(ind)));
    y(x<=-30) = 0;
    


%     y = double(x>0);
end

