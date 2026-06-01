function [U_airy] = Airy_disk(waist,P,X,Y)
%% All inputs are in dimensionless form

%% Calculate
r = sqrt(X.^2 + Y.^2);
U_airy = P * 2*besselj(1,1.22*pi*r/waist) ./ (1.22*pi*r/waist);

end

