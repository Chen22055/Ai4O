function [U] = Lens_version1(U0,f,X,Y)
%% This function is to simulate a single lens
%% Input: lambda in RS formula is the dimensionless length scale
%% Input: X and Y is meshgrid(xlist,ylist), all in dimensionless form
%% Input: U0 is the amplitude distrubution at the diffraction source
%% Input: f is the lens focal length, in dimensionless form

%% Prepare
k = 2*pi;
t = exp(-1i * k * (X.^2 + Y.^2)/2/f);
%% Calculate
U = repmat(t,1,1,size(U0,3)) .* U0;

end

