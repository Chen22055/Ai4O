function [U] = Rayleigh_Sommerfeld_SingleWavelength(U0,z,X,Y)
%% This function is to do Rayleigh-Sommerfeld intergral
%% Input: lambda in RS formula is the dimensionless length scale
%% Input: X and Y is meshgrid(xlist,ylist), all in dimensionless form
%% Input: U0 is the amplitude distrubution at the diffraction source
%% Input: z is the propagation distance, in dimensionless form

%% Prepare
k = 2*pi;
h = (z/1i) .* exp(1i*k*sqrt(z.^2 + X.^2 + Y.^2))./(z.^2+X.^2+Y.^2); %convolution kernel

%% Calculate
U = fftshift(ifft2(fft2(U0) .* fft2(h)));
% U = conv2(U0,h,'same');


end

