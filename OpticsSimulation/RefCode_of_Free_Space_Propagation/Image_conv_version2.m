function [U] = Image_conv_version2(U0,delta,X,Y,d)
%% This function is to do Rayleigh-Sommerfeld intergral
%% Input: lambda in RS formula is the dimensionless length scale
%% Input: X and Y is meshgrid(xlist,ylist), all in dimensionless form
%% Input: U0 is the amplitude distrubution at the diffraction source
%% Input: delta is the resolution, in dimensionless form
%% Input: d is in dimensionless form
%% Different from version1, this version considers the propagation of convolution kernel
physical_constants;
%% Prepare
r = sqrt(X.^2 + Y.^2);

h = 2*besselj(1,1.22*pi*r/delta) ./ (1.22*pi*r/delta); %convolution kernel for Intensity convolving
h = Rayleigh_Sommerfeld_SingleWavelength_version3(h,d,X,Y);
h = abs(h).^2;

% P = abs(fftshift(ifft2(h)));
% P_new = P .* exp(1i * pi * r.^2 * d / (0.2/lamb0) / (0.015/lamb0));
% 
% h = fftshift(abs(fft2(P_new)).^2);

%% Debug
figure;
m = mesh(h); view(0,90); m.FaceColor='interp'; m.EdgeColor='interp';

%% Calculate
% padrows = size(U0,1) + size(h,1) - 1;
% padcols = size(U0,2) + size(h,2) - 1;
% 
% pad_U0_1 = round((padrows-size(U0,1))/2)*2;
% pad_U0_2 = round((padcols-size(U0,2))/2)*2;
% pad_h_1 = round((padrows-size(h,1))/2)*2;
% pad_h_2 = round((padcols-size(h,2))/2)*2;
% 
% 
% U0_pad = padarray(U0,[pad_U0_1,pad_U0_2],'both');
% h_pad = padarray(h,[pad_h_1,pad_h_2],'both');
% 
% U_pad = fftshift(ifft2(fft2(U0_pad) .* fft2(h_pad)));
% U = U_pad(pad_U0_1+1:pad_U0_1+size(U0,1),pad_U0_2+1:pad_U0_2+size(U0,2));

U = conv2(U0,h,'same');

end

