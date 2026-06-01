function [U] = Rayleigh_Sommerfeld_SingleWavelength_version5(U0,z,X,Y)
%% This function is to do Rayleigh-Sommerfeld intergral
%% Input: lambda in RS formula is the dimensionless length scale
%% Input: X and Y is meshgrid(xlist,ylist), all in dimensionless form
%% Input: U0 is the amplitude distrubution at the diffraction source
%% Input: z is the propagation distance, in dimensionless form

%% Prepare
k = 2*pi;
h = (z/1i) .* exp(1i*k*sqrt(z.^2 + X.^2 + Y.^2))./(z.^2+X.^2+Y.^2); %convolution kernel

%% Calculate
% padrows = size(U0,1) + size(h,1) - 1;
% padcols = size(U0,2) + size(h,2) - 1;
% 
% pad_U0_1 = round((padrows-size(U0,1))/2)*2;
% pad_U0_2 = round((padcols-size(U0,2))/2)*2;
% pad_h_1 = round((padrows-size(h,1))/2)*2;
% pad_h_2 = round((padcols-size(h,2))/2)*2;
% 
% U0_pad = padarray(U0,[pad_U0_1,pad_U0_2],'both');
% h_pad = padarray(h,[pad_h_1,pad_h_2],'both');
% 
%% Add kernel
% x_list = X(1,:);
% y_list = Y(:,1)';
% x_list_new = linspace(x_list(1)*size(h_pad,2)/size(h,2),x_list(end)*size(h_pad,2)/size(h,2),size(h_pad,2));
% y_list_new = linspace(y_list(1)*size(h_pad,1)/size(h,1),y_list(end)*size(h_pad,1)/size(h,1),size(h_pad,1));
% [X_new,Y_new] = meshgrid(x_list_new,y_list_new);
% h_pad = (z/1i) .* exp(1i*k*sqrt(z.^2 + X_new.^2 + Y_new.^2))./(z.^2+X_new.^2+Y_new.^2); %convolution kernel

%% convolution by fft2
% U_pad = conv2(U0_pad,h_pad,'same');
% U = U_pad(pad_U0_1+1:pad_U0_1+size(U0,1),pad_U0_2+1:pad_U0_2+size(U0,2));

U = conv2(U0,h,'same');

end

