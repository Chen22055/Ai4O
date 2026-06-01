function [U_gau] = Gaussian_beam(waist,P,X,Y)
%% This function is to produce a Gaussian-beam amplitude distribution
%% Input: waist in dimensionless form
%% Input: P is a relative intensity number(not important)
%% Input: X and Y is meshgrid(xlist,ylist), all in dimensionless form

%% Prepare
xlist = X(1,:); lenx = length(xlist);
ylist = Y(:,1); leny = length(ylist);

% if mod(lenx,2)==0
%     centx = (xlist(lenx/2) + xlist(lenx/2+1))/2;
% else
%     centx = xlist((lenx+1)/2);
% end
% 
% if mod(leny,2)==0
%     centy = (ylist(leny/2) + ylist(leny/2+1))/2;
% else
%     centy = ylist((leny+1)/2);
% end
centx = 0;
centy = 0;

%% Calculate
U_gau = sqrt(2*P/pi) * (1/waist) * exp(-((X-centx).^2+(Y-centy).^2)/waist^2);


end

