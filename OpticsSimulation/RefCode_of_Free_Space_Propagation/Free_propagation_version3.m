function [Amp_all_prop] = Free_propagation_version3(Amp_all,z,X,Y)

%% Prepare
Amp_all_prop = zeros(size(Amp_all));

%% Calculation
for i=1:size(Amp_all,3)
    Amp_all_prop(:,:,i) = Rayleigh_Sommerfeld_SingleWavelength_version3(Amp_all(:,:,i),z,X,Y);
%     disp([num2str(i),' free propagation completed...']);
end

end

