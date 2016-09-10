function [mHat,SHat]=calcMLEstimatesNormalDistro(data)
%CALCMLESTIMATESNORMALDISTRO Estimate the ML parameters of a multivariate
%normal distribution
% INPUT ARGS
%   data:      lxN  matrix, whose columns are the data vectors.
%
% OUTPUT ARGS
%   mHat:  l-dimensional estimate of the mean vector of the distribution.
%   SHat:  lxl estimate of the covariance matrix of the distribution.
%
% 2014-SEPT AJ Rivera aj.rivera@jhu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[l,N]=size(data);

mHat=(1/N)*sum(data,2);
SHat=zeros(l);
for k=1:N
    SHat=SHat+(data(:,k)-mHat)*(data(:,k)-mHat)';
end
SHat=(1/N)*SHat;
end