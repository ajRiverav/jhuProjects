function [z]=evalPointInNormalDistro(x,m,S)
%evalPointInNormalDistro Estimate the ML parameters of a multivariate
%normal distribution
% INPUT ARGS
%   x:  l-dimensional column vector where the value of the normal
%       distribution will be evaluated.
%   m:  l-dimensional column vector corresponding to the mean vector of the
%       normal distribution.
%   S:  lxl matrix that corresponds to the covariance matrix of the 
%       normal distribution.
%
% OUTPUT ARGS
%   z:  the value of the normal distribution at x.
%
% 2014-SEPT AJ Rivera aj.rivera@jhu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l=size(m,1);
z=(2*pi)^(-l/2)*det(S)^(-1/2) * exp(-0.5*(x-m)'*inv(S)*(x-m)); %#ok<MINV>

end