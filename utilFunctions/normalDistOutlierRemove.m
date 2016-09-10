function [outliers,idx,prunedData]=normalDistOutlierRemove(data,thresh,useMean)
%NORMALDISTOUTLIERREMOVE Remove outliers from normally distributed dataset
% Detects and removes outliers from a normally distributed dataset by
% means of a thresholding technique. The threshold depends on
% the mean or median value and standard deviation of the dataset.
% INPUT ARGS
%   data:       normally distributed data.
%   thresh:     threshold defined as number of standard deviations 
%   useMean:    flag indicating whether the mean should be used. 
%
% OUTPUT ARGS
%   outliers:        outliers that have been detected.
%   idx:             indices of the outliers in the input data matrix.
%   prunedData:      dataset after outliers have been removed.
%
% 2014-SEPT AJ Rivera aj.rivera@jhu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargin<3 || isempty(useMean))
    useMean=false;
end

if(useMean)
    m=mean(data);
else
    m=median(data);
end
%use the unbisased standard deviation estimator
sd=std(data);
idx=find(data>m+(thresh*sd) | data<m-thresh*sd);
outliers=data(idx);
prunedData=data;
prunedData(idx)=[];
end