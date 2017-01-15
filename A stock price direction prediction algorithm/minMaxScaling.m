function [scaledData, minDatIn, minVal, maxVal] = minMaxScaling(datIn,lowerBound,upperBound)
%MINMAXSCALING  Scale data using the mixmax method with specified upper and lower
%               bounds. 

%TODO: more details and explain how to reconstruct exemplars
% 2016-4-17 AJ RIVERA, Johns Hopkins University

minDatIn= min(datIn(:));
dataOut = datIn - minDatIn;
maxVal=max(dataOut);
minVal=min(dataOut);
rangeVals=maxVal-minVal;
dataOut = (dataOut/rangeVals)*(upperBound-lowerBound);
scaledData = dataOut + lowerBound;

end

