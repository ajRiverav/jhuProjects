function [ scaledDataVec ] = scaleWithMinMax( dataVec,bottom,top )
%scaleWithMinMax Scales data such that every point falls
%				 in the range of [bottom,top] values
%
% 2015-March, AJ Rivera, aj.rivera@jhu.edu

if(~exist('bottom','var')),bottom=-1;end
if(~exist('top','var')),top=1;end

%guard against inf and -inf
infIdx=find(isinf(dataVec)==1);%where are inf and -inf
validIdx=setxor(infIdx,1:length(dataVec));%discard inf/-inf idx

for i=1:length(infIdx)%replace every inf or -inf with max or min
    if(dataVec(infIdx(i))>0)
        dataVec(infIdx(i))=2*max(dataVec(validIdx));%replace with max
    else
        dataVec(infIdx(i))=2*min(dataVec(validIdx));
    end
end

%proceed with scaling
maxVal = max(dataVec);
minVal = min(dataVec);

scaledDataVec = ...
    (top-bottom)/(maxVal-minVal)*(dataVec-maxVal)+top;

end

