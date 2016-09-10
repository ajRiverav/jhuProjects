function [z]=euclideanDistClassifier(m,X)
%EUCLIDEANDISTCLASSIFIER Classify accordingly to the Euclidean distance metric
% INPUT ARGS
%   m:  lxc matrix, whose i-th column corresponds to the mean of the c-th
%       class.
%   X:  lxN matrix whose columns are the data vectors to be classified.
%
% OUTPUT ARGS
%   z:  N-dimensional vector whose i-th element contains the label
%       of the class where the i-th data vector has been assigned.
%
% 2014-SEPT AJ Rivera aj.rivera@jhu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,c]=size(m);
[~,N]=size(X);

z(N)=0;
eucDist(c)=0;

for i=1:N
    for j=1:c
        eucDist(j)=sqrt((X(:,i)-m(:,j))'*(X(:,i)-m(:,j)));
    end
    [num,z(i)]=min(eucDist);
    
end
end