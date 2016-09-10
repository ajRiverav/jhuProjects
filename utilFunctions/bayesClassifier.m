function [z]=bayesClassifier(X,m,S,P)
%bayesClassifier Use a Beyesian classification rule to classify data
% INPUT ARGS
%   m:      lxc matrix, whose j-th column is the mean of the j-th class.
%   S:      lxlxc matrix, where S(:,:,j) corresponds to
%           the covariance matrix of the normal distribution of the j-th
%           class.
%   P:      c-dimensional vector, whose j-th component is the a priori
%           probability of the j-th class.
%   X:      lxN matrix, whose columns are the data vectors to be
%           classified.
%
% OUTPUT ARGS
%   z:      N-dimensional vector, whose i-th element is the label
%           of the class where the i-th data vector is classified.
%
% 2014-SEPT AJ Rivera aj.rivera@jhu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[~,c]=size(m);
[~,N]=size(X);

z(N)=0;
post(c)=0;

for i=1:N
    for j=1:c
        post(j)=P(j)*evalPointInNormalDistro(X(:,i),m(:,j),S(:,:,j));
    end
    [num,z(i)]=max(post);
end
end