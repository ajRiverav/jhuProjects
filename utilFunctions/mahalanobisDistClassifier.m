function z=mahalanobisDistClassifier(m,S,X)
%mahalanobisDistClassifier Classify accordingly to the Mahalanobist distance metric
% INPUT ARGS
%   m:  lxc matrix, whose i-th column corresponds to the
%       mean of the i-th class
%   S:  lxl matrix which corresponds to the matrix
%       involved in the Mahalanobis distance (when the classes have
%       the same covariance matrix, S equals to this common covariance
%       matrix).
%   X:  lxN matrix, whose columns are the data vectors to be classified.
%
% OUTPUT ARGS
%   z:  N-dimensional vector whose i-th component contains the label
%       of the class where the i-th data vector has been assigned.
%
% 2014-SEPT AJ Rivera aj.rivera@jhu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,c]=size(m);
[~,N]=size(X);

z(N)=0;
mahaDist(c)=0;

for i=1:N
    for j=1:c
        mahaDist(j)=sqrt((X(:,i)-m(:,j))'*inv(S)*(X(:,i)-m(:,j))); %#ok<MINV>
    end
    [~,z(i)]=min(mahaDist);
end
end