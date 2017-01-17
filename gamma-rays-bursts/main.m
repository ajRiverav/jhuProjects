function []=main()
clear all, close('all')
clc

%load data
load('GRBdata.mat')

%feat(1) is really column 5)
for i=5:length(FeatureLabels)
    featIdx(i-4)=i;
end
datOrig=GRBFeatures;clear GRBFeatures;%use short var name


%% VISUALIZE DATA
for fIdx=featIdx 
    figure;hist(datOrig(:,fIdx));
    title([' Feature #' num2str(fIdx)]);
end

disp('Click continue to close all figures and continue')
keyboard
close all
%% NORMALIZE WITH normalizeStd
dat=normalizeStdAJ(datOrig(:,featIdx(1):featIdx(end)));
dat=[datOrig(:,1:4),dat];%ONLY overwrite some of the columns

%% MBSAS Clustering
%% MBSAS Clustering
close all

caseIdx{1}=[5 7 8 9 18];
caseIdx{2}=[6 7 9 10 11 13];
caseIdx{3}=[8 9 12 13 14 16];
caseIdx{4}=5:8;
figure;hold on
h=waitbar(0,'Computing num of clusters vs theta...');
minTheta=1;maxTheta=10;%define range
clusters_mbsas_case1(maxTheta-minTheta,99)=0;
clusters_mbsas_case2(maxTheta-minTheta,99)=0;
clusters_mbsas_case3(maxTheta-minTheta,99)=0;
clusters_mbsas_case4(maxTheta-minTheta,99)=0;
i=1;
tic
for thetaIdx=minTheta:maxTheta
    waitbar(i/(maxTheta-minTheta),h,'Computing number of clusters vs. theta...')
    clusters_mbsas_case1(i,:)=MBSAS(dat(:,featIdx(caseIdx{1})),thetaIdx,50);
    clusters_mbsas_case2(i,:)=MBSAS(dat(:,featIdx(caseIdx{2})),thetaIdx,50);
    clusters_mbsas_case3(i,:)=MBSAS(dat(:,featIdx(caseIdx{3})),thetaIdx,50);
    clusters_mbsas_case4(i,:)=MBSAS(dat(:,featIdx(caseIdx{4})),thetaIdx,50);
    i=i+1;
end
toc
close(h)
%load('mbsas.mat');
%plot num of clusters vs. theta for all cases. 
plot(minTheta:maxTheta,max(clusters_mbsas_case1,[],2),...
    minTheta:maxTheta,max(clusters_mbsas_case2,[],2),'r',...
    minTheta:maxTheta,max(clusters_mbsas_case3,[],2),'black',...
    minTheta:maxTheta,max(clusters_mbsas_case4,[],2),'green')
legend('Features:[5 7 8 9 18]','Features [6 7 9 10 11 13]','Features [8 9 12 13 14 16]',...
   'Features [5 6 7 8]');
xlim([2 8]);xlabel('Theta');ylabel('Number of Clusters')
title('MBSAS Algorithm')

%pick best theta for every case
mbsas_theta_case1=5;labels.mbsas{1}=clusters_mbsas_case1(mbsas_theta_case1,:);
mbsas_theta_case2=4;labels.mbsas{2}=clusters_mbsas_case2(mbsas_theta_case2,:);
mbsas_theta_case3=5;labels.mbsas{3}=clusters_mbsas_case3(mbsas_theta_case3,:);
mbsas_theta_case4=3;labels.mbsas{4}=clusters_mbsas_case4(mbsas_theta_case4,:);

%get num of "intrinsic" clusters
for i=1:length(caseIdx)
    numClusters.mbsas(i)=max(labels.mbsas{i});
end
%compute avg. J3 value for each case
for i=1:size(caseIdx,2)%for every set of features
    allComb=combnk(unique(labels.mbsas{i}),2);%get all possible combinations
    for j=1:size(allComb,1)%for every combination pair, compute j3
        comb=allComb(j,:);%get current combination pair
        class1=datOrig(labels.mbsas{i}==comb(1),caseIdx{i})';
        class2=datOrig(labels.mbsas{i}==comb(2),caseIdx{i})';
        J3(j)=ScatterMatrices(class1,class2);
%         figure;scatter3(datOrig(:,caseIdx{i}(1)),...
%             datOrig(:,caseIdx{i}(4)),...
%             datOrig(:,caseIdx{i}(5)),...
%             25,labels.mbsas{i},'filled');grid on;
    end
    avgJ3.mbsas(i)=nanmean(J3);
    clear J3%self note= J3=the bigger, the better=
    %large between-class distance, small within-class variance
end


%% K-MEANS CLUSTERING
for m=1:50%for every num of cluster "hypothesis"
    for i=1:size(caseIdx,2) %for every set of features
        theta_init=rand(size(caseIdx{i},2),m);
        [theta{m,i},labels.kMeans{m,i},JkMeans{m,i}]=...
            k_means(dat(:,caseIdx{i})',theta_init);
    end
end
%check which m minimizes J using different feature sets
figure;plot(2:50,[JkMeans{2:end,1}],2:50,[JkMeans{2:end,2}],'r',...
    2:50,[JkMeans{2:end,3}],'black',2:50,[JkMeans{2:end,4}],'green');
legend('Features:[5 7 8 9 18]','Features [6 7 9 10 11 13]','Features [8 9 12 13 14 16]',...
   'Features [5 6 7 8]');
title('k-Means algorithm');xlabel('Num of Clusters');ylabel('Jm')

clear J3
%compute avg. J3 value for each case,
for m=2:50 %for every m
    for i=1:size(caseIdx,2)%for every set of features
    
        allComb=combnk(unique(labels.kMeans{m,i}),2);%get all possible combinations
        for j=1:size(allComb,1)%for every combination pair, compute j3
            comb=allComb(j,:);%get current combination pair
            class1=dat(labels.kMeans{m,i}==comb(1),caseIdx{i})';
            class2=dat(labels.kMeans{m,i}==comb(2),caseIdx{i})';
            J3{i,j,m}=ScatterMatrices(class1,class2);
        end
        avgJ3.kMeans(m,i)=nanmean([J3{i,:,m}]);
    end
    
    
end

%check with m maximizes J3
figure;plot(2:50,10*log10(avgJ3.kMeans(2:50,1)),...
    2:50,10*log10(avgJ3.kMeans(2:50,2)),'r',...
    2:50,10*log10(avgJ3.kMeans(2:50,3)),'black',...
    2:50,10*log10(avgJ3.kMeans(2:50,4)),'green')
legend('Features:[5 7 8 9 18]','Features [6 7 9 10 11 13]','Features [8 9 12 13 14 16]',...
   'Features [5 6 7 8]');
title('k-Means algorithm');xlabel('Num of Clusters');ylabel('avg. J3 (dB)')
xlim([3 10])
%% FUZZY C-MEANS CLUSTERING
q=1.5;%fuzzifier (>1)
for m=2:50%for every num of cluster "hypothesis" 
    for i=1:size(caseIdx,2) %for every set of features
        [thetaFcm{m,i},memship{m,i},Jfcm{m,i}]=...
            fuzzy_c_means(dat(:,caseIdx{i})',m,q);
    end
end

%Build labels
for m=2:50
    for i=1:size(caseIdx,2)
        [~,labels.fcm{m,i}]=max(memship{m,1},[],2);%highest membership is cluster idx
    end
end

clear J3
%compute avg. J3 value for each case,
for m=2:50 %for every m
    for i=1:size(caseIdx,2)%for every set of features
    
        allComb=combnk(unique(labels.fcm{m,i}),2);%get all possible combinations
        for j=1:size(allComb,1)%for every combination pair, compute j3
            comb=allComb(j,:);%get current combination pair
            class1=dat(labels.fcm{m,i}==comb(1),caseIdx{i})';
            class2=dat(labels.fcm{m,i}==comb(2),caseIdx{i})';
            J3{i,j,m}=ScatterMatrices(class1,class2);
        end
        avgJ3.fcm(m,i)=nanmean([J3{i,:,m}]);
    end
    
    
end

%check with m maximizes J3
figure;plot(2:50,10*log10(avgJ3.fcm(2:50,1)),...
    2:50,10*log10(avgJ3.fcm(2:50,2)),'r',...
    2:50,10*log10(avgJ3.fcm(2:50,3)),'black',...
    2:50,10*log10(avgJ3.fcm(2:50,4)),'green')
legend('Features:[5 7 8 9 18]','Features [6 7 9 10 11 13]','Features [8 9 12 13 14 16]',...
   'Features [5 6 7 8]');
title('Fuzzy C-Means algorithm');xlabel('Num of Clusters');ylabel('avg. J3 (dB)')
xlim([4 12])
disp('MBSAS Algorithm: Class separability using J3 is:')
disp(['   Case 1: ' num2str(10*log10(avgJ3.mbsas(1))) ' dB'])
disp(['   Case 2: ' num2str(10*log10(avgJ3.mbsas(2))) ' dB'])
disp(['   Case 3: ' num2str(10*log10(avgJ3.mbsas(3))) ' dB'])
disp(['   Case 4: ' num2str(10*log10(avgJ3.mbsas(4))) ' dB'])

disp('MBSAS Cluster distribution:')
for i=1:length(labels.mbsas)
    disp(['Case #' num2str(i)])
    for j=unique(labels.mbsas{i})
    disp(['  Cluster#' num2str(j) ':' num2str(sum(labels.mbsas{i}==j))])
    end
end

disp('k-Means Cluster distribution:')
idx=[8 6 8 4];
for i=1:4
    disp(['Case #' num2str(i)])
    for j=unique(labels.kMeans{idx(i),i})
    disp(['  Cluster#' num2str(j) ':' num2str(sum(labels.kMeans{idx(i),i}==j))])
    end
end

disp('Fuzzy C-Means Cluster distribution:')
idx=[7 9 7 6];
for i=1:4
    disp(['Case #' num2str(i)])
    for j=unique(labels.fcm{idx(i),i})'
    disp(['  Cluster#' num2str(j) ':' num2str(sum(labels.fcm{idx(i),i}==j))])
    end
end
disp('Click "exit debugging" to exit')
keyboard
return
function [classesNorm]=normalizeStdAJ(classes)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [classes]=normalizeStd(classes)
% Data normalization to zero mean and standard deviation 1,
% separately for each feature.
%
% INPUT ARGUMENTS:
%   classNorm:  dataset for class n, one pattern per column.
%
% OUTPUT ARGUMENTS:
%   classesNorm   normalized data for class n
%
% Modification History
%   Nov3 2014 - AJ Rivera
%       -created original
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:size(classes,2)%for every feautre

    meanOfFeature=mean(classes(:,i));
    stdOfFeature=std(classes(:,i));
    classesNorm(:,i)=(classes(:,i)-meanOfFeature)/stdOfFeature;
    
end

return
function [theta,U,obj_fun]=fuzzy_c_means(X,m,q)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [w, U, obj_fun]=fuzzy_c_means(X,m,q)
% This function applies the FCM algorithm by calling the corresponding
% MATLAB function, called "fcm". 
%
% INPUT ARGUMENTS:
%  X:   lxN matrix, each column of which corresponds to
%       an l-dimensional data vector.
%  m:   the number of clusters.
%  q:   the fuzzifier.
% 
% OUTPUT ARGUMENTS:
%  theta:   lxm matrix, each column of which corresponds to
%           a cluster representative, after the convergence of the
%           algorithm. 
%  U:       Nxm matrix containing in its i-th row
%           the ``grade of membership'' of the data vector xi to the m
%           clusters.  
%  obj_fun: a vector, whose t-th coordinate is the value of the cost
%           function, J, for the clustering produced at the t-th teration.
%
% (c) 2010 S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


X=X';
options(1)=q;
[theta,U,obj_fun] = fcm(X,m,options);
theta=theta';
U=U';
return
function out = distfcm(center, data)
%DISTFCM Distance measure in fuzzy c-mean clustering.
%	OUT = DISTFCM(CENTER, DATA) calculates the Euclidean distance
%	between each row in CENTER and each row in DATA, and returns a
%	distance matrix OUT of size M by N, where M and N are row
%	dimensions of CENTER and DATA, respectively, and OUT(I, J) is
%	the distance between CENTER(I,:) and DATA(J,:).
%
%       See also FCMDEMO, INITFCM, IRISFCM, STEPFCM, and FCM.

%	Roger Jang, 11-22-94, 6-27-95.
%       Copyright 1994-2002 The MathWorks, Inc. 
%       $Revision: 1.13 $  $Date: 2002/04/14 22:20:29 $

out = zeros(size(center, 1), size(data, 1));

% fill the output matrix

if size(center, 2) > 1,
    for k = 1:size(center, 1),
	out(k, :) = sqrt(sum(((data-ones(size(data, 1), 1)*center(k, :)).^2)'));
    end
else	% 1-D data
    for k = 1:size(center, 1),
	out(k, :) = abs(center(k)-data)';
    end
end

return
function [center, U, obj_fcn] = fcm(data, cluster_n, options)
%FCM Data set clustering using fuzzy c-means clustering.
%
%   [CENTER, U, OBJ_FCN] = FCM(DATA, N_CLUSTER) finds N_CLUSTER number of
%   clusters in the data set DATA. DATA is size M-by-N, where M is the number of
%   data points and N is the number of coordinates for each data point. The
%   coordinates for each cluster center are returned in the rows of the matrix
%   CENTER. The membership function matrix U contains the grade of membership of
%   each DATA point in each cluster. The values 0 and 1 indicate no membership
%   and full membership respectively. Grades between 0 and 1 indicate that the
%   data point has partial membership in a cluster. At each iteration, an
%   objective function is minimized to find the best location for the clusters
%   and its values are returned in OBJ_FCN.
%
%   [CENTER, ...] = FCM(DATA,N_CLUSTER,OPTIONS) specifies a vector of options
%   for the clustering process:
%       OPTIONS(1): exponent for the matrix U             (default: 2.0)
%       OPTIONS(2): maximum number of iterations          (default: 100)
%       OPTIONS(3): minimum amount of improvement         (default: 1e-5)
%       OPTIONS(4): info display during iteration         (default: 1)
%   The clustering process stops when the maximum number of iterations
%   is reached, or when the objective function improvement between two
%   consecutive iterations is less than the minimum amount of improvement
%   specified. Use NaN to select the default value.
%
%   Example
%       data = rand(100,2);
%       [center,U,obj_fcn] = fcm(data,2);
%       plot(data(:,1), data(:,2),'o');
%       hold on;
%       maxU = max(U);
%       % Find the data points with highest grade of membership in cluster 1
%       index1 = find(U(1,:) == maxU);
%       % Find the data points with highest grade of membership in cluster 2
%       index2 = find(U(2,:) == maxU);
%       line(data(index1,1),data(index1,2),'marker','*','color','g');
%       line(data(index2,1),data(index2,2),'marker','*','color','r');
%       % Plot the cluster centers
%       plot([center([1 2],1)],[center([1 2],2)],'*','color','k')
%       hold off;
%
%   See also FCMDEMO, INITFCM, IRISFCM, DISTFCM, STEPFCM.

%   Roger Jang, 12-13-94, N. Hickey 04-16-01
%   Copyright 1994-2002 The MathWorks, Inc. 
%   $Revision: 1.13 $  $Date: 2002/04/14 22:20:38 $

if nargin ~= 2 & nargin ~= 3,
	error('Too many or too few input arguments!');
end

data_n = size(data, 1);
in_n = size(data, 2);

% Change the following to set default options
default_options = [2;	% exponent for the partition matrix U
		100;	% max. number of iteration
		1e-5;	% min. amount of improvement
		1];	% info display during iteration 

if nargin == 2,
	options = default_options;
else
	% If "options" is not fully specified, pad it with default values.
	if length(options) < 4,
		tmp = default_options;
		tmp(1:length(options)) = options;
		options = tmp;
	end
	% If some entries of "options" are nan's, replace them with defaults.
	nan_index = find(isnan(options)==1);
	options(nan_index) = default_options(nan_index);
	if options(1) <= 1,
		error('The exponent should be greater than 1!');
	end
end

expo = options(1);		% Exponent for U
max_iter = options(2);		% Max. iteration
min_impro = options(3);		% Min. improvement
display = options(4);		% Display info or not

obj_fcn = zeros(max_iter, 1);	% Array for objective function

U = initfcm(cluster_n, data_n);			% Initial fuzzy partition
% Main loop
for i = 1:max_iter,
	[U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo);
% 	if display, 
% 		fprintf('Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
% 	end
	% check termination condition
	if i > 1,
		if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro, break; end,
	end
end

iter_n = i;	% Actual number of iterations 
obj_fcn(iter_n+1:max_iter) = [];
return
function U = initfcm(cluster_n, data_n)
%INITFCM Generate initial fuzzy partition matrix for fuzzy c-means clustering.
%   U = INITFCM(CLUSTER_N, DATA_N) randomly generates a fuzzy partition
%   matrix U that is CLUSTER_N by DATA_N, where CLUSTER_N is number of
%   clusters and DATA_N is number of data points. The summation of each
%   column of the generated U is equal to unity, as required by fuzzy
%   c-means clustering.
%
%       See also DISTFCM, FCMDEMO, IRISFCM, STEPFCM, FCM.

%   Roger Jang, 12-1-94.
%   Copyright 1994-2002 The MathWorks, Inc. 
%   $Revision: 1.11 $  $Date: 2002/04/14 22:21:58 $

U = rand(cluster_n, data_n);
col_sum = sum(U);
U = U./col_sum(ones(cluster_n, 1), :);
return
function [U_new, center, obj_fcn] = stepfcm(data, U, cluster_n, expo)
%STEPFCM One step in fuzzy c-mean clustering.
%   [U_NEW, CENTER, ERR] = STEPFCM(DATA, U, CLUSTER_N, EXPO)
%   performs one iteration of fuzzy c-mean clustering, where
%
%   DATA: matrix of data to be clustered. (Each row is a data point.)
%   U: partition matrix. (U(i,j) is the MF value of data j in cluster j.)
%   CLUSTER_N: number of clusters.
%   EXPO: exponent (> 1) for the partition matrix.
%   U_NEW: new partition matrix.
%   CENTER: center of clusters. (Each row is a center.)
%   ERR: objective function for partition U.
%
%   Note that the situation of "singularity" (one of the data points is
%   exactly the same as one of the cluster centers) is not checked.
%   However, it hardly occurs in practice.
%
%       See also DISTFCM, INITFCM, IRISFCM, FCMDEMO, FCM.

%   Roger Jang, 11-22-94.
%   Copyright 1994-2002 The MathWorks, Inc. 
%   $Revision: 1.13 $  $Date: 2002/04/14 22:21:02 $

mf = U.^expo;       % MF matrix after exponential modification
center = mf*data./((ones(size(data, 2), 1)*sum(mf'))'); % new center
dist = distfcm(center, data);       % fill the distance matrix
obj_fcn = sum(sum((dist.^2).*mf));  % objective function
tmp = dist.^(-2/(expo-1));      % calculate new U, suppose expo != 1
U_new = tmp./(ones(cluster_n, 1)*sum(tmp));
return
function [theta,bel,J]=k_means(X,theta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%  [theta,bel,J]=k_means(X,theta)
% This function implements the k-means algorithm, which requires
% as input the number of clusters underlying the data set. The algorithm
% starts with an initial estimation of the cluster representatives and
% iteratively tries to move them into regions that are "dense" in data
% vectors, so that a suitable cost function is minimized. This is
% achieved by performing (usually) a few passes on the data set. The
% algorithm terminates when the values of the cluster representatives
% remain unaltered between two successive iterations.
%
% INPUT ARGUMENTS:
%  X:       lxN matrix, each column of which corresponds to
%           an l-dimensional data vector.
%  theta:   a matrix, whose columns contain the l-dimensional (mean)
%           representatives of the clusters.
%
% OUTPUT ARGUMENTS:
%  theta:   a matrix, whose columns contain the final estimations of
%           the representatives of the clusters.
%  bel:     N-dimensional vector, whose i-th element contains the
%           cluster label for the i-th data vector.
%  J:       the value of the cost function (sum of squared Euclidean
%           distances of each data vector from its closest parameter
%           vector) that corresponds to the  estimated clustering.
%
% (c) 2010 S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[l,N]=size(X);
[l,m]=size(theta);
e=1;
iter=0;
while(e~=0)
    iter=iter+1;
    theta_old=theta;
    dist_all=[];
    for j=1:m
        dist=sum(((ones(N,1)*theta(:,j)'-X').^2)');
        dist_all=[dist_all; dist];
    end
    [q1,bel]=min(dist_all);
    J=sum(min(dist_all));
    
    for j=1:m
        if(sum(bel==j)~=0)
            theta(:,j)=sum(X'.*((bel==j)'*ones(1,l))) / sum(bel==j);
        end
    end
    e=sum(sum(abs(theta-theta_old)));
end
return
function [J3]=ScatterMatrices(class1,class2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [J3]=ScatterMatrices(class1,class2)
% Computes the J3 distance measure from the within-class and mixture
% scatter matrices.
%
% INPUT ARGUMENTS:
%   class1:     data of the first class, one pattern per column.
%   class2:     data of the second class, one pattern per column.
%
% OUTPUT ARGUMENTS:
%   J3:         J3 distance measure
%
% (c) 2010 S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class1=class1'; class2=class2';
z1=cov(class1,1);
z2=cov(class2,1);
NofPattInClass1=size(class1,1);
NofPattInClass2=size(class2,1);
N=NofPattInClass1+NofPattInClass2;
Sw=((NofPattInClass1/N)*z1+(NofPattInClass2/N)*z2);
classboth=[class1;class2];
Sm=cov(classboth,1);
J3=trace(inv(Sw)*Sm)/size(class1,2);
return
function [ softMaxed ] = softMax_aj( vals , r)
%softmax_aj Normalizes data using softmax method
%   Detailed explanation goes here

[~,numFeatures]=size(vals);

%Get means and standard deviations
miu=mean(vals);
stdDev=std(vals);

%Get softmax values
for i=1:numFeatures
    y=(vals(:,i)-miu(i))/(r*stdDev(i));
    softMaxed(:,i)=1./(1+exp(-y));
end
return
function res = MBSAS(X,theta,q)
% MBSAS - Modified Basic Sequential Algorithmic Scheme
%   
% Inputs: 
%   X : [ number_of_samples by number_of_features ] matrix of feature vectors
%   theta : threshold of dissimilarty 
%   q : maximum number of clusters 
% 
% Notes: 
%   For this scheme one needs to define the distance between a point x_i and a cluster C_j.
%   This implementation uses the cluster represntative is the mean and the distance between x_i and C_j 
%   is the Euclidean distance between x_i and the cluster reprentative.
% 
% Written by:
% -- 
% John L. Weatherwax                2007-07-01
% 
% email: wax@alum.mit.edu
% 
% Please send comments and especially bug reports to the
% above email address.
% 
%-----

N = size(X,1);
nFeatures = size(X,2); 

labels = zeros(1,N); % zero means the point is not yet labeled 

% Cluster determination:
% 
m=1;
labels(1)=1; 
for ii = 2:N, 
  % find C_k : d(x_ii,C_k) = min_{1 <= j <= m} d(x_ii,C_j)
  %
  [ d_x_i_C_k, k ] = findClosestCluster( ii, labels, X ); 
  
  if( (d_x_i_C_k > theta) && (m<q) )
    m=m+1;
    labels(ii)=m;
  end
end

% Pattern Classification:
% 
for ii=1:N,
  if( labels(ii)==0 )
    [ d_x_i_C_k, k ] = findClosestCluster( ii, labels, X ); 
    labels(ii) = k; 
  end
end

res = labels; 
return
function [d_x_i_C_k,k] = findClosestCluster( ii, labels, X )
% 
% Written by:
% -- 
% John L. Weatherwax                2007-07-01
% 
% email: wax@alum.mit.edu
% 
% Please send comments and especially bug reports to the
% above email address.
% 
%-----
  
ulabels = unique(labels);
if( ulabels(1)==0 ) 
  ulabels = ulabels(2:end); % drop the value of 0 which indicates this point has not been labeled
end
x_ii_to_cluster = [];
for lab = ulabels,
  inds = find( labels==lab );
  rep  = getClusterRepresentative( inds, X ); 
  d = sqrt( ( X(ii,:)' - rep )' * ( X(ii,:)' - rep ) ); 
  x_ii_to_cluster = [ x_ii_to_cluster, d ]; 
end
[d_x_i_C_k,mind] = min(x_ii_to_cluster); 
k = ulabels(mind); % the cluster index to which x_i is closest 
return
function rep = getClusterRepresentative(inds, X)
% 
% Written by:
% -- 
% John L. Weatherwax                2007-07-01
% 
% email: wax@alum.mit.edu
% 
% Please send comments and especially bug reports to the
% above email address.
% 
%-----

rep = mean( X(inds,:), 1 )';
return