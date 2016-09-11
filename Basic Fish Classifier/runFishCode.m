function runFishCode()
%% FISH CLASSIFIER
% TASK: To build a classifier based on 
%   1) Euclidean Distance
%   2) Mahalanobis Distance
%   3) Naive Bayes
% DESCRIPTION:
% We first do some exploratory data analysis. Then, outliers are removed. 
% Separability is analyzed using results from a t-test. 
% To parameterize the class distribution, maximum likelihood estimates are
% calculated (needed for the Mahalanobis classifier). Then,
% Conditional PDFs are estimated (needed for the Naive Bayes classifer). 
% Some parts of the code were exclusively written for a report. 
%
% 2014-SEPT AJ Rivera aj.rivera@jhu.edu

colordef black%set default bckgnd color to black

addpath(genpath('../utilFunctions/'))
%% IMPORT RAW DATA
filename='exemplars.csv';

[exemplarId,speciesOrig,weightOrig,len1Orig,len2Orig,len3Orig,heightOrig,widthOrig,sexOrig] = importFishData(filename);

legendMixed={'.red','owhite','^blue','+green','*magenta','xcyan','dyellow'};
legendSpecies={'Bream','Whitefish','Roach','NoName','Smelt','Pike','Perch'};
legendFeatures={'weight','len1','len2','len3','height','width'};
%% EXAMINE FEATURE DATA

%Do pie-chart to show species distribution in this dataset
charts_pieSpecies=[];
for i=1:7
    charts_pieSpecies=[charts_pieSpecies sum(speciesOrig==i)];
end
pie(charts_pieSpecies);legend(legendSpecies);

% After data examination, use indices of valid exemplars
validIdx=find(exemplarId~=14);%get all except 14
species = speciesOrig(validIdx);
weight = weightOrig(validIdx);
len1 = len1Orig(validIdx);
len2 = len2Orig(validIdx);
len3 = len3Orig(validIdx);
height = heightOrig(validIdx);
width = widthOrig(validIdx);
sex = sexOrig(validIdx);


% Because we want to analyze features on a per-species basis,
% let's go ahead and get the indices that correspond to each species
%(will be useful when investigating patterns)
% So, weight values for 1st species are
% are weight(idx{1}).
for i=1:7 %for every feature
    idx{i}=find(species==i)'; %find indices
end

%-------------

% plot weight (one dimension)
figure;plotFeatData(weight,[],[],idx,legendMixed);
title('Separability using "Weight" feature');xlabel('Weight(grams)');legend(legendSpecies)

% plot len1 (one dimension)
figure;plotFeatData(len1,[],[],idx,legendMixed);
title('Separability using "Lenght 1" feature');xlabel('Length 1(cm)');legend(legendSpecies)

% plot len2 (one dimension)
figure;plotFeatData(len2,[],[],idx,legendMixed);
title('Separability using "Length 2" feature');xlabel('Length 2(cm)');legend(legendSpecies)

% plot len3 (one dimension)
figure;plotFeatData(len3,[],[],idx,legendMixed);
title('Separability using "Length 3" feature');xlabel('Length 3(cm)');legend(legendSpecies)

% plot height (one dimension)
figure;plotFeatData(height,[],[],idx,legendMixed);
title('Separability using "Height" feature');xlabel('Height');legend(legendSpecies)

% plot width (one dimension)
figure;plotFeatData(width,[],[],idx,legendMixed);
title('Separability using "Width" feature');xlabel('Width');legend(legendSpecies)

% plot sex (one dimension)
figure;plotFeatData(sex,[],[],idx,legendMixed);
title('Separability using "Sex" feature');xlabel('Sex (0=F,1=M)');legend(legendSpecies)
xlim([-1.5 1.5])

colordef white
corrplot([weight,len1,len2,len3,height,width],'varNames',legendFeatures)
colordef black

% plot height vs. width
figure;plotFeatData(height,width,[],idx,legendMixed);
title('Separability using Height vs. Width feature space');
xlabel('Height'),ylabel('Width');legend(legendSpecies)

% plot height vs. len1
figure;plotFeatData(height,len1,[],idx,legendMixed);
title('Separability using Height vs. Len1 feature space');
xlabel('Height'),ylabel('Len1');legend(legendSpecies)

% plot height vs. weight
figure;plotFeatData(height,weight,[],idx,legendMixed);
title('Separability using Height vs. Weight feature space');
xlabel('Height'),ylabel('Weight');legend(legendSpecies)

% plot len1 vs. len3
figure;plotFeatData(len1,len3,[],idx,legendMixed);
title('Separability using Len1 vs. Len3 feature space');
xlabel('Len1'),ylabel('Len3');legend(legendSpecies)

figure;plotFeatData(len1,len2,[],idx,legendMixed);
title('Separability using Len1 vs. Len2 feature space');
xlabel('Len1'),ylabel('len2');legend(legendSpecies)

figure;plotFeatData(height,weight,len1,idx,legendMixed)
title('Separability using weight vs. height vs. Len1 feature space');
xlabel('height'),ylabel('weight');zlabel('len1');legend(legendSpecies)

% % capture video of 3d plot
% axis tight;
% %% Set up recording parameters (optional), and record
% OptionZ.FrameRate=56;OptionZ.Duration=10;OptionZ.Periodic=true;
% CaptureFigVid([-20,10;-110,10;-190,80;-290,10;-380,10], 'WellMadeVid',OptionZ)

%% REMOVE OUTLIERS 
%Remove outliers on a per class basis. Assumes each feature follows a
%Gaussian distribution, and thus, a value outside 2 standard deviations is
%an outlier. Median is used. 
for i=1:7
    removeIdxAll=[];
    [~,toRemoveIdx,~]=normalDistOutlierRemove(weight(idx{i}),2);
    if (i==3)
        toRemoveIdx=[toRemoveIdx 6]'; %number 7 has weight 0!
    end
    removeIdxAll=[removeIdxAll toRemoveIdx'];
    %find lenght1 outlcliers for this class
    [~,toRemoveIdx,~]=normalDistOutlierRemove(len1(idx{i}),2);
    removeIdxAll=[removeIdxAll toRemoveIdx'];
    %lenght2 outliers removal for this class
    [~,toRemoveIdx,~]=normalDistOutlierRemove(len2(idx{i}),2);
    removeIdxAll=[removeIdxAll toRemoveIdx'];
    %lenght3 outliers removal for this class
    [~,toRemoveIdx,~]=normalDistOutlierRemove(len3(idx{i}),2);
    removeIdxAll=[removeIdxAll toRemoveIdx'];
    %height outliers removal for this class
    [~,toRemoveIdx,~]=normalDistOutlierRemove(height(idx{i}),2);
    removeIdxAll=[removeIdxAll toRemoveIdx'];
    %width outliers removal for this class
    [~,toRemoveIdx,~]=normalDistOutlierRemove(width(idx{i}),2);
    removeIdxAll=[removeIdxAll toRemoveIdx'];
    
    idx{i}(unique(removeIdxAll))=[];
end

%% TEST SEPARABILITY USING A T-TEST

%Basically, perform a t-test of the hypothesis that two
%independent samples, in the feature vectors X and Y, come from distributions
%with equal means. A significance level of 5% is used. 

%Perform t-test for all combinations
combinations={[1 2],[1 3],[1 4],[1 5],[1 6],[1 7],...
                    [2 3],[2 4],[2 5],[2 6],[2 7],...
                          [3 4],[3 5],[3 6],[3 7],...
                                [4 5],[4 6],[4 7],...
                                      [5 6],[5 7],...
                                            [6 7]};
                
%Think about the loop below as:
%For combination of class 1 and 2, determine the accepted hypothesis (0 or 1)
%for all the features. 
%results in table form would be:
%                        weight           |           len1         | etc...
% (class1,class2)  weightTtest(1)=0 or 1  |   len1Ttest(1)=0 or 1  | etc..
% (1,3)            weightTtest(2)=0 or 1  |   len1Ttest(2)=0 or 1  | etc..
for i=1:length(combinations)
    ttest_weight(i)=ttest2(weight(idx{combinations{i}(1)}),weight(idx{combinations{i}(2)}),0.05);
    ttest_len1(i)=ttest2(len1(idx{combinations{i}(1)}),len1(idx{combinations{i}(2)}),0.05);
    ttest_len2(i)=ttest2(len2(idx{combinations{i}(1)}),len2(idx{combinations{i}(2)}),0.05);
    ttest_len3(i)=ttest2(len3(idx{combinations{i}(1)}),len3(idx{combinations{i}(2)}),0.05);
    ttest_height(i)=ttest2(height(idx{combinations{i}(1)}),height(idx{combinations{i}(2)}),0.05);
    ttest_width(i)=ttest2(width(idx{combinations{i}(1)}),width(idx{combinations{i}(2)}),0.05);
end

%% COMPUTE ML ESTIMATES FOR EVERY CLASS
for i=1:7
    [m_hat{i} S_hat{i}]=calcMLEstimatesNormalDistro([weight(idx{i})';len1(idx{i})';...
        len2(idx{i})';len3(idx{i})';height(idx{i})';...
        width(idx{i})']);
end

%%  COMPUTE CONDITIONAL PDFs

randn('seed',1)% use same seed for reproducibility
    
for i=1:7 %for every species
    %first term is: {1}=bream species, {4}= (1)=weight mean, (4)=len3 mean
    m=[m_hat{i}(1) m_hat{i}(4)]';
    % from the bream species cov mat, get correct var and covar.
    S=[S_hat{i}(1,1) S_hat{i}(1,4); S_hat{i}(4,1) S_hat{i}(4,4)];
    N=length(idx{i});%number of exemplars in this class
    X=mvnrnd(m,S,N)'; %generate distrib.
    
    figure;
    subplot(3,3,[1;2]);hist(X(1,:),10);hold on;modifyHistPlotColor('y');
    hist(weight(idx{i}),10);
    subplot(3,3,[6;9]);[counts,bins]=hist(X(2,:),10);
    barh(bins,counts);hold on;modifyHistPlotColor('y');
    [counts,bins]=hist(len3(idx{i}),10);barh(bins,counts);
    subplot(3,3,[4:5,7:8]);plot(X(1,:),X(2,:),'y+',weight(idx{i}),len3(idx{i}),'c+');
%     xlim([min([weight(idx{i})' X(1,:)]) max([weight(idx{i})' X(1,:)])])
%     ylim([min([len3(idx{i})' X(2,:)]) max([len3(idx{i})' X(2,:)])])
    
    %axes labeling and titles
    subplot(3,3,[1;2]);
    legend('Generated Weight Distribution','Actual Weigth Distribution');
    xlabel('Weight (grams)');title(['Histograms -- Species: ' legendSpecies{i}])
    subplot(3,3,[6;9]);
    legend('Generated Len3 Distribution','Actual Len3 Distribution');
    xlabel('Length 3 (cm)');
    subplot(3,3,[4:5,7:8]);
    legend('Generated Len3 Distribution','Actual Len3 Distribution');
    xlabel('Weight (grams)');ylabel('Length 3 (cm)')
end

%% CDF(Cumulative Density Function and ROC curve
%GET weight CDF for class 1
%when doing cdf, do it at the highest increment possible= min non-zero
%difference
incrmnts=abs(diff(weight));incrmnts=min(incrmnts(incrmnts~=0));
[counts,bins]=hist(weight(idx{1}),0:incrmnts:max(weight));
class1Pdf=counts./sum(counts);%now it sums to 1
class1Cdf=cumsum(class1Pdf);

%GET weight CDF for class 7
%when doing cdf, do it at the highest increment possible= min non-zero
%difference
[counts,bins]=hist(weight(idx{7}),0:incrmnts:max(weight));
class7Pdf=counts./sum(counts);%now it sums to 1
class7Cdf=cumsum(class7Pdf);
figure;plot(bins,class1Pdf,'-y',bins,class7Pdf,'-c')
xlabel('Weight (grams)');title('PDF')
legend('Bream Species','Perch Species')

figure;plot(bins,class1Cdf,'-y',bins,class7Cdf,'-c')
xlabel('Weight (grams)');title('CDF')
legend('Bream Species','Perch Species')

i=1;
class1Vals=weight(idx{1});
class7Vals=weight(idx{7});
for thres=bins(end:-1:1)

    %mentioned by Dr. Baumgart but apparently not needed
%     belowThres=sum(find(class1Vals<thres))+sum(find(class7Vals<thres));
%     aboveThres=sum(find(class1Vals>=thres))+sum(find(class7Vals>=thres));
    
    TP=sum(find(class1Vals>=thres));
    FP=sum(find(class7Vals>=thres));
    TN=sum(find(class7Vals<thres));
    FN=sum(find(class1Vals<thres));

    sens(i)=TP/(TP+FN);
    specif(i)=TN/(TN+FP);
    i=i+1;
end
figure;plot(1-specif,sens,'y',[0,1],[0,1],'--c')
title('ROC curve (Bream vs. Perch species separation by weight)');xlabel('Specifity');ylabel('Sensitivity')

%% BUILD EUCLIDEAN , MAHALANOBIS, AND NAIVE BAYES CLASSIFIER
%first, let's compute the S_hat (necessary for Bayes and Mahalanobis)
%using the selected features
for i=1:7
    %m_hat will have the same mean values as before, but it's easier this
    %way than to resize it to just have the selected features
     %uses four features
    [m_hat{i},S_hat{i}]=calcMLEstimatesNormalDistro([weight(idx{i})';...
         len3(idx{i})';height(idx{i})';width(idx{i})']);
%     
%     %uses 3 features
%     [m_hat{i},S_hat{i}]=calcMLEstimatesNormalDistro([weight(idx{i})';...
%         len3(idx{i})';height(idx{i})']);
    
%     %uses 2 features
%     [m_hat{i},S_hat{i}]=calcMLEstimatesNormalDistro([weight(idx{i})';...
%         len3(idx{i})']);
        %uses 6 features
%     [m_hat{i},S_hat{i}]=calcMLEstimatesNormalDistro([weight(idx{i})';...
%         len3(idx{i})';height(idx{i})';width(idx{i})';len1(idx{i})']);
end

%now, reload original dataset (contains outliers) to obtain
%true performance

species = speciesOrig;
weight = weightOrig;
len3 = len3Orig;
height = heightOrig;
width = widthOrig;

%Euclidean first
%euclidean classifier needs
%miu to be 4(features) by 7 (classes)
%the i-th row is the feature mean of the class in the i-th column
miu=[m_hat{:}];

%X1 is the list of exemplars to be labeled
%X1 has to be 4(feature) by N(total num of exemplar)
% each column n (exemplar n) has to have 4 rows (its computed feature);
%labels= N by 1
%for every exemplar, labels are the classified exemplars labels

%-------------------------RUN CLASSIFIERS--------------
%form exemplar data vector (used in all classifiers)
switch size(m_hat{1},1);
    case 4
        X1=[weight len3 height width]';
        
    case 3
        X1=[weight len3 height]';
    case 2
        X1=[weight len3]';
    case 5
        X1=[weight len3 height width len1]';
end

%EUCLIDEAN CLASSIFIER
labels=euclideanDistClassifier(miu,X1);
errorEucl=sum(species~=labels')/length(labels);%count mislabeled exemplars

%MAHALANOBIS CLASSIFIER
%assumming "same" covariance matrix, so it is
S=0;
for i=1:7
    S=S+(1/7)*S_hat{i};
end
labels=mahalanobisDistClassifier(miu,S,X1);
errorMaha=sum(species~=labels')/length(labels);

%BAYESIAN CLASSIFIER
%here covariance for every class is needed...woohooo finally some real action!
P=[];%a priori probab.
for i=1:7
    S(:,:,i)=S_hat{i};
    P=[P sum(species==i)];
end
P=P./length(species); %get a priori probabilities

labels=bayesClassifier(X1,miu,S,P);
errorBayes=sum(species~=labels')/length(labels);

figure;hold on;
bar(1,100*errorEucl);
bar(2,100*errorMaha,'r');
bar(3,100*errorBayes,'y');
set(gca, 'XTickLabelMode', 'Manual')
set(gca, 'XTick', [])
legend({'Euclidean Distance', 'Mahalanobis Distance', 'Bayes'});
title('Classifier Performance (% exemplars misclassified)')


text(1,(100*errorEucl+2),[num2str(100*errorEucl) '%'])
text(2,(100*errorMaha+2),[num2str(100*errorMaha) '%'])
text(3,(100*errorBayes+2),[num2str(100*errorBayes) '%'])




colordef white %reset default bkground color to white

disp('All done :)')
end















