% An algorithm to predict the stock price direction near a support level.
%
% Gets stock data, calculates features for each sample, and trains and
% evaluatates an artificial neural network. 
%
% 2016-4-17 AJ RIVERA, Johns Hopkins University


clear all
clc
close all

%% GET DATA IN THE FORM OF FEATURE VECTORS
% Call function that computes features using stock data. For example 
% a feature vector can be the ratio of the price to the 20-day simple moving average
% 
saveToFile = false; 
[sampleFeat, sampleLabels, featNames] = stockData2Samples(saveToFile);
sampleFeat=sampleFeat';

%% GET TRAINING, VALIDATION, and TEST SETS
[trainingSetIdx, validationSetIdx, testSetIdx ] = splitSetInto3Sets( sampleLabels, 0.7, 0.15);

%% SCALE TRAINING SET
% Now we need to scale the training set only. Then, using the scaling
% parameters computed during this process, we scale the test/validation
% sets. The scaling method used is minmax.
% The data is rescaled feature by feature. This way, all
% features have values ranging from 0 to 1.
upperBound = 1;lowerBound = 0;
numFeatures = size(sampleFeat,1); numSamples = size(sampleFeat,2);
scalingParams=zeros(3,1);  %preallocation
scaledsampleFeat = zeros(numFeatures,numSamples);  %preallocation
for curFeat=1:numFeatures
    %scale the training set and get the scaling parameters
    [scaledsampleFeat(curFeat,trainingSetIdx) , ...
        scalingParams(1), ...
        scalingParams(2), ...
        scalingParams(3)] = ...
        minMaxScaling(sampleFeat(curFeat,trainingSetIdx),lowerBound,upperBound);
    
    % Now scale validation and test set using the scaling parameters
    % from the testing set
    numerator = sampleFeat(curFeat,validationSetIdx)-scalingParams(1);
    denominator = scalingParams(3)-scalingParams(2);
    tmp = (numerator/denominator) * (upperBound-lowerBound) ;
    tmp = tmp + lowerBound;
    scaledsampleFeat(curFeat,validationSetIdx) = tmp;
    
    numerator = sampleFeat(curFeat,testSetIdx)-scalingParams(1);
    denominator = scalingParams(3)-scalingParams(2);
    tmp = (numerator/denominator) * (upperBound-lowerBound) ;
    tmp = tmp + lowerBound;
    scaledsampleFeat(curFeat,testSetIdx) = tmp;
    
end
%Clean up dishes as you eat.
clear upperBound lowerBound curFeat tmp ...
    numerator denominator scalingParams

%% TRAIN NEURAL NETWORK

%results log
filename = '__featureCombinationResults4Features_crossEntropy2.csv';
header = 'TRAININGALGORITHM,DATASET,TRAININGRUN,FEATURES,NUMNEURONSHIDDLAYER,THRESHOLD,AUC,FAR,TPR,FNR,TNR\n';
writeToFile(header,filename);


% Set-up Neural Network for training and evaluation
% Log results using feature combination of length 2, 
% using the Resilient Backprop algorithm, 10 neurons in 
% a single hidden layer, an a cross-entropy error function.
% Later, the logs will be evaluated and the best topology will be chosen. 

for i=2
    %find all possible combiations of length i
    combos=combnk(1:numFeatures,i);
    %for every combination, compute performance
    for j=1:size(combos,1)
        curCombo = combos(j,:);
        disp(['Training & Evaluation NN using feature(s): ' num2str(curCombo)]);
        tic
        trainAndEvalNN( scaledsampleFeat(curCombo,:),...
            sampleLabels,...
            trainingSetIdx,...
            validationSetIdx,...
            testSetIdx,...
            filename,...
            curCombo);
        toc
    end
end
