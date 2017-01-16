function [ trainingIdx , validationIdx , testIdx] = splitSetInto3Sets( labels, set1Pct, set2Pct )
%Split the data into 3 data sets, each one containing the same percentage 
%of samples from a class. This way one does not end up having a set with a
%disproportionate number of samples from a class relative to the other data
%sets. 
%
% 2016-4-17 AJ RIVERA, Johns Hopkins University


trueLabelsIdx = find(labels==1);
falseLabelsIdx = find(labels==0);
numTrueLabels = length(trueLabelsIdx);
numFalseLabels = length(falseLabelsIdx);

%get training elements from both classes
idx1= randsample(trueLabelsIdx, floor(set1Pct*numTrueLabels));
idx2 = randsample(falseLabelsIdx, floor(set1Pct*numFalseLabels));
trainingIdx = union(idx1, idx2);
%remove used elements
trueLabelsIdx = setxor(idx1, trueLabelsIdx);
falseLabelsIdx = setxor(idx2, falseLabelsIdx);

%get validation elements from both classes
idx1= randsample(trueLabelsIdx, floor(set2Pct*numTrueLabels));
idx2 = randsample(falseLabelsIdx, floor(set2Pct*numFalseLabels));
validationIdx = union(idx1, idx2);
%remove used elements
trueLabelsIdx = setxor(idx1, trueLabelsIdx);
falseLabelsIdx = setxor(idx2, falseLabelsIdx);

%get test set elements from both classes (just use remaining exemplars)
testIdx = union(trueLabelsIdx, falseLabelsIdx);

end

