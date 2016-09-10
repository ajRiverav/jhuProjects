function plotFeatData(feat1,feat2,feat3,classIdx,plotLegend)
%plotFeatData Plot feature data (max of 3 features allowed). 
% INPUT ARGS
%   feat1:  Nx1 vector containing values for feature 1
%   feat2:  Nx1 vector containing values for feature 2
%   feat3:  Nx1 vector containing values for feature 3
%   classIdx: cell array containing the indices to the exemplars for each class
%       i.e. (for 2 classes) [1x34 double]    [1x6 double]
%   plotLegend: cell array with the marker symbol and color for each class
%       i.e.  (for 2 classes) {'.red','owhite'}
%
% 2014-SEPT AJ Rivera aj.rivera@jhu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if     isempty(feat2), numAxes=1;
elseif isempty(feat3), numAxes=2;
else                   numAxes=3;
end
hold on;
switch numAxes
    case 1
        for i=1:length(classIdx)
            
            %plot one dimensional graph,
            plot(feat1(classIdx{i}),i*ones(length(classIdx{i}),1),plotLegend{i})
        end
        for i=1:length(classIdx)
            plot(feat1(classIdx{i}),zeros(length(classIdx{i}),1),plotLegend{i})
        end
        
    case 2
        for i=1:length(classIdx)
            plot(feat1(classIdx{i}),feat2(classIdx{i}),plotLegend{i})
        end
        
    case 3
        for i=1:length(classIdx)
            plot3(feat1(classIdx{i}),feat2(classIdx{i}),...
                feat3(classIdx{i}),plotLegend{i})
        end
end
%[frequency,value]=hist(height(idx{i}),1:max(height(idx{i})));
grid on;hold on;
end