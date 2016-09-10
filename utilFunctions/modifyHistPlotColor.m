function modifyHistPlotColor(color)
%modifyHistPlotColor Change histogram color and make it transparent
% INPUT ARGS
%   color: string denoting desired color (e.g. 'r', 'blue')
%
% OUTPUT ARGS
%   z:  N-dimensional vector whose i-th component contains the label
%       of the class where the i-th data vector has been assigned.
%
% 2014-SEPT AJ Rivera aj.rivera@jhu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%TODO: Parameterize function to accept figure handlers and other things. 

%change color
set(get(gca,'child'),'FaceColor','none','EdgeColor',color);
%make it transparent
h = findobj(gca,'Type','patch');
set(h,'facealpha',0.50);

end