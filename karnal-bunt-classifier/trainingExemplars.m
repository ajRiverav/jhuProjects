%Performs feature extraction on training images and saves features in training.mat

clc,close all, clear all

%indices of spores for every image
%i.e. sporeIdxTrain{1}=2 means for image 1, spore is object#2
sporesIdxTrain{1}=2;
sporesIdxTrain{2}=1;
sporesIdxTrain{3}=2;
sporesIdxTrain{4}=1;
sporesIdxTrain{5}=4;
sporesIdxTrain{6}=2;
sporesIdxTrain{7}=[1,2,4,23,42];
sporesIdxTrain{8}=[1,6];
sporesIdxTrain{9}=1;

[ratio , label , metricC] = deal([]);%init vars
for imNum=1:9
    
    I=imread(['Train' num2str(imNum) '.tif']);
    figure(1);imshow(I);
    
    %Preprocess image for better recognition
    level = graythresh(I);
    bw = im2bw(I,level);
    figure(2);imshow(bw);title('Black&White Image');
    bw2 = ~bwareaopen(~bw, 20);
    figure(3);imshow(bw2);title('Noise removed in B&W image')
    Igray = rgb2gray(I);
    
    %extract feature #1 
    %Feature #2 is: circle diameter
    [centres, radii, metric] = imfindcircles(~bw2, [6, 15]);
    
 
    for i=1:length(centres)
        figure(1);
        
        if ~isempty(find(sporesIdxTrain{imNum}==i))
        text(centres(i,1),centres(i,2),['<-' num2str(i)],...
            'EdgeColor','blue','Color','blue')
        else
            text(centres(i,1),centres(i,2),['<-' num2str(i)],...
            'EdgeColor','red','Color','red')
        end
            
        ci = [centres(i,:) radii(i)];     % center and radius of circle ([c_row, c_col, r])
        [xx,yy] = ndgrid((1:size(I,1))-ci(2),(1:size(I,2))-ci(1));
        mask = uint8((xx.^2 + yy.^2)<ci(3)^2);
        croppedI=Igray.*mask;
        
        below50=sum(sum(croppedI<50));
        bet151to255=sum(sum(croppedI>151));
        
        %feature #1: Histogram ratio
        
        ratio2(i)=(below50/bet151to255)/10000;%divide by 10k just to make value smaller
 
    end
    
    title('Objects detected (truth data applied, blue=target, red=clutter')
    
    sporesIdx=sporesIdxTrain{imNum};
    nonsporesIdx=setxor(1:length(ratio2),sporesIdx);
    
    figure(4);hold on;grid on
    title('Spores separability using feature #1 vs #2')
    plot(ratio2(sporesIdx),metric(sporesIdx),'b+');
    plot(ratio2(nonsporesIdx),metric(nonsporesIdx),'r+');
    xlabel('Feature #1: Histogram Ratio');ylabel('Feature #2: Circle-likeness')
    
    disp('Press CONTINUE (F5) to continue with next image')
    
    ratio=[ratio ratio2(sporesIdx) ratio2(nonsporesIdx)];
    metricC=[metricC metric(sporesIdx)' metric(nonsporesIdx)'];
    label=[label ones(1,length(sporesIdx)) zeros(1,length(nonsporesIdx))];
    
    keyboard
    clear ratio2
end

figure(4);legend('Spores','Non-spores')
%save features
if(1);save('training','ratio','metricC','label');end

