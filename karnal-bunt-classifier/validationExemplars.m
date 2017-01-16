%Performs feature extraction on validation images and saves features in validation.mat

clc,close all, clear all
sporesIdxValidt{1}=6;
sporesIdxValidt{2}=1;
sporesIdxValidt{3}=[11 1 3 14 18];
sporesIdxValidt{4}=[9 5 25];
sporesIdxValidt{5}=[1 17];

[ratio,label,metricC,centers,imNumber] = deal([]);%initialize to empty
for imNum=1:5
    
    I=imread(['Validate' num2str(imNum) '.tif']);
    figure(1);imshow(I);
    level = graythresh(I);
    bw = im2bw(I,level);
    figure(2);imshow(bw);title('Black&White Image');
    bw2 = ~bwareaopen(~bw, 20);
    figure(3);imshow(bw2);title('Noise removed in B&W image')
    Igray = rgb2gray(I);
    
    [centres, radii, metric] = imfindcircles(~bw2, [6, 15]);
    

    for i=1:length(centres)
        figure(1);
        if ~isempty(find(sporesIdxValidt{imNum}==i))
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
        
        below50=sum(sum(croppedI<50));%&croppedI~=0));
        bet50and151=sum(sum(croppedI>50&croppedI<151));
        bet151to255=sum(sum(croppedI>151));
        

        ratio2(i)=(below50/bet151to255)/10000;
    end
    title('Objects detected (truth data applied, blue=target, red=clutter')
    sporesIdx=sporesIdxValidt{imNum};
    nonsporesIdx=setxor(1:length(ratio2),sporesIdx);
    figure(4);hold on;grid on
    title('Spores separability using feature #1 vs #2')
    plot(ratio2(sporesIdx),metric(sporesIdx),'b+');
    plot(ratio2(nonsporesIdx),metric(nonsporesIdx),'r+');
    disp('Press CONTINUE (F5) to continue with next image')
    ratio=[ratio ratio2(sporesIdx) ratio2(nonsporesIdx)];
    metricC=[metricC metric(sporesIdx)' metric(nonsporesIdx)'];
    label=[label ones(1,length(sporesIdx)) zeros(1,length(nonsporesIdx))];
    centers=[centers;centres(sporesIdx,:);centres(nonsporesIdx,:)];
    imNumber=[imNumber repmat(imNum,1,length([sporesIdx,nonsporesIdx]))];
    keyboard
    clear  ratio2
end

figure(4);legend('Spores','Non-spores')
if(1);save('validate','ratio','metricC','label','centers','imNumber');end

