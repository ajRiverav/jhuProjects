function [exemFeat,exemLabel,featNames]=stockData2samples(save2File)

%% CONNECT TO YAHOO'S STOCK DATABASE
c=yahoo;

%% IDENTIFY STOCKS AND TO-BE-SAMPLES

%% TICKER LIST
n=1;

%bd means the stock price broke down below the support level price
%nbd means it did not break down.
%the indices point to the index (which ultimately correspond to a day)
%in which the event occured. This will be used to compute the features
%for this sample

%---- UTX = United Tech 1
TICKER{n}='UTX';
fromDate{n}='02/01/2013';
toDate{n}='11/27/2014';
bd{n}=[358] ; nbd{n}=[220 255  327];n=n+1;
%---- Boeing 2
TICKER{n}='BA';
fromDate{n}='01/01/2013';
toDate{n}='03/01/2015';
bd{n}=[396] ; nbd{n}=[322 373];n=n+1;
%---- Lockheed-Martin 3,4
TICKER{n}='LMT';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[594] ; nbd{n}=[441 463 477 495 508 520 572 493];n=n+1;
TICKER{n}='LMT';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[576] ; nbd{n}=[234 269 307 455 506 525];n=n+1;

%---- General Dynamics 5,6
TICKER{n}='GD';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[363] ; nbd{n}=[326 353  535 548 646 652 723];n=n+1;
TICKER{n}='GD';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[524 551 553 577];n=n+1;
%---- Raytheon 7,8
TICKER{n}='RTN';
fromDate{n}='01/01/2011';
toDate{n}='01/01/2013';
bd{n}=[343 472] ; nbd{n}=[211 320 400 442];n=n+1;
TICKER{n}='RTN';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[345 603] ; nbd{n}=[192 204 524 586];n=n+1;
%---- TransDigm Group Incorporated 9,10
TICKER{n}='TDG';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[303] ; nbd{n}=[201 220 ];n=n+1;
TICKER{n}='TDG';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[582] ; nbd{n}=[332 522 533 561 577];n=n+1;
%---- Rockwell Collins 11
TICKER{n}='COL';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[385] ; nbd{n}=[226];n=n+1;
%---- L3 12
TICKER{n}='LLL';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[562] ; nbd{n}=[523];n=n+1;

%---- Honeywell 13
TICKER{n}='HON';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[309 625] ; nbd{n}=[240 268 524 551 586 613 617];n=n+1;
%---- Spirit AeroSystems 14
TICKER{n}='SPR';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[710] ; nbd{n}=[640];n=n+1;
%---- Orbital ATK, Inc.
%no levels
%---- BAE Aerospace. 15,16
TICKER{n}='BEAV';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[337 351];n=n+1;
TICKER{n}='BEAV';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[690];n=n+1;
%---- Hexcel Corp. 17
TICKER{n}='HXL';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[540] ; nbd{n}=[266 493 651 669];n=n+1;
%---- HEICO Corp. 18,19
TICKER{n}='HEI';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[661 666 673 724];n=n+1;
TICKER{n}='HEI';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[576 629 662] ; nbd{n}=[541 550 559 617 659];n=n+1;
%---- BWX Technologies, Inc. 20,21
TICKER{n}='BWXT';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[573] ; nbd{n}=[501 527 564];n=n+1;
TICKER{n}='BWXT';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[397] ; nbd{n}=[348 357 620];n=n+1;
%---- Woodward, Inc. 22,23
TICKER{n}='WWD';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[433 543] ; nbd{n}=[426 463 496 524 535];n=n+1;
TICKER{n}='WWD';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[711 741 748];n=n+1;
% %---- Esterline Technologies Corp. 24
TICKER{n}='ESL';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[434 610] ; nbd{n}=[348 398 590];n=n+1;
%---- Moog Inc. 25,26
TICKER{n}='MOG-A';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[718] ; nbd{n}=[572 700];n=n+1;
TICKER{n}='MOG-A';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[576 641 680] ; nbd{n}=[511 524 634 675];n=n+1;
%---- KLX Inc.
% no data prior to 2015
%---- Triumph Group, Inc. 27,28
TICKER{n}='TGI';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[];n=n+1;
TICKER{n}='TGI';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[];n=n+1;
%---- Smith & Wesson Holding Corporation 29,30
TICKER{n}='SWHC';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[743] ; nbd{n}=[617 667 714 724 740];n=n+1;
TICKER{n}='SWHC';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[379] ; nbd{n}=[375 694 724];n=n+1;
%---- Sturm, Ruger & Co. Inc. 31,32
TICKER{n}='RGR';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[744] ; nbd{n}=[485 496 716];n=n+1;
TICKER{n}='RGR';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[666 708] ; nbd{n}=[651 667 694];n=n+1;
%---- DigitalGlobe, Inc. 33,34
TICKER{n}='DGI';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[544] ; nbd{n}=[535 542 615 643];n=n+1;
TICKER{n}='DGI';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[643] ; nbd{n}=[463 615 628];n=n+1;
%---- Kaman Corporation 35,36
TICKER{n}='KAMN';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[350 425 ] ; nbd{n}=[201 210 346 422];n=n+1;
TICKER{n}='KAMN';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[397 632] ; nbd{n}=[241 274 332 627];n=n+1;
%---- Astronics Corporation 37,38
TICKER{n}='ATRO';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[408 512 659] ; nbd{n}=[363 463 626 645];n=n+1;
TICKER{n}='ATRO';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[317 436 642] ; nbd{n}=[253 277 396 633 726 741];n=n+1;
%---- AAR Corp. 39
TICKER{n}='AIR';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[426 430] ; nbd{n}=[293 303 326 405];n=n+1;
%---- AeroVironment, Inc.
TICKER{n}='AVAV';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[];n=n+1;
TICKER{n}='AVAV';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[];n=n+1;
%---- National Presto Industries Inc.
TICKER{n}='NPK';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[];n=n+1;
TICKER{n}='NPK';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[];n=n+1;
%---- Ducommun Inc.
TICKER{n}='DCO';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[];n=n+1;
TICKER{n}='DCO';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[];n=n+1;
%---- LMI Aerospace Inc.
TICKER{n}='LMIA';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[];n=n+1;
TICKER{n}='LMIA';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[];n=n+1;
%---- CPI Aerostructures Inc.
TICKER{n}='CVU';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[];n=n+1;
TICKER{n}='CVU';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[];n=n+1;
%---- Air Industries Group
TICKER{n}='AIRI';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[];n=n+1;
TICKER{n}='AIRI';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[];n=n+1;
%---- Innovative Solutions & Support Inc.
TICKER{n}='ISSC';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[];n=n+1;
TICKER{n}='ISSC';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[];n=n+1;
%---- Astrotech Corp.
TICKER{n}='ASTC';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[];n=n+1;
TICKER{n}='ASTC';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[];n=n+1;
%---- Acorn Energy, Inc.
TICKER{n}='ACFN';
fromDate{n}='01/01/2010';
toDate{n}='01/01/2013';
bd{n}=[] ; nbd{n}=[];n=n+1;
TICKER{n}='ACFN';
fromDate{n}='01/01/2013';
toDate{n}='01/01/2016';
bd{n}=[] ; nbd{n}=[];n=n+1;


%% [FOR DEBUG PURPOSES] ROUTINE FOR SAMPLE IDENTIFICATION
if(0)
    tId=n-1;
    FIELDS={'Open','High','Low','Close','Volume','Adj Close'};
    t=1;op=2;hi=3;lo=4;cl=5;vol=6;adCl=7;
    dat=fetch(c,TICKER{tId},FIELDS,fromDate{tId},toDate{tId});
    dat=flipud(dat);
    dat(:,t)=1:length(dat(:,t));
    figure(1);
    if(1);plot(dat(:,t),dat(:,cl));end
    title([TICKER{tId} ' stock price from ' fromDate{tId} ...
        ' to ' toDate{tId}])
    grid on;grid minor;
    xlabel('day')
    ylabel('price ($)')
    figure(2);
    plot(dat(:,t),volroc(dat(:,vol)),'black',...
        dat(:,t),tsmovavg(volroc(dat(:,vol)),'s',50,1),'r--');
    if(1);
        for figNum=1:2;
            figure(figNum);hold on;
            if (figNum==1); xVals=dat(:,cl);end;
            if (figNum==2); xVals=volroc(dat(:,vol));end;
            for i=bd{end}
                plot(i,...
                    xVals(i),...
                    'Color','r','Marker','o');
            end
            for i=nbd{end}
                plot(i,...
                    xVals(i),...
                    'Color','b','Marker','o');
            end
        end
    end
    keyboard
end

numBreakdowns = sum(cellfun('length',bd));
numBounce = sum(cellfun('length',nbd));
numsamples = numBreakdowns + numBounce;
disp(['# samples ' num2str(numsamples)])
disp(['     Breakdowns: ' num2str(numBreakdowns) '(' num2str(100*numBreakdowns/numsamples) '%)']);
disp(['     Bounce  : ' num2str(numBounce) '(' num2str(100*numBounce/numsamples) '%)']);

%% GET STOCK DATA TO COMPUTE FEATURES
sampleCounter=1;%initialize
datRaw{n-1}=[];
featNames{1}=[];
%for every ticker ID (ticker ID is not unique)
for tId=1:n-1
    %Get this stock data
    FIELDS={'Open','High','Low','Close','Volume','Adj Close'};
    t=1;op=2;hi=3;lo=4;cl=5;vol=6;adCl=7;
    if(1)
        datRaw{tId}=fetch(c,TICKER{tId},FIELDS,fromDate{tId},toDate{tId});
    end
    %reverse matrix to have latest period at the end
    dat=flipud(datRaw{tId});
    
    %% [FOR DEBUG-PURPOSES] PLOT RAW DATA
    if(0)
        if(1);dat(:,t)=1:length(dat(:,t));end
        %figure(1);
        if(0);plot(dat(:,t),dat(:,op),dat(:,t),dat(:,cl));end
        if(1);plot(dat(:,t),dat(:,cl));end
        if(1);hold on;
            for i=bd{tId}
                plot(i,...
                    dat(i,cl),...
                    'Color','r','Marker','o');
            end
            for i=nbd{tId}
                plot(i,...
                    dat(i,cl),...
                    'Color','b','Marker','o');
            end
        end
        
        title([TICKER{tId} ' Stock price from ' fromDate{tId} ...
            ' to ' toDate{tId}])
        keyboard;close(gcf)
    end
    
    %% COMPUTE TECHNICAL INDICATORS USING RAW DATA.
    % Eventually these become features, but not yet...
    
    %% TECH.IND. :Relative Strength Index
    %   RSINDEX calculates the Relative Strength Index (RSI). The RSI is calculated
    %   based on a default 14-period period.
    d_rsindex=rsindex(dat(:,cl));
    
    %% TECH.IND. :Bollinger Bands
    %   [mid, uppr, lowr] = bollinger(data, wsize, wts, nstd) calculates the
    %   middle (mid), upper (uppr), and lower (lowr) bands that make up the
    %   Bollinger bands from the vector data.
    %   mid is the vector that represents the middle band, a simple moving
    %   average with a window size of wsize. uppr and lowr are vectors that
    %   represent the upper and lower bands. uppr is a vector representing
    %   the upper band that is +nstd times. lowr is a vector representing
    %   the lower band that is -nstd times.
    [d_bollinger_mid, d_bollinger_uppr_2sig,d_bollinger_lowr_2sig] = bollinger(dat(:,cl),20,0,2);
    [~, d_bollinger_uppr_1sig,d_bollinger_lowr_1sig] = bollinger(dat(:,cl),20,0,1);
    
    %% TECH.IND. : Moving Averages
    sma20=tsmovavg(dat(:,cl),'s',20,1);
    sma50=tsmovavg(dat(:,cl),'s',50,1);
    sma200=tsmovavg(dat(:,cl),'s',200,1);
    
    %% TECH.IND. : MACD
    [d_macd, d_ema9]=macd(dat(:,cl));
    
    %% TECH.IND. :Volume rate of change
    %   VROC = VOLROC(TVOLUME) calculates the volume rate-of-change, VROC,
    %   from the volume traded data TVOLUME.  The volume rate-of-change is
    %   calculated between the current volume and the volume 12 periods
    %   ago.
    d_volroc=volroc(dat(:,vol));
    d_volRoc_sma50 = tsmovavg (d_volroc , 's', 50,1);
    
    %% [FOR DEBUG-PURPOSES] PLOT tech inidicators
    if(0)
        dateIdx=1:length(dat(:,t));
        figIdx=1;
        figure(figIdx);figIdx=figIdx+1;
        plot(dateIdx,dat(:,cl),'black',...
            dateIdx,sma20,'black--',...
            dateIdx, sma50,'red--',...
            dateIdx, sma200,'blue--');
        plotBreakdownsAndBounce(bd{tId},nbd{tId},dat(:,cl));
        title([TICKER{tId} ' Stock price and SMAs from ' fromDate{tId} ' to ' toDate{tId}]);
        xlim([200 dateIdx(end)]); xlabel('days');ylabel('$')
        legend('price','sma20','sma50','sma200');
        
        figure(figIdx);figIdx=figIdx+1;
        plot(dateIdx,d_rsindex,'black',...
            dateIdx, repmat(70,1,length(dateIdx)),...
            dateIdx, repmat(30,1,length(dateIdx)))
        plotBreakdownsAndBounce(bd{tId},nbd{tId},d_rsindex);
        title('RSI');xlim([200 dateIdx(end)]);
        
        figure(figIdx);figIdx=figIdx+1;
        plot(dateIdx,dat(:,cl),'black',...
            dateIdx,d_bollinger_mid,'black--',...
            dateIdx, d_bollinger_uppr_1sig,'red--',...
            dateIdx, d_bollinger_lowr_1sig,'red',...
            dateIdx, d_bollinger_uppr_2sig,'blue--',...
            dateIdx, d_bollinger_lowr_2sig,'blue')
        legend('$','midBollinger','uppr1sig','lowr1sig','uppr2sig','lowr2sig')
        plotBreakdownsAndBounce(bd{tId},nbd{tId},dat(:,cl));
        title('Bollinger Bands');xlim([200 dateIdx(end)]);
        
        figure(figIdx);figIdx=figIdx+1;
        plot(dateIdx, abs(dat(:,cl)./sma20),'black',...
            dateIdx, abs(dat(:,cl)./sma50),'black--',...
            dateIdx, abs(dat(:,cl)./sma200),'black-.',...
            dateIdx, abs(sma20./sma50 ), 'red',...
            dateIdx, abs(sma50./sma200), 'red--',...
            dateIdx, abs(sma20./sma200), 'red-.',...
            dateIdx, repmat(1,1,length(dateIdx)),'black')
        legend('--$/sma20','--$/sma50','--$/sma200','sma20/sma50','sma50/sma200','sma20/sma200');
        plotBreakdownsAndBounce(bd{tId},nbd{tId},ones(dateIdx(end),1));
        title('$,SMA20/50/200');xlim([200 dateIdx(end)]);
        
        figure(figIdx);figIdx=figIdx+1;
        plot(dateIdx,d_volroc,'black',...
            dateIdx,tsmovavg(d_volroc,'s',50,1),'r--');
        plotBreakdownsAndBounce(bd{tId},nbd{tId},d_volroc);
        title('Volume Rate of Change');xlim([200 dateIdx(end)]);
        legend('volRoC','volRoc(sma50)');
        
        %keyboard;
    end
    
    %% COMPUTE FEATURES AND FEAT.VECTOR FOR sampleS
    % We start with breakdowns and then bounces.
    for listIdx=1:2
        if (listIdx==1)
            list = bd;
        elseif (listIdx==2)
            list = nbd;
        end
        for i=list{tId}
            featNum=1;
            %% FEATURES (LEADING INDICATORS)
            %% RSI
            %Feature #1
            F(featNum) = d_rsindex(i); featNames{featNum}='RSI';featNum=featNum+1;
            %Feature #2
            tmp= diff(sma20);
            F(featNum) = d_rsindex(i)*nanmean(tmp(i-10:i)); featNames{featNum}='RSI/(dSMA20/dt)';featNum=featNum+1;
            %% Trend in the last 20 days 3 Feature #2
            tmp= diff(sma20);
            %Feature #3
            F(featNum) = nanmean(tmp(i-10:i)); featNames{featNum}='dSMA20/dt';featNum=featNum+1;
            %% Moving Averate to Current price ratios:
            %Feature #4-6
            F(featNum) = dat(i,cl)/sma20(i);featNames{featNum}='$/SMA20';featNum=featNum+1;
            F(featNum) = dat(i,cl)/sma50(i);featNames{featNum}='$/SMA50';featNum=featNum+1;
            F(featNum) = dat(i,cl)/sma200(i);featNames{featNum}='$/SMA200';featNum=featNum+1;
            %% Moving Average ratios
            %Feature #7-9
            F(featNum) = sma20(i)/sma50(i);featNames{featNum}='SMA20/SMA50';featNum=featNum+1;
            F(featNum) = sma20(i)/sma200(i);featNames{featNum}='SMA20/SMA200';featNum=featNum+1;
            F(featNum) = sma50(i)/sma200(i);featNames{featNum}='SMA50/SMA200';featNum=featNum+1;
            %% Bollinger Band
            %         if (dat(i,cl) <= d_bollinger_lowr_1sig(i) && dat(i,cl) >= d_bollinger_lowr_2sig(i))
            %             F(featNum) = 1;
            %         elseif (dat(i,cl) <= d_bollinger_uppr_2sig(i) && dat(i,cl) >= d_bollinger_uppr_1sig(i))
            %             F(featNum) = -1;
            %         else
            %             F(featNum) = 0;
            %         end;featNames{featNum}='BB';featNum=featNum+1;
            %Feature #10-15
            F(featNum) = dat(i,cl)/d_bollinger_lowr_1sig(i);featNames{featNum}='$/BBL1sig';featNum=featNum+1;
            F(featNum) = dat(i,cl)/d_bollinger_lowr_2sig(i);featNames{featNum}='$/BBL2sig';featNum=featNum+1;
            F(featNum) = dat(i,cl)/d_bollinger_uppr_1sig(i);featNames{featNum}='$/BBU1sig';featNum=featNum+1;
            F(featNum) = dat(i,cl)/d_bollinger_uppr_2sig(i);featNames{featNum}='$/BBU2sig';featNum=featNum+1;
            F(featNum) = d_bollinger_lowr_1sig(i)/d_bollinger_uppr_1sig(i);featNames{featNum}='BBL1sig/BBU1sig';featNum=featNum+1;
            F(featNum) = d_bollinger_lowr_2sig(i)/d_bollinger_uppr_2sig(i);featNames{featNum}='BBL2sig/BBU2sig';featNum=featNum+1;
            
            %% MACD
            %Feature #16-17
            F(featNum) = d_macd(i);featNames{featNum}='MACD';featNum=featNum+1;
            F(featNum) = d_macd(i)/d_ema9(i);featNames{featNum}='MACD/EMA9';featNum=featNum+1;
            
            %% VOL RATE OF CHANGE
            F(featNum) = d_volroc(i);featNames{featNum}='dVol/dt';featNum=featNum+1;
            %             if (d_volroc(i) < d_volRoc_sma50(i))
            %                 F(featNum)=1;
            %             else
            %                 F(featNum)=-1;
            %             end; featNames{featNum}='dVolDt/sma50 as -1/+1';featNum=featNum+1;
            F(featNum) = d_volroc(i) - d_volRoc_sma50(i);featNames{featNum}='dVolDt-sma50';featNum=featNum+1;
            
            
            %% FORM FEATURE VECTOR FOR CURRENT SAMPLE
            exemFeat(sampleCounter,:)=F;
            if(listIdx == 1)
                exemLabel(1,sampleCounter)=1;
            elseif (listIdx == 2)
                exemLabel(1,sampleCounter)=0;
            end
            sampleCounter=sampleCounter+1;
        end
    end
end


disp('---------DONE GETTING DATA-----------')
if(save2File)
    disp('---------SAVING DATA to MATFILE: stockData2samples.mat -----------')
    save('stockData2samples.mat','exemLabel','featNames', 'exemFeat')
end

return
