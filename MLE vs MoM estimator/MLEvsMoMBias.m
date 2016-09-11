%Detection and Estimation Theory
 
close all;
experimentNum=1;
%% Generate 100 experiments using N observations from uniform pdf
% # of observations
for N=10:1000;
    clearvars -except N experimentNum MLE MoM
    theta=1;
    %Observations value
    x=rand(N,100);
    
    %MoM theta estimator
    %Given N observations, estimate theta for EACH EXPERIMENT,
    theta_MoM_estimator=(2/N)*sum(x(1:N,[1:100]'));
    %Now, get the theoretical and experimental means
    mean_theoretical_MoM=theta;
    mean_expmtl_MoM=mean(theta_MoM_estimator);
    
    %MLE theta estimator
    %Given N observations, estimate theta for EACH EXPERIMENT,
    theta_MLE_estimator=max(x(1:N,[1:100]'));
    %Now, get the theoretical and experimental means
    mean_theoretical_MLE=theta*N/(N+1);
    mean_expmtl_MLE=mean(theta_MLE_estimator);
    
    %Save the theoretical/experimental data point pairs for each estimator type
    %MLE(experiment_with_N_observations,<1=theoretical,2=experimental>)
    MLE(experimentNum,1)=mean_theoretical_MLE;
    MLE(experimentNum,2)=mean_expmtl_MLE;
    %MLE(experiment_with_N_observations,<1=theoretical,2=experimental>)
    MoM(experimentNum,1)=mean_theoretical_MoM;
    MoM(experimentNum,2)=mean_expmtl_MoM;
    
    experimentNum=experimentNum+1;
end
plot(10:N,MLE(:,1),'r-');hold on;
plot(10:N,MLE(:,2),'g-');hold on;
plot(10:N,MoM(:,1),'b-');hold on;
plot(10:N,MoM(:,2),'m-');hold on;
xlabel('Number of Observations');xlim([10 N])
ylabel('Expected value of estimator');
title('MLE and MoM theoretical and experimental expected value comparison')
legend('MLE theoretical','MLE experimental','MoM theoretical','MoM experimental')
