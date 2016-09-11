%Detection and Estimation Theory - Midterm Exam

close all;

experimentNum=1;
%% Generate 100 experiment using N observations from uniform pdf
% # of observations
for N=10:1000;
    clearvars -except N experimentNum MLE MoM
    theta=1;
    %Observations value
    x=rand(N,100);
    % %Sample Mean (for each experiment there is matlab mean=sample mean)
    % mean_sample=mean(x(1:N,[1:100]')); %take mean of each run and then average it
    
    %MoM theta estimator
    %Given N observations, estimate theta for EACH EXPERIMENT,
    theta_MoM_estimator=(2/N)*sum(x(1:N,[1:100]'));
    %Now, get the theoretical and experimental variance
    variance_theoretical_MoM=theta^2/(3*N);
    variance_expmtl_MoM=var(theta_MoM_estimator);
    
    %MLE theta estimator
    %Given N observations, estimate theta for EACH EXPERIMENT,
    theta_MLE_estimator=max(x(1:N,[1:100]'));
    %Now, get the theoretical and experimental means
    variance_theoretical_MLE=(N*theta^2)/((N+2)*(N+1)^2);
    variance_expmtl_MLE=var(theta_MLE_estimator);
    
    %Save the theoretical/experimental data point pairs for each estimator type
    %MLE(experiment_with_N_observations,<1=theoretical,2=experimental>)
    MLE(experimentNum,1)=variance_theoretical_MLE;
    MLE(experimentNum,2)=variance_expmtl_MLE;
    %MLE(experiment_with_N_observations,<1=theoretical,2=experimental>)
    MoM(experimentNum,1)=variance_theoretical_MoM;
    MoM(experimentNum,2)=variance_expmtl_MoM;
    
    experimentNum=experimentNum+1;
end
plot(10:N,MLE(:,1),'r-');hold on;
plot(10:N,MLE(:,2),'g-');hold on;
plot(10:N,MoM(:,1),'b-');hold on;
plot(10:N,MoM(:,2),'m-');hold on;
xlabel('Number of Observations');xlim([10 N])
ylabel('variance');
title('MLE and MoM theoretical and experimental variance comparison')
legend('MLE theoretical','MLE experimental','MoM theoretical','MoM experimental')





