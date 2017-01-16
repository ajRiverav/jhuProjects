function trainAndEvalNN(    scaledExemplars,...
                            targetLabels ,...
                            trainingSetIdx ,...
                            validationSetIdx,...
                            testSetIdx,...
                            filename, ...
                            featCombo)
%Trains and Evaluates a neural network
%
% 2016-4-17 AJ RIVERA, Johns Hopkins University

% Use Resilient backprop algorithm for training. 
trainingAlgorithms = {'trainrp'};

for i=1:length(trainingAlgorithms)
    
    curTrainAlgo = trainingAlgorithms{i};
    
    % Use 10 neurons in single hidden layer.
    for numNeuronsHidLayer= 10
        
        %Perform training 5 times since the initial weights matter when it
        %comes to NN performance.
        %Per Matlab, init(nn) initializes the nn. The initial weights
        %depend on which init function each layer is using. 
        %In this case, we are using initnw, which is a layer initialization 
        %function that initializes a layer's weights and biases according 
        %to the Nguyen-Widrow initialization algorithm. 
        %This algorithm chooses values in order to distribute the active 
        %region of each neuron in the layer approximately evenly across 
        %the layer's input space. The values contain a degree of randomness, 
        %so they are not the same each time this function is called, which
        %is what we want. 
        
        for trainingRun = 1:5
            
            nn = patternnet(numNeuronsHidLayer, curTrainAlgo);
            nn=init(nn);
            
            %only stop if validation error increases max_fail consecutive times
            %     OR
            %we reach 1,000 training epochs
            nn.trainParam.max_fail = 30;
            nn.trainParam.epochs=1000;
            
            nn.trainParam.showWindow=0;%turns off 'show training window feedback'
            nn.inputs{1}.processFcns={};%no processing Needed, inputs have been minmax'ed.
            
            %Now, we need to let the neural network know which indices in the
            %data correspond to the validation and training sets.
            nn.divideFcn = 'divideind';
            nn.divideParam.trainInd= trainingSetIdx;
            nn.divideParam.valInd = validationSetIdx;
            nn.divideParam.testInd = testSetIdx;
            
            %Train Neural neural Network
            [nn] = train(nn,scaledExemplars,targetLabels);
            
            %Now evaluate performance with the training, validation, and test
            %sets. See reason in paper
            %"Generating ROC Curves for Artificial Neural Networks", Woods & Bowyer
            
            set = {trainingSetIdx,validationSetIdx,testSetIdx};
            setStr = {'trainingSet','validationSet','testSet'};
            for idx = 3 %Only report results from test set. 
                
                predLabels = nn(scaledExemplars(:,set{idx}));
                
                %Compute confusion matrix
                [FAR,TPR,T,AUC] = perfcurve(targetLabels(set{idx}),predLabels,1);
                FNR = 1-TPR;
                TNR = 1-FAR;
                
                %Write results to a file
                % The data here allows us to construct a ROC curve
                % which is uncommon for NNs. See referenced paper above.
                
                %parfor threshold=1:length(T)
                for threshold=1:length(T)
                    
                    %curWorker=getCurrentWorker();
                    %id=curWorker.ProcessId;
                    id='1';
                    
                    str2 = [curTrainAlgo ...
                        ',' setStr{idx} ...
                        ',' num2str(trainingRun) ...
                        ',' num2str(featCombo) ...
                        ',' num2str(numNeuronsHidLayer) ...
                        ',' num2str(threshold) ...
                        ',' num2str(AUC) ...
                        ',' num2str(FAR(threshold)) ...
                        ',' num2str(TPR(threshold)) ...
                        ',' num2str(FNR(threshold)) ...
                        ',' num2str(TNR(threshold)) ...
                        '\n'];
                    
                    writeToFile(str2,filename,id);
                end
                
                
            end
            
        end

       
        

        
        
    end
end
end

