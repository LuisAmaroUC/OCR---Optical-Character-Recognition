function network = trainingNetwork()

    option = menu('Do you want to Train or Classify ')

    load('PerfectArial.mat');   %It is the target function. Is used when the input character is not perfect
    %load('Labels.mat'); %Is used to tell the network what the expected output is
    
    load('TesteDataOrder.mat'); %Data used to validate the network -> P10
    load('TesteDataOrder1.mat'); %Data used to test the network -> P11
    load('TesteDataOrder2.mat'); %Data used to test the network -> P12
    load('matrizTeste150.mat'); %Data used to test and validate the network
    
    load('TesteDataNoOrder.mat'); %Data used to validate the network -> P13
    load('TesteDataNoOrder1.mat'); %Data used to validate the network -> P14
    load('TesteDataNoOrder2.mat'); %Data used to validate the network -> P15
    load('matrizTeste150NoOrder.mat'); %Data used to test and validate the network
    
    %The use can train the network with 500 or 750 test cases
    dataSet = menu('Dataset: ','500','750');
    
    if(dataSet == 1) load('matriz500.mat'); column = 500;       
    else load('matriz750.mat'); column = 750;               
    end
    
    %Preencher a matriz target. Como sabemos que os dados vai ser inseridos
    %de forma ordenada basta preencher a matriz com 1´s e 0´s, sendo que
    %cada coluna vai ter 1 no valor do output esperado. Tudo isto de forma
    %ciclica ja que apenas temos 10 valores possíveis. Ficamos entao com
    %uma matriz com dimensoes de 10*500 ou 10*750 dependendo do dataset
    %escolhido
    matrizTargetOutputs = zeros(10,column);
    for i = 0 : column - 1
        matrizTargetOutputs(mod(i,10) + 1, i + 1) = 1; 
    end
    
    %Para a memoria associativa teremos de ter uma matriz 250*500 ou
    %250*750, para o caso de algum caracter de input n/ ser possivel de
    %identificar, iremos recorrer a esta matriz para "corrigir" esse input
    arrayTargetAssMem = repmat(Perfect,1,column/10);
    
    architecture = menu ('Arquitetura : ', 'Associative Memory + Classifier', 'Classifier Only');
    
    if(dataSet == 1) 
        if(architecture == 1)
            Wp = arrayTargetAssMem * pinv(matriz500); %os pesos vao ser iguis ao valor do output(256,(500,750)) desejado a multiplicar pela inversao dos inputs(256,(500,750)) 
            P2 = Wp * matriz500;                    %O valor do output vai ser igual ao valor do input multiplicado pelo pesos
            
            %FOR TESTING
            testData = Wp * P10;
            %TestData= Wp * P11;
            %TestData = Wp * P12;
            %TestData = Wp * matrizTeste150;
            
            % NO ORDER
            testDataNoOrder = Wp * P13;
            %TestDataNoOrder = Wp * P14;
            %TestDataNoOrder = Wp * P15;
            %TestDataNoOrder = Wp * matrizTeste150NoOrder;
        else 
            P2 = matriz500;
        end
            
    else
        if(architecture == 1)
            Wp = arrayTargetAssMem * pinv(matriz750); %os pesos vao ser iguis ao valor do output(256,(500,750)) desejado a multiplicar pela inversao dos inputs(256,(500,750)) 
            P2 = Wp * matriz750;   %O valor do output vai ser igual ao valor do input multiplicado pelo pesos
            
            %FOR TESTING
            TestData = Wp * P10;
            %TestData= Wp * P11;
            %TestData = Wp * P12;
            %TestData = Wp * matrizTeste150;
            
            % NO ORDER
            TestDataNoOrder = Wp * P13;
            %TestDataNoOrder = Wp * P14;
            %TestDataNoOrder = Wp * P15;
            %TestDataNoOrder = Wp * matrizTeste150NoOrder;
        else 
            P2 = matriz750;
        end
    end
    
    
    net = perceptron;   %Perceptrons are simple single-layer binary classifiers, which divide the input space with a linear decision boundary.
                                %The matrices P and T define the number of the inputs and outputs (and so the number of neurons).
                                %(default hardlim, learnp) can be changed.
    net = configure(net, P2,matrizTargetOutputs) 
            

    %sub-object properties
    %Devemos fazer "fine-tunning" de modo a tentar obter melhores
    %resultados?
    weights = rand(10,256); %256 inputs (pixels) and 10 neurons
    bias = rand(10,1); %1 bias value to each neuron, the value will be between 0 and 1

    net.IW{1,1} = weights;
    net.b{1,1} = bias;

    %try to initiaze the weights and biases at 0 to see what happens
    %net.IW{};
    %net.b{1,1};
    net.divideFcn = 'divideblock';
    net.divideParam.valRatio = 15/100;  %validation     Valudation. While training one verifies if the error in this set u; when the error augments from an interation to another, 
    %then the network is entering the overfitting condition, and the training should be stopped
    net.divideParam.trainRatio = 85/100; %training
            
    %net.divideParam.trainRatio = 70/100 %training
    %net.divideParam.valRatio = 15/100; %validation
    %net.divideParam.testRatio = 15/100; %testing
    
    
    if(architecture == 1)      %Associative Memory + Classifier
    
            activationFunction = menu ('Função de Ativação : ', 'Hardlim', 'Linear', 'Sigmoidal');

            if(activationFunction == 1)         %hardlim



                %Training Parameters

                %net.perfromParam.lr = 0.5  %learning rate| default value is 0.01
                net.trainParam.epochs = 300;   %The default is 1000 %The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
                net.trainParam.show = 35;   %The default is 25 %show| Epochs between displays
                %net.trainParam.goal = 1e-6;     %The default is 0 %goal=objective Performance ggoal
                net.performFcn = 'sse';         %criterion | (Sum Squared error)

                %Train 
                net = train(net, P2,matrizTargetOutputs)
                
                sim(net,P10)
                %sim(net,P13) % simulates the specified model using the parameter values specified in the structure ParameterStruct.
                
                if(dataSet == 1)  hardLimA1 = net;save hardLimA1;
                else  hardLimA1D2 = net;save hardLimA1D2;
                end


            elseif(activationFunction == 2)           %linear

                net.layers{1}.transferfcn = 'purelin'; %This is really necessary, since it is already the deafult function
                net.inputWeights{1}.learnFcn = 'learngd';  %incremental learning
                net.biases{1}.learnFcn = 'learngd';
                net.trainFcn = 'traingd'; %Batch training 
                
                %Training Parameters

                %net.perfromParam.lr = 0.5  %learning rate| default value is 0.01
                net.trainParam.epochs = 1000;   %The default is 1000 %The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
                net.trainParam.show = 35;   %The default is 25 %show| Epochs between displays
                %net.trainParam.goal = 1e-6;     %The default is 0 %goal=objective Performance ggoal
                net.performFcn = 'mse';         %criterion | (Mean Squared error) 
                %WHY WE USE MSE INSTEAD OF SSE IN HERE? WTF
                
                %Train
                net = train(net, P2, matrizTargetOutputs)
                
                sim(net,testData)
                %sim(net,testDataNoOrder) % simulates the specified model using the parameter values specified in the structure ParameterStruct.
                
                if(dataSet == 1)  hardLimA1 = net;save hardLimA1;
                else  hardLimA1D2 = net;save hardLimA1D2;
                end
                
            elseif(activationFunction == 3)     %Logsig (Sigmoid trasnfer Function)
                
                net.layers{1}.transferfcn = 'logsig'; 
                net.inputWeights{1}.learnFcn = 'learngd';  %incremental learning
                net.biases{1}.learnFcn = 'learngd';
                net.trainFcn = 'traingd'; %Batch training 
                
                %Training Parameters

                %net.perfromParam.lr = 0.5  %learning rate| default value is 0.01
                net.trainParam.epochs = 10000;   %The default is 1000 %The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
                net.trainParam.show = 35;   %The default is 25 %show| Epochs between displays
                %net.trainParam.goal = 1e-6;     %The default is 0 %goal=objective Performance ggoal
                net.performFcn = 'mse';         %criterion | (Mean Squared error) 
                %WHY WE USE MSE INSTEAD OF SSE IN HERE? WTF
                
                net = train(net, P2, matrizTargetOutputs);
                
                sim(net,P10)
                %sim(net,P13) % simulates the specified model using the parameter values specified in the structure ParameterStruct.

                if(dataSet == 1)  hardLimA1 = net;save hardLimA1;
                else  hardLimA1D2 = net;save hardLimA1D2;
                end
                
            else disp("Invalid INPUT")
                
            end
        
    else     %Classifier
        
            activationFunction = menu ('Função de Ativação : ', 'Hardlim', 'Linear', 'Sigmoidal');

            if(activationFunction == 1)         %hardlim

                %Training Parameters

                %net.perfromParam.lr = 0.5  %learning rate| default value is 0.01
                net.trainParam.epochs = 300;   %The default is 1000 %The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
                net.trainParam.show = 35;   %The default is 25 %show| Epochs between displays
                %net.trainParam.goal = 1e-6;     %The default is 0 %goal=objective Performance ggoal
                net.performFcn = 'sse';         %criterion | (Sum Squared error)

                %Train 
                net = train(net, P2,matrizTargetOutputs)
                
                sim(net,P10)
               %sim(net,P13) % simulates the specified model using the parameter values specified in the structure ParameterStruct.

                
                if(dataSet == 1)  hardLimA2 = net;save hardLimA2;
                else  hardLimA2D2 = net;save hardLimA2D2;
                end


            elseif(activationFunction == 2)           %linear

                net.layers{1}.transferfcn = 'purelin'; %This is really necessary, since it is already the deafult function
                net.inputWeights{1}.learnFcn = 'learngd';  %incremental learning
                net.biases{1}.learnFcn = 'learngd';
                net.trainFcn = 'traingd'; %Batch training 
                
                %Training Parameters

                %net.perfromParam.lr = 0.5  %learning rate| default value is 0.01
                net.trainParam.epochs = 300;   %The default is 1000 %The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
                net.trainParam.show = 35;   %The default is 25 %show| Epochs between displays
                %net.trainParam.goal = 1e-6;     %The default is 0 %goal=objective Performance ggoal
                net.performFcn = 'mse';         %criterion | (Mean Squared error) 
                %WHY WE USE MSE INSTEAD OF SSE IN HERE? WTF
                
                %Train
                net = train(net, P2, matrizTargetOutputs)
                
                sim(net,P10)
                %sim(net,P13) % simulates the specified model using the parameter values specified in the structure ParameterStruct.
               
                
                if(dataSet == 1) pureLinA2 = net;save pureLinA2;
                else pureLinA2D2 = net;save pureLinA22D2;
                end
                
            elseif(activationFunction == 3)     %Logsig (Sigmoid trasnfer Function)
                
                net.layers{1}.transferfcn = 'logsig'; 
                net.inputWeights{1}.learnFcn = 'learngd';  %incremental learning
                net.biases{1}.learnFcn = 'learngd';
                net.trainFcn = 'traingd'; %Batch training 
                
                %Training Parameters

                %net.perfromParam.lr = 0.5  %learning rate| default value is 0.01
                net.trainParam.epochs = 1000;   %The default is 1000 %The number of epochs define the number of times that the learning algorithm will work trhough the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
                net.trainParam.show = 35;   %The default is 25 %show| Epochs between displays
                %net.trainParam.goal = 1e-6;     %The default is 0 %goal=objective Performance ggoal
                net.performFcn = 'mse';         %criterion | (Mean Squared error) 
                %WHY WE USE MSE INSTEAD OF SSE IN HERE? WTF
                
                net = train(net, P2, matrizTargetOutputs);
                
                sim(net,P10)
                %sim(net,P13) % simulates the specified model using the parameter values specified in the structure ParameterStruct.
      
                
                if(dataSet == 1) logSigA2 = net;save logSigA2;
                else logSigA2 = net;save logSigA2D2;
                end
                
            else disp("Invalid INPUT")
                
            end
        
    end
        