function network = myclassify(data, filled)
    
    architecture = menu('Neural Network Architecture:','Associative Memory + Classifier', 'Classifier Only');
    act_fun_option = menu('Type of Activation Function: ', 'Binary', 'Linear', 'Sigmoidal');
    dataSet = menu('Training Data size : ', '500','750')
    if(architecture == 1)
        if(act_fun_option == 1)
            
            if(dataSet == 1) load('hardLimA1.mat');
            else load('hardLimA1D2.mat');
            end
            
        elseif(act_fun_option == 2)
            
            if(dataSet == 1) load('pureLinA1.mat');
            else load('pureLinA1D2.mat');
            end
            
        else
            if(dataSet == 1) load('logSigA1.mat');
            else load('logSigA1D2.mat');
            end
            
        end
    else
        if(act_fun_option == 1)
                        
            if(dataSet == 1) load('hardLimA2.mat');
            else load('hardLimA2D2.mat');
            end
            
        elseif(act_fun_option == 2)
                        
            if(dataSet == 1) load('hardLimA2.mat');
            else load('hardLimA2D2.mat');
            end
            
        else
                        
            if(dataSet == 1) load('hardLimA2.mat');
            else load('hardLimA2D2.mat');
            end
            
        end
    end
    
results=net(data);
[x,y]=max(results);
network=int64(y(filled));
        
end
        

        
    
    
    
        
    
    
    
    
    
    
    
    