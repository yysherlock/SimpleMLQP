%% Netural network Assignment 1

% author: Luo zhiyi(0130339024)
%         ShanghaiJiaoTong University, department of Computing, SEIEE-3-341

inputSize = 2;
outputSize = 1;
hiddenSize = 10;

lambda = 0.001; % regularization parameter
alpha = 0.1; % learning rate
momentum = 0.5; % Momentum
epsilon = 1e-4;

[traindata,testdata] = dataloading();
%% Step 0: Let us have a look at the data
% Here we can plot the original training data out
drawTrainData(traindata);

%% Step 1:
% Obtain random paratemers theta
close;

%% Step 4: Train MLQP Model

theta = initializeParameters(hiddenSize,inputSize,outputSize);
maxIter = 200 * size(traindata,1);
iter = 1;
index = 1;
timeBegin = clock;

while (iter < maxIter  && cost > epsilon)
    
    if mod(index,m+1)==0
        index = 1;
    else
        index = mod(index,m+1);
    end
    
    [cost,grad] = onlineCost(theta,lambda,inputSize,outputSize,hiddenSize,traindata,index);
    
    theta = theta - alpha * grad;

    index = index + 1;
    iter = iter + 1;
end
opttheta = theta;
timeEnd = clock;
% The optimal parameters were stored in the opttheta as a vector.

%% Step 5: Test the Model

% biclassifier, set threshold to 0.5


[result,label] = applyModel(testdata,opttheta,inputSize,outputSize,hiddenSize);

[correct,correctindex,m,precision] = precisionCalculate(result,label);

re = [label' result' correctindex'];
disp(re);
disp([correct,m,precision]);
% calculate the time
timeCost = timeEnd - timeBegin;
str=sprintf('This online learning algorithm costs %s seconds.',num2str(timeCost(6)));
disp(str);
% drawResult(testdata,result);





