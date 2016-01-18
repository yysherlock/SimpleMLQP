%% Netural network Assignment 1

% author: Luo zhiyi(0130339024)
%         ShanghaiJiaoTong University, department of Computing, SEIEE-3-341

inputSize = 2;
outputSize = 1;
hiddenSize = 10;

lambda = 0.0001; % regularization parameter
alpha = 0.2; % learning rate
momentum = 0.5; % Momentum

[traindata,testdata] = dataloading();
%% Step 0: Let us have a look at the data
% Here we can plot the original training data out
%drawData(traindata);
drawData(testdata);

%% Step 1:
% Obtain random paratemers theta
%close;
theta = initializeParameters(hiddenSize,inputSize,outputSize);


%% Step 2:
[cost,grad] = batchCost(theta,lambda,inputSize,outputSize,hiddenSize,traindata);
%[cost,grad] = onlineCost(theta,lambda,inputSize,outputSize,hiddenSize,traindata);

%% Step 3: Gradient Checking
% Check1: computeNumericalGradient is correct or not.
checkNumericalGradient();

% Check2: batchCost is correct or not
%numgrad = computeNumericalGradient(@(x)batchCost(x,lambda,inputSize,outputSize,...
 %   hiddenSize,traindata),theta);
numgrad = computeNumericalGradient(@(x)batchCost(x,lambda,inputSize,outputSize,...
    hiddenSize,traindata),theta);

disp([numgrad grad]);

diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff);

%% Step 4: Train MLQP Model
timeBegin = clock;
theta = initializeParameters(hiddenSize,inputSize,outputSize);

addpath ../minFunc/
options.Method = 'lbfgs'; % use L-BFGS to optimize the cost function.
                          % Generally, for minFunc to work, we need a
                          % function pointer with two outputs: the function
                          % value and the gradient. batchCost.m satisfies
                          % this.
                          
options.maxIter = 400;    % Maximum number of iterations of L-BFGS to run.
options.display = 'on';

% [opttheta,cost] = minFunc(@(p) batchCost(p,lambda,inputSize,outputSize,...
%     hiddenSize,traindata),theta,options);
[opttheta,cost] = minFunc(@(p) batchCost(p,lambda,inputSize,outputSize,...
    hiddenSize,traindata),theta,options);

% The optimal parameters were stored in the opttheta as a vector.
timeEnd = clock;
%% Step 5: Test the Model

% biclassifier, set threshold to 0.5


[result,label] = applyModel(testdata,opttheta,inputSize,outputSize,hiddenSize);

[correct,correctindex,m,precision] = precisionCalculate(result,label);

re = [label' result' correctindex'];
disp(re);
disp([correct,m,precision]);
% calculate the time
timeCost = timeEnd - timeBegin;
str=sprintf('This batch learning algorithm costs %s seconds.',num2str(timeCost(6)));
disp(str);
drawResult(testdata,result,1);

% optimal parameters 

rx = zeros(100000,1);
ry = zeros(100000,1);

% draw decision region
for i = 1:100000
    rx(i,1) = -4 + (4 - (-4))*rand(1);
    ry(i,1) = -4 + (4 - (-4))*rand(1);
end

[sol] = ff(opttheta,[rx,ry],inputSize,outputSize,hiddenSize);

close;
drawResult([rx,ry],sol,3);

