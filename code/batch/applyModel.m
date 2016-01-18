function [result,label] = applyModel(testdata,opttheta,inputSize,outputSize,hiddenSize)

U1 = reshape(opttheta(1:hiddenSize*inputSize),hiddenSize,inputSize);
U2 = reshape(opttheta(hiddenSize*inputSize+1:hiddenSize*(inputSize+outputSize)),outputSize,hiddenSize);
V1 = reshape(opttheta(hiddenSize*(inputSize+outputSize)+1:hiddenSize*(2*inputSize+outputSize)),hiddenSize,inputSize);
V2 = reshape(opttheta(hiddenSize*(2*inputSize+outputSize)+1:2*hiddenSize*(inputSize+outputSize)),outputSize,hiddenSize);
b1 = opttheta(2*hiddenSize*(inputSize+outputSize)+1:2*hiddenSize*(inputSize+outputSize)+hiddenSize);
b2 = opttheta(2*hiddenSize*(inputSize+outputSize)+hiddenSize+1:end);

inputData = testdata(:,1:2)'; % traindata: 96 x 3, inputData: 2 x 96 -
                                     % InputData: each column is an example
label = testdata(:,3)'; % 1 x 96 dimension

m = size(testdata,1);

% FeedForward pass
Z1 = U1*(inputData.^2)+V1*inputData+repmat(b1,1,m); % 10 x 96
A1 = sigmoid(Z1); % 10 x 96
Z2 = U2*(A1.^2) + V2*A1 + repmat(b2,1,m); % 1 x 96
A2 = sigmoid(Z2); % 1 x 96

result = A2;
end


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
