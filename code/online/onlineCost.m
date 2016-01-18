function [cost,grad] = onlineCost(theta,lambda,inputSize,outputSize,hiddenSize,traindata,index)

% InoutSize and OutputSize are all two dimensions, say visibleSize.
% hiddenSize is 10.
% The input theta is a vector because minFunc expects the parameters to be
% a vector. We first convert theta to the {U1,U2,V1,V2,b1,b2} matrix/vector
% format, so that it is convinient to be processed.


U1 = reshape(theta(1:hiddenSize*inputSize),hiddenSize,inputSize);
U2 = reshape(theta(hiddenSize*inputSize+1:hiddenSize*(inputSize+outputSize)),outputSize,hiddenSize);
V1 = reshape(theta(hiddenSize*(inputSize+outputSize)+1:hiddenSize*(2*inputSize+outputSize)),hiddenSize,inputSize);
V2 = reshape(theta(hiddenSize*(2*inputSize+outputSize)+1:2*hiddenSize*(inputSize+outputSize)),outputSize,hiddenSize);
b1 = theta(2*hiddenSize*(inputSize+outputSize)+1:2*hiddenSize*(inputSize+outputSize)+hiddenSize);
b2 = theta(2*hiddenSize*(inputSize+outputSize)+hiddenSize+1:end);

% LMS Cost
% Initialize the cost and grads to zeros
cost = 0;
U1grad = zeros(hiddenSize,inputSize);
U2grad = zeros(outputSize,hiddenSize);
V1grad = zeros(hiddenSize,inputSize);
V2grad = zeros(outputSize,hiddenSize);
b1grad = zeros(hiddenSize,1);
b2grad = zeros(outputSize,1);

m = size(traindata,1); % number of training examples
inputData = traindata(:,1:2)'; % traindata: 96 x 3, inputData: 2 x 96 -
% InputData: each column is an example
label = traindata(:,3)'; % 1 x 96 dimension

% We can design the structured nn to improve the code. I'll write this later.
% Here, we just on the way.

% FeedForward pass
Z1 = U1*(inputData.^2)+V1*inputData+repmat(b1,1,m); % 10 x 96
A1 = sigmoid(Z1); % 10 x 96
Z2 = U2*(A1.^2) + V2*A1 + repmat(b2,1,m); % 1 x 96
A2 = sigmoid(Z2); % 1 x 96

error = label - A2; % 1 x 96
% batch: calculate the gradient using all train examples
% i = randi(m);

%for i = 1:m
i = index;
%i=randi(m);
data = inputData(:,i); % 2 x 1
z1 = Z1(:,i); % 10 x 1
z2 = Z2(:,i); % 1
a1 = A1(:,i); % 10 x 1
a2 = A2(:,i); % 1

%     % LMS
%     delta2 = -fgradient(z2)*error(:,i); % 1
%     delta1 = fgradient(z1).*(2*U2'.*a1 + V2') * delta2; % 10 x 1
%     U2grad = U2grad + delta2*(a1.^2)';% 1 x 10
%     U1grad = U1grad + delta1*(data.^2)';% 10 x 2
%     V2grad = V2grad + delta2*a1';% 1 x 10
%     V1grad = V1grad + delta1*data';% 10 x 2
%     b2grad = b2grad + delta2;
%     b1grad = b1grad + delta1;

% logistic cost function
delta2 = -fgradient(z2)*error(:,i)/(a2*(1-a2)); % 1
delta1 = fgradient(z1).*(2*U2'.*a1 + V2') * delta2; % 10 x 1
U2grad = U2grad + delta2*(a1.^2)';% 1 x 10
U1grad = U1grad + delta1*(data.^2)';% 10 x 2
V2grad = V2grad + delta2*a1';% 1 x 10
V1grad = V1grad + delta1*data';% 10 x 2
b2grad = b2grad + delta2;
b1grad = b1grad + delta1;

%end

U2grad = U2grad + lambda*U2;U1grad = U1grad + lambda*U1;
V2grad = V2grad + lambda*V2;V1grad = V1grad + lambda*V1;
%b1grad = b1grad;b2grad = b2grad;

% LMS cost
% cost = 0.5 * (1/m) * (sum(error.^2)) + 0.5*lambda*(sum(sum(U1.*U1))+sum(sum(U2.*U2))+sum(sum(V1.*V1))+sum(sum(V2.*V2)));

% logistic cost
cost = -sum(label.*log(a2) + (1-label).*(log(1-a2)))+ 0.5*lambda*(sum(sum(U1.*U1))+sum(sum(U2.*U2))+sum(sum(V1.*V1))+sum(sum(V2.*V2)));
grad = [U1grad(:);U2grad(:);V1grad(:);V2grad(:);b1grad(:);b2grad(:)];

end


function sigm = sigmoid(x)
sigm = 1 ./ (1 + exp(-x));
end

function sigGradient = fgradient(Z)
sigGradient = sigmoid(Z) .* (1 - sigmoid(Z));
end


