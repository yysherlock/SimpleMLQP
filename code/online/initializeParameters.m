function theta = initializeParameters(hiddenSize,inputSize,outputSize)

%% Initialize parameters randomly based on layer sizes.
r = sqrt(6) / sqrt(inputSize + outputSize + hiddenSize + 1);
U1 = rand(hiddenSize,inputSize) * 2 * r - r;
U2 = rand(outputSize,hiddenSize) * 2 * r - r;

V1 = rand(hiddenSize,inputSize) * 2 * r - r;
V2 = rand(outputSize,hiddenSize) * 2 * r - r;

b1 = zeros(hiddenSize,1);
b2 = zeros(outputSize,1);

theta = [U1(:);U2(:);V1(:);V2(:);b1(:);b2(:)];

end


