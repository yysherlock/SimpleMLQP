function [traindata,testdata] = dataloading()

load 'two_spiral_train.txt';
load 'two_spiral_test.txt';

load 'xordata.txt';

traindata = two_spiral_train;
%traindata = xordata;
testdata = two_spiral_test;

% traindata(traindata(:,3)>0.5,3)=0.9;
% traindata(traindata(:,3)<0.5,3)=0.1;
% 
% testdata(testdata(:,3)>0.5,3)=0.9;
% testdata(testdata(:,3)<0.5,3)=0.1;


% Q1:
% normalize or not? squash the data into [0,1]? Since the range of the two
% features are similar, I do not squash them into [0,1]

% Q2:
% How to deal with the label information? Since this is the two class
% classification problem, the 0 or 1 label may make the training process
% unefficient.
%

end