function [correct,correctindex,m,precision] = precisionCalculate(result,label)

m = size(result,2);
correct = 0;
correctindex=zeros(1,m);

for i = 1:m
    if (result(:,i) >= 0.5 && label(:,i)>0.5) || (result(:,i) < 0.5 && label(:,i)<0.5)
        correct = correct + 1;
        correctindex(:,i) = 1;
    end
    
end

precision = correct/m;
end