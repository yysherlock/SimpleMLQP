function [] = drawData(traindata)
figure(1);
indexcol1 = find(traindata(:,3)>0.5);
x1=traindata(indexcol1,1);
y1=traindata(indexcol1,2);
indexcol0=find(traindata(:,3)<0.5);
x0=traindata(indexcol0,1);
y0=traindata(indexcol0,2);
subplot(1,2,1);
plot(x1,y1,'ro');
hold on;
plot(x0,y0,'bx');
title('Figure 2a)');
hold off;
end