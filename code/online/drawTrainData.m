function [] = drawTrainData(traindata)

indexcol1 = find(traindata(:,3)>0.5);
x1=traindata(indexcol1,1);
y1=traindata(indexcol1,2);
indexcol0=find(traindata(:,3)<0.5);
x0=traindata(indexcol0,1);
y0=traindata(indexcol0,2);
plot(x1,y1,'o')
hold on;
plot(x0,y0,'x')

end