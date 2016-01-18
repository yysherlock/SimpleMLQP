function [] = drawResult(inputdata,result,figureindex)
figure(figureindex);
indexcol1 = find(result>0.5);
x1=inputdata(indexcol1,1);
y1=inputdata(indexcol1,2);
indexcol0=find(result<0.5);
x0=inputdata(indexcol0,1);
y0=inputdata(indexcol0,2);
if figureindex == 1
    subplot(1,2,2)
    title('Figure 2b)');
else
    title('Figure 3');
end
plot(x1,y1,'ro');
hold on;
plot(x0,y0,'bx');

end