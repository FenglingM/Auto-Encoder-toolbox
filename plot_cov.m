x=1:5:100;
y1 = XCOV(x,:);
y2 = XCOV_SDC(x,:);
plot(x,y1,x,y2);