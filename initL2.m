

AE{2}.kernelsize = 11;  
AE{2}.imgchannel = 1;
AE{2}.inputsize =  AE{2}.kernelsize ^2 *AE{2}.imgchannel;
AE{2}.outputsize = 500;  
AE{2}.poolsize = 6; 
AE{2}.numpatches = 300000;
AE{2}.spartarget = 0.01;
AE{2}.sparterm = 1;
AE{2}.wdterm = 3e-3;
AE{2}.dropoutrate = 0;
AE{2}.denoiserate = 0;
AE{2}.epsilon = 0;
AE{2}.pooltype = 'max';
AE{2}.type = 'relu';
AE{2}.ZCAF = 0; % whiten flag
AE{2}.NORMF = 1; % normalization flag
AE{2}.decov_para = 0; 
AE{2}.decov_p = 0; 

AE{2}.options.Method = 'sgd';
AE{2}.options.epochs = 100;
AE{2}.options.batchsize = 200;
AE{2}.options.learnrate = 0.01;
AE{2}.options.usegpu = 0;
AE{2}.options.gama =0.00001;
AE{2}.options.power = 0.75;