tic;diary('.\MNIST_log.txt');  diary on;
setpath;
%%
load mnist_uint8;
traindata = reshape(train_x',28,28,1,60000);
traindata = single(traindata)/255;  
testdata = reshape(test_x',28,28,1,10000);
testdata  = single(testdata)/255;
trainlabel = single(train_y);
testlabel  = single(test_y);

y1 = zeros(size(trainlabel,1),1);
for i = 1:size(trainlabel,1)
    y1(i,1) = find(train_y(i,:)==1);
end
trainlabel = y1;
y2 = zeros(size(testlabel,1),1);
for i = 1:size(testlabel,1)
    y2(i,1) = find(test_y(i,:)==1);
end
testlabel  = y2;
toc;  fprintf('data loaded! \n');
clear y1 y2 train_x train_y test_x test_y;

% 随机选取部分样本
% select_num = 10000;
% for i = 1: 10
%    index =  find((trainlabel)== i);
%    randnum = randperm(size(index,1));
%    index = index(randnum(1:(select_num / 10)));  
%    if i == 1
%         select_x = index';
%    else 
%        select_x = [select_x index'];
%    end
% end
% randnum = randperm(select_num);
% select_x = select_x(:,randnum);
% 
% traindata = traindata(:,:,:,select_x);
% trainlabel = trainlabel(select_x,:);
% clear select_x select_y;
%%% -----------从这里开始， 适用于所有data  ----------------------------------------

%% train DAE
initL2
AE{2}
AE{2}.options
    

for i = 2: numel(AE)
 
    
    [patches, AE] = cpu_conv_patch(traindata, AE, i);
    if AE{i}.NORMF == 1
        patches = normalizeData(patches);
        load normal.mat;
        AE{i}.mean_patch = mean_patch;  AE{i}.pstd = pstd;
    end
  
    if AE{i}.ZCAF == 1
        [patches, ZCAWhite] = ZCAwhitenData(patches, AE{i}.epsilon);  %在这里保存了针对训练数据得到的ZCAWhite矩阵
        AE{i}.ZCA = ZCAWhite;
    end
    
     
    initL2;
    theta = initializeParameters( AE{i});
    theta=trainAE(theta, AE{i}, AE{i}.options, patches);
    theta = gather(theta);
    AE{i}.W = reshape(theta(1:AE{i}.outputsize * AE{i}.inputsize), AE{i}.outputsize, AE{i}.inputsize);
    AE{i}.b= theta(2*AE{i}.outputsize*AE{i}.inputsize+1:2*AE{i}.outputsize*AE{i}.inputsize+AE{i}.outputsize);
   
   
    AE{i}.W = AE{i}.W';
    if AE{i}.ZCAF == 1
        AE{i}.W =  AE{i}.ZCA' * AE{i}.W ; 
    end
    
    W = AE{i}.W;  b = AE{i}.b;  str = ['parasL', num2str(i)];  save (str, 'W', 'b');
    toc;
end

%% feature extraction (convolution and pooling)


clear patches;
tic;
initL2;


if AE{2}.options.usegpu            
    AE = gpu_conv_feature(traindata, AE);
    fea_train = AE{end}.final_fea;            
    AE  = gpu_conv_feature(testdata,  AE);
    fea_test = AE{end}.final_fea;            
else
    AE = cpu_conv_feature(traindata, AE);
    fea_train = AE{end}.final_fea;            
    AE  = cpu_conv_feature(testdata,  AE);
    fea_test = AE{end}.final_fea;            
end 
toc;

clear AE;
clear traindata testdata;
save('fea_all.mat','-v7.3','fea_train','fea_test');


%% Linear SVM (using Liblinear) (need large memory)

% model = train(double(trainlabel),sparse(double(fea_train'))   , '-q'  );
% 
% fprintf('Train');
% [pred, train_accuracy, decision_values] = predict(double(trainlabel), sparse(double( fea_train' ) ), model);
% toc;
% train_acc = train_accuracy(1);
% 
% fprintf('Test');
% [pred, test_accuracy, decision_values] = predict(double(testlabel), sparse(double( fea_test' ) ), model);
% toc;
% test_acc = test_accuracy(1);


%% Optional: softmax classification

% load fea_all;

options.maxIter = 600;
softmaxLambda = 1e-4;%
numClasses = 10;
numfeatures = size(fea_train,1);
softmaxModel = softmaxTrain(numfeatures, numClasses, softmaxLambda, double(fea_train), (trainlabel), options);%                            
[predi] = softmaxPredict(softmaxModel, double(fea_test));
accu = (predi(:) == (testlabel(:)));

acc = sum(accu(:))/numel(accu);

%% 
save('acc.mat','-v7.3', 'train_acc','test_acc');

clear   
diary off;
