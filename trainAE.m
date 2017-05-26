%% 
function theta=trainAE( theta, AE, options, patches)

% for i =1: options.epochs
%     np=randperm(size(patches,2));
%     for j = 1 : size(patches,2) / options.batchsize       
%         batch = patches(:, np((j - 1) * options.batchsize + 1 : j * options.batchsize) );
%         [theta, cost] = minFunc( @(p) AEcost(p, AE, batch) , theta, options); 
%     end
%     
%     fprintf('%d epoch have passed! mean cost is  %f   \n', i, cost);
% end
L = single( zeros(size(patches,2) / options.batchsize,1) );
vel = zeros(size(theta), 'single');
if options.usegpu ==1
    theta = gpuArray(single(theta));
    vel = gpuArray(single(vel));
    L = gpuArray(single(L));
end

mean_cost = 10000;
learnrate = options.learnrate;
iteration = 0;

n=1;

for i =1: options.epochs
    if i<2
        options.momentum = 0.5;
    else
         options.momentum = 0.9;
    end

    np=randperm(size(patches,2));
    for j = 1 : floor( size(patches,2) / options.batchsize )   
        
        iteration =iteration +1;
        if i>3
            learnrate = options.learnrate * ( (1 + options.gama * iteration)^(-options.power) );
        end
        
        batch = patches(:, np((j - 1) * options.batchsize + 1 : j * options.batchsize) );
        if options.usegpu ==1
            batch = gpuArray(single(batch));
        end
            
        [cost,grad] = AEcost(theta, AE,  batch);
        vel = vel * options.momentum + learnrate * grad;
        theta = theta - vel;
        L(j)= cost;
       
    end 
  
% 保存每个epoch的loos和互协方差
%     u = mean(a, 2); % 计算隐含层输出的平均值, 1*h       
%     I = a - repmat(u, [1,size(a,2)]); % 减去均值, m*h
%     C = (1/size(a,2)).*((I)* I');   
%     COV_SDC(i) = sum(C(:) .^2 );
%     
%     COV{i}.C = C;
%     LOSS_SDC(i,:) = LOSS_ITEM(:);
    
    if mean(L(:)) > mean_cost
        options.power = options.power * 1.1;
    end
    mean_cost = mean(L(:));
    fprintf('%d epoch have passed! mean cost is  %f   learnrate is %f  \n', i, mean_cost, learnrate);
   
    if AE.imgchannel ==3
        W = reshape(theta(1:AE.outputsize * AE.inputsize), AE.outputsize, AE.inputsize);
        displayColorNetwork(gather(W') );
        drawnow;
    elseif AE.imgchannel == 1
        W = reshape(theta(1:AE.outputsize * AE.inputsize), AE.outputsize, AE.inputsize);
        display_network(gather(W') );
        drawnow; 
    end
  
end

% save COV_SDC COV_SDC;
% save LOSS_SDC LOSS_SDC;

    
