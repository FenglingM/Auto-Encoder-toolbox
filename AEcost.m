function [cost,grad] = AEcost(theta, AE,  data)
visibleSize = AE.inputsize;
hiddenSize = AE.outputsize;
sparsityParam = AE.spartarget;
beta = AE.sparterm;
lambda = AE.wdterm;

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

m = size(data, 2);%
X = data ;
Y = data ;

% feedforward
if AE.denoiserate > 0
   X = X .* ( rand(size(X))> AE.denoiserate );  % masking noise
%     X = X + AE.denoiserate *  ( std(data(:)) * randn(size(X)) + mean(data(:)) ) ;  % gaussian noise
end
z2 = W1 * X + repmat(b1, [1, m]);
a2 = activefunc(z2, AE.type);

if AE.dropoutrate > 0
    a2 = a2 .* ( rand(size(a2))> AE.dropoutrate );
end
z3 = W2 * a2 + repmat(b2, [1, m]);
% z3 = sigmoid(z3);% 如果是lineardecoder，就不要这句，如果是autoencoder，就要这句
a3 = z3;

rhohats = mean(a2,2);
rho = sparsityParam;
% KLsum = sum(rho * log(rho ./ rhohats) + (1-rho) * log((1-rho) ./ (1-rhohats)));

% errors and mean square cost
squares = (a3 - Y).^2;
squared_err_J = (1/2) * (1/m) * sum(squares(:));
% weight_decay_J = (lambda/2) * (sum(W1(:).^2) + sum(W2(:).^2));
% sparsity_J = beta * KLsum;

% 

% 计算DeCov项
% h_size = size(a2, 1); % 实际是隐含层+1
h = a2; % 去除 一列1. 得到的矩阵，每一行都是一个隐含层的输出，一共m行,m*h 

if AE.decov_p > 0       
         h = h .* repmat(( rand(size(a2,1),1) > AE.decov_p ),[1 m]);
end

u = mean(h, 2); % 计算隐含层输出的平均值, 1*h       
I = h - repmat(u, [1,m]); % 减去均值, m*h
C = (1/m).*((I)* I');   
cost_decov = 0.5 * AE.decov_para * ( sum(C(:) .^2 ) - sum(diag(C) .^ 2) );
        
% cost = squared_err_J + weight_decay_J + sparsity_J;%
cost = squared_err_J + cost_decov ;

% ===================================================================================================
% back propogation

% 计算DeCov项的delta
m = size(a2, 2) ;
delta_decov = (C * I);
diag_cov = diag( diag(C) );
delta_decov = delta_decov - diag_cov * I;
delta_decov = (2 * (m-1)/m) * AE.decov_para  * delta_decov; 
% delta_decov = [zeros(size(a2,1),1) delta_decov'];

        
delta3 = -(Y - a3);  
% delta3 = delta3 .* a3.*(1-a3); % 加了这句就是auteencoder，不加就是linear decoder
beta_term = beta * (- rho ./ rhohats + (1-rho) ./ (1-rhohats));
delta2 = ((W2' * delta3) + repmat(beta_term, [1,m]) + delta_decov) .* d_active_fun(a2, AE.type );
W2grad = (1/m) * delta3 * a2' + lambda * W2;
b2grad = (1/m) * sum(delta3, 2);
W1grad = (1/m) * delta2 * X' + lambda * W1;
b1grad = (1/m) * sum(delta2, 2);

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

