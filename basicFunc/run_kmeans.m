function centroids = run_kmeans(X, k, iterations)
% X 即patches, e.g. 400000* 108 每一行为一个样本
% k 为质心个数
% iterations 为迭代次数

  x2 = sum(X.^2,2);%每一个样本元素的平方和，x2这里指每个样本点与原点之间的欧式距离。
  centroids = randn(k,size(X,2))*0.1;%X(randsample(size(X,1), k), :); 程序中传进来的k为1600，即有1600个聚类类别
  BATCH_SIZE=1000;
  
  
  for itr = 1:iterations%iterations为kemans聚类迭代的次数
    fprintf('K-means iteration %d / %d\n', itr, iterations);
    
    c2 = 0.5*sum(centroids.^2,2);%c2表示类别中心点到原点之间的欧式距离

    summation = zeros(k, size(X,2));
    counts = zeros(k, 1);
    
    loss =0;
    
    for i=1:BATCH_SIZE:size(X,1) %X输入的参数为50000，所以该循环能够进行50次
      lastIndex=min(i+BATCH_SIZE-1, size(X,1));%lastIndex=1000,2000,3000,...
      m = lastIndex - i + 1;%m=1000,2000,3000,...
      %这种算法也是求每个样本的标签，因为求min(a-b)^2等价于求min(a^2+b^2-2*a*b)等价于求max(a*b-0.5*a^2-0.5*b^2),假设a为输入数据矩阵，而b为初始化中心点样本
      %则每次从a中取出一个数据与b中所有中心点作比较时，此时a中的该数据可以忽略不计，只跟b有关。即原式等价于求max(a*b-0.5*a^2)
      [val,labels] = max(bsxfun(@minus,centroids*X(i:lastIndex,:)',c2));%val为BATCH_SIZE大小的行向量（1000*1），labels为每个样本经过一次迭代后所属的类别标号
      loss = loss + sum(0.5*x2(i:lastIndex) - val');%求出loss没什么用
      
      S = sparse(1:m,labels,1,m,k,m); % labels as indicator matrix，最后一个参数为最大非0个数
      summation = summation + S'*X(i:lastIndex,:);%1600*108
      counts = counts + sum(S,1)';%1600*1的列向量，每个元素代表属于该类样本的个数
    end


    centroids = bsxfun(@rdivide, summation, counts);%步骤2，move centroids
    
    % just zap empty centroids so they don't introduce NaNs everywhere.
    badIndex = find(counts == 0);
    centroids(badIndex, :) = 0;%防止出现无穷大的情况
  end