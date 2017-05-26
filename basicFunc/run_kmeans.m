function centroids = run_kmeans(X, k, iterations)
% X ��patches, e.g. 400000* 108 ÿһ��Ϊһ������
% k Ϊ���ĸ���
% iterations Ϊ��������

  x2 = sum(X.^2,2);%ÿһ������Ԫ�ص�ƽ���ͣ�x2����ָÿ����������ԭ��֮���ŷʽ���롣
  centroids = randn(k,size(X,2))*0.1;%X(randsample(size(X,1), k), :); �����д�������kΪ1600������1600���������
  BATCH_SIZE=1000;
  
  
  for itr = 1:iterations%iterationsΪkemans��������Ĵ���
    fprintf('K-means iteration %d / %d\n', itr, iterations);
    
    c2 = 0.5*sum(centroids.^2,2);%c2��ʾ������ĵ㵽ԭ��֮���ŷʽ����

    summation = zeros(k, size(X,2));
    counts = zeros(k, 1);
    
    loss =0;
    
    for i=1:BATCH_SIZE:size(X,1) %X����Ĳ���Ϊ50000�����Ը�ѭ���ܹ�����50��
      lastIndex=min(i+BATCH_SIZE-1, size(X,1));%lastIndex=1000,2000,3000,...
      m = lastIndex - i + 1;%m=1000,2000,3000,...
      %�����㷨Ҳ����ÿ�������ı�ǩ����Ϊ��min(a-b)^2�ȼ�����min(a^2+b^2-2*a*b)�ȼ�����max(a*b-0.5*a^2-0.5*b^2),����aΪ�������ݾ��󣬶�bΪ��ʼ�����ĵ�����
      %��ÿ�δ�a��ȡ��һ��������b���������ĵ����Ƚ�ʱ����ʱa�еĸ����ݿ��Ժ��Բ��ƣ�ֻ��b�йء���ԭʽ�ȼ�����max(a*b-0.5*a^2)
      [val,labels] = max(bsxfun(@minus,centroids*X(i:lastIndex,:)',c2));%valΪBATCH_SIZE��С����������1000*1����labelsΪÿ����������һ�ε����������������
      loss = loss + sum(0.5*x2(i:lastIndex) - val');%���lossûʲô��
      
      S = sparse(1:m,labels,1,m,k,m); % labels as indicator matrix�����һ������Ϊ����0����
      summation = summation + S'*X(i:lastIndex,:);%1600*108
      counts = counts + sum(S,1)';%1600*1����������ÿ��Ԫ�ش������ڸ��������ĸ���
    end


    centroids = bsxfun(@rdivide, summation, counts);%����2��move centroids
    
    % just zap empty centroids so they don't introduce NaNs everywhere.
    badIndex = find(counts == 0);
    centroids(badIndex, :) = 0;%��ֹ�������������
  end