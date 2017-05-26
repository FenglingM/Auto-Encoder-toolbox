
function patches = samplepatches(X, patchsize, num_patches)
% 输入x是一个row* col * channel * numimages 的4维矩阵

[row , col, channel,numimages] = size(X);%这里是256*256
patchonepic =   ceil(1.1* num_patches / numimages ) ;
patches = zeros(patchsize*patchsize * channel, numimages*patchonepic);
for i = 1:numimages %
   
    for j = 1:patchonepic %
        xPos = randi([1,row-patchsize+1]);%随机获取一个坐标，这里randi([a,b])表示随机取一个范围为[a,b]的整数
        yPos = randi([1, col-patchsize+1]);
        patches(:,(i-1)*patchonepic+j) = reshape(X(xPos:xPos+patchsize-1,yPos:yPos+patchsize-1,:, i),patchsize*patchsize * channel, 1);      
    end
    
end
kk=randperm(size(patches, 2));
if num_patches > size(patches, 2)
    error('patch number too large!\n ');
else
    patches = patches(:, kk(1:num_patches));
end
