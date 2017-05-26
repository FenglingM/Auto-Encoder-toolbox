
function patches = samplepatches(X, patchsize, num_patches)
% ����x��һ��row* col * channel * numimages ��4ά����

[row , col, channel,numimages] = size(X);%������256*256
patchonepic =   ceil(1.1* num_patches / numimages ) ;
patches = zeros(patchsize*patchsize * channel, numimages*patchonepic);
for i = 1:numimages %
   
    for j = 1:patchonepic %
        xPos = randi([1,row-patchsize+1]);%�����ȡһ�����꣬����randi([a,b])��ʾ���ȡһ����ΧΪ[a,b]������
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
