function patches = normalizeData(patches)%���ֹ�һ��Ϊȫ�ֵĹ�һ����ƽ��ֵ���ǰ�����ģ����Ǳ�׼�����ȫ�ֵı�׼��
mean_patch = mean(patches,2);
patches = bsxfun(@minus, patches, mean_patch);
if size(patches, 1) >  1000 && size(patches,2) > 100000
    kk = randperm(size(patches,2));
    temp = patches(:,kk(1:100000));
    pstd = 3* std(temp(:));
else
    
    pstd = 3 * std(patches(:));%patches(:)��ʾ�Ѿ���չ����һ�У�������reshape��Ȼ������std�����������׼�
end
% patches = max(min(patches, pstd), -pstd) / pstd;%������һ����patches���Ϊpstd����СΪ-pstd���ٳ���pstd���õ����Ϊ1����СΪ-1.
% patches = (patches + 1) * 0.45 + 0.05;

save normal.mat mean_patch pstd;
end