function patches = normalizeData(patches)%这种归一化为全局的归一化，平均值还是按列求的，但是标准差就是全局的标准差
mean_patch = mean(patches,2);
patches = bsxfun(@minus, patches, mean_patch);
if size(patches, 1) >  1000 && size(patches,2) > 100000
    kk = randperm(size(patches,2));
    temp = patches(:,kk(1:100000));
    pstd = 3* std(temp(:));
else
    
    pstd = 3 * std(patches(:));%patches(:)表示把矩阵展开成一列，类似于reshape，然后这里std对整个列求标准差。
end
% patches = max(min(patches, pstd), -pstd) / pstd;%经过这一步，patches最大为pstd，最小为-pstd，再除以pstd，得到最大为1，最小为-1.
% patches = (patches + 1) * 0.45 + 0.05;

save normal.mat mean_patch pstd;
end