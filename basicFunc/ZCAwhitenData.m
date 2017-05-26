function [patches, ZCAWhite] = ZCAwhitenData(patches, epsilon)

numpatches=size(patches,2);
sigma = patches * patches' / numpatches;
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';%
patches = ZCAWhite * patches;

end
