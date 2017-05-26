function Y = transform(X , type, ratio)
% X is a 4-D img, eg., 32*32*3*10000;
% Y is the transformed 4-D image, also 32*32*3*10000;
% type : translation, flip, noise.
[~, col, ~, ~] = size(X);
Y = zeros(size(X));

% ratio = 0.1;
switch type
    case 'translation'
        n = ceil( ratio * col) ;
        Y(:, n+1: col, :,:) = X(:, 1: col - n, :,:);   % 向右平移 n 个单位
    case 'flip'
        Y (:, 1: col, :,:) = X(:, col: -1: 1, :, :);
        
    case 'noise'
        Y = X + ratio * randn(size(X));
        
end



end