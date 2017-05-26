






% input configuration
%-----------------------------------------------------------------------
% img = randn(225,225,3,9000,'single');
% filter = randn(10,10,3,1000,'single');
% bias = randn(1000,1,'single');
% pool_size = 54;

% layer 是所要取patch的那层的指针。是对前一层的特征图取patch
function    [patches, AE] = cpu_conv_patch(img, AE, layer )

[AE{1}.conv_dim, ~, AE{1}.outputsize , num_images] = size(img);
AE{1}.pool_dim = AE{1}.conv_dim;
  
for i=2:numel(AE)
    AE{i}.conv_dim = ( AE{i-1}.pool_dim - AE{i}.kernelsize + 1 ) ;
    AE{i}.pool_dim = floor( AE{i}.conv_dim / AE{i}.poolsize );
    AE{i}.pool_nth = AE{i}.outputsize * AE{i}.pool_dim^2;
    AE{i}.a = zeros(AE{i}.pool_dim, AE{i}.pool_dim, AE{i}.outputsize, 'single');
end
%   AE{end}.final_fea = zeros(AE{end}.pool_dim * AE{end}.pool_dim * AE{end}.outputsize, num_images, 'single');   
  

 patches = zeros(AE{layer}.inputsize, AE{layer}.numpatches ,'single');
 
for j=1: num_images

    if mod(j,10000) == 0
        fprintf('%d \n',j); 
    end
    AE{1}.a = img(:,:,:,j);
    
  t = ceil( AE{layer}.numpatches / num_images );
  t_patch = samplepatches( AE{layer-1}.a, AE{layer}.kernelsize, t );
  patches(:, (j-1) * t + 1 : j*t) = t_patch;
      

end
size(patches)
 kk=randperm(size(patches, 2));
 patches = patches(:, kk(1:AE{layer}.numpatches));
        
end



        
    


