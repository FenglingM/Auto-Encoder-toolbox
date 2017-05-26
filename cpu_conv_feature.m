% cpu_conv_feature
% input configuration
%-----------------------------------------------------------------------
% img = randn(225,225,3,9000,'single');
% filter = randn(10,10,3,1000,'single');
% bias = randn(1000,1,'single');
% pool_size = 54;


function AE = cpu_conv_feature(img, AE )
img = single(img);
[AE{1}.conv_dim, ~, ~, num_images] = size(img);
AE{1}.pool_dim = AE{1}.conv_dim;
AE{1}.a = single( img(:,:,:,1) );
for i=2:numel(AE)
    AE{i}.conv_dim = ( AE{i-1}.pool_dim - AE{i}.kernelsize + 1 ) ; 
    AE{i}.pool_dim = AE{i}.conv_dim / AE{i}.poolsize;
    AE{i}.conv_nth = AE{i}.outputsize * AE{i}.conv_dim^2;    
    AE{i}.pool_dim = floor(AE{i}.conv_dim / AE{i}.poolsize);
    AE{i}.a = zeros(AE{i}.pool_dim, AE{i}.pool_dim, AE{i}.outputsize, 'single');
    convlocation{i} =  ( calconvlocation(AE{i-1}.a , AE{i}.kernelsize) )';
%     poollocation{i}   = mypoollocation( zeros( AE{i-1}.conv_dim  - AE{i}.kernelsize +1, AE{i-1}.conv_dim - AE{i}.kernelsize +1, AE{i}.outputsize   ),  AE{i}.poolsize);
end

AE{end}.final_fea = zeros(AE{end}.pool_dim * AE{end}.pool_dim * AE{end}.outputsize, num_images, 'single');
    
for j=1: num_images

    if mod(j,1000)==0
        fprintf('%d \n',j);  toc;
    end
    AE{1}.a = img(:,:,:,j);
    for i=2:numel(AE)
         
         coldata = AE{i-1}.a(convlocation{i}); % eg: (84*84)*243
         % ------------------normalization -------------------------------
         if AE{i}.NORMF == 1 && i<= numel(AE)
%              coldata = bsxfun(@minus, coldata, AE{i}.mean_patch');
%              coldata = max(min(coldata, AE{i}.pstd), -AE{i}.pstd) / AE{i}.pstd;

         end
         % --------------------------------------------------------------- 
         temp_conv = coldata *  AE{i}.W ;   
         temp_conv = bsxfun(@plus, temp_conv, AE{i}.b'); 
         temp_conv = activefunc(temp_conv, AE{i}.type);
         conv_feature = reshape(temp_conv, AE{i}.conv_dim, AE{i}.conv_dim, AE{i}.outputsize);
         
         switch AE{i}.pooltype
             case 'max'
                 [AE{i}.a, ~]= MaxPooling(conv_feature, single([AE{i}.poolsize AE{i}.poolsize])); 
             case 'mean'
%                  AE{i}.a = mean(  conv_feature(poollocation{i})  );  
                 [AE{i}.a, ~]= MeanPooling(conv_feature, single([AE{i}.poolsize AE{i}.poolsize])); 
         end

    end
    AE{end}.final_fea(:,j) = AE{end}.a(:);
    
end
        
                
        
end



        
    


