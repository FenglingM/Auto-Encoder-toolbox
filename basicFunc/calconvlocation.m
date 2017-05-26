function myconvlocation=calconvlocation(x, patchDim)
% ����һ����ά����;���˴�С����������������ʹ�õ������ݶ�Ӧ��ԭ���ݵ�λ��
% ���� im*in*imagechannel*num_images , patchdim,  
% ���  �� patchDim * patchDim * imagechannel ��*�� (im - patchDim + 1)*(in - patchDim + 1)* num_images �� �Ķ�ά����

numdim = ndims(x) ;
    if numdim ~=4
        if numdim ==3
            image(:,:,:,1)=x;
        else
            if numdim ==2
                image(:,:,1,1)=x;
            end
        end
    end
    if numdim == 4
        image = x;
    end
    
% assert(ndims(image)==4, ' ERROR: Please check if the input data has 4 dimension: row col channel num_images !');
[im,in,imagechannel, num_images]=size(image);
locate=zeros( patchDim*patchDim ,(im - patchDim + 1)*(in - patchDim + 1));
convlocation =zeros(patchDim * patchDim * imagechannel, (im - patchDim + 1)*(in - patchDim + 1)  );
count=0;
for j=1:in - patchDim + 1
    for i=1:im - patchDim + 1
        b=[];
        a=im*(j-1)+i: im*(j-1)+i+patchDim-1;
        a=a';
        for t=1:patchDim
            b=[b;a+im*(t-1)];
        end
    count=count+1;
    locate(:,count)=b;
    end
end


for i=1:imagechannel
convlocation((i-1)*( patchDim * patchDim)+1 : i*( patchDim * patchDim) , :) = locate+im*in*(i-1);
end

myconvlocation = zeros(patchDim * patchDim * imagechannel, (im - patchDim + 1)*(in - patchDim + 1)* num_images);
 for i=1: num_images
     myconvlocation(:, (i-1)* (im - patchDim + 1)*(in - patchDim + 1) +1: i* (im - patchDim + 1)*(in - patchDim + 1)  ) = convlocation + im*in*imagechannel*(i-1);
 end

 
end