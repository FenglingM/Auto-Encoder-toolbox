function poollocation=mypoollocation(convfeatures,pooldim)

[im, in, imagechannel]=size(convfeatures);
poollocate=zeros( pooldim*pooldim ,(im)*(in)/(pooldim*pooldim));
count=0;
for j=1:in/pooldim
   for i=1:im/pooldim
        b=[];
        a=im*pooldim*(j-1)+(i-1)*pooldim+1:im*pooldim*(j-1)+(i-1)*pooldim+pooldim;
        a=a';
        for t=1:pooldim
            b=[b;a+im*(t-1)];
        end
       count=count+1;
       poollocate(:,count)=b;
   end
end

for i=1:imagechannel
    poollocation(:,:,i)=poollocate+im*in*(i-1);
end


end