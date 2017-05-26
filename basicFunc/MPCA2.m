function [tUs, odrIdx, TXmean, Wgt]  = MPCA2(TX,gndTX,testQ,Ps,maxK)
odrIdx=[]; Wgt=[];
%TX: (N+1)-dimensional tensor of Tensor Sample Dimension x NumSamples
N=ndims(TX)-1;%The order of samples.
IsTX=size(TX);
Is=IsTX(1:N);%The dimensions of the tensor
numSpl=IsTX(N+1);%Number of samples

%%%%%%%%%%%%%Zero-Mean%%%%%%%%%%
TXmean=mean(TX,N+1);%The mean
TX=TX-repmat(TXmean,[ones(1,N), numSpl]);%Centering

%The full projection for initialization
Qs=ones(N,1)*testQ;
Us=cell(N,1);
tUs=cell(N,1);
Lmds=cell(N,1);
for n=1:N
    In=Is(n);Phi=zeros(In,In);
    for m=1:numSpl
        switch N
            case 2
                Xm=TX(:,:,m);
            case 3
                Xm=TX(:,:,:,m);
            case 4
                Xm=TX(:,:,:,:,m);
            otherwise
                error('Order N not supported. Please modify the code here or email hplu@ieee.org for help.')
        end
        tX=tensor(Xm);
        tXn=tenmat(tX,n);
        Xn=tXn.data;
        Phi=Phi+Xn*Xn';
    end
    [Un,Lmdn]=eig(Phi);
    Lmd=diag(Lmdn);
    [stLmd,stIdx]=sort(Lmd,'descend');
    Us{n}=Un(:,stIdx);
    tUs{n}=Us{n}';
    Lmds{n}=Lmd(stIdx);
end

%Cumulative distribution of eigenvalues
cums=cell(N,1);
for n=1:N
    In=length(Lmds{n});
    cumLmds=zeros(In,1);
    Lmd=Lmds{n};
    cumLmds(1)=Lmd(1);
    for in=2:In
        cumLmds(in)=cumLmds(in-1)+Lmd(in);
    end
    cumLmds=cumLmds./sum(Lmd);
    cums{n}=cumLmds;
end

%MPCA Iterations
if maxK>0
    tPs=cell(N,1);
    pUs=cell(N,1);
    %%%%%%%%%%%%%Determine Rn, the dimension of projected space%%%%
    for n=1:N
        cum=cums{n};
        idxs=find(cum>=Qs(n)/100);
 %%%%%% Ps(n)=idxs(1);
        tUn=tUs{n};
        tPn=tUn(1:Ps(n),:);
        tPs{n}=tPn;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for iK=1:maxK
        for n=1:N
            In=Is(n);
            Phi=double(zeros(In,In));
            for m=1:numSpl
                switch N
                    case 2
                        Xm=TX(:,:,m);
                    case 3
                        Xm=TX(:,:,:,m);
                    case 4
                        Xm=TX(:,:,:,:,m);
                    otherwise
                        error('Order N not supported. Please modify the code here or email hplu@ieee.org for help.')
                end
                tX=ttm(tensor(Xm),tPs,-n);
                tXn=tenmat(tX,n);
                Xn=tXn.data;
                Phi=Phi+Xn*Xn';
            end
            Pn=Ps(n);
            Phi=double(Phi);
            if Pn<In
                option=struct('disp',0);
                [pUs{n},pLmdn]=eigs(Phi,Pn,'lm',option);
                pLmds{n}=diag(pLmdn);
            else
                [pUn,pLmdn]=eig(Phi);
                pLmd=diag(pLmdn);
                [stLmd,stIdx]=sort(pLmd,'descend');
                pUs{n}=pUn(:,stIdx(1:Pn));
                pLmds{n}=pLmd(stIdx(1:Pn));
            end
            tPs{n}=pUs{n}';
        end
    end
    Us=pUs;
    tUs=tPs;
    Lmds=pLmds;
    Is=Ps;
else
    if testQ<100
        error('At least one iteration is needed');
    end
end

%Calculate the weight tensor Wgt
% Wgt=zeros(Is);
% switch N
%     case 2
%         for i1=1:Is(1)
%             for i2=1:Is(2)
%                 Wgt(i1,i2)=sqrt(Lmds{1}(i1)*Lmds{2}(i2));
%             end
%         end
%     case 3
%         for i1=1:Is(1)
%             for i2=1:Is(2)
%                 for i3=1:Is(3)
%                     Wgt(i1,i2,i3)=sqrt(Lmds{1}(i1)*Lmds{2}(i2)*Lmds{3}(i3));
%                 end
%             end
%         end
%     case 4
%         for i1=1:Is(1)
%             for i2=1:Is(2)
%                 for i3=1:Is(3)
%                     for i4=1:Is(4)
%                         Wgt(i1,i2,i3,i4)=sqrt(Lmds{1}(i1)*Lmds{2}(i2)*Lmds{3}(i3)*Lmds{4}(i4));
%                     end
%                 end
%             end
%         end
%     otherwise
%         error('Order N not supported. Please modify the code here or email hplu@ieee.org for help.')
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Yps=ttm(tensor(TX),tUs,1:N);%MPCA projections of samples TX
% vecDim=1;
% for n=1:N, vecDim=vecDim*Is(n); end
% vecYps=reshape(Yps.data,vecDim,numSpl); %vectorization of Yps
% %%%%%%%%%%%%%%Now vecYps contains the feature vectors for training data
% 
% if max(gndTX)<0%%%%%%%%%%%%%%%%%%%%%%%%Sort by Variance%%%%%%%%%%%%%%%%%%%%
%     TVars=diag(vecYps*vecYps');
%     [stTVars,odrIdx]=sort(TVars,'descend');
% else%%%%%%%%%%%%%%%Sort according to Fisher's discriminality%%%%%%%%%%%%%%%
%     classLabel = unique(gndTX);
%     nClass = length(classLabel);%Number of classes
%     ClsIdxs=cell(nClass);
%     Ns=zeros(nClass,1);
%     for i=1:nClass
%         ClsIdxs{i}=find(gndTX==classLabel(i));
%         Ns(i)=length(ClsIdxs{i});
%     end
%     Ymean=mean(vecYps,2);
%     TSW=zeros(vecDim,1);
%     TSB=zeros(vecDim,1);
%     for i=1:nClass
%         clsYp=vecYps(:,ClsIdxs{i});
%         clsMean=mean(clsYp,2);
%         FtrDiff=clsYp-repmat(clsMean,1,Ns(i));
%         TSW=TSW+sum(FtrDiff.*FtrDiff,2);
%         meanDiff=clsMean-Ymean;
%         TSB=TSB+Ns(i)*meanDiff.*meanDiff;
%     end
%     FisherRatio=TSB./TSW;
%     [stRatio,odrIdx]=sort(FisherRatio,'descend');
% end