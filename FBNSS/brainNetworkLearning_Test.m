clear; clc;
warning off all
root=cd; addpath(genpath([root '/DATA'])); addpath(genpath([root '/FUN']));

load fMRI80; data1 = fMRImciNc; clear fMRImciNc;

nDegree=size(data1{1},1);
nSubj=length(lab);
nROI=size(data1{1},2);
label=input('SR[1],PC[2],SR+SS[3]:');

% Network learning based on sparse representation(SR) - SLEP
if label==1
    %Parameter setting for SLEP
    ex=-5:5;
    lambda=2.^ex;
    disp('Press any key:'); pause;
    nPar=length(lambda);
    brainNetSet=cell(1,nPar);
    
    opts=[];
    opts.init=2;% Starting point: starting from a zero point here
    opts.tFlag=0;% termination criterion
    % abs( funVal(i)- funVal(i-1) ) ¡Ü .tol=10e?4 (default)
    %For the tFlag parameter which has 6 different termination criterion.
    % 0 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol.
    % 1 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol ¡Á max(funVal(i),1).
    % 2 ? funVal(i) ¡Ü .tol.
    % 3 ? kxi ? xi?1k2 ¡Ü .tol.
    % 4 ? kxi ? xi?1k2 ¡Ü .tol ¡Á max(||xi||_2, 1).
    % 5 ? Run the code for .maxIter iterations.
    opts.nFlag=0;% normalization option: 0-without normalization
    opts.rFlag=0;% regularization % the input parameter 'rho' is a ratio in (0, 1)
    opts.rsL2=0; % the squared two norm term in min  1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * ||x||_1
    fprintf('\n mFlag=0, lFlag=0 \n');
    opts.mFlag=0;% treating it as compositive function
    opts.lFlag=0;% Nemirovski's line search
    
    for L=1:nPar
        brainNet=zeros(nROI,nROI,nSubj);
        for i=1:nSubj
            tmp=data1{i}(:,(1:nROI));
            % tmp=tmp-repmat(mean(tmp')',1,nROI);% data centralization
            tmp=tmp-repmat(mean(tmp)',1,nDegree)';% centrlization
            
            currentNet=zeros(nROI,nROI);
            for j=1:nROI
                y=[tmp(:,j)];
                A=[tmp(:,setdiff(1:nROI,j))];
                [x, funVal1, ValueL1]= LeastR(A, y, lambda(L), opts);
                currentNet(setdiff(1:nROI,j),j) = x;
            end
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{L}=brainNet;
        fprintf('Done %d/%d networks!\n',L,nPar);
    end
    save('brainNetSet_SR_UNC.mat','brainNetSet','lab','-v7.3');
end

if label==2
    lambda=[5 10 15 20 15 30 35 40 45 50 55  60 65 70 75 80 85 90 95 99 99.2 99.5 99.9]; % the values lies in [0,100] denoting the sparsity degree
    
    disp('Press any key:'); pause;
    nPar=length(lambda);
    brainNetSet=cell(1,nPar);
    for L=1:nPar
        brainNet=zeros(nROI,nROI,nSubj);
        for i=1:nSubj
            currentNet=corrcoef(data1{i}(:,(1:nROI)));
            currentNet=currentNet-diag(diag(currentNet));% no link to oneself
            threhold=prctile(abs(currentNet(:)),lambda(L)); % fractile quantile
            currentNet(find(abs(currentNet)<=threhold))=0;
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{L}=brainNet;
        fprintf('Done %d/%d networks!\n',L,nPar);
    end
    save('brainNetSet_PC_UNC.mat','brainNetSet','lab');
end

if label==3
    ex=-5:5;
    lambda=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
    gamma=lambda;
    disp('Press any key:'); pause;
    nL=length(lambda);
    nZ=length(gamma);
    brainNetSet=cell(nZ,nL);
    
    for iL=1:nL
        for iZ=1:nZ
            brainNet=zeros(nROI,nROI,nSubj);
            for i=1:nSubj
                tmp=data1{i}(:,1:nROI);%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                tmp=tmp-repmat(mean(tmp)',1,nDegree)';% centrlization
                Maxt=max(max(tmp));
                Mint=min(min(tmp));
                tmp=tmp./(Maxt-Mint);%normalized
                currentNet = SelfpaceLeastR(tmp,lambda(iL),gamma(iZ));
                currentNet = (currentNet + currentNet')/2;
                currentNet=currentNet-diag(diag(currentNet));
                brainNet(:,:,i)=currentNet;
            end
            brainNetSet{iZ,iL}=brainNet;
            fprintf('Done lambda=%d gamma=%d networks!\n',iL,iZ);
        end
    end
    save('brainNetSet_SPLPCSR_UNC.mat','brainNetSet','lab','-v7.3');
end
