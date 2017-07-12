function W = SelfpaceSLR(data,lambda1,lambda2,gamma)

%% Function SelfpaceLeastR
%      Least Squares Loss with the L1-norm Regularization and selfpace
% under the SLEP package
%% Problem
%
%  min  1/2 v|| data - data * W||^2 + lambda1 * ||W||_1+ lambda2 * ||W||_*-gamma*||v||_1
%  W (ii) ~= 0
%

%
%% Input parameters:
%
%  data-         Matrix of size m x n
%                data can be a dense matrix
%                         a sparse matrix
%                         or a DCT matrix
%  label -        Response vector (of size mx1)
%  lambda -        L_1 norm regularization parameter (lambda >=0)
%  gamma-      selfpace regularization parameter (gamma >=0, )
%
%% Output parameters:
%  W-         Solution
%
%% Copyright WeikaiLi@cqjtu  email :leeweikai@outlook.com
%
[sample,nROI]=size(data);
vold=zeros(1,sample);
vnew=ones(1,sample);
J_old = rand(1,1);
J_new = rand(1,1);
maxIter = 100;
Iter=1;
epsilon=10^-4;
W_tmp=zeros(nROI,nROI);

%%         SPL
while vold~=vnew %end condition
   
     nDegree=sum(vnew);
    tmp_old=diag(vnew)*data;
    index=find(vnew==0);
    tmp=tmp_old(setdiff(1:sample,index),:);% this place used the hard threshold and filter out the noise sample
    tmp=tmp-repmat(mean(tmp)',1,nDegree)';% centrlization
    %parameters for trace + sparse algorithms
    opt.z=lambda2;
    % learning networks based a revised version of accel_grad_mlr by combining sparse and low-rank regularizers.
    [currentNet, fval_vec, itr_counter] = accel_grad_mlr_qiao(tmp,tmp,lambda1,opt);
    currentNet=currentNet-diag(diag(currentNet));
    W_tmp=currentNet;
    
    %     S_val=sum(sum(abs(W_tmp)));
    %      S_val=sum(sum(abs(W_tmp)));
    %     L_val=trace(W_tmp);
    %     [U,D,V] = svd(W_tmp,0);
    %     D = diag(D);
    %     Lval = sum(D);
    %     Q_val=(sum(abs(v)));
    %     J_new = diag(v)*norm(data-data*W_tmp,'fro')^2+lambda1*S_val+lambda2*L_val-gamma*Q_val;
    
    error=abs(sum(((data-data*W_tmp))'))';
    [d c]=kmeans(error,2);
    ind=find(c==max(c));
    index= find(d==ind);
    num=length(index);
    sorterror=sort(error(index));
    threshold=sorterror(ceil(num*gamma));
    vold=vnew;
    vnew=vold.*(error'<threshold);%updata V
    
    %     sorterror=sort(error);
    %     threshold=sorterror(ceil(sample*gamma));
    %     v=v.*(error<threshold);%updata V
    %     gamma=gamma*1.1;
    %     lambda=lambda*1.1;%updata gamma and lambda
    Iter=Iter+1;
    
    if Iter>maxIter
        fprintf('.\n')
        break;
    end
end
W=W_tmp;
