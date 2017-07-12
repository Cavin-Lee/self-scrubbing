function W=PSOfirst(A,L)
%% ��ջ���


%% ��������
w=0.9;%Ȩֵ ��Ӱ��PSO ��ȫ����ֲ����������� ֵ�ϴ�ȫ����������ǿ���ֲ�����������;��֮����ֲ�����������ǿ����ȫ����������������
c1=0.1;%���ٶȣ�Ӱ�������ٶ�
c2=0.1;
nROI=size(A,2);
dim=nROI*nROI;
swarmsize=1;%����Ⱥ��ģ����ʾ��100����Ŀռ�
maxiter=200;%���ѭ��������Ӱ��ʱ��
minfit=0.001;%��С��Ӧֵ
vmax=0.01;
vmin=-0.01;
ub=ones(1,dim);%���������������
lb=-ones(1,dim);%����������С����

%% ��Ⱥ��ʼ��

swarm=rand(swarmsize,dim)*2-1;%����Ⱥλ�þ���
vstep=rand(swarmsize,dim)*(vmax-vmin)+vmin;%����Ⱥ�ٶȾ���
fswarm=zeros(swarmsize,1);%Ԥ��վ��󣬴����Ӧֵ
for i=1:swarmsize
    X=swarm(i,:);
   
     fswarm(i)=fitness(X,A,L);
    %fswarm(i,:)=feval(jn,swarm(i,:));%������Ⱥλ�õĵ�i��Ϊ���룬����ֵ����Ӧ�������Ӧֵ
end

%% ���弫ֵ��Ⱥ�弫ֵ
[bestf,bestindex]=min(fswarm);%�����Ӧֵ�е���С��Ӧֵ���ͣ������ڵ�����
gbest=swarm;%��ʱ�ĸ������Ž�Ϊ�Լ�
fgbest=fswarm;%��ʱ�ĸ���������Ӧֵ
zbest=swarm(bestindex,:);%�������еĶ�Ӧ�Ľ�������У�ȫ����ѽ�
fzbest=bestf;%ȫ��������Ӧֵ


%% ����Ѱ��
iter=0;



while((iter<maxiter)&&(fzbest>minfit))
    for j=1:swarmsize
        % �ٶȸ���
        vstep(j,:)=w*vstep(j,:)+c1*rand*(gbest(j,:)-swarm(j,:))+c2*rand*(zbest-swarm(j,:));
        if vstep(j,:)>vmax  
            vstep(j,:)=vmax;%�ٶ�����
        end
        if vstep(j,:)<vmin
            vstep(j,:)=vmin;
        end
        % λ�ø���
        swarm(j,:)=swarm(j,:)+vstep(j,:);
        for k=1:dim
            if swarm(j,k)>ub(k)
                swarm(j,k)=ub(k);%λ������
            end
            if swarm(j,k)<lb(k)
                swarm(j,k)=lb(k);
            end
        end
       
        % ��Ӧֵ        
         X=swarm(j,:);
       
         fswarm(j,:)=fitness(X,A,L);
        % ���ڴ˴�����Լ��������������Լ���������������Ӧֵ����
        
        %
        % �������Ÿ���
        if fswarm(j)>fgbest(j) %�����ǰ�ĺ���ֵ�ȸ�������ֵС
            gbest(j,:)=swarm(j,:);%�������Ž����
            fgbest(j)=fswarm(j);%��������ֵ����
        end
        % Ⱥ�����Ÿ���
        if fswarm(j)>fzbest%�����ǰ�ĺ���ֵ��Ⱥ������ֵ��
            zbest=swarm(j,:);%Ⱥ�����Ž����
            fzbest=fswarm(j);%Ⱥ������ֵ����
        end
    end
    iter=iter+1;
   

 
 
end

W=reshape(zbest,nROI,nROI);
return
function z=fitness(X,A,L)
Wp=reshape(X,size(A,2),size(A,2));
z=0.5*norm(Wp-A'*A,'fro')^2+L*sum(sum(abs(Wp)));

return

