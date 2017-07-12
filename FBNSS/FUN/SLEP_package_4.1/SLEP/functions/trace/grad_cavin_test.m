

function [W] = grad_cavin_test(Xtrain,Lamda)

W_init = Xtrain'*Xtrain;
Wp=W_init;
fval_old=0.5*norm(Wp-Xtrain'*Xtrain,'fro')^2;
Max_Iteration=1000;
epsilon=10^-5;
fval=0;
L=100;
Iteration=1;
while(abs((fval_old-fval)/fval_old)>epsilon||Iteration<Max_Iteration)
    fval_old=fval;
	[Wp sval]=ComputeQS(Xtrain,Wp,L,Lamda);
	fval=0.5*norm(Wp-Xtrain'*Xtrain,'fro')^2+sval;
	Iteration=Iteration+1;
    
end
W=Wp;

return;

function [Wp sval]= ComputeQS(X,W,L,lambda)

[W1] = ComputeGradStep(X,W,L,lambda);

Wp=sign(W1).*max(abs(W1)-lambda/L,0);
sval=lambda*sum(sum(abs(Wp)));
return


%%
function [W1] = ComputeGradStep(X,W,L,lambda)

delta_W = ComputeDerivative(X,W);
W1 = W-(lambda/L)*delta_W;

return;
%%%
function dev = ComputeDerivative(X,W)

dev = W-X'*X;

return;