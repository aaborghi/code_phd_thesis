function [Er,Ar,Br,Cr,Dr,s,left,right,converged] = ctfirka_mimo(H,dH,left,right,r,init,maxiter,conv_tol,a)
% [Er,Ar,Br,Cr,Dr,s,converged] = ctfirka_mimo(H,dH,left,right,r,init,maxiter,conv_tol)
% 
% Implements the conformal TF-IRKA algorithm for MIMO systems
% 
% % INPUT %
% H = function handle of the function to approximate
% dH = derivative of the function to approximate
% left, right = left and right tangential directions respectively of size
% (qxr) and (mxr)
% r = reduced order
% init = initial interpolation points
% maxiter = maximum number of iterations
% conv_tol = tolerance
% 
% % OUTPUT %
% Er, Ar, Br, Cr, Dr = system matrices of Cr*((s*Er-Ar)\Br) + Dr
% s = final interpolation conditions
% converged = flag indicating if the algorithm converged

s = init;
[q,m] = size(H(0));


cauchy = gallery('cauchy',s,-s);
cauchy(isinf(cauchy)|isnan(cauchy)) = 0;
H_eval = zeros(r,1);
dH_eval = zeros(r,1);
dsH_eval = zeros(r,1);
ll = zeros(m,r);
rr = zeros(q,r);
for j = 1:r
    H_eval(j) = (left(:,j)')*H(s(j))*right(:,j);
    dH_eval(j) = (left(:,j)')*dH(s(j))*right(:,j);
    dsH_eval(j) = H_eval(j) + s(j)*dH_eval(j);
    ll(:,j) = H(s(j))'*left(:,j);
    rr(:,j) = H(s(j))*right(:,j);
end
L = -(diag(H_eval) * cauchy - cauchy * diag(H_eval));
Er = L - diag(dH_eval);
Ls = -(diag(s.*H_eval) * cauchy - cauchy * diag(s.*H_eval));
Ar = Ls - diag(dsH_eval);
Br = ll';
Cr = rr; 

% s = init; [q,m] = size(H(0));
% Ar = rand(r,r);
% Cr = rand(q,r);
% Br = rand(r,m);

iter = 0;
conv_crit = inf;

while(conv_crit > conv_tol && iter < maxiter)
    iter = iter+1;
    [X,Ei,Y] = eig(Ar,Er);
    % X = X/(Y'*Er*X);
    % Dr = H(0) + Cr*X*((Ei)\(Y'*Br));
    
    Dr = H(0)+Cr*((Ar)\Br);
   
    % Dr = 0;  
    
    ll = zeros(r,m);
    rr = zeros(q,r);
    L = zeros(r,r);
    M = zeros(r,r);
    for i = 1:r
        ll(i,:) = left(:,i)'*H(s(i));
        for j = 1:r
            rr(:,j) = H(s(j))*right(:,j);
            L(i,j) = (left(:,i)'*(H(s(i))-H(s(j)))*right(:,j))/(s(i)-s(j));
            M(i,j) = (left(:,i)'*(s(i)*H(s(i))-s(j)*H(s(j)))*right(:,j))/(s(i)-s(j));
            if s(i)==s(j)
                L(i,j) = left(:,i)'*dH(s(i))*right(:,i);
                M(i,j) = left(:,i)'*(s(i)*dH(s(i)) + H(s(i)))*right(:,i);
            end
        end
    end
    
    Ar = left'*Dr*right-M;
    Er = -L;
    Br = ll - left'*Dr;
    Cr = rr - Dr*right;

    s_old = s;

    [V,S,W] = eig(Ar,Er);

    V = V/(W'*Er*V);
    
    s = a(diag(S));
    left = Cr * V;
    right = Br' * W;

    conv_crit = norm(sort(s)-sort(s_old))/norm(s_old);
    fprintf('Iteration %d - Convergence %f \n', iter, conv_crit);
    if isnan(conv_crit)
        fprintf('Not converged \n');
    end
    
end
converged = (conv_crit < conv_tol)

end