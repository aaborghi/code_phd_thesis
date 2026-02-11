function [Er,Ar,Br,Cr,s,left,right,converged] = tfirka_mimo(H,dH,left,right,r,init,maxiter,conv_tol,a)
% [Er,Ar,Br,Cr,Dr,s,converged] = ctfirka_mimo(H,dH,left,right,r,init,maxiter,conv_tol)
% 
% Implements TF-IRKA for MIMO systems
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

[q,m] = size(H(0));
s = init;
iter = 0;
conv_crit = inf;

while(conv_crit > conv_tol && iter < maxiter)
    iter = iter+1;
    
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
    
    Ar = -M;
    Er = -L;
    Br = ll;
    Cr = rr;

    s_old = s;

    [X,S,Y] = eig(Ar,Er);
    X = X/(Y'*Er*X);

    % sigma = diag(S);
    % for k=1:r
    %     s(k) = a(sigma(k));
    %     % left(:,k) = Cr*X(:,k);
    %     % right(:,k) = Br'*Y(:,k);
    % end
    
    s = a(diag(S));
    left = Cr * X;
    right = Br' * Y;
    
    % right = (Y' * Er * Br)'; 
    % right = ((Er * X) \ Br)';

    % right = (X \ Br)'; 


    % right = (W'*Br)';
    
    conv_crit = norm(sort(s)-sort(s_old))/norm(s_old);
    fprintf('Iteration %d - Convergence %f \n', iter, conv_crit);
    if isnan(conv_crit)
        fprintf('Not converged \n');
    end
    
end
converged = (conv_crit < conv_tol)

end