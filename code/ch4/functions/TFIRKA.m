function [Er,Ar,Br,Cr,sigma,converged] = TFIRKA(H,dH,r,inter,init,maxiter,tol)
% [Er,Ar,Br,Cr,sigma,converged] = TFIRKA(H,dH,r,inter,init,maxiter,tol)
% % INPUT %
% H = function handle of the function to approximate
% dH = derivative of the function to approximate
% r = reduced order
% inter = function for the interpolation points (take -conj(x))
% init = initial interpolation points
% maxiter = maximum number of iterations
% tol = tolerance
% % OUTPUT %
% Er, Ar, Br, Cr= system matrices of Cr*((s*Er-Ar)\Br)
% sigma = final interpolation conditions
% converged = flag indicating if the algorithm converged

iter = 0;
conv_crit = inf;
conv_tol = tol;
s = init;
while(conv_crit > conv_tol && iter < maxiter)
    iter = iter+1;   
    cauchy = gallery('cauchy',s,-s);
    cauchy(isinf(cauchy)|isnan(cauchy)) = 0;
    H_eval = zeros(r,1);
    dH_eval = zeros(r,1);
    dsH_eval = zeros(r,1);
    for j = 1:r
        H_eval(j) = H(s(j));
        dH_eval(j) = dH(s(j));
        dsH_eval(j) = H_eval(j) + s(j)*dH_eval(j);
    end
    L = -(diag(H_eval) * cauchy - cauchy * diag(H_eval));
    Er = L - diag(dH_eval);
    Ls = -(diag(s.*H_eval) * cauchy - cauchy * diag(s.*H_eval));
    Ar = Ls - diag(dsH_eval);
    Br = H_eval;
    Cr = H_eval.';   

    s_old = s;

    [~,S] = eig(Ar,Er);
    s = diag(S);
    for i=1:r
        snew(i) = inter(s(i));
    end
    s = snew.';
    conv_crit = norm(sort(s)-sort(s_old))/norm(s_old);
end
sigma = s;
fprintf('Iterations %d', iter);
converged = (conv_crit < conv_tol);
end