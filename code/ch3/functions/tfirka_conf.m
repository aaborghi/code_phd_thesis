function [Er,Ar,Br,Cr,Dr,s,converged] = tfirka_conf(H,dH,r,init,maxiter,conv_tol)
% [Er,Ar,Br,Cr,Dr,s,converged] = tfirka_conf(H,dH,r,init,maxiter,conv_tol)
% 
% Implements the conformal TF-IRKA algorithm for SISO systems
% 
% % INPUT %
% H = function handle of the function to approximate
% dH = derivative of the function to approximate
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
Cr = H_eval.';  %H_eval.'; 

iter = 0;
conv_crit = inf;

while(conv_crit > conv_tol && iter < maxiter)
    iter = iter+1;
    Dr = H(0)+Cr*(Ar\Br);
    %Dr = 0;
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
    Ar = Ls - diag(dsH_eval) + ones(size(Ls))*Dr;
    Br = H_eval-ones(size(H_eval))*Dr;
    Cr = H_eval.'-ones(size(H_eval.'))*Dr;    %H_eval.'-ones(size(H_eval.'))*Dr;   

    s_old = s;

    [~,S] = eig(Ar,Er);
    s = 1./(diag(S'));
    conv_crit = norm(sort(s)-sort(s_old))/norm(s_old);
    fprintf('Iteration %d - Convergence %f \n', iter, conv_crit);
    
end
converged = (conv_crit < conv_tol);

end