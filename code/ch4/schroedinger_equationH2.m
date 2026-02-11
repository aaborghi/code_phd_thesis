%% schroedinger h2

clear; clc;
rng(42)
addpath('functions');
% constructing the Schroedinger equation
nx = 1000; %1000
m = 2;
q = 2;
xa = 0;
xb = 1;
nu = 1;
hx = (xb-xa)/(nx+1);
xd = xa+hx:hx:xb-hx;
ex = ones(nx,1);
I = speye(nx);
Laplace_x = 1/hx^2*spdiags([ex -2*ex ex], -1:1, nx, nx);
e1x = I(:,1);
enx = I(:,nx);
O = sparse(nx,nx);
A = nu*Laplace_x;
B = zeros(nx,m);
C = zeros(q,nx);

for i = 1:nx
    if i*hx >= 0.4 && i*hx <= 0.5
        B(i,1) = 1;  
    end
    if i*hx >= 0.5 && i*hx <= 0.6
        B(i,2) = 1; 
    end
    if i*hx >= 0.1 && i*hx <= 0.3
        C(1,i) = hx;
    end
    if i*hx >= 0.7 && i*hx <= 0.9
        C(2,i) = hx;
    end
end
C = sparse(C);
B = sparse(B);
n = size(A,1);
A=-1i*A;

% define conformal map
phi = @(x) -1i*x;
dphi = @(x) -1i;

% construct the ROM with conformalBT
rset = 4:2:28;

for i = 1:size(rset,2)
    r = rset(i)
    % Running conformalIRKA
    a = @(z) (conj(z)); 
    % init = -500i -1000i*rand(r,1);
    init = 500*randn(r/2,1) -1000i*rand(r/2,1); init = [init;conj(init)];
    [Ar_,Br_,Cr_,~] = conformalIRKA(A,B,C,r,a,init,1000);

    [~,H2A_cirka(i)] = H2Anorm(A,B,C,Ar_,Br_,Cr_,phi,dphi);

    a = @(z) -(conj(z)); 
    init = 1i*init;
    [Ar_2,Br_2,Cr_2,~] = conformalIRKA(A,B,C,r,a,init,1000);

    [~,H2A_irka(i)] = H2Anorm(A,B,C,Ar_2,Br_2,Cr_2,phi,dphi);

end


% Plots
figure()
semilogy(rset, H2A_cirka, 'r-*', 'Linewidth', 1.5); hold on
semilogy(rset, H2A_irka, 'b-o', 'Linewidth', 1.5);
xlabel('$r$', 'Interpreter','latex');
ylabel('$\mathcal{H}_2(\bar{P}^c)$ error', 'Interpreter','latex');
legend('cIRKA','IRKA','fontsize',20, 'interpreter','latex', 'Location', 'northeast')
xlim([rset(1),rset(end)]);
ax = gca;
ax.FontSize = 14;
