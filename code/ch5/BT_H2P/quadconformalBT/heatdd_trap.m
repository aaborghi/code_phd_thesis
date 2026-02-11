%% Testing on heat equation
clear; clc;
rng(42)

% constructing the heat equation
n = 200;
nx = n;
xa = 0;
xb = 1;
hx = (xb-xa)/(nx+1);
ex = ones(nx,1);
II = speye(nx);
Laplace_x = 1/hx^2*spdiags([ex -2*ex ex], -1:1, nx, nx);
A = Laplace_x;
C = zeros(1,nx);
for i = 1:nx
    if i*hx >= 0 && i*hx <= 1
        C(i) = hx;
    end
end
B = zeros(nx,1);
B(end) = 1/hx^2;

G = @(z) C*((z*eye(n) - A)\B);
dG = @(z) -C*(((z*eye(n) - A)^2)\B);

% Running Algorithm conformalBT with disk
r = 10; 
c = -1.7e5; 
R = 1.7e5;



%%
psi = @(z) c + R * (z+1)./(z-1);
% psiinv = @(z) ((z-c)./R + 1)./((z-c)./R - 1);
dpsi = @(z) R*(-2)./(z-1).^2;

dynamics = @(t,x,A,B) A*x;
options = odeset('RelTol',1e-8,'AbsTol',1e-12);
[t1,x] = ode23(dynamics,linspace(0,1,1000),B,options,A,B);
y1 = C*x.';


Nj = [30, 60, 200];

figure()
for q = 1:3
    N = Nj(q)
  
    [qo, wo, qc, wc] = trap_rule([-3;5], N, false);
    qo = -1i*qo;
    qc = -1i*qc;

    wo = wo.*(sqrt(abs(dpsi(1i*qo))));
    wc = wc.*(sqrt(abs(dpsi(1i*qc))));

    Gevalc = zeros(size(qc));
    Gevalo = zeros(size(qo));
    
    L = zeros(size(qo,2),size(qc,2));
    M = zeros(size(qo,2),size(qc,2));


    for j = 1:length(qc)
        Gevalc(j) = G(psi(qc(j)*1i));
        h(j) = wc(j)*Gevalc(j);
        for k = 1:length(qo)
            Gevalo(k) = G(psi(qo(k)*1i));
            L(k,j) = - wc(j)*wo(k) * (Gevalo(k)-Gevalc(j))/(psi(qo(k)*1i)-psi(qc(j)*1i));
            M(k,j) = - wc(j)*wo(k) * (psi(qo(k)*1i)*Gevalo(k)-psi(qc(j)*1i)*Gevalc(j))/(psi(qo(k)*1i)-psi(qc(j)*1i));
            g(k) = wo(k)*Gevalo(k);
    
            if qc(j) == qo(k)
                L(k,j) = - wc(j)*wo(k) * dG(psi(qo(k)*1i));
                M(k,j) = - wc(j)*wo(k) * (psi(qo(k)*1i)*dG(psi(qo(k)*1i)) + G(psi(qo(k)*1i)));
            end
        end
    end
    
    [Z,S,Y] = svd(L, 'econ');
    Z1 = Z(:,1:r); S1 = S(1:r,1:r); Y1 = Y(:,1:r);
    Ar = sqrt(S1)\((Z1')*M*Y1)/sqrt(S1);
    br = sqrt(S1)\((Z1')*h.');
    cr = (g*Y1)/sqrt(S1);

    [~,xr] = ode23(dynamics,linspace(0,1,1000),br,options,Ar,br);
    yr = cr*xr.';
    error = abs(y1-yr);

    subplot(1,2,1)
    semilogy(t1,abs((error))./abs(y1),'-', 'color', [1-q/4, 1-q/4, 1-q/4]); hold on 
    xlabel('time', 'interpreter','latex');
    title('error dynamics', 'interpreter','latex');
    subplot(1,2,2)
    semilogy(diag(S), 'k.', 'color', [1-q/4, 1-q/4, 1-q/4]); hold on
    xlabel('index', 'interpreter','latex');
    title('Hankel singular values', 'interpreter','latex');

end
Gr = @(z) cr*((z*eye(r)-Ar)\br);



psiinv = @(x) ((x-c*speye(n))/R+speye(n))/((x-c*speye(n))/R-speye(n));
Alyap = psiinv(A);
Blyap = sqrt(2*R) * ((c*eye(n)+R*eye(n)-A) \ (B));
Clyap = sqrt(2*R) * (((c*eye(n)+R*eye(n)-A)') \ (C'));

U = lyapchol(Alyap, Blyap);
L = lyapchol(Alyap', Clyap);

[Z,S,Y] = svd(L*U', 'econ');

Z1 = Z(:,1:r);
Y1 = Y(:,1:r);
S1 = S(1:r,1:r);  S1half = sqrt(S1);

Wr = L'*Z1/S1half;
Vr = U'*Y1/S1half;

Ar2 = Wr'*A*Vr;
br2 = Wr'*B;
cr2 = C*Vr;

[~,xr2] = ode23(dynamics,linspace(0,1,1000),br2,options,Ar2,br2);
yr2 = cr2*xr2.';
error2 = abs(y1-yr2);

subplot(1,2,1)
semilogy(t1,abs((error2))./abs(y1),'r--'); hold off
subplot(1,2,2)
semilogy(diag(S), 'ro'); hold off
xlim([0,40]);


figure()
plot(real(psi(qc*1i)),imag(psi(qc*1i)), 'b.'); hold on
plot(real(eig(Ar)),imag(eig(Ar)), 'kx');
plot(real(eig(Ar2)),imag(eig(Ar2)), 'ro');
plot(real(eigs(A,n)),imag(eigs(A,n)), 'k.');
legend('quadrature nodes', '$\Lambda(A_r)$ conformal quadBT', '$\Lambda(A_r)$ conformalBT', '$\Lambda(A)$', 'interpreter', 'latex')
hold off

