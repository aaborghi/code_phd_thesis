%% Testing
clear all; clc; rng(42);
nx = 500; 
n = 2*nx; 
m = 2;
q = 2;
xa = 0;
xb = 1;
nu = 1;
damping = 0;
h = (xb-xa)/(nx+1);
xd = xa+h:h:xb-h;
ex = ones(nx,1);
I = speye(nx);
Laplace_x = (1/h^2)*spdiags([ex -2*ex ex], -1:1, nx, nx);

Laplace_x(end:end) = -(1/h^2); 

O = sparse(nx,nx);
A = nu*Laplace_x;
A = [O,I;A,-damping*I];

B = zeros(nx,1); B(end) =  1/h; 
B = [zeros(nx,1);B];
B = sparse(B);
C = zeros(1,nx); C(1) =  1/h; 
C = [C,zeros(1,nx)];
C = sparse(C);
    

G = @(s) (2)./(exp(s) + exp(-s));
dG = @(s) -2.*(exp(s)-exp(-s))./((exp(-s)+exp(s)).^2);

addpath('functions');

% % construct conformal map implicit euler
step = 2e-2;
phi = @(z) (1-z)./step;
dphi = @(z) -1./step;

% algorithm 1
r = 10;
init = 0.1*randn(r,1)+0.1i*randn(r,1); init = (init./abs(init)).*rand(r,1);
H = @(s) G(phi(s));
dH = @(s) dG(phi(s)).*dphi(s);
[Er,Ar,Br,Cr,Dr,sigma,~] = tfirka_conf(H,dH,r,init,1000,1e-6);

figure()
plot_phase(H, [0.5,-1.4,1,-1]); hold on
p(1) = plot(real(eig(Ar,Er)),imag(eig(Ar,Er)),'ko','DisplayName','$\lambda_j$','markersize',10,'linewidth',2);
p(2) = plot(real(exp(1i*linspace(0,2*pi,100))),imag(exp(1i*linspace(0,2*pi,100))),'k-','linewidth',2);
p(3) = plot(real(sigma),imag(sigma),'k.','DisplayName','$\sigma$','markersize',15,'linewidth',2);
xlim([0.5,1.4]), ylim([-0.7,0.7]);
legend(p([1,3]),'Interpreter','latex','FontSize',20);
hold off
c = colorbar;
c.FontSize = 20;
ax = gca;


%% Discrete dynamical system
time_interval = 50;
time = time_interval/step;
n = size(A,1);
x = zeros(n,time);
xr = zeros(r,time);
u = zeros(1,time); 

inputu_d = @(t) exp(-(t*step-3).^2./0.5);
u(1:time) = inputu_d(linspace(0,time,time));

inputu = @(t) exp(-(t-3).^2./0.5);
dynamics = @(t,x,A,B) A*x+B*inputu(t);
options = odeset('RelTol',1e-8,'AbsTol',1e-12); 
[t1,x_] = ode23(dynamics,linspace(0,time_interval,1000),zeros(n,1),options,A,B);
y1 = C*x_.';


% computing the discrete-time dynamics
for k = 2:1:time
    xr(:,k) = (Ar\Er)*xr(:,k-1) - (Ar\Br)*u(k);
    x(:,k) = (speye(n)-A*step)\(x(:,k-1) + step*B*u(k));
end
y = C*x;
yr = Cr*xr + Dr*u;

Gr = @(s) Cr*((s*Er-Ar)\(Br))+Dr;
G_ = @(s) C*((phi(s)*speye(n)-A)\(B));
funerror = @(z) abs((G_(exp(1i*z))-Gr(exp(1i*z)))).^2;
H2A_bound = sqrt((1/(2*pi))*integral(funerror,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true))*norm(u,2);
funerror2 = @(z) abs((H(exp(1i*z))-Gr(exp(1i*z)))).^2;
H2A_bound2 = sqrt((1/(2*pi))*integral(funerror2,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true))*norm(u,2);

space = 5;
figure()
subplot(3,1,1)
plot(1:space:time,y(1:space:end),'k-'); hold on
plot(1:space:time,real(yr(1:space:end)),'r--');
plot(t1(1:space:end)/step, y1(1:space:end), 'b:');
legend('$y_k$','$\widehat{y}_k$','interpreter','latex');
subplot(3,1,2)
plot(1:space:time, inputu_d(1:space:time), 'b--');
legend('$u_k$', 'interpreter', 'latex');
subplot(3,1,3)
semilogy(1:space:time, abs(y(1:space:end)-yr(1:space:end)), 'r-'); hold on
semilogy(1:space:time, ones(size(1:space:time))*H2A_bound, 'k-');
semilogy(1:space:time, ones(size(1:space:time))*H2A_bound2, 'k:');
legend('$|y_k-\hat{y}_k|$','$\ell_\infty$ error bound', '$\ell_\infty$ error bound','interpreter','latex', 'location', 'southeast');
xlabel('$k$','interpreter','latex');





r_range = 4:1:14;
fomfom = @(z) H(exp(1i*z))*(H(exp(1i*z)))';
H2fomfom = sqrt((1/(2*pi))*integral(fomfom,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true));

for i = 1:1:size(r_range,2)
    r = r_range(i)
    %TFIRKA
    init = 0.1*randn(r,1)+0.1i*randn(r,1); init = (init./abs(init)).*rand(r,1);
    [Er,Ar,Br,Cr,Dr,sigma,converged] = tfirka_conf(H,dH,r,init,10000,1e-6);
    count = 0;
    while ~converged && count<5
        count = count+1;
        init = 0.1*randn(r,1)+0.1i*randn(r,1); init = (init./abs(init)).*rand(r,1);
        [Er,Ar,Br,Cr,Dr,sigma,converged] = tfirka_conf(H,dH,r,init,10000,1e-6);
    end
    Gr = @(s) Cr*((s*Er-Ar)\(Br))+Dr;
    funerror = @(z) trace((H(exp(1i*z))-Gr(exp(1i*z)))*(H(exp(1i*z))-Gr(exp(1i*z)))');
    H2D_IRKA(i) = sqrt((1/(2*pi))*integral(funerror,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true))./H2fomfom;

end


figure()
semilogy(r_range,H2D_IRKA,'k-*');
legend('$H^2(A)$ relative error','Interpreter','latex','FontSize',14);
hold off
xlabel('$r$','interpreter','latex');





