%% schroedginer trajectories

clear; clc; rng(42);
addpath('functions');
% Constructing the discretized Schroedinger equation
nx = 1000; %1000
m = 2;
q = 2;
xa = 0;
xb = 1;
nu = 1;
hx = (xb-xa)/(nx+1);
ex = ones(nx,1);
Laplace_x = 1/hx^2*spdiags([ex -2*ex ex], -1:1, nx, nx);
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

r = 8;
% Running conformalIRKA
a = @(z) (conj(z)); 
% init = -500i -1000i*rand(r,1);
init = 500*randn(r/2,1) -1000i*rand(r/2,1); init = [init;conj(init)];
[Ar,Br,Cr,sigma] = conformalIRKA(A,B,C,r,a,init,500);

eig(Ar)

a = @(z) -(conj(z)); 
init = 1i*init;
% init = 10+100i*rand(r,1);
[Ar_2,Br_2,Cr_2,~] = conformalIRKA(A,B,C,r,a,init,1000);


% Computing systems output trajectories
inputu = @(t) exp(-(t-1).^2./0.1)-2*exp(-(t-4).^2./0.1) + 1.5*exp(-(t-6).^2./0.1);
dynamics = @(t,x,A,B) A*x+B*ones(m,1)*inputu(t);
options = odeset('RelTol',1e-8,'AbsTol',1e-12);
[t1,x] = ode23(dynamics,linspace(0,10,1000),zeros(nx,1),options,A,B);
y1 = C*x.';
[~,xr] = ode23(dynamics,linspace(0,10,1000),zeros(r,1),options,Ar,Br);
yr = Cr*xr.';
error = y1-yr;
[~,xr2] = ode23(dynamics,linspace(0,10,1000),zeros(r,1),options,Ar_2,Br_2);
yr2 = Cr_2*xr2.';
error2 = y1-yr2;
gaussinput = inputu(t1); 

froerror = zeros(1,size(t1,1));
froerror2 = zeros(1,size(t1,1));
for i = 1:size(t1,1)
    froerror(i) = norm(y1(:,i) - yr(:,i),'fro')/norm(y1(:,i),'fro');
    froerror2(i) = norm(y1(:,i) - yr2(:,i),'fro')/norm(y1(:,i),'fro');
end


%% Plots
figure()
set(gcf,'position',[100,100,1100,800])
subplot(4,1,1)
plot(t1(1:5:end),real(y1(1,1:5:end)),'r-', 'Linewidth', 3); hold on
plot(t1(1:5:end),real(yr(1,1:5:end)),'b--', 'Linewidth', 3);
plot(t1(1:5:end),imag(y1(1,1:5:end)),'r-.', 'Linewidth', 3);
plot(t1(1:5:end),imag(yr(1,1:5:end)),'b:', 'Linewidth', 3);
title(['\fontsize{14}{0}\selectfont Schr\"odinger equation'],'Interpreter','latex')
ax = gca;
ax.FontSize = 18; 
subplot(4,1,2)
plot(t1(1:5:end),real(y1(2,1:5:end)),'-', 'color', [0.5,1,1], 'Linewidth', 3); hold on
plot(t1(1:5:end),real(yr(2,1:5:end)),'--', 'color', [1,1,0.5], 'Linewidth', 3);
plot(t1(1:5:end),imag(y1(2,1:5:end)),'-.', 'color', [0.5,1,1], 'Linewidth', 3);
plot(t1(1:5:end),imag(yr(2,1:5:end)),':', 'color', [1,1,0.5],'Linewidth', 3);
ax = gca;
ax.FontSize = 18; 
legend({['\fontsize{13}{0}\selectfont Re$\{y(t)\}$'],['\fontsize{13}{0}\selectfont Re$\{\widehat{y}_r(t)\}$'],['\fontsize{13}{0}\selectfont Im$\{y(t)\}$'],['\fontsize{13}{0}\selectfont Im$\{\widehat{y}_r(t)\}$']},'fontsize',20, 'interpreter','latex', 'Location', 'northwest', 'NumColumns',4)
subplot(4,1,3)
semilogy(t1(1:5:end),froerror(1:5:end),'k', 'Linewidth', 3); hold on
semilogy(t1(1:5:end),froerror2(1:5:end),'m--', 'Linewidth', 3);
ax = gca;
ax.FontSize = 18; 
ylabel(['\fontsize{14}{0}\selectfont $|(y(t)-\hat{y}_r(t))/y(t)|$'], 'interpreter','latex')
legend('cIRKA','IRKA','fontsize',22, 'interpreter','latex', 'Location', 'southeast', 'NumColumns',2)
subplot(4,1,4)
plot(t1(1:5:end),gaussinput(1:5:end),'k:', 'Linewidth', 3);
ylabel(['\fontsize{14}{0}\selectfont $u(t)$'], 'interpreter','latex')
xlabel(['\fontsize{14}{0}\selectfont time [s]'],'interpreter','latex')
ax = gca;
ax.FontSize = 18; 


figure()
set(gcf,'position',[100,100,1100,800])
subplot(2,1,1)
semilogy(t1(1:5:end),froerror(1:5:end),'r', 'Linewidth', 3); hold on
semilogy(t1(1:5:end),froerror2(1:5:end),'b--', 'Linewidth', 3);
ax = gca;
ax.FontSize = 18; 
ylabel(['\fontsize{14}{0}\selectfont $|(y(t)-\hat{y}_r(t))/y(t)|$'], 'interpreter','latex')
legend('cIRKA','IRKA','fontsize',22, 'interpreter','latex', 'Location', 'southeast', 'NumColumns',2)
subplot(2,1,2)
plot(t1(1:5:end),gaussinput(1:5:end),'k:', 'Linewidth', 3);
ylabel(['\fontsize{14}{0}\selectfont $u(t)$'], 'interpreter','latex')
xlabel(['\fontsize{14}{0}\selectfont time [s]'],'interpreter','latex')
ax = gca;
ax.FontSize = 18; 
