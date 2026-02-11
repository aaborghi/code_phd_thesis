%% random example
clear all; close all; clc; rng(42);

addpath('functions');
t = 2*pi*(0:500)'/500; 
% Disk parameters
R = 1.51;
c = -0.8;
Z = c+ R*exp(1i*t); 


n = 100;
% Triangle matrix
alpha = zeros(n,1);
gamma = ones(n,1);
beta = zeros(n,1);
a = 0.25*ones(n,1);
A = -spdiags([a beta alpha gamma], -2:1, n, n) -0.8*eye(n);


B = ones(n,1);
C = ones(1,n);



ro = 4;
% CIRKA
init = c + exp(1i*linspace(0,2*pi,ro));
[Ar, Br, Cr, sigma] = conformalIRKA(A,B,C,ro, @(z) (c+(R^2)/(conj(z)-conj(c))), init, 1000);
% IRKA
init2 = 2 + 1i*rand(ro/2,1); init2 = [init2;conj(init2)];
[Ar2, Br2, Cr2, sigma2] = conformalIRKA(A,B,C,ro, @(z) -conj(z), init, 1000);

eigAr = eig(Ar);
eigAr2 = eig(Ar2);
eigA = eig(full(A));


figure()
plot(real(Z), imag(Z),'k--'), axis square, hold on
plot(real(eigA), imag(eigA), '.','color',[0.5 0.5 0.5],'markersize',10);
plot(real(eigAr), imag(eigAr), 'ro'); 
plot(real(eigAr2), imag(eigAr2), 'bo'); 
plot(real(sigma), imag(sigma), 'r.', 'markersize',15); hold off
legend('$\partial P$','$\Lambda(\mathbf{A})$','$\Lambda(\mathbf{A}_r)$ cIRKA', '$\Lambda(\mathbf{A}_r)$ IRKA','$\sigma$','fontsize',20, 'interpreter','latex', 'Location', 'northeast')
xlabel('Re($z$)', 'fontsize',20, 'interpreter','latex')
ylabel('Im($z$)', 'fontsize',20, 'interpreter','latex')
axis equal


% bode on P and im
wi = Z;
omega = [logspace(-1,2,100)];

G = @(s) C*((s*speye(n) - A)\B);
Gr = @(s) Cr*((s*eye(ro) - Ar)\Br);
Gr2 = @(s) Cr2*((s*eye(ro) - Ar2)\Br2);
Geval_P = [];
Greval_P = [];
Gr2eval_P = [];
for j = 1:length(wi)
    Geval_P(j) = G(wi(j));
    Greval_P(j) = Gr(wi(j));
    Gr2eval_P(j) = Gr2(wi(j));
end
Geval_im = [];
Greval_im = [];
Gr2eval_im = [];
for j = 1:length(omega)
    Geval_im(j) = G(1i*omega(j));
    Greval_im(j) = Gr(1i*omega(j));
    Gr2eval_im(j) = Gr2(1i*omega(j));
end

figure()
subplot(1,2,1);
semilogy(t,abs(Geval_P - Greval_P)./abs(Geval_P),'r-'); hold on
semilogy(t,abs(Geval_P - Gr2eval_P)./abs(Geval_P),'b--'); hold off
xlabel('$z$', 'fontsize',20, 'interpreter','latex')
ylabel('rel. error', 'fontsize',20, 'interpreter','latex')
title('Evaluation on P','interpreter','latex');
subplot(1,2,2);
loglog(omega,abs(Geval_im - Greval_im)./abs(Geval_im),'r-'); hold on
loglog(omega,abs(Geval_im - Gr2eval_im)./abs(Geval_im),'b--'); hold off
title('Evaluation on Imaginary axis','interpreter','latex');
legend('cIRKA', 'IRKA', 'fontsize',20, 'interpreter','latex', 'Location', 'northeast')
xlabel('$i\omega$', 'fontsize',20, 'interpreter','latex')


contour_transfer_function(@(z) log10(abs(Gr(z)-G(z))),[-6,6],[-6,6],200); hold on
c = colorbar;
plot(real(Z),imag(Z),'--', 'color', [0.8,0.8,0.8], 'linewidth', 2);
plot(real(sigma),imag(sigma),'r.', 'markersize', 14);
axis equal
c.FontSize = 20;
ax = gca;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% H2 computation
R = 1.51;
c = -0.8;
Z = c+ R*exp(1i*t); 
rng(42)
u = @(t) real(c)+R*cos(t);
v = @(t) imag(c)+R*sin(t); 
tau = @(t) (u(t)+1i*v(t));
du = @(t) -R*sin(t);
dv = @(t) R*cos(t);
dtau = @(t) (du(t)+1i*dv(t));

rset = 2:2:18; 

fom = @(x) (C*((x*speye(n)-A)\B)); 
fomfom = @(z) fom(z)*(fom(z))';
H2P_fomfom = sqrt((1/(2*pi))*integral(@(z) fomfom(tau(z)).*abs(dtau(z)) ,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true));
sys = ss(A,B,C,0);
H2_fomfom = norm(sys,2);

for i = 1:size(rset,2)
    ro = rset(i);
    disp(['r=',num2str(ro)]);

    % Running cIRKA
    init = -10 + 10i*(rand(ro,1)-0.5);
    [Ar, Br, Cr, sigma, converged] = conformalIRKA(A,B,C,ro, @(z) (c+(R^2)/(conj(z)-conj(c))), init, 1000);
    count = 0;
    while ~converged && count<10
        count = count + 1;
        init = -5 + 3i*(rand(ro/2,1)-0.5); init = [init;conj(init)];
        [Ar, Br, Cr, sigma, converged] = conformalIRKA(A,B,C,ro, @(z) (c+(R^2)/(conj(z)-conj(c))), init, 1000);
    end
    rom = @(x) (Cr*((tau(x)*speye(ro)-Ar)\Br));
    error = @(x) fom(tau(x))-rom(x);
    funerror = @(z) error(z)*(error(z))' .* abs(dtau(z));
    H2P_cIRKA(i) = sqrt((1/(2*pi))*integral(funerror,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true))./H2P_fomfom;   
    sys_error = ss([A, zeros(n,ro); zeros(ro,n), Ar],[B ; Br],[C, -Cr],0);
    H2_cIRKA(i) = norm(sys_error,2)./H2_fomfom;   
    
    % Running IRKA
    init = 2 + 1i*(rand(ro,1)-0.5);
    [Ar2, Br2, Cr2, sigma2, converged] = conformalIRKA(A,B,C,ro, @(z) -conj(z), init, 1000);
    count = 0;
    while ~converged && count<10
        count = count + 1;
        init = 2 + 1i*(rand(ro,1)-0.5);
        [Ar2, Br2, Cr2, sigma2, converged] = conformalIRKA(A,B,C,ro, @(z) -conj(z), init, 1000);
    end
    rom2 = @(x) (Cr2*((1i*(x)*speye(ro)-Ar2)\Br2));
    error = @(x) fom(1i*(x))-rom(x);
    funerror = @(z) error(z)*(error(z))';
    H2P_IRKA(i) = sqrt((1/(2*pi))*integral(funerror,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true))./H2P_fomfom;   
    sys_error2 = ss([A, zeros(n,ro); zeros(ro,n), Ar2],[B ; Br2],[C, -Cr2],0);
    H2_IRKA(i) = norm(sys_error2,2)./H2_fomfom;   
    
    
end


figure()
subplot(1,2,1)
semilogy(rset, H2P_cIRKA, 'r-*'); hold on
semilogy(rset, H2P_IRKA, 'b-o'); hold off
xlim([rset(1),rset(end)]);
ylabel('$\|H-\widehat{H}\|/\|H\|$', 'fontsize',20, 'interpreter','latex')
xlabel('$r$', 'fontsize',20, 'interpreter','latex')
subplot(1,2,2)
semilogy(rset, H2_cIRKA, 'r-*'); hold on
semilogy(rset, H2_IRKA, 'b-o'); hold off
xlim([rset(1),rset(end)]);
ylabel('$\|H-\widehat{H}\|/\|H\|$', 'fontsize',20, 'interpreter','latex')
xlabel('$r$', 'fontsize',20, 'interpreter','latex')







%% Testing another shape and use Schwarz with AAA

clear all; close all; clc; rng(42);
addpath('functions');
t = 2*pi*(0:500)'/500; 
x = 1*sin(t)+0.8*cos(2*t);
y = 1.3*cos(t); 
Z = 0.9*(x+1i*y)-0.6; 

inpoly = @(z,w) inpolygon(real(z),imag(z),real(w),imag(w));
[r,pol] = aaa(conj(Z),Z,'tol',1e-14); ii = inpoly(pol,Z);

M = [1.1,1.2,1.3,1.4];
figure()
plot(real(Z(1:5:end)), imag(Z(1:5:end)), 'k--'); hold on
for i = 1:4
    scale = M(i);
    interior1 = (Z(50:4:200)+1/scale)/scale*0.9-1;
    interior2 = (Z(300:4:360)+1)/scale - 1;
    interior3 = (Z(390:4:450)+1)/scale - 1;
    interior = [interior1; interior2; interior3];
    exterior = conj(r(interior));
    plot(real(interior), imag(interior), '.', 'color', [1-1/i 1-1/i 1-1/i]);
    plot(real(exterior), imag(exterior), '.', 'color', [1.1-1/i 0 1.1-1/i]);
end
xlabel('Re($z$)', 'fontsize',20, 'interpreter','latex')
ylabel('Im($z$)', 'fontsize',20, 'interpreter','latex')
axis equal


n = 100;
% Triangle matrix
alpha = zeros(n,1);
gamma = ones(n,1);
beta = zeros(n,1);
a = 0.25*ones(n,1);
A = -spdiags([a beta alpha gamma], -2:1, n, n) -0.8*eye(n);
B = ones(n,1);
C = ones(1,n);





ro = 4;
init = 0.5 + 1i*(rand(ro/2,1)-0.5); init = [init;conj(init)];
[Ar, Br, Cr, sigma] = conformalIRKA(A,B,C,ro, @(x) conj(r(x)), init, 1000);


eigAr = eig(Ar);
eigA = eig(full(A));


figure()
plot(real(Z(1:5:end)), imag(Z(1:5:end)),'k--'), axis square, hold on
plot(real(eigA), imag(eigA), '.','color',[0.5 0.5 0.5],'markersize',10);
plot(real(eigAr), imag(eigAr), 'rx', 'markersize', 15); 
plot(real(sigma), imag(sigma), 'r.', 'markersize', 15); hold off
legend('$\partial A$','$\Lambda(\mathbf{A})$','$\Lambda(\mathbf{A}_r)$ cIRKA','$\sigma$','fontsize',20, 'interpreter','latex', 'Location', 'northeast')
xlabel('Re($z$)', 'fontsize',20, 'interpreter','latex')
ylabel('Im($z$)', 'fontsize',20, 'interpreter','latex')
axis equal

% bode on P
wi = Z;
G = @(s) C*((s*speye(n) - A)\B);
Gr = @(s) Cr*((s*eye(ro) - Ar)\Br);
Geval = [];
Greval = [];
for j = 1:size(wi,1)
    Geval(j) = G(wi(j));
    Greval(j) = Gr(wi(j));
end

figure()
subplot(2,1,1);
semilogy(t,abs(Geval),'k-'); hold on
semilogy(t,abs(Greval),'r--'); hold off
subplot(2,1,2);
semilogy(t,abs(Geval - Greval)./abs(Geval),'r-');
legend('cIRKA', 'fontsize',20, 'interpreter','latex', 'Location', 'northeast')
xlabel('$z$', 'fontsize',20, 'interpreter','latex')
ylabel('rel. error', 'fontsize',20, 'interpreter','latex')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = @(t) 1*sin(t)+0.8*cos(2*t);
v = @(t) 1.3*cos(t); 
tau = @(t) 0.9*(u(t)+1i*v(t))-0.6;
du = @(t) 1*sin(t)-0.8*sin(t);
dv = @(t) -1.3*sin(t);
dtau = @(t) 0.8*(du(t)+1i*dv(t));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% H2 computation
rset = 2:1:10; 

for i = 1:size(rset,2)
    ro = rset(i);
    disp(['r=',num2str(ro)]);

    % Running CIRKA
    init = -10 + 10i*(rand(ro,1)-0.5); %init = [init;conj(init)];
    [Ar, Br, Cr, sigma, converged] = conformalIRKA(A,B,C,ro, @(x) conj(r(x)), init, 1000);
    count = 0;
    while ~converged && count<5
        count = count + 1;
        init = -5 + 3i*(rand(ro,1)-0.5);
        [Ar, Br, Cr, sigma, converged] = conformalIRKA(A,B,C,ro, @(x) conj(r(x)), init, 1000);
    end
    fom = @(x) (C*((tau(x)*speye(n)-A)\B)); 
    rom = @(x) (Cr*((tau(x)*speye(ro)-Ar)\Br));
    error = @(x) fom(x)-rom(x);
    funerror = @(z) error(z)*(error(z))' .* abs(dtau(z));
    fomfom = @(z) fom(z)*(fom(z))' .* abs(dtau(z));
    H2P_(i) = sqrt((1/(2*pi))*integral(funerror,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true))./sqrt((1/(2*pi))*integral(fomfom,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true));   
end


figure()
semilogy(rset, H2P_, 'r-x');
ylabel('$\|H-\widehat{H}\|/\|H\|$', 'fontsize',20, 'interpreter','latex')
xlabel('$r$', 'fontsize',20, 'interpreter','latex')