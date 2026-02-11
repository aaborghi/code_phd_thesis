%% Testing example
clear all; clc; rng(31);
addpath('functions');

h = 0.001;
phi = @(z) (2/h)*(1-z)./(1+z);
dphi = @(z) -(2/h)*2./(1+z).^2;
dphisqrt= @(z) sqrt((2/h))*(1i*sqrt(2)./(1+z));
ddphi = @(z) (2/h)*4./(1+z).^3;


% https://www.slicot.org/20-site/126-benchmark-examples-for-model-reduction
system = load("CDplayer.mat");
A = full(system.A);
n = size(A,1);
E = speye(n);
B = (system.B);
C = (system.C);
q = size(C,1);
m = size(B,2);

%% TF-IRKA
G = @(s) C*((s*E-A)\B);
dG = @(s) -(C/(s*E-A))*E*(((s*E-A)\B));

H = @(s) G(s);
dH = @(s) dG(s);



fomfom = @(z) trace((H(phi(exp(1i*z))))*(H(phi(exp(1i*z))))');
H2fomfom = sqrt((1/(2*pi))*integral(fomfom,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true));
r_range = 2:1:10;
for i = 1:1:size(r_range,2)
    r = r_range(i)
    init = 1+ 1*rand(r,1)+1i*(randn(r,1)-0.5);
    left = rand(q,r) + 1i*rand(q,r);
    right = rand(m,r) + 1i*rand(m,r);
    [Er,Ar,Br,Cr,sigma,left,right,conv] = tfirka_mimo(H,dH,left,right,r,init,500,1e-6, @(z) -conj(z));
    count = 1;
    test = real(Er\Ar);
    while conv == 0 && count < 5 || isnan(test(1,1))
        init = 1+ 1*rand(r,1)+1i*(randn(r,1)-0.5);
        left = rand(q,r) + 1i*rand(q,r);
        right = rand(m,r) + 1i*rand(m,r);
        [Er,Ar,Br,Cr,sigma,left,right,conv] = tfirka_mimo(H,dH,left,right,r,init,1000,1e-6, @(z) -conj(z));
        count = count + 1;
        test = real(Er\Ar);
    end
    if conv == 1
        Gr2 = @(s) Cr*((s*Er-Ar)\(Br));
        funerror = @(z) trace((H(phi(exp(1i*z)))-Gr2(phi(exp(1i*z))))*(H(phi(exp(1i*z)))-Gr2(phi(exp(1i*z))))');
        H2_tfirka(i) = sqrt((1/(2*pi))*integral(funerror,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true))./H2fomfom;
    else 
        H2_tfirka(i) = nan;
    end
end


%% CT-IRKA
unitcircle = exp(1i*linspace(0,2*pi,1000));

G = @(s) C*((s*E-A)\B);
dG = @(s) -(C/(s*E-A))*E*(((s*E-A)\B));


H = @(s) G(phi(s));
dH = @(s) dG(phi(s)).*dphi(s);

fomfom = @(z) trace((H(exp(1i*z)))*(H(exp(1i*z)))');
H2fomfom = sqrt((1/(2*pi))*integral(fomfom,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true));
sigma = [];
r_range = 2:1:10;
for i = 1:1:size(r_range,2)
    r = r_range(i)
    init = 0.1*randn(r,1)+0.1i*randn(r,1); init = (init./abs(init)).*rand(r,1);
    left = rand(q,r) + 1i*rand(q,r);
    right = rand(m,r) + 1i*rand(m,r);
    [Er,Ar,Br,Cr,Dr,sigma,left,right,conv] = ctfirka_mimo(H,dH,left,right,r,init,500,1e-6, @(z) 1./conj(z));
    while conv == 0
        left = rand(q,r) + 1i*rand(q,r);
        right = rand(m,r) + 1i*rand(m,r);
        init = sigma;
        [Er,Ar,Br,Cr,Dr,sigma,left,right,conv] = ctfirka_mimo(H,dH,left,right,r,init,1000,1e-6, @(z) 1./conj(z));
    end
    Gr = @(s) Cr*((s*Er-Ar)\(Br))+Dr;
    funerror = @(z) trace((H(exp(1i*z))-Gr(exp(1i*z)))*(H(exp(1i*z))-Gr(exp(1i*z)))');
    H2_ctirka(i) = sqrt((1/(2*pi))*integral(funerror,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true))./H2fomfom;

end

figure()
semilogy(r_range,H2_tfirka,'b-o'); hold on
semilogy(r_range,H2_ctirka,'r-x'); 
legend('$H_2(A)$ error TF-IRKA', '$H_2(A)$ error CT-IRKA','Interpreter','latex','FontSize',14);
hold off



%% Discrete dynamical system

r = 6;
G = @(s) C*((s*E-A)\B);
dG = @(s) -(C/(s*E-A))*E*(((s*E-A)\B));

init = 0.1*randn(r,1)+0.1i*randn(r,1); init = (init./abs(init)).*rand(r,1);
left = rand(q,r) + 1i*rand(q,r);
right = rand(m,r) + 1i*rand(m,r);
H = @(s) G(phi(s));
dH = @(s) dG(phi(s)).*dphi(s);
[Er,Ar,Br,Cr,Dr,sigma,left,right,conv] = ctfirka_mimo(H,dH,left,right,r,init,1000,1e-6, @(z) 1./conj(z));

time = 5e3;
x = zeros(n,time);
xr = zeros(r,time);
u = zeros(m,time); 
u(:,10) = [1;1]; % discrete-time impulse
I = speye(n);

% computing the discrete-time dynamics
for k = 3:1:time
    x(:,k) = (I-(h/2)*A)\((I+(h/2)*A)*x(:,k-1)+ B*(h/2)*(u(:,k)+u(:,k-1)));
    xr(:,k) = (Ar\Er)*xr(:,k-1) - (Ar\Br)*u(:,k);
end
y = C*x;
yr = Cr*xr + Dr*u;

% computing the l-infinity error bound
Gr = @(s) Cr*((s*Er-Ar)\(Br))+Dr;
funerror = @(z) trace((H(exp(1i*z))-Gr(exp(1i*z)))*(H(exp(1i*z))-Gr(exp(1i*z)))');
H2D_bound = sqrt((1/(2*pi))*integral(funerror,0,2*pi,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true))*(sqrt(sum(u(:).^2)));

skip = 20;
for k = 1:size(y,2)
    normtwo_y(k) = norm(y(:,k) - yr(:,k),2);
end
figure()
semilogy(1:skip:time,normtwo_y(1:skip:time),'r--'); hold on
semilogy(1:skip:time,ones(1,length(1:skip:time))*H2D_bound,'b:');
legend('$|y_k-\widehat{y}_k|$','boundary','interpreter','latex');
title('Error and bound');

figure()
plot(1:time,y, 'k'); hold on
plot(1:time,yr, 'r:'); 





