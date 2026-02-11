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
c = -1.7e5; 
R = 1.7e5;


% compute gramians conformalBT
psiinv = @(x) ((x-c*speye(n))/R+speye(n))/((x-c*speye(n))/R-speye(n));

Alyap = psiinv(A);
Blyap = sqrt(2*R) * ((c*eye(n)+R*eye(n)-A) \ (B));
Clyap = sqrt(2*R) * (((c*eye(n)+R*eye(n)-A)') \ (C'));

U2 = lyapchol(Alyap, Blyap);
L2 = lyapchol(Alyap', Clyap);
[Zcbt,Scbt,Ycbt] = svd(L2*U2', 'econ');

%%
psi = @(z) c + R * (z+1)./(z-1);
dpsi = @(z) R*(-2)./(z-1).^2;

Nj = [30, 60, 200];
rset = 4:2:20;


H2Pquad = zeros(length(Nj), length(rset));
H2Pcbt = zeros(1,length(rset));


for i = 1:size(rset,2)
    r = rset(i)

    % Running conformalBT
    Z2 = Zcbt(:,1:r);
    Y2 = Ycbt(:,1:r);
    S2 = Scbt(1:r,1:r);  S2half = sqrt(S2);
    
    Wr = L2'*Z2/S2half;
    Vr = U2'*Y2/S2half;

    Ar2 = Wr'*A*Vr;
    Br2 = Wr'*B;
    Cr2 = C*Vr;

    [H2Pcbt(i),~] = H2Anorm(A,B,C,Ar2,Br2,Cr2,psi,dpsi);

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
    
        g = zeros(size(wo'));
        h = zeros(size(wc'));

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
    
        [H2Pquad(q,i),~] = H2Anorm(A,B,C,Ar,br,cr,psi,dpsi);
    
    
    end

end


figure()
semilogy(rset,H2Pcbt, 'r-o'); hold on
semilogy(rset,H2Pquad(1,:), '-x', 'color', [0.8,0.8,0.8]);
semilogy(rset,H2Pquad(2,:), '-x', 'color', [0.5,0.5,0.5]);
semilogy(rset,H2Pquad(3,:), '-x', 'color', [0.2,0.2,0.2]);
legend('conformalBT', 'quadcBT $N=30$', 'quadcBT $N=60$', 'quadcBT $N=200$' ,'interpreter', 'latex');
hold off

