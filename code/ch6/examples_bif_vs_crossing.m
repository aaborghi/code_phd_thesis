%% Defining eigenvalue trajectories

x_cross = linspace(-1,1,5);
l1 = @(p) -p; l2 = @(p) p; %crossing

left = 1 - (1 - 0) * logspace(0, -2, ceil(20));
right = 1 + (2 - 1) * logspace(-2, 0, floor(20));
x_bif = 1i*unique([left right]);
% x_bif = 1i*linspace(0,2,100);
l1_ = @(p) sqrt(1+p.^2); l2_ = @(p) -sqrt(1+p.^2); %bifurcation


figure()
subplot(1,3,1)
plot(l1(x_cross),x_cross, 'r-'); hold on
plot(l2(x_cross),x_cross, 'b--'); hold off
legend('\lambda_1', '\lambda_2');
title('crossing');
xlabel('\lambda');
ylabel('p');
subplot(1,3,2)
plot(imag(l1_(x_bif)), imag(x_bif), 'r-'); hold on
plot(imag(l2_(x_bif)), imag(x_bif), 'b--'); hold off
title('bifurcation');
xlabel('re(\lambda)');
subplot(1,3,3)
plot(real(l1_(x_bif)), imag(x_bif), 'r-'); hold on
plot(real(l2_(x_bif)), imag(x_bif), 'b--'); hold off
xlabel('im(\lambda)');
