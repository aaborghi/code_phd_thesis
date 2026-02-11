%% Mappings
h = 1;
phi1 = @(z) (1/h)*(1-z);
phi2 = @(z) (2/h)*(1-z)./(1+z);
phi3 = @(z) (1/h)*((z.^2)/2 - 2*(z) + 3/2);
phi4 = @(z) (1/h)*((z.^4)/4 - (4/3)*(z.^3) + 3*(z.^2) - 4*z + 25/12);

figure()
subplot(2,2,1)
for i = 0:1:10
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,1000));
    radius = linspace(0,1,100) .* exp(1i*2*pi*i/10);
    mapped_circ = phi1(circle);
    mapped_radius = phi1(radius);
    plot(real(mapped_circ),imag(mapped_circ),'k-'); hold on
    plot(real(mapped_radius),imag(mapped_radius),'k-'); 
end
unitcircle = exp(1i*linspace(0,2*pi,1000));
mapped_circ = phi1(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('Implicit Euler');
axis equal
hold off
subplot(2,2,2)
for i = 0:1:10
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,1000));
    radius = linspace(0,1,100) .* exp(1i*2*pi*i/10);
    mapped_circ = phi2(circle);
    mapped_radius = phi2(radius);
    plot(real(mapped_circ),imag(mapped_circ),'k-'); hold on
    plot(real(mapped_radius),imag(mapped_radius),'k-'); 
end
unitcircle = exp(1i*linspace(0,2*pi,1000));
mapped_circ = phi2(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('Midpoint');
axis([-10,10,-10,10]);
hold off
subplot(2,2,3)
for i = 0:1:10
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,1000));
    radius = linspace(0,1,100) .* exp(1i*2*pi*i/10);
    mapped_circ = phi3(circle);
    mapped_radius = phi3(radius);
    plot(real(mapped_circ),imag(mapped_circ),'k-'); hold on
    plot(real(mapped_radius),imag(mapped_radius),'k-'); 
end
unitcircle = exp(1i*linspace(0,2*pi,1000));
mapped_circ = phi3(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('BDF2');
axis equal
hold off
subplot(2,2,4)
for i = 0:1:10
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,1000));
    radius = linspace(0,1,100) .* exp(1i*2*pi*i/10);
    mapped_circ = phi4(circle);
    mapped_radius = phi4(radius);
    plot(real(mapped_circ),imag(mapped_circ),'k-'); hold on
    plot(real(mapped_radius),imag(mapped_radius),'k-'); 
end
unitcircle = exp(1i*linspace(0,2*pi,1000));
mapped_circ = phi4(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('BDF4');
axis equal
hold off


%% Anti-conformal reflection
h = 1;
phi1 = @(z) (1/h)*(1-z);
phi2 = @(z) (2/h)*(1-z)./(1+z);
phi3 = @(z) (1/h)*((z.^2)/2 - 2*(z) + 3/2);
phi4 = @(z) (1/h)*((z.^4)/4 - (4/3)*(z.^3) + 3*(z.^2) - 4*z + 25/12);

Z = phi4(exp(1i*linspace(0,2*pi,1000)));
schwarz = aaa(conj(Z),Z,'tol',1e-15);

Reul = @(z) conj(z./(z-1));
Rtrap = @(z) -conj(z);
Rbdf2 = @(z) 0.5*conj((3*(2*z+1) - 8*sqrt(2*z+1) + 5)./(2*z + 5 -4*sqrt(2*z+1)));
Rbdf4 = @(z) conj(schwarz(z));

figure()
subplot(2,2,1)
for i = 4:1:9
    col = (i/10)^2;
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,100));
    mapped_circ = phi1(circle);
    anticonformal = Reul(mapped_circ);
    plot(real(mapped_circ),imag(mapped_circ),'.','color',[col,col,col],'markersize',5); hold on
    plot(real(anticonformal),imag(anticonformal),'.','color',[col,col,1],'markersize',5); hold on
end
unitcircle = exp(1i*linspace(0,2*pi,1000));
mapped_circ = phi1(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('Implicit Euler');
axis equal
hold off
subplot(2,2,2)
for i = 1:1:9
    col = (i/11)^2;
    mapped_circ = 1i*linspace(-8,8,20) + (i)*0.5;
    anticonformal = Rtrap(mapped_circ);
    plot(real(mapped_circ),imag(mapped_circ),'.','color',[1-col,1-col,1-col],'markersize',5); hold on
    plot(real(anticonformal),imag(anticonformal),'.','color',[1-col,1-col,1],'markersize',5); hold on
end
unitcircle = exp(1i*linspace(0,2*pi,1000));
mapped_circ = phi2(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('Midpoint');
axis([-10,10,-10,10]);
hold off
subplot(2,2,3)
for i = 5:0.5:8.5
    col = (i/10)^2;
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,100));
    mapped_circ = phi3(circle);
    anticonformal = Rbdf2(mapped_circ);
    plot(real(mapped_circ),imag(mapped_circ),'.','color',[col,col,col],'markersize',5); hold on
    plot(real(anticonformal),imag(anticonformal),'.','color',[col,col,1],'markersize',5); hold on
end
unitcircle = exp(1i*linspace(0,2*pi,1000));
mapped_circ = phi3(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('BDF2');
axis equal
hold off
subplot(2,2,4)
for i = 8:0.2:10
    col = (i/11)^(4);
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,100));
    mapped_circ = phi4(circle);
    anticonformal = Rbdf4(mapped_circ);
    plot(real(mapped_circ),imag(mapped_circ),'.','color',[col,col,col],'markersize',5); hold on
    plot(real(anticonformal),imag(anticonformal),'.','color',[col,col,1],'markersize',5); hold on
end
unitcircle = exp(1i*linspace(0,2*pi,1000));
mapped_circ = phi4(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('BDF4');
axis equal
hold off


