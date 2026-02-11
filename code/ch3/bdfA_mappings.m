%% Mappings BDF
h = 1;
phi1 = @(z) (1-z);
phi2 = @(z) ((z.^2)/2 - 2*(z) + 3/2);
phi3 = @(z) (11/6)*(-(2/11)*(z.^3) + (9/11)*(z.^2) - (18/11)*(z) + 1);
phi4 = @(z) ((z.^4)/4 - (4/3)*(z.^3) + 3*(z.^2) - 4*z + 25/12);
phi5 = @(z) (137/60)*((-12/137)*(z.^5) + (75/137)*(z.^4) - (200/137)*(z.^3) + (300/137)*(z.^2) - (300/137)*(z) + 1);
phi6 = @(z) (147/60)*((10/147)*(z.^6) - (72/147)*(z.^5) + (225/147)*(z.^4) - (400/147)*(z.^3) + (450/147)*(z.^2) - (360/147)*(z) + 1);

N = 200;
figure()
subplot(2,3,1)
for i = 0:1:10
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,N));
    radius = linspace(0,1,100) .* exp(1i*2*pi*i/10);
    mapped_circ = phi1(circle);
    mapped_radius = phi1(radius);
    plot(real(mapped_circ),imag(mapped_circ),'k-'); hold on
    plot(real(mapped_radius),imag(mapped_radius),'k-'); 
end
unitcircle = exp(1i*linspace(0,2*pi,N));
mapped_circ = phi1(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('BDF1 or implicit Euler');
axis equal
hold off
subplot(2,3,2)
for i = 0:1:10
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,N));
    radius = linspace(0,1,100) .* exp(1i*2*pi*i/10);
    mapped_circ = phi2(circle);
    mapped_radius = phi2(radius);
    plot(real(mapped_circ),imag(mapped_circ),'k-'); hold on
    plot(real(mapped_radius),imag(mapped_radius),'k-'); 
end
unitcircle = exp(1i*linspace(0,2*pi,N));
mapped_circ = phi2(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('BDF2');
axis equal
hold off
subplot(2,3,3)
for i = 0:1:10
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,N));
    radius = linspace(0,1,100) .* exp(1i*2*pi*i/10);
    mapped_circ = phi3(circle);
    mapped_radius = phi3(radius);
    plot(real(mapped_circ),imag(mapped_circ),'k-'); hold on
    plot(real(mapped_radius),imag(mapped_radius),'k-'); 
end
unitcircle = exp(1i*linspace(0,2*pi,N));
mapped_circ = phi3(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('BDF3');
axis equal
hold off
subplot(2,3,4)
for i = 0:1:10
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,N));
    radius = linspace(0,1,100) .* exp(1i*2*pi*i/10);
    mapped_circ = phi4(circle);
    mapped_radius = phi4(radius);
    plot(real(mapped_circ),imag(mapped_circ),'k-'); hold on
    plot(real(mapped_radius),imag(mapped_radius),'k-'); 
end
unitcircle = exp(1i*linspace(0,2*pi,N));
mapped_circ = phi4(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('BDF4');
axis equal
hold off
subplot(2,3,5)
for i = 0:1:10
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,N));
    radius = linspace(0,1,100) .* exp(1i*2*pi*i/10);
    mapped_circ = phi5(circle);
    mapped_radius = phi5(radius);
    plot(real(mapped_circ),imag(mapped_circ),'k-'); hold on
    plot(real(mapped_radius),imag(mapped_radius),'k-'); 
end
unitcircle = exp(1i*linspace(0,2*pi,N));
mapped_circ = phi5(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('BDF5');
axis equal
hold off
subplot(2,3,6)
for i = 0:1:10
    circle = ((i/10)) .* exp(1i*linspace(0,2*pi,N));
    radius = linspace(0,1,100) .* exp(1i*2*pi*i/10);
    mapped_circ = phi6(circle);
    mapped_radius = phi6(radius);
    plot(real(mapped_circ),imag(mapped_circ),'k-'); hold on
    plot(real(mapped_radius),imag(mapped_radius),'k-'); 
end
unitcircle = exp(1i*linspace(0,2*pi,N));
mapped_circ = phi6(unitcircle);
plot(real(mapped_circ),imag(mapped_circ),'r-','linewidth',2);
title('BDF6');
axis equal
hold off
