function plot_phase(f, box, res)
%PLOT_PHASE   Phase plot of a complex function.
%   PLOT_PHASE(F, BOX) plots the phase of the function handle F in the box
%   defined by BOX (optional, by default BOX = 1.5); see interpret_plot_box.
%
%   PLOT_PHASE(F, BOX, RES) specifies the resolution (RES = 1024 by default).
%   If length(res) = 1, then xres = yres = res, if length(res) = 2, then 
%   [xres, yres] = res.

% Creating adequate bounds for plot-domain.
if ( ( nargin < 2 ) || isempty(box) )
    box = 1.5;
end

box = [-box, box, -box, box];
xmin = box(1);
xmax = box(2);
ymin = box(3);
ymax = box(4);

if ( ( nargin == 3 ) && ~isempty(res) )
    if ( length(res) == 1 )
        xres = res;
        yres = res;
    elseif ( length(res) == 2 )
        xres = res(1);
        yres = res(2);
    else
        warning('Wrong size of res, switching to default resolution.')
        xres = 1024; yres = 1024;
    end
else
    %xres = 256; yres = 256;
     xres = 1024; yres = 1024;
end

% Creating meshgrid
x = linspace(xmin, xmax, xres);
y = linspace(ymin, ymax, yres);
[x, y] = meshgrid(x, y);
z = x + 1i*y;

% Evaluating function
fz = f(z);

% Phaseplot
% p = surf(real(z), imag(z), 0*fz, angle(-fz));
p = surf(real(z), imag(z), zeros(size(fz)), angle(-fz));
set(p, 'EdgeColor', 'none');
clim([-pi, pi]);
colormap hsv(256)

view(0,90), axis equal, axis off

hold on
contour(real(z), imag(z), angle(-fz), 0, 'LineColor', 'w')
% This has no actual impact on the plot itself, but preserves the square
% appearance of the figure.
hold off

end