function contour_transfer_function(G, sigma_range, omega_range, resolution)
    % G: transfer function
    % sigma_range: [min_sigma, max_sigma]
    % omega_range: [min_omega, max_omega]
    % resolution: number of points in each dimension (e.g., 200)

    % Create grid in complex s-plane
    sigma = linspace(sigma_range(1), sigma_range(2), resolution);
    omega = linspace(omega_range(1), omega_range(2), resolution);
    [SIGMA, OMEGA] = meshgrid(sigma, omega);
    S = SIGMA + 1i*OMEGA;

    % Evaluate the transfer function at each point in S
    H = zeros(size(S));
    for i = 1:numel(S)
        s = S(i);
        H(i) = G(s);
    end

    % Plot the magnitude of H(s)
    figure;
    contourf(SIGMA, OMEGA, H, 20); % 50 contour levels
    colorbar;
    set(gca, 'FontSize', 16);
    xlabel('Re($z$)','interpreter','latex','fontsize',20);
    ylabel('Im($z$)','interpreter','latex','fontsize',20);
end