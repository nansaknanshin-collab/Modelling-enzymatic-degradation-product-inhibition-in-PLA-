function [Si_matrix, Ti_matrix] = correctsoboldimensional11()
% =========================================================
% Sobol sensitivity analysis for PDE Weight Loss (WL)
%
% CONSISTENT convention:
%   C_j = A_B^(j)  (start from A, replace column j with B)
%
% Estimators consistent with A_B^(j):
%   First-order  (Saltelli 2010):
%       Si = mean( yB .* (yCj - yA) ) / Var(Y)
%   Total-order  (Jansen 1999):
%       Ti = mean( (yB - yCj).^2 ) / (2*Var(Y))
%            ^^^
%            yB NOT yA — C_j agrees with B on all cols except j,
%            so (yB - yCj) isolates only parameter j's effect.
%
% - NO experimental data needed (not fitting)
% - Linear sampling (NO log)
% - Bounds = [nominal/factor, nominal*factor]
% =========================================================

    % -------------------- Time points of interest (hours) --------------------
    time_points = [0, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72];
    time_points = unique(time_points(:))';

    % -------------------- Parameter names --------------------
    param_names = {'$k_1$', '$k_{m1}$', '$k_3$', '$k_{m3}$', '$k_{conv}$', ...
                   '$k_{degC}$', '$k_{degA}$', '$D_{e0}$'};

    % -------------------- Nominal parameter values --------------------
    nominal_values = [ ...
        0.344396,    ...   % k1
        1.19601e-05, ...   % km1
        0.189894,    ...   % k3
        0.0255093,   ...   % km3
        0.0438992,   ...   % kconv
        1.12821,     ...   % kdegC
        0.00438777,  ...   % kdegA
        4.59226      ...   % De0
    ];

    % -------------------- Linear bounds (NO log) --------------------
    factor       = 10.0;  % reduce to 5 or 3 if many solver failures
    lower_bounds = max(nominal_values ./ factor, 1e-15);
    upper_bounds = nominal_values .* factor;

    % -------------------- Sobol settings --------------------
    sample_points = 256;
    num_params    = numel(nominal_values);

    settings.model         = @(p) simulate_WL_PDE(p, time_points);
    settings.lower_bounds  = lower_bounds;
    settings.upper_bounds  = upper_bounds;
    settings.sample_points = sample_points;

    fprintf('\n=== Sobol (Saltelli 2010 + Jansen 1999) for PDE Weight Loss ===\n');
    fprintf('Convention : C_j = A_B^(j)  (start from A, replace col j with B)\n');
    fprintf('Si estimator: mean(yB*(yCj-yA))  / Var(Y)      [Saltelli 2010]\n');
    fprintf('Ti estimator: mean((yB-yCj)^2)   / (2*Var(Y))  [Jansen  1999]\n');
    fprintf('Times (hours): %s\n', mat2str(time_points));
    fprintf('M=%d, p=%d => total PDE solves = (p+2)*M = %d\n', ...
            sample_points, num_params, (num_params+2)*sample_points);
    fprintf('Bounds: lb = nominal/%.0f, ub = nominal*%.0f (linear)\n', factor, factor);

    % -------------------- Run Sobol --------------------
    out = saltelli_sobol_time_series(settings);

    Si_matrix = out.Si;   % [num_times x num_params]
    Ti_matrix = out.Ti;   % [num_times x num_params]

    % -------------------- Plot First-order indices --------------------
    figure('Color','w'); hold on;
    for j = 1:num_params
        plot(time_points, Si_matrix(:,j), '-o', 'LineWidth', 3, 'MarkerSize', 6, ...
             'DisplayName', param_names{j});
    end
    legend(param_names, 'Location','northeast','Interpreter','latex', ...
           'FontSize',14,'NumColumns',2);
    xlabel('Time (hours)','FontSize',18);
    ylabel('First-order Sobol Index  $S_i$','Interpreter','latex','FontSize',18);
    title('First-order Sobol indices — Weight Loss','FontSize',14);
    ylim([0 1]); grid on; set(gca,'FontSize',16);
    exportgraphics(gcf, fullfile(pwd,'sobol_first_order_WL.png'), 'Resolution',600);

    % -------------------- Plot Total-order indices --------------------
    figure('Color','w'); hold on;
    for j = 1:num_params
        plot(time_points, Ti_matrix(:,j), '-s', 'LineWidth', 3, 'MarkerSize', 6, ...
             'DisplayName', param_names{j});
    end
    legend(param_names, 'Location','northeast','Interpreter','latex', ...
           'FontSize',14,'NumColumns',2);
    xlabel('Time (hours)','FontSize',18);
    ylabel('Total-order Sobol Index  $T_i$','Interpreter','latex','FontSize',18);
    title('Total-order Sobol indices — Weight Loss','FontSize',14);
    ylim([0 1]); grid on; set(gca,'FontSize',16);
    exportgraphics(gcf, fullfile(pwd,'sobol_total_order_WL.png'), 'Resolution',600);

    % -------------------- Save CSVs --------------------
    writematrix(time_points(:),  fullfile(pwd,'sobol_time_points_hours.csv'));
    writematrix(Si_matrix,       fullfile(pwd,'sobol_Si_WL.csv'));
    writematrix(Ti_matrix,       fullfile(pwd,'sobol_Ti_WL.csv'));

    fprintf('Saved: sobol_first_order_WL.png, sobol_total_order_WL.png\n');
    fprintf('Saved: sobol_time_points_hours.csv, sobol_Si_WL.csv, sobol_Ti_WL.csv\n');

end


% =========================================================================
% PDE simulation -> WL vector at requested time points
% =========================================================================
function WL_vec = simulate_WL_PDE(pars, time_points)

    N_GRID   = 25;
    L_DOMAIN = 1.0;
    dx       = L_DOMAIN / (N_GRID - 1);

    C0_POLY = 0.302;
    A0_POLY = 0.698;
    E_INIT  = 0.0;
    P_INIT  = 0.0;
    E_BULK  = 0.5;
    ALPHA_E = 1.0;

    time_points = unique(time_points(:))';

    E0  = E_INIT  * ones(1, N_GRID);
    C0  = C0_POLY * ones(1, N_GRID);
    A0  = A0_POLY * ones(1, N_GRID);
    EC0 = zeros(1, N_GRID);
    EA0 = zeros(1, N_GRID);
    P0  = P_INIT  * ones(1, N_GRID);

    E0(1) = E_BULK;

    y0   = [E0, C0, A0, EC0, EA0, P0]';
    opts = odeset('RelTol',1e-6,'AbsTol',1e-9,'MaxStep',0.5);

    rhs  = @(t,y) rhs_pde_mol(t, y, pars, N_GRID, dx, ...
                               C0_POLY, A0_POLY, E_BULK, ALPHA_E);
    try
        [~, Y] = ode15s(rhs, time_points, y0, opts);
    catch
        WL_vec = nan(1, numel(time_points));
        return;
    end

    idxC = (1:N_GRID) + 1*N_GRID;
    idxA = (1:N_GRID) + 2*N_GRID;

    Cmat   = Y(:, idxC);
    Amat   = Y(:, idxA);
    M0     = mean(Cmat(1,:) + Amat(1,:));
    M      = mean(Cmat + Amat, 2);
    WL_vec = 100.0 * (1.0 - M ./ max(M0, 1e-12));
    WL_vec = WL_vec(:)';

end


% =========================================================================
% PDE RHS (method-of-lines)
% =========================================================================
function dydt = rhs_pde_mol(~, y, pars, N_GRID, dx, C0_POLY, A0_POLY, E_BULK, ALPHA_E)

    k1    = pars(1);
    km1   = pars(2);
    k3    = pars(3);
    km3   = pars(4);
    kconv = pars(5);
    kdegC = pars(6);
    kdegA = pars(7);
    De0   = pars(8);

    y  = y(:);
    E  = y(          1 :   N_GRID);
    C  = y(  N_GRID+1 : 2*N_GRID);
    A  = y(2*N_GRID+1 : 3*N_GRID);
    EC = y(3*N_GRID+1 : 4*N_GRID);
    EA = y(4*N_GRID+1 : 5*N_GRID);
    P  = y(5*N_GRID+1 : 6*N_GRID);   %#ok<NASGU>

    E(1) = E_BULK;

    denom0 = C0_POLY + A0_POLY;
    phi    = 1.0 - (C + A + EC + EA) ./ max(denom0, 1e-12);
    phi    = min(max(phi, -0.5), 2.0);

    De    = max(De0 .* (1.0 + ALPHA_E .* phi), 1e-15);
    diffE = diffusion_dirichlet_neumann_varD(E, De, E_BULK, N_GRID, dx);

    dE  =  diffE ...
           - k1.*E.*C  + km1.*EC ...
           - k3.*E.*A  + km3.*EA ...
           + kconv.*EC ...
           + kdegC.*EC + kdegA.*EA;

    dC  = -k1.*E.*C  + km1.*EC;
    dA  = -k3.*E.*A  + km3.*EA  + kconv.*EC;
    dEC =  k1.*E.*C  - km1.*EC  - kconv.*EC - kdegC.*EC;
    dEA =  k3.*E.*A  - km3.*EA  - kdegA.*EA;
    dP  =  kdegC.*EC + kdegA.*EA;

    dE(1) = 0.0;

    dydt = [dE; dC; dA; dEC; dEA; dP];
end


% =========================================================================
% Variable-coefficient diffusion operator
% Dirichlet u(1)=u_left,  Neumann du/dx(L)=0
% =========================================================================
function div = diffusion_dirichlet_neumann_varD(u, D, u_left, N_GRID, dx)

    u    = u(:);
    D    = D(:);
    u(1) = u_left;

    flux             = zeros(N_GRID+1, 1);
    D_iface          = 0.5*(D(1:end-1) + D(2:end));
    flux(2:N_GRID)   = D_iface .* (u(2:end) - u(1:end-1)) / dx;
    flux(N_GRID+1)   = 0.0;

    div              = zeros(N_GRID, 1);
    div(2:N_GRID-1)  = (flux(3:N_GRID)     - flux(2:N_GRID-1)) / dx;
    div(N_GRID)      = 2.0*(flux(N_GRID+1) - flux(N_GRID))     / dx;
    div(1)           = 0.0;
end


% =========================================================================
% Saltelli-Sobol for VECTOR output
%
% Construction : C_j = A_B^(j)
%                All columns from A, column j replaced by B(:,j)
%
% Estimators consistent with A_B^(j):
%   Si [Saltelli 2010] = mean( yB .* (yCj - yA) ) / Var(Y)
%   Ti [Jansen  1999]  = mean( (yB - yCj).^2 )   / (2*Var(Y))
%                                ^^^
%                                yB NOT yA.
% =========================================================================
function out = saltelli_sobol_time_series(settings)

    model = settings.model;
    lb    = settings.lower_bounds(:)';
    ub    = settings.upper_bounds(:)';
    M     = settings.sample_points;
    p     = numel(lb);

    rng(123);
    sob = sobolset(p, 'Skip',1e3, 'Leap',1e2);
    sob = scramble(sob, 'MatousekAffineOwen');
    AB  = net(sob, 2*M);

    A_mat = lb + AB(1:M,   :) .* (ub - lb);
    B_mat = lb + AB(M+1:2*M,:) .* (ub - lb);

    % C_j = A_B^(j): all columns from A, column j from B
    C_arr = zeros(M, p, p);
    for j = 1:p
        C_arr(:,:,j) = A_mat;        % start from A
        C_arr(:,j,j) = B_mat(:,j);   % replace col j with B
    end

    y_test = model(A_mat(1,:));
    T      = numel(y_test);

    yA = nan(M, T);
    yB = nan(M, T);
    yC = nan(M, p, T);

    for i = 1:M
        try
            yA(i,:) = model(A_mat(i,:));
        catch
        end
        try
            yB(i,:) = model(B_mat(i,:));
        catch
        end
    end

    for j = 1:p
        for i = 1:M
            try
                tmp       = model(C_arr(i,:,j));        % 1 x T
                yC(i,j,:) = reshape(tmp, 1, 1, T);      % 1 x 1 x T
            catch
            end
        end
    end

    % Remove samples where any evaluation failed
    valid = all(isfinite(yA), 2) & all(isfinite(yB), 2);
    for j = 1:p
        valid = valid & all(isfinite(squeeze(yC(:,j,:))), 2);
    end

    n_valid = sum(valid);
    fprintf('Sobol valid samples: %d / %d\n', n_valid, M);

    if n_valid < max(50, floor(0.2*M))
        error('Too many failed PDE solves (%d valid of %d). Reduce factor or M.', ...
              n_valid, M);
    end

    yA = yA(valid, :);       % [Mv x T]
    yB = yB(valid, :);       % [Mv x T]
    yC = yC(valid, :, :);    % [Mv x p x T]

    % Variance estimated from A samples
    f0    = mean(yA, 1);               % [1 x T]
    var_y = mean(yA.^2, 1) - f0.^2;   % [1 x T]
    var_y = max(var_y, eps);

    Si = zeros(T, p);
    Ti = zeros(T, p);

    for j = 1:p
        yCj = squeeze(yC(:,j,:));   % [Mv x T]

        % Saltelli (2010) first-order — consistent with C_j = A_B^(j)
        Si(:,j) = ( mean( yB .* (yCj - yA), 1 ) ./ var_y ).';

        % Jansen (1999) total-order — consistent with C_j = A_B^(j)
        % *** yB here, NOT yA ***
        Ti(:,j) = ( mean( (yB - yCj).^2,    1 ) ./ (2 .* var_y) ).';
    end

    out.Si = max(min(Si, 1), 0);
    out.Ti = max(min(Ti, 1), 0);

end
