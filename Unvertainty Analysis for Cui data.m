function CuiDimensionalboxplotttgh()
    % Cui reversible inhibition fit with DE-MCz in log(k) via unit cube
    % Green band: 95% prediction uncertainty (parameter uncertainty only)
    % Red curve: posterior mean prediction (no-change)
    % Blue curve: posterior mean prediction (daily refresh)
    % Pairwise / candle: Hegyesi style, no Q-filtering.

    % -------------------- MCMC settings --------------------
    nit       = 250;      % increase (e.g. 1500) for final results
    nthin     = 5;
    nchain    = 5;
    n_walkers = nchain * 8;
    gamma_val = [];
    jitter    = [];

    % -------------------- Data (Cui) -----------------------
    % No buffer change
    t_no_raw = [9.6, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    y_no_raw = [12.1, 12.1, 11.9, 11.9, 11.9, 11.8, 11.4, 11.1, 10.7, 8.7, 0.0];
    [t_no, idx_no] = sort(t_no_raw);
    y_no = y_no_raw(idx_no);

    % Daily refresh
    t_daily = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8.6];
    y_daily = [0, 10, 14.5, 24.2, 31.8, 40.1, 47.7, 60.6, 68, 73.1];

    % -------------------- Bounds (linear) ------------------
    lower = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3];
    upper = [1e3,  1e3,  1e3,  1e3,  1e3];

    % -------------------- PML (Hegyesi dimensional) --------
    % [k1, k2, k3, k4, k5]
    pml = [9.3e1, 6.1e1, 2.8e1, 7.0e1, 3.6e-2];

    fprintf('\n=== CUI DATA (log-bounds unit-cube DE-MCz; PML from Hegyesi) ===\n');

    % -------------------- Run DE-MCz (log-space via unit cube) ---------
    [samples, ~] = run_block_cui_unitcube( ...
        t_no, y_no, t_daily, y_daily, lower, upper, pml, ...
        nit, nthin, n_walkers, gamma_val, jitter);

    % samples are in LINEAR space (k1..k5)

    % ---------- Posterior representative parameters (geometric mean) ---
    log_samp    = log(samples);
    best_params = exp(mean(log_samp,1));   % geometric mean (more central for log-uniform)

    % ---------- Parameter labels ---------------------------------------
    param_labels = {'$k_1$','$k_2$','$k_3$','$k_4$','$k_5$'};

    fprintf('\nPML (Hegyesi) used for init (day^{-1}):\n');
    for i = 1:5
        fprintf('k%d_PML = %.6g\n', i, pml(i));
    end

    fprintf('\nPosterior geometric means (day^{-1}):\n');
    for i = 1:5
        fprintf('k%d_geom_mean = %.6g\n', i, best_params(i));
    end

    fprintf('\nPosterior means and 95%% credible intervals (all samples):\n');
    for i = 1:size(samples, 2)
        post_mean = mean(samples(:,i));
        plow      = prctile(samples(:,i),  2.5);
        phigh     = prctile(samples(:,i), 97.5);
        fprintf('k%d: mean = %.6g, 95%% CI = [%.6g, %.6g]\n', ...
                i, post_mean, plow, phigh);
    end

    % -------------------- Predictions & 95% bands ----------------------
    % Fixed time grids for each experiment
    tgrid_no    = linspace(min(t_no),    max(t_no),    600);
    tgrid_daily = linspace(min(t_daily), max(t_daily), 600);

    [preds_no, preds_daily] = compute_predictions_cui( ...
        samples, t_no, t_daily, tgrid_no, tgrid_daily);

    % 95% prediction-uncertainty *bands from parameter uncertainty only*
    ci_low_no   = prctile(preds_no,   2.5, 1);
    ci_high_no  = prctile(preds_no,  97.5, 1);
    mean_no     = mean(preds_no, 1);        % posterior mean prediction

    ci_low_day  = prctile(preds_daily,  2.5, 1);
    ci_high_day = prctile(preds_daily, 97.5, 1);
    mean_day    = mean(preds_daily, 1);     % posterior mean prediction

    % -------------------- Combined plot (Hegyesi style) ----------------
    plot_combined_red_no_blue_daily( ...
        t_no, y_no, tgrid_no, ci_low_no, ci_high_no, mean_no, ...
        t_daily, y_daily, tgrid_daily, ci_low_day, ci_high_day, mean_day);

    % -------------------- Candle-like CI panel (no Q-filter) -----------
    plot_candle_CI(samples, param_labels);

    % -------------------- Pairwise posterior (Hegyesi-style) -----------
    plot_pairwise_hegyesi(samples, best_params, param_labels);
end

% =====================================================================
% Predictions for Cui (Hegyesi-style, fixed grids)
% =====================================================================
function [preds_no, preds_daily] = compute_predictions_cui( ...
        samples, ~, t_daily, tgrid_no, tgrid_daily)

    nsamp = size(samples,1);

    preds_no    = zeros(nsamp, numel(tgrid_no));
    preds_daily = zeros(nsamp, numel(tgrid_daily));

    for i = 1:nsamp
        theta = samples(i,:);  % linear params [k1..k5]

        % ---------- No-buffer-change on fixed grid ----------
        WL_no_fine = simulate_nochange(theta, tgrid_no);
        preds_no(i,:) = WL_no_fine(:).';   % row

        % ---------- Daily-refresh on fixed grid -------------
        % 1. simulate "ridged" daily curve across actual day boundaries
        [t_full, WL_daily_full] = simulate_daily_full(theta, t_daily);
        % 2. interpolate onto fixed grid (like Hegyesi)
        preds_daily(i,:) = interp1(t_full, WL_daily_full, ...
                                   tgrid_daily, 'pchip', 'extrap');
    end
end

% =====================================================================
% Combined plot (uses already-computed CIs and mean predictions)
% =====================================================================
function plot_combined_red_no_blue_daily( ...
        t_no, y_no, tgrid_no, ci_low_no, ci_high_no, mean_no, ...
        t_daily, y_daily, tgrid_daily, ci_low_day, ci_high_day, mean_day)

    figure; hold on; grid on; set(gcf,'Color','w');

    greenShade = [0 0.7 0];

    % 95% band for no-buffer-change (prediction uncertainty)
    fill([tgrid_no, fliplr(tgrid_no)], ...
         [ci_low_no, fliplr(ci_high_no)], ...
         greenShade, 'EdgeColor','none', 'FaceAlpha',0.35, ...
         'DisplayName','95% prediction uncertainty');

    % 95% band for daily-refresh (same colour, hidden in legend)
    fill([tgrid_daily, fliplr(tgrid_daily)], ...
         [ci_low_day, fliplr(ci_high_day)], ...
         greenShade, 'EdgeColor','none', 'FaceAlpha',0.20, ...
         'HandleVisibility','off');

    % Posterior mean curves (red=no-change, blue=daily)
    plot(tgrid_no,    mean_no,  'r-', 'LineWidth', 3, ...
        'DisplayName','Reversible inhibition Model (no buffer change)');
    plot(tgrid_daily, mean_day, 'b-', 'LineWidth', 3, ...
        'DisplayName','Reversible inhibition Model (daily change)');

    % Data points
    plot(t_no,    y_no,    'ko', 'MarkerSize', 5, 'MarkerFaceColor','k', ...
         'DisplayName','No buffer change data');
    plot(t_daily, y_daily, 'ks', 'MarkerSize', 5, 'MarkerFaceColor','k', ...
         'DisplayName','Daily buffer change data');

    xlabel('Time (days)', 'FontSize', 20);
    ylabel('Relative Weight Loss (%)', 'FontSize', 20);
    ylim([0 100]);
    set(gca,'FontSize',16);
    legend('Location','northwest','FontSize',10,'Box','off');

    save_fig('Cui_Combined_RedNochange_BlueDaily.png');
end

% =====================================================================
% Candle-like posterior CI panel (with per-parameter legend)
% =====================================================================
function plot_candle_CI(samples, param_labels)
    means = mean(samples,1);
    lo    = prctile(samples,  2.5, 1);
    hi    = prctile(samples, 97.5, 1);

    n    = numel(param_labels);
    cmap = lines(n);
    w    = 0.28;

    figure('Name','Posterior CI (candle-like)'); hold on; grid on; set(gcf,'Color','w');

    hCandles = gobjects(1,n);
    for i = 1:n
        x1 = i - w/2; x2 = i + w/2;
        xv = [x1 x2 x2 x1];
        yv = [lo(i) lo(i) hi(i) hi(i)];
        hCandles(i) = patch(xv, yv, cmap(i,:), ...
                            'EdgeColor', cmap(i,:), ...
                            'LineWidth', 1.2, ...
                            'DisplayName', sprintf('%s (95\\%% CI)', param_labels{i}));
        % posterior mean tick
        plot([x1 x2], [means(i) means(i)], 'r-', 'LineWidth', 2);
    end

    xlim([0.5, n+0.5]);
    set(gca,'XTick',1:n, 'XTickLabel',param_labels, ...
            'TickLabelInterpreter','latex','FontSize',12);
    xlabel('Parameters','Interpreter','latex','FontSize',18);
    ylabel('Value of rate constant (day$^{-1}$)', 'Interpreter','latex','FontSize',18);

    % add a single "Mean" legend item
    hMean = plot(nan, nan, 'r-', 'LineWidth', 1, 'DisplayName', 'Mean');

    lgd = legend([hCandles, hMean], ...
                 'Location','northeast', 'Interpreter','latex','Box','off');
    lgd.NumColumns    = 1;
    lgd.ItemTokenSize = [10 10];

    save_fig('Cui_PosteriorCI_Candle.png');
end

% =====================================================================
% Simulators (reversible inhibition), linear params k = [k1..k5]
% =====================================================================
function WL_no = simulate_nochange(theta, t_vec)
    E0 = 0.08; S0 = 1.0; y0 = [E0; S0; 0; 0; 0];
    [~, sol] = ode45(@(tt,yy) ode_model_linear(tt,yy,theta), t_vec, y0);
    WL_no = 100 * sol(:,5);
end

function [t_full, WL_daily] = simulate_daily_full(theta, t_vec)
    E0  = 0.08; S0 = 1.0;
    n   = numel(t_vec);
    t_full       = [];
    P_cum_full   = [];
    P_cum_offset = 0;

    for i = 1:(n-1)
        t_start = t_vec(i);
        t_end   = t_vec(i+1);
        dt      = t_end - t_start;

        t_local = linspace(0, dt, 200);      % within-day resolution
        y0_day  = [E0; S0; 0; 0; 0];

        [~, sol_day] = ode45(@(tt,yy) ode_model_linear(tt,yy,theta), t_local, y0_day);
        P_local = sol_day(:,5);

        t_segment     = t_local + t_start;
        P_cum_segment = P_cum_offset + P_local.';  % row

        if i == 1
            t_full     = t_segment;
            P_cum_full = P_cum_segment;
        else
            t_full     = [t_full,     t_segment(2:end)];
            P_cum_full = [P_cum_full, P_cum_segment(2:end)];
        end

        P_cum_offset = P_cum_offset + P_local(end);
    end

    WL_daily = 100 * P_cum_full;
end

% Reversible inhibition ODE (linear parameters)
function dydt = ode_model_linear(~, y, p)
    k1 = p(1); k2 = p(2); k3 = p(3); k4 = p(4); k5 = p(5);
    E   = y(1); S = y(2); ES = y(3); ESP = y(4); P = y(5);

    dE   = -k1*E*S + (k2+k3)*ES;
    dS   = -k1*E*S + k2*ES;
    dES  =  k1*E*S - (k2+k3)*ES - k4*ES*P + k5*ESP;
    dESP =  k4*ES*P - k5*ESP;
    dP   =  k3*ES - k4*ES*P + k5*ESP;

    dydt = [dE; dS; dES; dESP; dP];
end

% =====================================================================
% Unit-cube DE-MCz block (log-bounds + PML like Hegyesi)
% =====================================================================
function [saved_k, saved_logprob] = run_block_cui_unitcube( ...
        t_no, y_no, t_daily, y_daily, lower, upper, pml, ...
        nit, nthin, n_walkers, gamma_val, jitter)

    d  = numel(lower);

    % --- LOG-SPACE BOUNDS ---
    lb = log(lower(:)');   % log(lower)
    ub = log(upper(:)');   % log(upper)

    % --- PML in log-space, then to unit cube -----------------
    log_pml = log(pml(:)');                 % 1×d
    u_pml   = (log_pml - lb) ./ (ub - lb);  % map to [0,1]^d

    % Initialize walkers around PML in unit cube
    init_u = repmat(u_pml, n_walkers, 1) + 0.1*randn(n_walkers, d);
    init_u = reflect01(init_u);  % keep in [0,1]

    % History for DEMCz proposals
    Z_u = init_u;

    logpost_u = @(u) logposterior_cui_unit(u, lb, ub, t_no, y_no, t_daily, y_daily);

    cur_logp = zeros(n_walkers,1);
    for j = 1:n_walkers
        cur_logp(j) = logpost_u(init_u(j,:));
    end

    [saved_u, saved_logprob] = DEMCz_unit( ...
        logpost_u, init_u, Z_u, nit, nthin, cur_logp, gamma_val, jitter);

    % Map unit cube → log-parameters → linear parameters
    saved_logk = box_from_unit(saved_u, lb, ub);  % log(k)
    saved_k    = exp(saved_logk);                 % linear k
end

% Box <-> unit helpers
function k = box_from_unit(u, lb, ub)
    % NOTE: here k is actually log(k) when lb,ub are logs
    k = lb + u .* (ub - lb);
end

function u = unit_from_box(k, lb, ub) %#ok<DEFNU>
    u = (k - lb) ./ (ub - lb);
end

% -------------------- LOG-SPACE LIKELIHOOD -----------------
function logp = logposterior_cui_unit(u, lb, ub, t_no, y_no, t_daily, y_daily)
    % u in [0,1]^d
    if any(u<0) || any(u>1)
        logp = -inf;
        return;
    end
    try
        % Map unit cube to LOG-parameter box, then exponentiate
        theta_log = box_from_unit(u, lb, ub);  % log(k)
        theta     = exp(theta_log);            % linear k = [k1..k5]

        % No-change at observed times
        WL_no = simulate_nochange(theta, t_no);

        % Daily: full ridged curve, then sample it at observed times
        [t_full_day, WL_full_day] = simulate_daily_full(theta, t_daily);
        WL_daily = interp1(t_full_day, WL_full_day, t_daily, 'pchip');

        r_no    = WL_no(:)    - y_no(:);
        r_daily = WL_daily(:) - y_daily(:);

        % Gaussian likelihood with fixed sigma = 0.8
        sigma  = 0.8;
        sigma2 = sigma^2;
        logp   = -0.5 * (sum(r_no.^2) + sum(r_daily.^2)) / sigma2;
        % Normalisation constant -n*log(sigma*sqrt(2*pi)) is omitted (constant in theta)

    catch
        logp = -inf;
    end
end

% -------------------- DE-MCz in unit cube -----------------------------
function [saved_u, saved_logprob] = DEMCz_unit( ...
        logtarget_u, states_u, Z_u, n, n_thin, current_logprob, gamma_schedule, jitter)

    [m,d] = size(states_u);
    if nargin<7 || isempty(gamma_schedule)
        gamma_schedule = 2.38/sqrt(2*d);
    end
    if nargin<8 || isempty(jitter)
        jitter = 1e-5 .* ones(1,d);
    end
    if nargin<5 || isempty(n_thin), n_thin = 1; end

    saved_u       = [];
    saved_logprob = [];
    Z = [Z_u; states_u];

    for it = 1:n
        for j = 1:m
            mZ = size(Z,1);
            idx = randperm(mZ,2);
            diffvec = Z(idx(1),:) - Z(idx(2),:);

            prop = states_u(j,:) + gamma_schedule * diffvec + jitter .* randn(1,d);
            prop = reflect01(prop); % keep in [0,1]

            prop_lp = logtarget_u(prop);
            if log(rand) < (prop_lp - current_logprob(j))
                states_u(j,:)      = prop;
                current_logprob(j) = prop_lp;
            end
        end

        if mod(it, n_thin)==0
            Z = [Z; states_u];
            saved_u       = [saved_u; states_u];              %#ok<AGROW>
            saved_logprob = [saved_logprob; current_logprob]; %#ok<AGROW>
        end
    end
end

function x = reflect01(x)
    x = mod(x, 2);
    over = x > 1;
    x(over) = 2 - x(over);
end

% =====================================================================
% Pairwise posterior (Hegyesi style, no centering / clamp)
% =====================================================================
function plot_pairwise_hegyesi(samples, best_params, param_labels)
    n_dim = size(samples,2);

    figure('Name','Cui Pairwise Posterior','Position',[100,100,1200,900]);
    n_rows = ceil(sqrt(n_dim-1));
    n_cols = ceil((n_dim-1)/n_rows);

    for i = 1:(n_dim-1)
        subplot(n_rows,n_cols,i); hold on; box on;

        xdata = samples(:,i);
        ydata = samples(:,i+1);

        scatter(xdata, ydata, ...
                150, 'b', 'filled', 'MarkerFaceAlpha',0.6);
        plot(best_params(i), best_params(i+1), 'r*', ...
             'MarkerSize',15, 'LineWidth',2);  % geometric-mean parameters

        % --- NEW: tighten axes to data range so 0 is not included -------
        x_min = min(xdata); x_max = max(xdata);
        y_min = min(ydata); y_max = max(ydata);

        % small padding but still strictly positive
        px = 0.05*(x_max - x_min);
        py = 0.05*(y_max - y_min);
        xlim([x_min - px, x_max + px]);
        ylim([y_min - py, y_max + py]);

        xlabel(param_labels{i},   'Interpreter','latex','FontSize',20,'FontWeight','bold');
        ylabel(param_labels{i+1}, 'Interpreter','latex','FontSize',20,'FontWeight','bold');
        grid on;
        ax = gca;
        ax.FontSize = 16; ax.FontWeight = 'bold';
        ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
        ax.XAxis.TickLabelFormat = '%.1e'; 
        ax.YAxis.TickLabelFormat = '%.1e';
    end

    save_fig('Cui_PairwisePosterior.png');
end

% =====================================================================
% Save helper
% =====================================================================
function save_fig(fname)
    if ispc
        downloads_dir = fullfile(getenv('USERPROFILE'),'Downloads');
    else
        downloads_dir = fullfile(getenv('HOME'),'Downloads');
    end
    if ~exist(downloads_dir,'dir'), mkdir(downloads_dir); end
    exportgraphics(gcf, fullfile(downloads_dir, fname), 'Resolution', 1200);
    fprintf('✅ Saved figure: %s\n', fullfile(downloads_dir, fname));
end
