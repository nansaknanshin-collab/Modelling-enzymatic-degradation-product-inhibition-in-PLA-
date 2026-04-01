function correcthegyesiDimensionalboxplot()
    % Fit reversible inhibition model to Hegyesi data using DE-MCz
    % Includes Q0.95, prediction uncertainty, pairwise, and posterior CI plot

    % -------------------- MCMC settings --------------------
    nit       = 250;               % iterations per chain
    nthin     = 5;                 % thinning rate
    nchain    = 5;
    n_walkers = nchain * 8;        % total parallel chains
    gamma_val = [];                % default gamma schedule
    jitter    = [];                % default jitter

    % -------------------- Data (Hegyesi) --------------------
    time  = [0, 1, 3, 6, 7, 8, 12, 14, 15];
    data  = [0, 11.3, 12.4, 14, 14.4, 24.1, 26.6, 28.1, 28.3];
    error1 = [0.0, 0.2, 0.9, 1.1];       % for t < 7
    error2 = [1.0, 1.2, 1.3, 1.0, 0.7];  % for t >= 7
    errs   = [error1, error2];

    % -------------------- Initial conditions --------------------
    % Reversible inhibition model: [E, S, ES, ESP, P]
    ic        = [0.03, 0.3, 0,    0,     0   ];     % before phase change
    ic_change = [0.03, 0.3, 0.07, 0,     0.144];    % after t = 7
    t_change  = 7;

    % -------------------- Prior guess (Hegyesi best-fit) --------------------
    % [k1, k2, k3, k4, k5] in day^-1
    pml = [9.3e1, 6.1e1, 2.8e1, 7.0e1, 3.6e-2];

    fprintf('\n=== HEGYESI DATA: COMBINED PHASES (reversible inhibition) ===\n');

    % -------------------- Run MCMC --------------------
    [samples_log, ~] = run_block_combined( ...
        time, data, errs, ic, ic_change, t_change, pml, ...
        nit, nthin, n_walkers, gamma_val, jitter);

    % Back-transform to linear space
    samples     = exp(samples_log);
    best_params = mean(samples, 1);   % posterior mean

    % Print posterior means
    fprintf('\nPosterior means for Hegyesi reversible inhibition model (day^{-1}):\n');
    fprintf('k1 = %.6f\n', best_params(1));
    fprintf('k2 = %.6f\n', best_params(2));
    fprintf('k3 = %.6f\n', best_params(3));
    fprintf('k4 = %.6f\n', best_params(4));
    fprintf('k5 = %.6f\n', best_params(5));

    % Parameter labels
    param_labels = {'$k_1$','$k_2$','$k_3$','$k_4$','$k_5$'};

    % -------------------- Posterior CI plot --------------------
    plot_posterior_CI(samples, param_labels);

    % -------------------- Q95 calculation --------------------
    Q95 = compute_Q95(samples, best_params, time, ic, ic_change, t_change);
    fprintf('\nQ_{0.95} = %.4f\n', Q95);

    % -------------------- Predictions --------------------
    tgrid = linspace(min(time), max(time), 600);
    preds = compute_predictions(samples, ic, ic_change, t_change, tgrid);
    ci_low  = prctile(preds,  2.5, 1);
    ci_high = prctile(preds, 97.5, 1);

    figure; hold on; grid on;
    xlabel('Time (days)','FontSize',20);
    ylabel('Relative Weight Loss (%)','FontSize',20);

    errorbar(time, data, errs, errs, 'o', ...
        'MarkerSize',8, 'Color','k', 'MarkerFaceColor','k', ...
        'LineStyle','none', 'DisplayName','PLA–Proteinase K data');

    fill([tgrid fliplr(tgrid)], [ci_low fliplr(ci_high)], ...
        [0 0.7 0], 'EdgeColor','none', 'FaceAlpha',0.4, ...
        'DisplayName','Prediction uncertainty');

    plot(tgrid, mean(preds,1), 'r-', 'LineWidth',3, ...
        'DisplayName','Reversible inhibition model');

    legend('Location','northwest','FontSize',10,'Box','off');
    save_fig('Hegyesi_PredictionUncertainty.png');

    % -------------------- Pairwise posterior --------------------
    plot_pairwise(samples, best_params, param_labels);
end

% =====================================================================
% Run DE-MCz for combined phases
% =====================================================================
function [saved_states, saved_logprob] = run_block_combined( ...
        time, data, errs, ic, ic_change, t_change, pml, ...
        nit, nthin, n_walkers, gamma_val, jitter)

    % Same bounds as in CuiDimensionalboxplot
    lower  = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3];
    upper  = [233, 153, 70, 175, 0.09];

    lb = log(lower);
    ub = log(upper);

    d        = numel(pml);
    init_pos = log(pml) + 0.1 * randn(n_walkers, d);
    Z        = init_pos;

    logpost = @(theta) logposterior_combined( ...
        theta, lb, ub, time, data, errs, ic, ic_change, t_change);

    [states_out, saved_logprob] = DEMCz( ...
        logpost, init_pos, Z, nit, nthin, {1:d}, [], [], gamma_val, jitter);

    saved_states = states_out;
end

% =====================================================================
% Log-posterior
% =====================================================================
function logp = logposterior_combined(theta, lb, ub, ...
                                      time, data, errs, ...
                                      ic, ic_change, t_change)
    if any(theta < lb) || any(theta > ub)
        logp = -inf;
        return;
    end
    try
        % Phase 1
        t1 = time(time < t_change);
        [~, sol1] = ode45(@(tt,yy) ode_model(tt,yy,theta), t1, ic);

        % Phase 2
        t2 = time(time >= t_change);
        [~, sol2] = ode45(@(tt,yy) ode_model(tt,yy,theta), t2, ic_change);

        % Combine in time order (time vectors are t1 and t2)
        times_all = [t1, t2];
        y_all     = [sol1(:,5); sol2(:,5)] * 100;  % P -> % weight loss

        y_model = interp1(times_all, y_all, time, 'linear', 'extrap');

        sigma = errs(:);
        resid = (y_model(:) - data(:)) ./ sigma;
        logp  = -0.5 * sum(resid.^2) - sum(log(sigma));
    catch
        logp = -inf;
    end
end

% =====================================================================
% ODE model (reversible inhibition)
% =====================================================================
function dydt = ode_model(~, y, log_params)
    p  = exp(log_params);
    k1 = p(1); k2 = p(2); k3 = p(3); k4 = p(4); k5 = p(5);

    E   = y(1);
    S   = y(2);
    ES  = y(3);
    ESP = y(4);
    P   = y(5);

    dE   = -k1*E*S + (k2 + k3)*ES;
    dS   = -k1*E*S + k2*ES;
    dES  =  k1*E*S - (k2 + k3)*ES - k4*ES*P + k5*ESP;
    dESP =  k4*ES*P - k5*ESP;
    dP   =  k3*ES - k4*ES*P + k5*ESP;

    dydt = [dE; dS; dES; dESP; dP];
end

% =====================================================================
% Prediction helper
% =====================================================================
function preds = compute_predictions(samples, ic, ic_change, t_change, tgrid)
    nsamp = size(samples,1);
    npnts = numel(tgrid);
    preds = zeros(nsamp, npnts);

    for i = 1:nsamp
        theta = log(samples(i,:));
        idx1  = tgrid < t_change;
        idx2  = tgrid >= t_change;
        try
            [~, sol1] = ode45(@(tt,yy) ode_model(tt,yy,theta), tgrid(idx1), ic);
            [~, sol2] = ode45(@(tt,yy) ode_model(tt,yy,theta), tgrid(idx2), ic_change);
            times_all = [tgrid(idx1), tgrid(idx2)];
            y_all     = [sol1(:,5); sol2(:,5)] * 100;
            preds(i,:) = interp1(times_all, y_all, tgrid, 'linear', 'extrap');
        catch
            preds(i,:) = zeros(1,npnts);
        end
    end
end

% =====================================================================
% Q95 calculation
% =====================================================================
function Q95 = compute_Q95(samples, best_params, time, ic, ic_change, t_change)
    nsamp = size(samples,1);

    idx1 = time < t_change;
    idx2 = time >= t_change;

    [~, sol1] = ode45(@(tt,yy) ode_model(tt,yy,log(best_params)), time(idx1), ic);
    [~, sol2] = ode45(@(tt,yy) ode_model(tt,yy,log(best_params)), time(idx2), ic_change);

    times_all = [time(idx1), time(idx2)];
    p_best    = interp1(times_all, [sol1(:,5); sol2(:,5)] * 100, ...
                        time, 'linear', 'extrap');

    Q_vals = nan(nsamp,1);
    for i = 1:nsamp
        [~, s1] = ode45(@(tt,yy) ode_model(tt,yy,log(samples(i,:))), time(idx1), ic);
        [~, s2] = ode45(@(tt,yy) ode_model(tt,yy,log(samples(i,:))), time(idx2), ic_change);

        p_i = interp1(times_all, [s1(:,5); s2(:,5)] * 100, ...
                      time, 'linear', 'extrap');

        valid = (p_i > 1e-8) & (p_best > 1e-8);
        if any(valid)
            log_diff   = log2(p_i(valid) ./ p_best(valid));
            Q_vals(i)  = sum(log_diff.^2);
        end
    end

    Q95 = prctile(Q_vals, 95);
end

% =====================================================================
% Supervisor-style Posterior CI plot
% =====================================================================
function plot_posterior_CI(samples, param_names)
    means = mean(samples,1);
    lower = prctile(samples,  2.5,1);
    upper = prctile(samples, 97.5,1);

    cmap = lines(numel(param_names));
    figure; hold on;

    h_leg = gobjects(1,numel(param_names));

    for i = 1:numel(param_names)
        rectangle('Position',[i-0.15, lower(i), 0.3, upper(i)-lower(i)], ...
                  'FaceColor',cmap(i,:), 'EdgeColor',cmap(i,:), 'LineWidth',1.2);
        plot([i-0.15 i+0.15],[means(i) means(i)], 'r-', 'LineWidth',2);
        h_leg(i) = patch(NaN,NaN,cmap(i,:), 'EdgeColor',cmap(i,:));
    end

    xlim([0 numel(param_names)+1]);
    set(gca,'XTick',1:numel(param_names),'XTickLabel',param_names, ...
        'TickLabelInterpreter','latex','FontSize',16);
    xlabel('Parameters','FontSize',16,'Interpreter','latex');
    ylabel('Value of rate constant (day$^{-1}$)','FontSize',16,'Interpreter','latex');
    grid on;

    legend_labels = cell(1,numel(param_names));
    for i = 1:numel(param_names)
        legend_labels{i} = sprintf('%s (95\\%% CI)', param_names{i});
    end
    h_mean = plot(NaN,NaN,'r-','LineWidth',2);
    lgd = legend([h_leg h_mean], [legend_labels {'Mean'}], ...
                 'Location','northeast','Interpreter','latex', ...
                 'FontSize',9,'NumColumns',2);
    lgd.ItemTokenSize = [10,8];

    save_fig('Hegyesi_PosteriorCI_colouredLegend.png');
end

% =====================================================================
% Pairwise posterior
% =====================================================================
function plot_pairwise(samples, best_params, param_labels)
    n_dim = size(samples,2);
    figure('Name','Pairwise Posterior','Position',[100,100,1200,1000]);
    n_rows = ceil(sqrt(n_dim-1));
    n_cols = ceil((n_dim-1)/n_rows);

    for i = 1:(n_dim-1)
        subplot(n_rows,n_cols,i); hold on; box on;
        scatter(samples(:,i), samples(:,i+1), 50, 'b', 'filled', ...
                'MarkerFaceAlpha',0.4);
        plot(best_params(i), best_params(i+1), 'r*', ...
             'MarkerSize',12,'LineWidth',2);
        xlabel(param_labels{i},  'Interpreter','latex','FontSize',14);
        ylabel(param_labels{i+1},'Interpreter','latex','FontSize',14);
        grid on;
    end
    save_fig('Hegyesi_PairwisePosterior.png');
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
    exportgraphics(gcf, fullfile(downloads_dir,fname), 'Resolution',1200);
    fprintf('✅ Saved %s to %s\n', fname, downloads_dir);
end

% =====================================================================
% DE-MCz Sampler
% =====================================================================
function [saved_states,saved_logprob] = DEMCz( ...
        logtarget, states, Z, n, n_thin, blockindex, ...
        current_logprob, temperature_schedule, gamma_schedule, jitter)

    [m,d] = size(states);
    mZ0 = size(Z,1);

    if nargin<10 || isempty(jitter)
        jitter = 1e-5 .* ones(1,d);
    end
    if nargin<8 || isempty(temperature_schedule)
        temperature_schedule = [];
    end
    if nargin<7 || isempty(current_logprob)
        current_logprob = zeros(m,1);
        for j = 1:m
            current_logprob(j) = logtarget(states(j,:));
        end
    end
    if nargin<6 || isempty(blockindex)
        blockindex = {1:d};
    end
    if nargin<9 || isempty(gamma_schedule)
        gamma_schedule = zeros(numel(blockindex),1);
        for k = 1:numel(blockindex)
            gamma_schedule(k) = 2.38 / sqrt(2 * numel(blockindex{k}));
        end
    end
    if nargin<5 || isempty(n_thin)
        n_thin = 1;
    end

    temperature_schedule = temperature_schedule(:);
    if numel(temperature_schedule) < n
        temperature_schedule = [temperature_schedule; ...
                                ones(n-numel(temperature_schedule),1)];
    end
    gamidx = mod(0:(n-1), size(gamma_schedule,2)) + 1;
    save_logprob = [];

    for i = 1:n
        for j = 1:m
            for k = 1:numel(blockindex)
                idx = blockindex{k};
                mZ = size(Z,1) - m;
                if mZ < 2
                    n1 = randi(size(Z,1));
                    n2 = randi(size(Z,1)-1);
                    if n2 >= n1, n2 = n2 + 1; end
                else
                    n1 = randi(mZ);
                    n2 = randi(mZ-1);
                    if n2 >= n1, n2 = n2 + 1; end
                end
                diff      = zeros(1,d);
                gamma_k   = gamma_schedule(k,gamidx(i));
                diff(idx) = gamma_k .* (Z(n1,idx) - Z(n2,idx)) + ...
                            jitter(idx).*randn(1,numel(idx));

                proposal  = states(j,:) + diff;
                prop_logp = logtarget(proposal);
                if temperature_schedule(i)*log(rand) < (prop_logp - current_logprob(j))
                    states(j,:)         = proposal;
                    current_logprob(j)  = prop_logp;
                end
            end
        end
        if mod(i,n_thin) == 0
            Z           = [Z; states];
            save_logprob = [save_logprob; current_logprob];
        end
    end

    num_discard = mZ0;
    if size(Z,1) >= num_discard
        saved_states = Z((num_discard+1):end,:);
    else
        saved_states = [];
    end
    if ~isempty(save_logprob)
        saved_logprob      = save_logprob;
        saved_logprob(1)   = [];
    else
        saved_logprob = [];
    end
end
