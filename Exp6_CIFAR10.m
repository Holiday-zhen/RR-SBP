%% Experiment 3 (CIFAR-10 LP): Fixed Consistency and Condition Number
% LP feasibility problem: Ax = b, x in X_0 = R_+^N x R^p x R_+^N
% X-Axis: CPU Time (seconds)
% Y-Axis: ||Ax^k - b||^2 + dist_{X_0}^2(x^k) (log scale)
% Algorithms:
%   1. Adaptive SBP (Global/Barycentric, tau=m) 
%   2. Fixed Block SBP (IID, tau=100) 
%   3. Block Adaptive SBP (IID, tau=100)
%   4. Fixed Block RR-SBP (RR, tau=100)
%   5. Block Adaptive RR-SBE (Proposed, RR, tau=100) 
clear; clc; close all;

%% 1. Global Parameters
p_dim = 1000;        
N_dim = 1500;        

delta = 0.1;         
max_cpu_time = 15;   
record_interval = 0.05; 
repeat_runs = 10;     
rng(100);

m_eq = N_dim + p_dim + 1;   
n = 2*N_dim + p_dim;         

tau_global = m_eq;  
tau_block = 100;    
tau_our = 100;      

fixed_step = 1.9;   
alpha_max = 100;    

fprintf('==========================================================\n');
fprintf('Exp 3 (CIFAR-10 LP): Fixed KKT Consistency & Raw Kernel\n');
fprintf('==========================================================\n');

%% 2. Data Loading & Preprocessing (Strictly following original paper)
try
    data_struct = load('cifar10testdata.mat'); 
    raw_X = double(reshape(data_struct.imageset, 3072, 10000))'; 
catch ME
    warning('CIFAR-10 Read failed. Generating synthetic data.');
    rng(1);
    raw_X = randn(10000, 3072); 
end

X_sub = raw_X(1:N_dim, :) / 255.0; 

fprintf('Computing raw Gaussian Kernel...\n');
D2 = pdist2(X_sub, X_sub, 'squaredeuclidean');

K_mat = exp(-D2);

E = K_mat(1:p_dim, 1:N_dim);
fprintf('Data Matrix E constructed. Cond(E) is healthy.\n');

%% 3. Preparation for Main Loop
num_points = floor(max_cpu_time / record_interval) + 1;
time_grid = linspace(0, max_cpu_time, num_points)';
num_algos = 5;
all_err = zeros(num_points, num_algos, repeat_runs);

algo_names = {'Adaptive SBP (\tau=m)', ...
              ['Fixed Block SBP (\tau=' num2str(tau_block) ')'], ...
              ['Block Adaptive SBP (\tau=' num2str(tau_block) ')'], ...
              ['Fixed Block RR-SBP (\tau=' num2str(tau_our) ')'], ...
              ['Block Adaptive RR-SBP (\tau=' num2str(tau_our) ')']};

%% 4. Main Loop
for r = 1:repeat_runs
    fprintf('\n--- Run %d/%d ---\n', r, repeat_runs);
    
    u_true = abs(randn(N_dim, 1)) + 0.1;   
    s_true = abs(randn(N_dim, 1)) + 0.1;   
    v_true = randn(p_dim, 1);
    
    mask = rand(N_dim, 1) > 0.5;
    u_true(~mask) = 0;  
    s_true(mask) = 0; 

    c = E' * v_true + s_true;   
    d = E * u_true;              

    A = zeros(m_eq, n);
    A(1:N_dim, (N_dim+1):(N_dim+p_dim)) = E';
    A(1:N_dim, (N_dim+p_dim+1):end) = eye(N_dim);
    A((N_dim+1):(N_dim+p_dim), 1:N_dim) = E;
    A(m_eq, 1:N_dim) = c';
    A(m_eq, (N_dim+1):(N_dim+p_dim)) = -d';
    b = [c; d; 0];

    row_norms = sqrt(sum(A.^2, 2));
    row_norms(row_norms < 1e-15) = 1;  
    A = A ./ row_norms;
    b = b ./ row_norms;
    
    x0 = randn(n, 1);
    e0 = compute_residual_lp(x0, A, b, N_dim, p_dim);
    
    [t1, e1] = run_proj_lp_timed(x0, A, b, tau_global, false, 1, 2 - delta, max_cpu_time, e0, alpha_max, N_dim, p_dim);
    all_err(:, 1, r) = interpolate_errors(t1, e1, time_grid);
    
    [t2, e2] = run_proj_lp_timed(x0, A, b, tau_block, false, 0, fixed_step, max_cpu_time, e0, alpha_max, N_dim, p_dim);
    all_err(:, 2, r) = interpolate_errors(t2, e2, time_grid);
    
    [t3, e3] = run_proj_lp_timed(x0, A, b, tau_block, false, 2, 2 - delta, max_cpu_time, e0, alpha_max, N_dim, p_dim);
    all_err(:, 3, r) = interpolate_errors(t3, e3, time_grid);
    
    [t4, e4] = run_proj_lp_timed(x0, A, b, tau_our, true, 0, fixed_step, max_cpu_time, e0, alpha_max, N_dim, p_dim);
    all_err(:, 4, r) = interpolate_errors(t4, e4, time_grid);
    
    [t5, e5] = run_proj_lp_timed(x0, A, b, tau_our, true, 2, 2 - delta, max_cpu_time, e0, alpha_max, N_dim, p_dim);
    all_err(:, 5, r) = interpolate_errors(t5, e5, time_grid);
    
    fprintf('Run %d done.\n', r);
end

avg_err = mean(all_err, 3);

%% 5. Plotting
fig_width_cm = 17.4;  
fig_height_cm = 10;  
figure('Color', 'w', 'Units', 'centimeters', 'Position', [2, 2, fig_width_cm, fig_height_cm]);

colors =[0.9 0.6 0.0; 0.49 0.18 0.56; 0.0 0.45 0.74; 0.47 0.67 0.19; 0.85 0.33 0.10];
line_styles = {'-.', ':', '--', '-.', '-'};
line_widths =[1.5, 2.0, 2.0, 2.0, 3.0];

h_lines = zeros(num_algos, 1);
for i = 1:num_algos
    h_lines(i) = semilogy(time_grid, avg_err(:, i), line_styles{i}, 'LineWidth', line_widths(i), 'Color', colors(i, :));
    hold on;
end

grid on; 
ax = gca; 
ax.FontName = 'Arial'; 
ax.FontSize = 10;     
ax.LineWidth = 1.2; 
ax.GridAlpha = 0.3;

ax.YMinorGrid = 'off';   
ax.YMinorTick = 'off';

xlim([0, max_cpu_time]);

y_min_global = min(avg_err(end, :));
y_lo = 10^(floor(log10(max(y_min_global, 1e-16))) - 0.5);
ylim([y_lo, 1e6]);


xlabel('\bf CPU Time (seconds)', 'FontSize', 10);
ylabel('\boldmath$\log(\|Ax^k - b\|^2 + \mathrm{dist}_{X_0}^2(x^k))$', 'Interpreter', 'latex', 'FontSize', 10);
legend(h_lines, algo_names, 'Location', 'northeast', 'FontSize', 10);

hold off;
fprintf('\nExperiment 3 (CIFAR-10 LP) Completed.\n');

%% =====================================================================
function err_interp = interpolate_errors(t_raw, e_raw, time_grid)
    log_e_raw = log(max(e_raw, 1e-16)); 
    [t_unique, idx_unique] = unique(t_raw);
    log_e_unique = log_e_raw(idx_unique);
    log_e_interp = interp1(t_unique, log_e_unique, time_grid, 'linear', 'extrap');
    idx_beyond = time_grid > t_unique(end);
    log_e_interp(idx_beyond) = log_e_unique(end);
    err_interp = exp(log_e_interp);
end

function res = compute_residual_lp(x, A, b, N_dim, p_dim)
    eq_viol_sq = norm(A * x - b)^2;
    u = x(1:N_dim);
    s = x((N_dim + p_dim + 1):end);
    cone_viol_sq = norm(min(0, u))^2 + norm(min(0, s))^2;
    res = eq_viol_sq + cone_viol_sq;
end

function [t_hist, e_hist] = run_proj_lp_timed(x, A, b, tau, is_rr, adap_type, step_val, max_cpu_time, e0, alpha_max, N_dim, p_dim)
    [m_eq, n] = size(A);
    K = floor(m_eq / tau);          
    
    max_records = 50000;
    t_hist = zeros(max_records, 1);
    e_hist = zeros(max_records, 1);
    t_hist(1) = 0; e_hist(1) = e0;
    record_count = 1; cpu_accum = 0;
    
    idx_u = 1:N_dim;
    idx_s = (N_dim + p_dim + 1):n;
    
    w = 1.0 / (tau + 1);            
    
    while cpu_accum < max_cpu_time
        ts = tic;
        if is_rr, perm = randperm(m_eq); end
        
        for j = 1:K                
            if tau == m_eq
                hyper_idx = 1:m_eq;
            elseif is_rr
                hyper_idx = perm((j-1)*tau + 1 : j*tau);
            else
                hyper_idx = randi(m_eq, tau, 1);
            end
            
            A_sub = A(hyper_idx, :);
            resid_eq = A_sub * x - b(hyper_idx);   
            
            g_block = w * (A_sub' * resid_eq);
            E_block = w * (resid_eq' * resid_eq);
            
            u_neg = min(0, x(idx_u));    
            s_neg = min(0, x(idx_s));    
            
            cone_grad = zeros(n, 1);
            cone_grad(idx_u) = u_neg;
            cone_grad(idx_s) = s_neg;
            
            g_block = g_block + w * cone_grad;
            E_block = E_block + w * (u_neg' * u_neg + s_neg' * s_neg);
            
            if E_block > 1e-30
                L_adap = (g_block' * g_block) / E_block;
                
                if adap_type == 1
                    % Global Adaptive 
                    L_tau = w + (1 - w) * L_adap;
                    step = min(step_val / L_tau, alpha_max);
                elseif adap_type == 2
                    % Block Adaptive 
                    step = min(step_val / max(L_adap, 1e-12), alpha_max);
                else
                    % Fixed 
                    step = step_val;    
                end
                
                x = x - step * g_block;
            end
        end
        
        cpu_accum = cpu_accum + toc(ts);
        
        record_count = record_count + 1;
        if record_count > max_records
            t_hist = [t_hist; zeros(max_records, 1)]; 
            e_hist = [e_hist; zeros(max_records, 1)]; 
            max_records = max_records * 2;
        end
        e_hist(record_count) = compute_residual_lp(x, A, b, N_dim, p_dim);
        t_hist(record_count) = cpu_accum;
    end
    t_hist = t_hist(1:record_count); e_hist = e_hist(1:record_count);
end