% 实验C: 边界不稳定系统对比
clear; close all; clc;

% Laplacian系统
n = 3; m = 3;
A = [1.01, 0.01, 0; 0.01, 1.01, 0.01; 0, 0.01, 1.01];
B = eye(3);
Q = eye(3); R = eye(3);
noise_std = sqrt(1/100);

% 真实LQR
[K_true, S_true, ~] = dlqr(A, B, Q, R);
K_true = -K_true;
J_true = trace(S_true);

T_total = 150;
T_init = 20;

% DeePO方法
rng(1);
x_deepo = randn(n, 1);

% 初始数据收集(增强激励)
X_init = zeros(n, T_init);
U_init = zeros(m, T_init);
X_next_init = zeros(n, T_init);

for t = 1:T_init
    u = 0.2 * randn(m, 1);  % 增强激励强度
    w = noise_std * randn(n, 1);
    x_next = A * x_deepo + B * u + w;
    
    X_init(:, t) = x_deepo;
    U_init(:, t) = u;
    X_next_init(:, t) = x_next;
    x_deepo = x_next;
end

% 初始化DeePO(更稳健的方法)
D0 = [U_init; X_init];
S_deepo = D0 * D0';
T_deepo = X_next_init * D0';
t_current = T_init;

Phi = S_deepo / t_current;
X1bar = T_deepo / t_current;
U0bar = Phi(1:m, :);
X0bar = Phi(m+1:end, :);

% 检查数据质量
if cond(Phi) > 1e12 || rank(D0) < m+n
    warning('数据质量差，调整初始化');
    % 使用保守初始化
    K_deepo = -0.05 * eye(m, n);
else
    % 使用协方差参数化初始化
    try
        V = Phi \ [zeros(m, n); eye(n)];
        V = V + pinv(X0bar) * (eye(n) - X0bar * V);
        K_deepo = U0bar * V;
        
        % 检查初始稳定性
        Acl_test = X1bar * V;
        if max(abs(eig(Acl_test))) >= 0.98
            K_deepo = -0.05 * eye(m, n);
        end
    catch
        K_deepo = -0.05 * eye(m, n);
    end
end

% CE-LQR初始化
rng(1);
x_ce = randn(n, 1);
X_data = X_init;
U_data = U_init;
X_next_data = X_next_init;

% 在线对比
J_deepo = zeros(T_total - T_init, 1);
J_ce = zeros(T_total - T_init, 1);
time_deepo = zeros(T_total - T_init, 1);
time_ce = zeros(T_total - T_init, 1);

for k = 1:(T_total - T_init)
    % DeePO更新
    tic;
    v = 0.1 * randn(m, 1);
    u_deepo = K_deepo * x_deepo + v;
    w = noise_std * randn(n, 1);
    x_next_deepo = A * x_deepo + B * u_deepo + w;
    
    d = [u_deepo; x_deepo];
    S_deepo = S_deepo + d * d';
    T_deepo = T_deepo + x_next_deepo * d';
    t_current = t_current + 1;
    
    Phi = S_deepo / t_current;
    X1bar = T_deepo / t_current;
    U0bar = Phi(1:m, :);
    X0bar = Phi(m+1:end, :);
    
    % 稳健的参数更新
    try
        V = Phi \ [K_deepo; eye(n)];
        constraint_error = norm(X0bar * V - eye(n), 'fro');
        if constraint_error > 1e-3
            V = V + pinv(X0bar) * (eye(n) - X0bar * V);
        end
        
        K_current = U0bar * V;
        Acl = X1bar * V;
        rho = max(abs(eig(Acl)));
        
        if rho < 0.98 && ~any(isnan(Acl(:))) && ~any(isinf(Acl(:)))
            Sigma = dlyap(Acl, eye(n));
            Qc = Q + K_current' * R * K_current;
            P = dlyap(Acl', Qc);
            J_deepo_current = trace(P);
            
            % 检查结果有效性
            if isfinite(J_deepo_current) && J_deepo_current > 0
                G = (U0bar' * R * U0bar) + (X1bar' * P * X1bar);
                grad = 2 * (G * V * Sigma);
                
                % 检查梯度有效性
                if ~any(isnan(grad(:))) && ~any(isinf(grad(:)))
                    Proj = eye(m+n) - pinv(X0bar) * X0bar;
                    step = Proj * grad;
                    
                    V_new = V - 0.6 * step;  
                    V_new = V_new + pinv(X0bar) * (eye(n) - X0bar * V_new);
                    
                    K_new = U0bar * V_new;
                    Acl_new = X1bar * V_new;
                    rho_new = max(abs(eig(Acl_new)));
                    
                    if rho_new < 0.98 && ~any(isnan(K_new(:)))
                        K_deepo = K_new;
                    end
                end
            else
                J_deepo_current = Inf;
            end
        else
            J_deepo_current = Inf;
        end
    catch
        J_deepo_current = Inf;
    end
    
    time_deepo(k) = toc;
    J_deepo(k) = J_deepo_current;
    x_deepo = x_next_deepo;
    
    % CE-LQR更新
    tic;
    if size(X_data, 2) >= n + m
        Phi_reg = [U_data; X_data]';
        Y_reg = X_next_data';
        
        % 正则化回归
        lambda_reg = 1e-3;
        Theta = (Phi_reg' * Phi_reg + lambda_reg * eye(m+n)) \ (Phi_reg' * Y_reg);
        Theta = Theta';
        
        B_est = Theta(:, 1:m);
        A_est = Theta(:, m+1:end);
        
        try
            Co_est = ctrb(A_est, B_est);
            if rank(Co_est) >= n && max(abs(eig(A_est))) < 1.5
                [K_ce_new, ~, ~] = dlqr(A_est, B_est, Q, R);
                K_ce = -K_ce_new;
                
                Acl_ce = A_est + B_est * K_ce;
                rho_ce = max(abs(eig(Acl_ce)));
                
                if rho_ce < 0.98
                    Qc_ce = Q + K_ce' * R * K_ce;
                    P_ce = dlyap(Acl_ce', Qc_ce);
                    J_ce_current = trace(P_ce);
                    
                    if ~isfinite(J_ce_current) || J_ce_current <= 0
                        K_ce = -0.1 * eye(m, n);
                        J_ce_current = Inf;
                    end
                else
                    K_ce = -0.1 * eye(m, n);
                    J_ce_current = Inf;
                end
            else
                K_ce = -0.1 * eye(m, n);
                J_ce_current = Inf;
            end
        catch
            K_ce = -0.1 * eye(m, n);
            J_ce_current = Inf;
        end
    else
        K_ce = -0.1 * eye(m, n);
        J_ce_current = Inf;
    end
    
    time_ce(k) = toc;
    J_ce(k) = J_ce_current;
    
    v = 0.1 * randn(m, 1);
    u_ce = K_ce * x_ce + v;
    w = noise_std * randn(n, 1);
    x_next_ce = A * x_ce + B * u_ce + w;
    
    % 更新CE数据(滑动窗口)
    window_size = 60;
    if size(X_data, 2) >= window_size
        X_data = [X_data(:, 2:end), x_ce];
        U_data = [U_data(:, 2:end), u_ce];
        X_next_data = [X_next_data(:, 2:end), x_next_ce];
    else
        X_data = [X_data, x_ce];
        U_data = [U_data, u_ce];
        X_next_data = [X_next_data, x_next_ce];
    end
    
    x_ce = x_next_ce;
end

% 结果分析
valid_deepo = isfinite(J_deepo) & J_deepo < 1e6 & J_deepo > 0;
valid_ce = isfinite(J_ce) & J_ce < 1e6 & J_ce > 0;

fprintf('实验C结果:\n');
if sum(valid_deepo) > 0
    fprintf('DeePO平均代价: %.4f\n', mean(J_deepo(valid_deepo)));
    fprintf('DeePO稳定比例: %.1f%%\n', sum(valid_deepo)/length(J_deepo)*100);
else
    fprintf('DeePO平均代价: 无有效数据\n');
end

if sum(valid_ce) > 0
    fprintf('CE-LQR平均代价: %.4f\n', mean(J_ce(valid_ce)));
    fprintf('CE-LQR稳定比例: %.1f%%\n', sum(valid_ce)/length(J_ce)*100);
else
    fprintf('CE-LQR平均代价: 无有效数据\n');
end

fprintf('DeePO平均时间: %.2f ms\n', mean(time_deepo) * 1000);
fprintf('CE-LQR平均时间: %.2f ms\n', mean(time_ce) * 1000);
fprintf('真实LQR代价: %.4f\n', J_true);

% 性能对比
figure;
subplot(1,2,1);
% 只绘制有效数据
if sum(valid_deepo) > 0
    valid_idx_deepo = find(valid_deepo);
    plot(valid_idx_deepo, J_deepo(valid_idx_deepo), 'b-', 'LineWidth', 2);
    hold on;
end
if sum(valid_ce) > 0
    valid_idx_ce = find(valid_ce);
    plot(valid_idx_ce, J_ce(valid_idx_ce), 'r-', 'LineWidth', 2);
    hold on;
end
plot([1, length(J_deepo)], [J_true, J_true], 'k--');
xlabel('时间步');
ylabel('代价');
title('性能对比');
legend('DeePO', 'CE-LQR', 'LQR*');
grid on;

subplot(1,2,2);
plot(time_deepo * 1000, 'b-', 'LineWidth', 2);
hold on;
plot(time_ce * 1000, 'r-', 'LineWidth', 2);
xlabel('时间步');
ylabel('计算时间 (ms)');
title('计算效率对比');
legend('DeePO', 'CE-LQR');
grid on;

% 详细的控制效果对比仿真
T_sim = 80;
x0 = [0.5; -0.3; 0.8]; % 边界不稳定系统的初始状态

% DeePO控制仿真
x_deepo_sim = zeros(n, T_sim+1);
u_deepo_sim = zeros(m, T_sim);
x_deepo_sim(:, 1) = x0;

% 直接使用在线学习循环结束时得到的最终控制器
K_deepo_final = K_deepo;

% DeePO控制仿真
for t = 1:T_sim
    u_deepo_sim(:, t) = K_deepo_final * x_deepo_sim(:, t);
    w = noise_std * randn(n, 1);
    x_deepo_sim(:, t+1) = A * x_deepo_sim(:, t) + B * u_deepo_sim(:, t) + w;
end

% CE-LQR控制仿真
x_ce_sim = zeros(n, T_sim+1);
u_ce_sim = zeros(m, T_sim);
x_ce_sim(:, 1) = x0;

% 直接使用在线学习循环结束时得到的最终控制器
K_ce_final = K_ce;

for t = 1:T_sim
    u_ce_sim(:, t) = K_ce_final * x_ce_sim(:, t);
    w = noise_std * randn(n, 1);
    x_ce_sim(:, t+1) = A * x_ce_sim(:, t) + B * u_ce_sim(:, t) + w;
end

% 真实LQR控制仿真
x_true_sim = zeros(n, T_sim+1);
u_true_sim = zeros(m, T_sim);
x_true_sim(:, 1) = x0;

for t = 1:T_sim
    u_true_sim(:, t) = K_true * x_true_sim(:, t);
    w = noise_std * randn(n, 1);
    x_true_sim(:, t+1) = A * x_true_sim(:, t) + B * u_true_sim(:, t) + w;
end

% 控制效果对比图
figure;
subplot(2,3,1);
plot(0:T_sim, x_deepo_sim(1,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_ce_sim(1,:), 'r--', 'LineWidth', 2);
plot(0:T_sim, x_true_sim(1,:), 'k:', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('x_1');
title('状态x_1');
legend('DeePO', 'CE-LQR', '真实LQR');
grid on;

subplot(2,3,2);
plot(0:T_sim, x_deepo_sim(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_ce_sim(2,:), 'r--', 'LineWidth', 2);
plot(0:T_sim, x_true_sim(2,:), 'k:', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('x_2');
title('状态x_2');
legend('DeePO', 'CE-LQR', '真实LQR');
grid on;

subplot(2,3,3);
plot(0:T_sim, x_deepo_sim(3,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_ce_sim(3,:), 'r--', 'LineWidth', 2);
plot(0:T_sim, x_true_sim(3,:), 'k:', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('x_3');
title('状态x_3');
legend('DeePO', 'CE-LQR', '真实LQR');
grid on;

subplot(2,3,4);
plot(1:T_sim, u_deepo_sim(1,:), 'b-', 'LineWidth', 2);
hold on;
plot(1:T_sim, u_ce_sim(1,:), 'r--', 'LineWidth', 2);
plot(1:T_sim, u_true_sim(1,:), 'k:', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('u_1');
title('控制输入u_1');
legend('DeePO', 'CE-LQR', '真实LQR');
grid on;

subplot(2,3,5);
plot(1:T_sim, u_deepo_sim(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(1:T_sim, u_ce_sim(2,:), 'r--', 'LineWidth', 2);
plot(1:T_sim, u_true_sim(2,:), 'k:', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('u_2');
title('控制输入u_2');
legend('DeePO', 'CE-LQR', '真实LQR');
grid on;

subplot(2,3,6);
plot(1:T_sim, u_deepo_sim(3,:), 'b-', 'LineWidth', 2);
hold on;
plot(1:T_sim, u_ce_sim(3,:), 'r--', 'LineWidth', 2);
plot(1:T_sim, u_true_sim(3,:), 'k:', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('u_3');
title('控制输入u_3');
legend('DeePO', 'CE-LQR', '真实LQR');
grid on;

sgtitle('实验C: 边界不稳定系统控制效果对比');

% 计算仿真性能指标
cost_deepo_sim = sum(diag(x_deepo_sim(:,1:T_sim)' * Q * x_deepo_sim(:,1:T_sim)) + diag(u_deepo_sim' * R * u_deepo_sim));
cost_ce_sim = sum(diag(x_ce_sim(:,1:T_sim)' * Q * x_ce_sim(:,1:T_sim)) + diag(u_ce_sim' * R * u_ce_sim));
cost_true_sim = sum(diag(x_true_sim(:,1:T_sim)' * Q * x_true_sim(:,1:T_sim)) + diag(u_true_sim' * R * u_true_sim));

fprintf('\n仿真总代价对比:\n');
fprintf('DeePO: %.4f\n', cost_deepo_sim);
fprintf('CE-LQR: %.4f\n', cost_ce_sim);
fprintf('真实LQR: %.4f\n', cost_true_sim);

% 保存结果
save('exp_C_results.mat', 'J_deepo', 'J_ce', 'time_deepo', 'time_ce', 'J_true', 'valid_deepo', 'valid_ce');