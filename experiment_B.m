% 实验B: 在线自适应示例
clear; close all; clc;

% 系统参数(与实验A相同)
n = 4; m = 2;
A = [-0.13, 0.14, -0.29, 0.28;
     0.48,  0.09,  0.41, 0.30;
     -0.01, 0.04,  0.17, 0.43;
     0.14,  0.31, -0.29, -0.10];
B = [1.63, 0.93; 0.26, 1.79; 1.46, 1.18; 0.77, 0.11];
Q = eye(n); R = eye(m);

% 在线参数
t0 = 8;
T_online = 200;
eta = 0.01;
sigma_levels = [0.1, 0.01, 0.001];

[K_true, S_true, ~] = dlqr(A, B, Q, R);
K_true = -K_true;
J_true = trace(S_true);

results = cell(length(sigma_levels), 1);

for noise_idx = 1:length(sigma_levels)
    sigma = sigma_levels(noise_idx);
    rng(noise_idx);
    
    % 初始数据收集
    x = randn(n, 1);
    X_init = zeros(n, t0);
    U_init = zeros(m, t0);
    X_next_init = zeros(n, t0);
    
    for t = 1:t0
        u = randn(m, 1);
        w = sigma * rand(n, 1);
        x_next = A * x + B * u + w;
        
        X_init(:, t) = x;
        U_init(:, t) = u;
        X_next_init(:, t) = x_next;
        x = x_next;
    end
    
    % 初始化
    D0 = [U_init; X_init];
    S = D0 * D0';
    T = X_next_init * D0';
    t_current = t0;
    
    % 初始控制器
    Phi = S / t_current;
    V = Phi \ [zeros(m, n); eye(n)];
    X0bar = Phi(m+1:end, :);
    V = V + pinv(X0bar) * (eye(n) - X0bar * V);
    U0bar = Phi(1:m, :);
    K = U0bar * V;
    
    % 在线更新
    J_history = zeros(T_online, 1);
    regret = 0;
    
    for k = 1:T_online
        % 控制输入
        v = randn(m, 1);
        u = K * x + v;
        w = sigma * rand(n, 1);
        x_next = A * x + B * u + w;
        
        % 递推更新
        d = [u; x];
        S = S + d * d';
        T = T + x_next * d';
        t_current = t_current + 1;
        
        % 更新矩阵
        Phi = S / t_current;
        X1bar = T / t_current;
        U0bar = Phi(1:m, :);
        X0bar = Phi(m+1:end, :);
        
        % 参数更新
        V = Phi \ [K; eye(n)];
        V = V + pinv(X0bar) * (eye(n) - X0bar * V);
        
        % 一步梯度更新
        K_current = U0bar * V;
        Acl = X1bar * V;
        
        if max(abs(eig(Acl))) < 1
            try
                Sigma = dlyap(Acl, eye(n));
                Qc = Q + K_current' * R * K_current;
                P = dlyap(Acl', Qc);
                J = trace(P);
                
                G = (U0bar' * R * U0bar) + (X1bar' * P * X1bar);
                grad = 2 * (G * V * Sigma);
                
                Proj = eye(m+n) - pinv(X0bar) * X0bar;
                step = Proj * grad;
                V_new = V - eta * step;
                V_new = V_new + pinv(X0bar) * (eye(n) - X0bar * V_new);
                
                K_new = U0bar * V_new;
                if max(abs(eig(X1bar * V_new))) < 1
                    K = K_new;
                end
            catch
                J = Inf;
            end
        else
            J = Inf;
        end
        
        J_history(k) = J;
        regret = regret + max(0, J - J_true);
        x = x_next;
    end
    
    results{noise_idx} = struct('sigma', sigma, 'J_history', J_history, 'regret', regret);
end

% 在线性能对比
figure;
colors = {'b', 'r', 'g'};
for i = 1:length(sigma_levels)
    valid = isfinite(results{i}.J_history) & results{i}.J_history < 1e6;
    plot(find(valid), results{i}.J_history(valid), colors{i}, 'LineWidth', 2);
    hold on;
end
plot([1, T_online], [J_true, J_true], 'k--', 'LineWidth', 1);
xlabel('时间步');
ylabel('代价');
title('在线自适应性能');
legend('σ=0.1', 'σ=0.01', 'σ=0.001', 'LQR*');
grid on;

% 选择最好的噪声水平进行详细仿真对比
best_idx = 2; % σ=0.01
sigma_best = sigma_levels(best_idx);

% 重新仿真以获取状态轨迹
rng(best_idx);
x = randn(n, 1);
T_sim = 100;

% 收集初始数据
X_init = zeros(n, t0);
U_init = zeros(m, t0);
X_next_init = zeros(n, t0);

for t = 1:t0
    u = randn(m, 1);
    w = sigma_best * rand(n, 1);
    x_next = A * x + B * u + w;
    
    X_init(:, t) = x;
    U_init(:, t) = u;
    X_next_init(:, t) = x_next;
    x = x_next;
end

% 初始化控制器
D0 = [U_init; X_init];
S = D0 * D0';
T = X_next_init * D0';
t_current = t0;

Phi = S / t_current;
V = Phi \ [zeros(m, n); eye(n)];
X0bar = Phi(m+1:end, :);
V = V + pinv(X0bar) * (eye(n) - X0bar * V);
U0bar = Phi(1:m, :);
K = U0bar * V;

% 记录轨迹
x_history = zeros(n, T_sim+1);
u_history = zeros(m, T_sim);
x_history(:, 1) = x;

% 对比: 固定LQR控制
x_lqr = x;
x_lqr_history = zeros(n, T_sim+1);
u_lqr_history = zeros(m, T_sim);
x_lqr_history(:, 1) = x_lqr;

for k = 1:T_sim
    % DeePO在线控制
    v = randn(m, 1);
    u = K * x + v;
    w = sigma_best * rand(n, 1);
    x_next = A * x + B * u + w;
    
    u_history(:, k) = u;
    x_history(:, k+1) = x_next;
    
    % 更新控制器(简化版)
    if k <= T_sim - t0
        d = [u; x];
        S = S + d * d';
        T = T + x_next * d';
        t_current = t_current + 1;
        
        Phi = S / t_current;
        X1bar = T / t_current;
        U0bar = Phi(1:m, :);
        X0bar = Phi(m+1:end, :);
        
        V = Phi \ [K; eye(n)];
        V = V + pinv(X0bar) * (eye(n) - X0bar * V);
        K = U0bar * V;
    end
    
    % 固定LQR控制
    u_lqr = K_true * x_lqr + v;
    x_lqr_next = A * x_lqr + B * u_lqr + w;
    
    u_lqr_history(:, k) = u_lqr;
    x_lqr_history(:, k+1) = x_lqr_next;
    
    x = x_next;
    x_lqr = x_lqr_next;
end

% 状态输入对比图
figure;
subplot(2,3,1);
plot(0:T_sim, x_history(1,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_lqr_history(1,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('x_1');
title('状态x_1 (σ=0.01)');
legend('DeePO在线', '固定LQR');
grid on;

subplot(2,3,2);
plot(0:T_sim, x_history(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_lqr_history(2,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('x_2');
title('状态x_2');
legend('DeePO在线', '固定LQR');
grid on;

subplot(2,3,3);
plot(0:T_sim, x_history(3,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_lqr_history(3,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('x_3');
title('状态x_3');
legend('DeePO在线', '固定LQR');
grid on;

subplot(2,3,4);
plot(0:T_sim, x_history(4,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_lqr_history(4,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('x_4');
title('状态x_4');
legend('DeePO在线', '固定LQR');
grid on;

subplot(2,3,5);
plot(1:T_sim, u_history(1,:), 'b-', 'LineWidth', 2);
hold on;
plot(1:T_sim, u_lqr_history(1,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('u_1');
title('控制输入u_1');
legend('DeePO在线', '固定LQR');
grid on;

subplot(2,3,6);
plot(1:T_sim, u_history(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(1:T_sim, u_lqr_history(2,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('u_2');
title('控制输入u_2');
legend('DeePO在线', '固定LQR');
grid on;

sgtitle('实验B: 在线自适应控制效果对比');

% 保存结果
save('exp_B_results.mat', 'results', 'sigma_levels', 'T_online');
