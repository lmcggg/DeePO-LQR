% 实验A: 离线收敛示例
clear; close all; clc;

% 系统参数
n = 4; m = 2;
A = [-0.13, 0.14, -0.29, 0.28;
     0.48,  0.09,  0.41, 0.30;
     -0.01, 0.04,  0.17, 0.43;
     0.14,  0.31, -0.29, -0.10];
B = [1.63, 0.93; 0.26, 1.79; 1.46, 1.18; 0.77, 0.11];
Q = eye(n);
R = eye(m);

% 生成数据
t = 8;
rng(1);
U0 = randn(m, t);
X0 = randn(n, t);
X1 = A * X0 + B * U0;

% 真实LQR解
[K_true, S_true, ~] = dlqr(A, B, Q, R);
K_true = -K_true;
J_true = trace(S_true);

% DeePO优化
options.eta = 0.1;
options.max_iter = 200;
[K_opt, J_opt, history] = deepo_core(X0, U0, X1, Q, R, options);

% 结果
fprintf('实验A结果:\n');
fprintf('真实LQR代价: %.6f\n', J_true);
fprintf('DeePO代价: %.6f\n', J_opt);
fprintf('增益误差: %.6f\n', norm(K_opt - K_true, 'fro'));
fprintf('迭代次数: %d\n', length(history.J));

% 收敛曲线
figure;
subplot(1,2,1);
semilogy(history.J, 'b-', 'LineWidth', 2);
hold on;
plot([1, length(history.J)], [J_true, J_true], 'r--');
xlabel('迭代次数');
ylabel('代价');
title('收敛曲线');
legend('DeePO', '真实LQR');
grid on;

subplot(1,2,2);
plot(history.rho, 'g-', 'LineWidth', 2);
hold on;
plot([1, length(history.rho)], [1, 1], 'k--');
xlabel('迭代次数');
ylabel('谱半径');
title('稳定性');
grid on;

% 控制效果对比
T_sim = 50;
x0 = [1; -1; 0.5; -0.5];

% DeePO控制仿真
x_deepo = zeros(n, T_sim+1);
u_deepo = zeros(m, T_sim);
x_deepo(:, 1) = x0;
for t = 1:T_sim
    u_deepo(:, t) = K_opt * x_deepo(:, t);
    x_deepo(:, t+1) = A * x_deepo(:, t) + B * u_deepo(:, t);
end

% 真实LQR控制仿真
x_true = zeros(n, T_sim+1);
u_true = zeros(m, T_sim);
x_true(:, 1) = x0;
for t = 1:T_sim
    u_true(:, t) = K_true * x_true(:, t);
    x_true(:, t+1) = A * x_true(:, t) + B * u_true(:, t);
end

% 控制效果对比图
figure;
subplot(2,3,1);
plot(0:T_sim, x_deepo(1,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_true(1,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('x_1');
title('状态x_1');
legend('DeePO', '真实LQR');
grid on;

subplot(2,3,2);
plot(0:T_sim, x_deepo(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_true(2,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('x_2');
title('状态x_2');
legend('DeePO', '真实LQR');
grid on;

subplot(2,3,3);
plot(0:T_sim, x_deepo(3,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_true(3,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('x_3');
title('状态x_3');
legend('DeePO', '真实LQR');
grid on;

subplot(2,3,4);
plot(0:T_sim, x_deepo(4,:), 'b-', 'LineWidth', 2);
hold on;
plot(0:T_sim, x_true(4,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('x_4');
title('状态x_4');
legend('DeePO', '真实LQR');
grid on;

subplot(2,3,5);
plot(1:T_sim, u_deepo(1,:), 'b-', 'LineWidth', 2);
hold on;
plot(1:T_sim, u_true(1,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('u_1');
title('控制输入u_1');
legend('DeePO', '真实LQR');
grid on;

subplot(2,3,6);
plot(1:T_sim, u_deepo(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(1:T_sim, u_true(2,:), 'r--', 'LineWidth', 2);
xlabel('时间步');
ylabel('u_2');
title('控制输入u_2');
legend('DeePO', '真实LQR');
grid on;

sgtitle('实验A: 控制效果对比');

% 性能指标
cost_deepo = sum(diag(x_deepo(:,1:T_sim)' * Q * x_deepo(:,1:T_sim)) + diag(u_deepo' * R * u_deepo));
cost_true = sum(diag(x_true(:,1:T_sim)' * Q * x_true(:,1:T_sim)) + diag(u_true' * R * u_true));
fprintf('仿真总代价 - DeePO: %.4f, 真实LQR: %.4f\n', cost_deepo, cost_true);

% 保存结果
save('exp_A_results.mat', 'K_opt', 'J_opt', 'history', 'K_true', 'J_true');
