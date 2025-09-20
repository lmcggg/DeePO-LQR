function [K_opt, J_opt, history] = deepo_core(X0, U0, X1, Q, R, options)
% DeePO核心算法实现

if nargin < 6, options = struct(); end

% 默认参数
eta = getfield_or_default(options, 'eta', 0.01);
max_iter = getfield_or_default(options, 'max_iter', 500);
tol = getfield_or_default(options, 'tol', 1e-5);

[n, t] = size(X0);
m = size(U0, 1);

% 构建数据矩阵
D0 = [U0; X0];
Phi = (D0 * D0') / t;
X1bar = (X1 * D0') / t;
U0bar = (U0 * D0') / t;
X0bar = (X0 * D0') / t;

% 初始化(增强数值稳定性)
if cond(Phi) > 1e12
    warning('数据矩阵条件数过大');
    V = [zeros(m, n); eye(n)];
else
    V = Phi \ [zeros(m, n); eye(n)];
end
constraint_error = norm(X0bar * V - eye(n), 'fro');
if constraint_error > 1e-6
    V = V + pinv(X0bar) * (eye(n) - X0bar * V);
end

% 投影矩阵
Proj = eye(m+n) - pinv(X0bar) * X0bar;

% 历史记录
history.J = zeros(max_iter, 1);
history.rho = zeros(max_iter, 1);

% 初始化变量
J = Inf;
iter = 0;

for iter = 1:max_iter
    K = U0bar * V;
    Acl = X1bar * V;
    rho = max(abs(eig(Acl)));
    
    if rho >= 1
        eta = eta * 0.5;
        if eta < 1e-8, break; end
        continue;
    end
    
    % Lyapunov方程(增加异常处理)
    try
        Sigma = dlyap(Acl, eye(n));
        Qc = Q + K' * R * K;
        P = dlyap(Acl', Qc);
        J = trace(P);
        
        % 检查结果有效性
        if ~isfinite(J) || J <= 0
            eta = eta * 0.5;
            continue;
        end
        
        % 梯度计算
        G = (U0bar' * R * U0bar) + (X1bar' * P * X1bar);
        grad = 2 * (G * V * Sigma);
        
        % 检查梯度有效性
        if any(isnan(grad(:))) || any(isinf(grad(:)))
            eta = eta * 0.5;
            continue;
        end
    catch
        eta = eta * 0.5;
        if eta < 1e-8, break; end
        continue;
    end
    
    % 投影更新
    step = Proj * grad;
    V_new = V - eta * step;
    V_new = V_new + pinv(X0bar) * (eye(n) - X0bar * V_new);
    
    % 稳定性检查(更严格)
    Acl_new = X1bar * V_new;
    rho_new = max(abs(eig(Acl_new)));
    if rho_new < 0.99 && ~any(isnan(Acl_new(:)))
        V = V_new;
    else
        eta = eta * 0.8;  % 减小步长
    end
    
    history.J(iter) = J;
    history.rho(iter) = rho;
    
    if norm(step, 'fro') < tol, break; end
end

K_opt = U0bar * V;

% 计算最终代价
K_final = U0bar * V;
Acl_final = X1bar * V;
rho_final = max(abs(eig(Acl_final)));

if rho_final < 1
    try
        Sigma_final = dlyap(Acl_final, eye(n));
        Qc_final = Q + K_final' * R * K_final;
        P_final = dlyap(Acl_final', Qc_final);
        J_opt = trace(P_final);
    catch
        J_opt = Inf;
    end
else
    J_opt = Inf;
end

history.J = history.J(1:iter);
history.rho = history.rho(1:iter);

end

function val = getfield_or_default(s, field, default)
if isfield(s, field)
    val = s.(field);
else
    val = default;
end
end
