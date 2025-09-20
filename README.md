# MATLAB Implementation of DeePO for LQR Control

This repository contains a MATLAB implementation of the paper:

**Data-Enabled Policy Optimization for Direct Adaptive Learning of the LQR**
*Authors: Feiran Zhao, Florian Dörfler, Alessandro Chiuso, Keyou You*
*Publication: IEEE Transactions on Automatic Control*
*DOI: [10.1109/TAC.2025.3569597](https://doi.org/10.1109/TAC.2025.3569597)*

The code reproduces the core algorithm (DeePO - Data-enabled Policy Optimization) and the simulation experiments presented in the paper, including offline convergence, online adaptive learning, and comparison with the Certainty-Equivalence LQR (CE-LQR) method.

## Core Methodology

The DeePO algorithm provides a direct, data-driven method for solving the Linear Quadratic Regulator (LQR) problem without explicit system identification. The key idea is a novel policy parameterization based on sample covariances.

### 1. The LQR Problem

The standard discrete-time LQR problem aims to find a state-feedback gain `K` that minimizes the H₂-norm of the closed-loop system, which is equivalent to the cost function:
$$
C(K) = \text{Tr}((Q + K^T R K)\Sigma_K)
$$
where `Σₖ` is the closed-loop state covariance matrix satisfying the Lyapunov equation:
$$
\Sigma_K = I_n + (A + BK)\Sigma_K(A + BK)^T
$$

### 2. Covariance Parameterization

Instead of identifying `A` and `B`, DeePO uses a batch of input-state data `D_0 = [U_0^T, X_0^T]^T`. The policy is re-parameterized using a new variable `V` and the sample covariance matrix `Φ`.

The sample covariance matrix `Φ` is defined as:
$$
\Phi := \frac{1}{t} D_0 D_0^T = \frac{1}{t} \begin{bmatrix} U_0 \\ X_0 \end{bmatrix} \begin{bmatrix} U_0^T & X_0^T \end{bmatrix}
$$

The policy gain `K` is linked to the new decision variable `V ∈ ℝ^{(n+m)×n}` through the linear constraint:
$$
\begin{bmatrix} K \\ I_n \end{bmatrix} = \Phi V
$$
This allows the control gain to be recovered as `K = U_0 D_0^T/t \cdot V`.

### 3. The Data-Driven LQR Problem (Offline)

With this new parameterization, the original LQR problem is reformulated into a direct data-driven optimization problem in terms of `V`:
$$
\min_{V} J(V) := \text{Tr}((Q + V^T \bar{U}_0^T R \bar{U}_0 V)\Sigma_V)
$$
subject to:
$$
\Sigma_V = I_n + (\bar{X}_1 V) \Sigma_V (\bar{X}_1 V)^T, \quad \bar{X}_0 V = I_n
$$
where `bar{X}₁`, `bar{U}₀`, `bar{X}₀` are data-based covariance matrices (e.g., `bar{X}₁ = X₁D₀ᵀ/t`).

### 4. Projected Gradient Descent

The optimization problem is solved using projected gradient descent. The gradient of `J(V)` has a closed-form solution:
$$
\nabla J(V) = 2 (\bar{U}_0^T R \bar{U}_0 + \bar{X}_1^T P_V \bar{X}_1) V \Sigma_V
$$
where `Pᵥ` and `Σᵥ` are solutions to their respective Lyapunov equations.

The update rule for `V` is then:
$$
V_{k+1} = V_k - \eta \cdot \Pi_{X_0} (\nabla J(V_k))
$$
where `η` is the learning rate and `Π_{X₀}` is the projection operator that enforces the constraint `bar{X}₀V = Iₙ`.

### 5. Online Adaptive Algorithm

For adaptive learning, the algorithm performs one step of projected gradient descent at each time step `t`, using all data collected up to that point. The key steps are (as in Algorithm 1 of the paper):

1.  Given the current policy `Kₜ`, solve for its parameter representation `Vₜ₊₁` using the newly updated covariance matrix `Φₜ₊₁`:
    $$
    V_{t+1} = \Phi_{t+1}^{-1} \begin{bmatrix} K_t \\ I_n \end{bmatrix}
    $$
2.  Perform a single projected gradient descent step on `Vₜ₊₁` to get an improved `V'_{t+1}`:
    $$
    V'_{t+1} = V_{t+1} - \eta \cdot \Pi_{X_{0,t+1}} (\nabla J_{t+1}(V_{t+1}))
    $$
3.  Update the control gain for the next time step:
    $$
    K_{t+1} = \bar{U}_{0,t+1} V'_{t+1}
    $$

## Code Structure and Usage

The repository is organized into a core algorithm file and three experiment scripts.

### File Structure

-   `deepo_core.m`: This function implements the core **offline** DeePO algorithm using projected gradient descent. It takes data matrices (`X0`, `U0`, `X1`), LQR weights (`Q`, `R`), and optimization options as input, and returns the optimized gain `K_opt`.
-   `experiment_A.m`: This script reproduces the simulation from Section VI-A of the paper. It demonstrates the **offline convergence** of the DeePO algorithm on a randomly generated stable system.
-   `experiment_B.m`: This script reproduces the simulation from Section VI-B. It demonstrates the **online adaptive learning** performance of DeePO under different noise levels.
-   `experiment_C.m`: This script reproduces the simulation from Section VI-C. It compares the online performance of DeePO against a standard **Certainty-Equivalence LQR (CE-LQR)** controller on a marginally unstable system.

### How to Run

1.  Ensure you have MATLAB installed with the Control System Toolbox (for `dlqr` and `dlyap` functions).
2.  Open any of the experiment scripts (`experiment_A.m`, `experiment_B.m`, or `experiment_C.m`).
3.  Run the script by pressing `F5` or clicking the "Run" button.
4.  The script will generate plots comparing the performance of the controllers and print quantitative results to the MATLAB command window.

## Experimental Results


### Experiment A: Offline Convergence

<img width="1671" height="905" alt="image" src="https://github.com/user-attachments/assets/310e0da4-19d4-4583-b61d-2212619a414c" />

<img width="539" height="408" alt="image" src="https://github.com/user-attachments/assets/933b6b59-8e10-4851-86b6-cce0ca281ef8" />



### Experiment B: Online Adaptive Learning

<img width="1440" height="761" alt="image" src="https://github.com/user-attachments/assets/9572d4f6-5d28-456c-a967-af02ff8c12c6" />

<img width="523" height="396" alt="image" src="https://github.com/user-attachments/assets/3b56af9a-e7f4-440e-829c-09b0edfd5303" />


### Experiment C: Comparison with CE-LQR

<img width="720" height="392" alt="image" src="https://github.com/user-attachments/assets/b0323e4b-9dfe-4586-b46a-538ab2f0898a" />
<img width="720" height="393" alt="image" src="https://github.com/user-attachments/assets/63e6541f-1333-42e9-adaf-f15481ce874e" />





    doi={10.1109/TAC.2025.3569597}
}
```
