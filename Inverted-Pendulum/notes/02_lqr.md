# 02 · Discrete LQR, and why it is the Rosetta stone for RL

## From one timestep to the control rate
MuJoCo gives $x_{k+1} = A x_k + B u_k$ for one 2 ms step. The policy acts at 50 Hz and holds $u$ for $N = 10$ steps (zero-order hold):

$$A_N = A^N, \qquad B_N = \Big(\sum_{i=0}^{N-1} A^i\Big) B.$$

(`hold_discretize`.) For a continuous model this is the same as $\exp\!\begin{bmatrix}A_c & B_c\\ 0 & 0\end{bmatrix}T$.

## The problem
$$\min_{u_0,u_1,\dots}\; \sum_{k=0}^{\infty} x_k^\top Q x_k + u_k^\top R u_k \quad \text{s.t. } x_{k+1} = A x_k + B u_k.$$

## Derivation by dynamic programming
Guess $V(x) = x^\top P x$. The Bellman equation

$$V(x) = \min_u\; x^\top Q x + u^\top R u + V(Ax + Bu)$$

has a quadratic right-hand side in $u$; setting the gradient to zero gives

$$u^* = -Kx, \qquad K = (R + B^\top P B)^{-1} B^\top P A,$$

and substituting back yields the **discrete algebraic Riccati equation**

$$P = Q + A^\top P A - A^\top P B (R + B^\top P B)^{-1} B^\top P A.$$

`dlqr` solves it with `scipy.linalg.solve_discrete_are`. The guess is self-consistent, so the optimal value function really is quadratic and the optimal policy really is linear.

## The RL dictionary
LQR is the one continuous-state MDP we can solve exactly, so every RL idea has a closed form here:

| RL object | LQR form |
|---|---|
| reward $r(x,u)$ | $-(x^\top Q x + u^\top R u)$ |
| value $V^\pi$ of $u = -Kx$ | $x^\top P_K x$, $P_K$ solves the Lyapunov eq. $P_K = Q + K^\top R K + (A-BK)^\top P_K (A-BK)$ |
| Q-function $Q^\pi(x,u)$ | quadratic in $(x,u)$: $\begin{bmatrix}x\\u\end{bmatrix}^\top \begin{bmatrix}Q + A^\top P_K A & A^\top P_K B\\ B^\top P_K A & R + B^\top P_K B\end{bmatrix}\begin{bmatrix}x\\u\end{bmatrix}$ |
| greedy improvement $\arg\min_u Q^\pi(x,u)$ | $K' = (R+B^\top P_K B)^{-1}B^\top P_K A$ |
| policy iteration | Hewer's algorithm (Lyapunov solve, then the update above); converges quadratically from any stabilizing $K_0$ |
| value iteration | iterating the Riccati map from $P_0 = 0$ |
| policy gradient | $\nabla_K J(K) = 2\big[(R + B^\top P_K B)K - B^\top P_K A\big]\Sigma_K$, with $\Sigma_K$ the state covariance; gradient descent converges globally despite $J$ being non-convex in $K$ (Fazel et al., 2018) |
| discount $\gamma$ | replace $A, B$ by $\sqrt\gamma A, \sqrt\gamma B$ |

**Exercises** (good warm-ups before the RL notes, all short numpy scripts on top of `LQRController`):
1. Implement Hewer's policy iteration and check it converges to `dlqr`'s $K$.
2. Fit the quadratic Q-function from simulated transitions by least squares (LSTD-Q), then improve greedily. That is model-free policy iteration, i.e. what DQN/DDPG approximate with networks.
3. Estimate $\nabla_K J$ by finite differences of rollout returns and compare with the formula above. That is a zeroth-order policy gradient.

## What LQR can't do
It is local: with saturation $|u|\le u_{\max}$ the region of attraction is bounded, and from hanging down $x$ is nowhere near the linearization point. That gap is what swing-up controllers, and RL, have to fill. A useful number for later: the LQR cost-to-go $x^\top P x$ is an upper bound on what a learned policy can save near upright, and comparing a learned critic to it near $x=0$ is a sanity check.
