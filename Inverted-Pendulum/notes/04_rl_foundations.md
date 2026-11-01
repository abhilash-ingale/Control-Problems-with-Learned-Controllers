# 04 · RL foundations for this problem

## The MDP
- State: $s = (q, \dot q)$ (+ cart $x, \dot x$). Observation `joints` uses $(\sin q, \cos q, \dot q)$, a continuous, bounded encoding of the angles, so the observation is a bijection of the state: **fully observed**.
- Action: $a \in [-1, 1]$, scaled to $u_{\max}$, held for $\Delta t = 20$ ms.
- Transition: deterministic MuJoCo step; the only randomness is the initial state.
- Reward: $r = \tfrac1n\sum_k \cos\varphi_k - c_u a^2 - c_v\|\dot q\|^2$ (`reward_fn`).
- Horizon: 500 steps, truncated, not terminated. Value targets must bootstrap through truncation; treating time-outs as terminal is a classic bug.

**Discount and effective horizon.** $\sum_t \gamma^t = 1/(1-\gamma)$ steps. At 50 Hz, $\gamma=0.99$ looks $\approx 2$ s ahead and $\gamma = 0.995$ about 4 s. A swing-up with $\tau_{\max} = 1$ N m takes several swings; if the horizon is shorter than one swing, the immediate cost of pushing *away* from upright dominates and pumping never gets learned. Compute the swing period and pick $\gamma$ accordingly.

## Bellman equations
$$V^\pi(s) = \mathbb E_{a\sim\pi}\big[r(s,a) + \gamma V^\pi(s')\big],\qquad Q^*(s,a) = r(s,a) + \gamma \max_{a'} Q^*(s',a').$$

Everything below is a way to approximate one of these with function approximators and samples. Notes/02 has the exact solution for the linear-quadratic case.

## Policy gradient theorem
For a stochastic policy $\pi_\theta$ and $J(\theta) = \mathbb E[\sum_t \gamma^t r_t]$:

$$\nabla_\theta J = \mathbb E_\pi\Big[\sum_t \gamma^t\, \nabla_\theta \log \pi_\theta(a_t\mid s_t)\, \big(Q^\pi(s_t,a_t) - b(s_t)\big)\Big].$$

Any baseline $b(s)$ leaves it unbiased because $\mathbb E_a[\nabla\log\pi] = 0$. The choice of the estimator for $Q - b$ (Monte Carlo return, TD error, GAE($\lambda$)) is a bias-variance knob and the main difference between REINFORCE, A2C and PPO.

## Deterministic policy gradient
For $a = \mu_\theta(s)$: $\nabla_\theta J = \mathbb E_s\big[\nabla_\theta\mu_\theta(s)\,\nabla_a Q(s,a)|_{a=\mu_\theta(s)}\big]$. It needs a differentiable critic but no likelihood ratio, which makes off-policy learning from a replay buffer natural (DDPG, TD3).

## Map of the algorithms (one note each, added as implemented)

| # | Algorithm | Learns | Data | Key idea to derive |
|---|---|---|---|---|
| 10 | REINFORCE | $\pi$ | on-policy, MC | score-function estimator, variance |
| 11 | A2C | $\pi, V$ | on-policy, TD | baseline, advantage, n-step returns |
| 12 | PPO | $\pi, V$ | on-policy, reused for K epochs | GAE($\lambda$), trust region → clipped surrogate |
| 13 | DQN | $Q$ | off-policy, replay | fitted Q-iteration, target network; discretized torque, n = 1 only |
| 14 | DDPG | $\mu, Q$ | off-policy | deterministic PG |
| 15 | TD3 | $\mu, Q_1, Q_2$ | off-policy | overestimation bias of $\max$, clipped double Q |
| 16 | SAC | $\pi, Q_1, Q_2$ | off-policy | max-entropy objective, soft Bellman operator, temperature tuning |

Each note: (1) derivation from the objective, (2) what the estimator's bias/variance is, (3) the few lines of code that implement the key equation, (4) findings on this environment (learning curves over seeds, what the policy does physically, comparison with LQR / energy swing-up).

## Questions to answer with experiments
1. Does a learned $n=1$ policy pump energy like notes/03 predicts? Compare action vs. $(E^*-E)\dot\theta$.
2. Near upright, is the learned policy close to the LQR gain $K$? Linearize the network ($\partial\mu/\partial s$ at $s=0$) and compare; compare the critic with $-x^\top P x$.
3. How does sample efficiency scale with $n$, and does it track the PBH margins from `scripts/linear_analysis.py`?
4. Does `joints+tip` help, given that it adds no information (notes/01)?
