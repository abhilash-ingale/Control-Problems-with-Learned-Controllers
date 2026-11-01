# 01 · Dynamics of the n-link chain

## Coordinates
Links point up at q = 0. Joint angles are relative, so link k's absolute angle from vertical is

$$\varphi_k = \sum_{i=1}^{k} q_i, \qquad \varphi = T q,\quad T = \text{lower-triangular ones}.$$

For the cart, add the slider position $x$ in front of $q$.

## Equations of motion
With centre-of-mass positions $p_k(q)$ and rotational inertias $I_k$ about each centre of mass, the Lagrangian $\mathcal L = K - U$ gives the manipulator equation

$$M(q)\,\ddot q + C(q,\dot q)\,\dot q + D\dot q + \nabla U(q) = b\,u,$$

- $M(q) = \sum_k m_k J_k^\top J_k + I_k\, t_k t_k^\top$ with $J_k = \partial p_k/\partial q$ and $t_k$ = row $k$ of $T$,
- $U(q) = g\sum_k m_k z_k(q)$ with $z_k$ the COM height,
- $D$ = joint damping, $b = e_1$ (base torque) or $e_x$ (cart force).

Only one entry of $b$ is non-zero, so for $n\ge2$ (base torque) or $n\ge1$ (cart) there are fewer inputs than degrees of freedom: the system is **underactuated**. The passive joints can only be moved through the inertial coupling in $M(q)$.

### n = 1 explicitly
$$I_p\,\ddot\theta = m g l_c \sin\theta - b\dot\theta + \tau, \qquad I_p = I_{c} + m l_c^2.$$

With the defaults ($m=1$, $l=0.5$, capsule radius 0.02) $l_c = 0.25$, so the peak gravity torque is $m g l_c \approx 2.45$ N m, while $\tau_{\max} = 1$ N m. Static lifting only works for $|\theta| < \arcsin(1/2.45) \approx 24°$; everything else needs energy pumping.

## Energy
$$E = \tfrac12 \dot q^\top M(q)\dot q + U(q), \qquad \dot E = \dot q^\top b\,u - \dot q^\top D\dot q.$$

(The $C$ term drops out because $\dot M - 2C$ is skew-symmetric.) This is checked by `tests/test_env.py::test_passive_energy`: with $u=0$, energy is conserved without damping and decreasing with it.

**Exercise.** Show that for the base-torque chain the input only does work through the first joint, $\dot E = \dot q_1 \tau$, no matter how many links there are. What does this say about which states let you inject energy?

## Linearization at upright
At $q=\dot q=0$, $u=0$: $C\dot q$ is second order and $\nabla U \approx G q$ with $G = \partial^2 U/\partial q^2$ (negative definite at upright). With $x = (q, \dot q)$:

$$\dot x = \underbrace{\begin{bmatrix}0 & I\\ -M_0^{-1}G & -M_0^{-1}D\end{bmatrix}}_{A_c} x + \underbrace{\begin{bmatrix}0\\ M_0^{-1}b\end{bmatrix}}_{B_c} u.$$

Each unstable eigenvalue is roughly $\sqrt{\lambda}$ for a positive eigenvalue $\lambda$ of $-M_0^{-1}G$, so a chain of $n$ links has $n$ unstable poles (the cart adds a double integrator at 0). Upper links have shorter effective lengths and faster unstable poles.

In code we never write $A_c$ by hand: `control/lqr.py::linearize_upright` asks MuJoCo for the finite-difference Jacobian of one timestep. For $n=1$, `test_single_pendulum_linearization_matches_analytic` checks it against $e^{A_c\Delta t}$.

## Controllability
Upright is locally stabilizable iff every unstable mode is controllable. The PBH test says mode $\lambda$ is controllable iff $\operatorname{rank}[A-\lambda I,\; B] = 2n$; `pbh_margins` reports $\sigma_{\min}$ of that matrix as a *graded* measure. Run

```bash
python scripts/linear_analysis.py
```

and note how the margin of the slowest unstable mode shrinks as $n$ grows (about $4\times10^{-2}$ for $n=1$ down to $8\times10^{-4}$ for $n=4$ with base torque) and how much input a small tip error needs. That is the analytical version of "the quadruple pendulum on one base torque may be beyond what RL can do here". For $n=4$, a 0.05 rad error on the last joint already asks for $1.5\,u_{\max}$ (base torque) or $2.2\,u_{\max}$ (cart).

The underactuation is visible in the closed loop too: starting $n=4$ from a 0.02 rad perturbation, LQR swings the *first* link out to about 0.24 rad before everything settles. The only way to move the upper links is to move the lower ones away from upright first.

## Tip orientation as an observation
$\varphi_\text{tip} = \sum_i q_i$ and $\omega_\text{tip} = \sum_i \dot q_i$ are *linear functions of the state*. They add no information to a full-state observation: with `obs_mode="joints"` the problem is already an MDP. What they change is the **representation**: a network gets the quantity that matters most for balance (the top of the chain) without having to learn the sum. Whether that helps sample efficiency is an empirical question for M3, and it only matters for $n\ge2$ (for $n=1$ it duplicates $q_1$).
