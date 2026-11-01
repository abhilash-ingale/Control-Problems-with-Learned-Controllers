"""Linearization at the upright equilibrium and discrete-time LQR.

State convention for everything here: x = [q - q*, qdot] with q* = 0 (upright,
cart at the origin), ordered as MuJoCo orders qpos/qvel (slider first for the cart).
See notes/02_lqr.md for the derivations.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from pendulum_rl.envs.chain_env import PendulumChainEnv, wrap_angle


def linearize_upright(model: mujoco.MjModel, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """One-timestep discrete linearization x_{t+1} = A x_t + B u_t at upright, u = 0.

    Central finite differences of one mj_step, so A and B describe exactly the integrator
    the simulator runs (RK4), not an idealized continuous model. (mujoco.mjd_transitionFD
    does the same but does not support RK4.) All joints are hinges/slides, so qpos is a
    plain vector and x = [qpos, qvel].
    """
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu

    def step(x, u):
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:], data.ctrl[:] = x[:nq], x[nq:], u
        mujoco.mj_step(model, data)
        return np.concatenate([data.qpos, data.qvel])

    x0, u0 = np.zeros(nq + nv), np.zeros(nu)
    A = np.column_stack([(step(x0 + eps * e, u0) - step(x0 - eps * e, u0)) / (2 * eps) for e in np.eye(nq + nv)])
    B = np.column_stack([(step(x0, u0 + eps * e) - step(x0, u0 - eps * e)) / (2 * eps) for e in np.eye(nu)])
    return A, B


def hold_discretize(A: np.ndarray, B: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Compose n_steps of x+ = A x + B u with u held constant (zero-order hold).

    A_N = A^N,  B_N = (A^{N-1} + ... + A + I) B.
    """
    A_N = np.eye(A.shape[0])
    B_N = np.zeros_like(B)
    for _ in range(n_steps):
        B_N = A @ B_N + B
        A_N = A @ A_N
    return A_N, B_N


def dlqr(A, B, Q, R) -> tuple[np.ndarray, np.ndarray]:
    """Infinite-horizon discrete LQR. Returns (K, P) with u = -K x and V(x) = x^T P x."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def controllability_matrix(A, B) -> np.ndarray:
    n = A.shape[0]
    blocks, AkB = [], B
    for _ in range(n):
        blocks.append(AkB)
        AkB = A @ AkB
    return np.hstack(blocks)


def pbh_margins(A, B) -> list[tuple[complex, float]]:
    """Popov-Belevitch-Hautus test for each unstable mode.

    For each eigenvalue lambda of A with |lambda| >= 1 returns
    sigma_min([A - lambda I, B]). Zero means that mode is uncontrollable; small means
    it is barely controllable (large inputs needed). This is better conditioned than the
    rank of the controllability matrix, whose columns A^k B blow up.
    """
    n = A.shape[0]
    out = []
    for lam in np.linalg.eigvals(A):
        if abs(lam) >= 1.0 - 1e-9:
            M = np.hstack([A - lam * np.eye(n), B])
            out.append((lam, float(np.linalg.svd(M, compute_uv=False)[-1])))
    return out


def error_state(env: PendulumChainEnv) -> np.ndarray:
    q = env.data.qpos.copy()
    q[env.hinge_idx] = wrap_angle(q[env.hinge_idx])
    return np.concatenate([q, env.data.qvel])


class LQRController:
    """Discrete LQR at the control rate of `env` (ZOH over the env's frame skip).

    Works in physical units (N m or N) and returns a normalized action in [-1, 1].
    Default weights: angles 10, cart position 1, velocities 1; R = (r / u_max)^2-scaled
    so that saturating the actuator costs about as much as a 1 rad angle error.
    """

    def __init__(self, env: PendulumChainEnv, Q=None, R=None):
        nv = env.model.nv
        A1, B1 = linearize_upright(env.model)
        self.A, self.B = hold_discretize(A1, B1, env.frame_skip)
        if Q is None:
            pos_w = np.full(nv, 10.0)
            if env.is_cart:
                pos_w[0] = 1.0
            Q = np.diag(np.concatenate([pos_w, np.ones(nv)]))
        if R is None:
            R = np.array([[10.0 / env.ctrl_limit**2]])
        self.Q, self.R = Q, R
        self.K, self.P = dlqr(self.A, self.B, Q, R)
        self.ctrl_limit = env.ctrl_limit

    def closed_loop_eigs(self) -> np.ndarray:
        return np.linalg.eigvals(self.A - self.B @ self.K)

    def cost_to_go(self, env: PendulumChainEnv) -> float:
        x = error_state(env)
        return float(x @ self.P @ x)

    def __call__(self, env: PendulumChainEnv) -> np.ndarray:
        u = -self.K @ error_state(env)
        return np.clip(u / self.ctrl_limit, -1.0, 1.0)
