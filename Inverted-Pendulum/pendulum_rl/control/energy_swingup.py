"""Energy-pumping swing-up with an LQR catch, for the single pendulum (n = 1, base torque).

Pendulum (theta = 0 upright):  I_p theta'' = m g l_c sin(theta) - b theta' + tau
Energy:                         E = 1/2 I_p theta'^2 + m g l_c cos(theta),   E* = m g l_c
Power balance:                  dE/dt = theta' (tau - b theta')

Choosing tau = k (E* - E) theta' + b theta' gives dE/dt = k (E* - E) theta'^2, so the
Lyapunov function V = 1/2 (E - E*)^2 is non-increasing. The actuator saturates, which
turns this into near bang-bang pumping. See notes/03_energy_swingup.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pendulum_rl.control.lqr import LQRController
from pendulum_rl.envs.chain_env import PendulumChainEnv, wrap_angle


@dataclass(frozen=True)
class SinglePendulumParams:
    m: float  # link mass
    l_c: float  # pivot -> centre of mass
    I_p: float  # moment of inertia about the pivot axis
    b: float  # joint damping
    g: float

    @property
    def E_star(self) -> float:
        return self.m * self.g * self.l_c

    def energy(self, theta: float, theta_dot: float) -> float:
        return 0.5 * self.I_p * theta_dot**2 + self.m * self.g * self.l_c * np.cos(theta)


def single_pendulum_params(env: PendulumChainEnv) -> SinglePendulumParams:
    """Read the physical parameters MuJoCo actually compiled (capsule end caps included)."""
    m_ = env.model
    body = m_.body("link1").id
    m = float(m_.body_mass[body])
    l_c = float(np.linalg.norm(m_.body_ipos[body]))
    # Capsule along z: principal axes coincide with the body frame, rotation is about y.
    I_p = float(m_.body_inertia[body][1]) + m * l_c**2
    b = float(m_.dof_damping[env.hinge_idx[0]])
    return SinglePendulumParams(m=m, l_c=l_c, I_p=I_p, b=b, g=env.cfg.gravity)


class EnergySwingUp:
    def __init__(self, env: PendulumChainEnv, k: float = 5.0, catch_angle: float = 0.3, release_angle: float = 0.8):
        if env.cfg.n_links != 1 or env.is_cart:
            raise ValueError("EnergySwingUp is the n = 1, base_torque baseline")
        self.p = single_pendulum_params(env)
        self.lqr = LQRController(env)
        self.k, self.catch_angle, self.release_angle = k, catch_angle, release_angle
        self.u_max = env.ctrl_limit
        self.balancing = False

    def reset(self):
        self.balancing = False

    def __call__(self, env: PendulumChainEnv) -> np.ndarray:
        theta = float(wrap_angle(env.data.qpos[0]))
        theta_dot = float(env.data.qvel[0])

        # Hysteresis between the two modes keeps the switch from chattering.
        if self.balancing and abs(theta) > self.release_angle:
            self.balancing = False
        elif not self.balancing and abs(theta) < self.catch_angle:
            self.balancing = True
        if self.balancing:
            return self.lqr(env)

        E_err = self.p.E_star - self.p.energy(theta, theta_dot)
        tau = self.k * E_err * theta_dot + self.p.b * theta_dot
        # Kick off from rest at the bottom, where theta_dot = 0 gives no pumping direction.
        if abs(theta_dot) < 1e-3:
            tau = self.u_max
        return np.array([np.clip(tau / self.u_max, -1.0, 1.0)])
