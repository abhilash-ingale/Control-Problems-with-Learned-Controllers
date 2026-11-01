"""Linear analysis of the upright equilibrium for every (n, actuation).

Prints, per configuration:
- the unstable open-loop poles (as continuous-time rates, 1/s),
- the PBH controllability margin of each unstable mode,
- the LQR gain and the slowest closed-loop pole,
- the input needed to correct a 0.05 rad tilt of the last link (|K x| / u_max).

This is the "how hard is this problem, before any learning?" table for notes/01_dynamics.md.
"""

import itertools

import numpy as np

from pendulum_rl.control import LQRController, pbh_margins
from pendulum_rl.envs import ACTUATIONS, PendulumChainEnv


def main():
    np.set_printoptions(precision=3, suppress=True)
    for n, act in itertools.product([1, 2, 3, 4], ACTUATIONS):
        env = PendulumChainEnv(n_links=n, actuation=act)
        lqr = LQRController(env)
        dt = env.dt
        unstable = [(np.log(complex(lam)) / dt, s) for lam, s in pbh_margins(lqr.A, lqr.B)]
        cl = lqr.closed_loop_eigs()
        slowest = np.max(np.abs(cl))

        x = np.zeros(lqr.A.shape[0])
        x[env.model.nv - 1] = 0.05  # tilt the last joint
        effort = float(np.abs(lqr.K @ x)[0]) / env.ctrl_limit

        print(f"n={n} {act:11s}  u_max={env.ctrl_limit:5.1f}")
        for rate, s in unstable:
            print(f"    unstable pole {rate.real:7.2f} 1/s   PBH margin {s:.2e}")
        print(f"    K = {lqr.K.ravel()}")
        print(f"    slowest closed-loop |z| = {slowest:.4f}  (time const {-dt / np.log(slowest):.2f} s)")
        print(f"    effort for 0.05 rad on last joint: {effort:.2f} x u_max")


if __name__ == "__main__":
    main()
