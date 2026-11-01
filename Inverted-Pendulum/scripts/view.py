"""Watch a controller live in the interactive MuJoCo viewer.

On macOS the viewer must run under `mjpython` (installed with mujoco):

    mjpython scripts/view.py --controller energy --n-links 1
    mjpython scripts/view.py --controller lqr --n-links 3 --actuation cart --init up
    mjpython scripts/view.py --controller zero --n-links 2 --init up    # passive chaos

Double-click a body and Ctrl+right-drag to push it; the controller reacts.
"""

import argparse
import time

import mujoco.viewer

from evaluate import make_controller  # sibling script; scripts/ is on sys.path when run directly
from pendulum_rl.envs import PendulumChainEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["lqr", "energy", "zero"], default="energy")
    ap.add_argument("--n-links", type=int, default=1)
    ap.add_argument("--actuation", choices=["base_torque", "cart"], default="base_torque")
    ap.add_argument("--init", choices=["down", "up"], default="down")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Long episode: the viewer runs until you close it.
    env = PendulumChainEnv(n_links=args.n_links, actuation=args.actuation, init=args.init, episode_seconds=1e6)
    ctrl = make_controller(args.controller, env)
    env.reset(seed=args.seed)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            env.step(ctrl(env))
            viewer.sync()
            time.sleep(max(0.0, env.dt - (time.perf_counter() - t0)))  # real time


if __name__ == "__main__":
    main()
