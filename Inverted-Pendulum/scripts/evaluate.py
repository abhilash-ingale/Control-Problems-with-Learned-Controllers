"""Evaluate a controller: metrics over seeds and an optional mp4.

    python scripts/evaluate.py --controller energy --n-links 1 --video runs/energy_n1.mp4
    python scripts/evaluate.py --controller lqr --n-links 3 --actuation cart --init up

RL policies get added as further --controller choices in M2.
"""

import argparse
from pathlib import Path

import numpy as np

from pendulum_rl.common.evaluate import rollout, save_video
from pendulum_rl.control import EnergySwingUp, LQRController
from pendulum_rl.envs import PendulumChainEnv


def make_controller(name: str, env: PendulumChainEnv):
    if name == "lqr":
        return LQRController(env)
    if name == "energy":
        return EnergySwingUp(env)
    if name == "zero":
        return lambda env: np.zeros(1)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["lqr", "energy", "zero"], default="energy")
    ap.add_argument("--n-links", type=int, default=1)
    ap.add_argument("--actuation", choices=["base_torque", "cart"], default="base_torque")
    ap.add_argument("--obs-mode", choices=["joints", "joints+tip"], default="joints")
    ap.add_argument("--init", choices=["down", "up"], default="down")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--video", type=str, default=None, help="write an mp4 of the first seed")
    args = ap.parse_args()

    env = PendulumChainEnv(
        n_links=args.n_links,
        actuation=args.actuation,
        obs_mode=args.obs_mode,
        init=args.init,
        episode_seconds=args.seconds,
        render_mode="rgb_array" if args.video else None,
    )
    ctrl = make_controller(args.controller, env)

    for seed in range(args.seeds):
        ro = rollout(env, ctrl, seed=seed, record=bool(args.video) and seed == 0)
        print(
            f"seed {seed}: return {ro.episode_return:8.2f}  "
            f"frac upright {ro.frac_upright():.2f}  time to upright {ro.time_to_upright():.2f} s"
        )
        if ro.frames:
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            save_video(ro.frames, args.video, fps=round(1 / env.dt))
            print(f"  video -> {args.video}")
    env.close()


if __name__ == "__main__":
    main()
