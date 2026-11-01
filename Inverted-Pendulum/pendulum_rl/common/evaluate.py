"""Rollouts, metrics and video for any controller.

A controller is a callable `controller(env) -> action`. Classical baselines read the
full MuJoCo state from `env`; an RL policy can be wrapped as
`lambda env: policy(env._get_obs())`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pendulum_rl.envs.chain_env import PendulumChainEnv, wrap_angle

UPRIGHT_TOL = 0.2  # rad: every link within this of vertical counts as "upright"


@dataclass
class Rollout:
    t: np.ndarray
    link_angles: np.ndarray  # (T, n) absolute, wrapped
    action: np.ndarray  # (T,)
    reward: np.ndarray  # (T,)
    frames: list = field(default_factory=list)

    @property
    def episode_return(self) -> float:
        return float(self.reward.sum())

    def upright_mask(self, tol: float = UPRIGHT_TOL) -> np.ndarray:
        return np.all(np.abs(self.link_angles) < tol, axis=1)

    def frac_upright(self, tol: float = UPRIGHT_TOL) -> float:
        return float(self.upright_mask(tol).mean())

    def time_to_upright(self, tol: float = UPRIGHT_TOL) -> float:
        """First time after which the chain stays upright until the end (inf if never)."""
        mask = self.upright_mask(tol)
        if not mask[-1]:
            return float("inf")
        last_bad = np.flatnonzero(~mask)
        return float(self.t[last_bad[-1] + 1]) if last_bad.size else 0.0


def rollout(env: PendulumChainEnv, controller, seed: int | None = None, record: bool = False) -> Rollout:
    env.reset(seed=seed)
    if hasattr(controller, "reset"):
        controller.reset()
    ts, angles, actions, rewards, frames = [], [], [], [], []
    done = False
    while not done:
        a = np.asarray(controller(env), dtype=np.float64).reshape(1)
        _, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ts.append(env.data.time)
        angles.append(wrap_angle(env.link_angles()))
        actions.append(a[0])
        rewards.append(r)
        if record:
            frames.append(env.render())
    return Rollout(np.array(ts), np.array(angles), np.array(actions), np.array(rewards), frames)


def save_video(frames, path: str, fps: int = 50):
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps)
