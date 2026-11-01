"""Gymnasium environment for n-link pendulum swing-up and balance."""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from pendulum_rl.envs.model import ChainConfig, build_mjcf

OBS_MODES = ("joints", "joints+tip")


def wrap_angle(x):
    """Wrap to [-pi, pi)."""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def reward_fn(env: "PendulumChainEnv", action: np.ndarray) -> float:
    """Uprightness minus small control and velocity penalties.

    Kept as a free function so reward shaping can be swapped out
    (``PendulumChainEnv(reward=my_fn)``) without touching the env.
    """
    upright = np.mean(np.cos(env.link_angles()))  # in [-1, 1], 1 = all links up
    ctrl_cost = 1e-3 * float(np.sum(action**2))
    vel_cost = 1e-4 * float(np.sum(env.hinge_qvel() ** 2))
    r = upright - ctrl_cost - vel_cost
    if env.cfg.actuation == "cart":
        r -= 1e-2 * env.data.qpos[0] ** 2
    return float(r)


class PendulumChainEnv(gym.Env):
    """Swing-up (or balance) of an n-link pendulum chain.

    Action: one scalar in [-1, 1], scaled to the torque/force limit.
    Observation ("joints"):     [sin q, cos q, qdot] for each hinge (+ [x, xdot] for the cart).
    Observation ("joints+tip"): the above + [sin phi_tip, cos phi_tip, omega_tip] read from
                                MuJoCo sensors on the tip site.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        n_links: int = 1,
        actuation: str = "base_torque",
        obs_mode: str = "joints",
        init: str = "down",
        control_hz: float = 50.0,
        episode_seconds: float = 10.0,
        init_noise: float = 0.05,
        reward=reward_fn,
        render_mode: str | None = None,
        **model_kwargs,
    ):
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}")
        if init not in ("down", "up"):
            raise ValueError("init must be 'down' or 'up'")
        self.cfg = ChainConfig(n_links=n_links, actuation=actuation, **model_kwargs)
        self.obs_mode, self.init, self.init_noise = obs_mode, init, init_noise
        self.reward = reward
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(round(1.0 / (control_hz * self.cfg.timestep)))
        self.dt = self.frame_skip * self.cfg.timestep
        self.max_steps = int(round(episode_seconds / self.dt))
        self.ctrl_limit = self.cfg.ctrl_limit

        self.is_cart = self.cfg.actuation == "cart"
        self.hinge_idx = np.arange(1, self.model.nq) if self.is_cart else np.arange(self.model.nq)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        obs_dim = 3 * n_links + (2 if self.is_cart else 0) + (3 if obs_mode == "joints+tip" else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64)

        self._sensor = {
            name: (self.model.sensor(name).adr[0], self.model.sensor(name).dim[0])
            for name in ("tip_zaxis", "tip_angvel", "tip_pos")
        }
        self._renderer = None
        self._steps = 0

    # ---- state accessors -------------------------------------------------------
    def hinge_qpos(self) -> np.ndarray:
        return self.data.qpos[self.hinge_idx].copy()

    def hinge_qvel(self) -> np.ndarray:
        return self.data.qvel[self.hinge_idx].copy()

    def link_angles(self) -> np.ndarray:
        """Absolute angle of each link from upright: phi_k = q_1 + ... + q_k."""
        return np.cumsum(self.hinge_qpos())

    def sensor(self, name: str) -> np.ndarray:
        adr, dim = self._sensor[name]
        return self.data.sensordata[adr : adr + dim].copy()

    def tip_angle(self) -> float:
        """Absolute tilt of the tip from the sensor: z axis of the tip frame is (sin phi, 0, cos phi)."""
        z = self.sensor("tip_zaxis")
        return float(np.arctan2(z[0], z[2]))

    def tip_rate(self) -> float:
        # Planar motion: the angular velocity is along y in both world and site frames.
        return float(self.sensor("tip_angvel")[1])

    def _get_obs(self) -> np.ndarray:
        q, qd = self.hinge_qpos(), self.hinge_qvel()
        parts = [np.sin(q), np.cos(q), qd]
        if self.is_cart:
            parts.append([self.data.qpos[0], self.data.qvel[0]])
        if self.obs_mode == "joints+tip":
            z = self.sensor("tip_zaxis")
            parts.append([z[0], z[2], self.tip_rate()])
        return np.concatenate(parts).astype(np.float64)

    # ---- gym API ----------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        n = self.cfg.n_links
        q = self.np_random.uniform(-self.init_noise, self.init_noise, size=n)
        if self.init == "down":
            q[0] += np.pi
        qd = self.np_random.uniform(-self.init_noise, self.init_noise, size=n)
        self.data.qpos[self.hinge_idx] = q
        self.data.qvel[self.hinge_idx] = qd
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(1), -1.0, 1.0)
        self.data.ctrl[0] = a[0] * self.ctrl_limit
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # mj_step evaluates sensors *before* integrating; recompute so they match the new state.
        mujoco.mj_forward(self.model, self.data)
        self._steps += 1

        reward = self.reward(self, a)
        terminated = False
        if self.is_cart and abs(self.data.qpos[0]) > self.cfg.rail_limit:
            terminated = True
            reward -= 100.0
        truncated = self._steps >= self.max_steps
        info = {"upright": float(np.mean(np.cos(self.link_angles())))}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="side")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
