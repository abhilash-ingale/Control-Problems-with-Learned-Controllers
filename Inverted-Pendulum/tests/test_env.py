import itertools

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pendulum_rl.envs import ACTUATIONS, OBS_MODES, PendulumChainEnv, wrap_angle

ALL_CONFIGS = list(itertools.product([1, 2, 3, 4], ACTUATIONS, OBS_MODES))


@pytest.mark.parametrize("n,actuation,obs_mode", ALL_CONFIGS)
def test_check_env(n, actuation, obs_mode):
    env = PendulumChainEnv(n_links=n, actuation=actuation, obs_mode=obs_mode)
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("n,actuation", itertools.product([1, 2, 3, 4], ACTUATIONS))
def test_tip_sensor_equals_sum_of_joint_angles(n, actuation):
    """phi_tip (from framezaxis) = q_1 + ... + q_n, and omega_tip = qdot_1 + ... + qdot_n."""
    env = PendulumChainEnv(n_links=n, actuation=actuation, obs_mode="joints+tip", init_noise=0.5)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        env.step(rng.uniform(-1, 1, size=1))
        assert np.isclose(wrap_angle(env.tip_angle() - env.hinge_qpos().sum()), 0.0, atol=1e-9)
        assert np.isclose(env.tip_rate(), env.hinge_qvel().sum(), atol=1e-9)


def _total_energy(env):
    mujoco.mj_energyPos(env.model, env.data)
    mujoco.mj_energyVel(env.model, env.data)
    return env.data.energy.sum()


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_passive_energy(n):
    """Zero input: energy is conserved without damping and never increases with damping.

    dE/dt = -qdot^T D qdot <= 0 for joint damping D, so any energy gain is integrator error.
    """
    for damping in (0.0, 0.01):
        env = PendulumChainEnv(n_links=n, init="up", init_noise=0.3, joint_damping=damping)
        env.reset(seed=1)
        E0 = _total_energy(env)
        energies = []
        for _ in range(250):  # 5 s, chaotic for n >= 2
            env.step(np.zeros(1))
            energies.append(_total_energy(env))
        energies = np.array(energies)
        scale = abs(E0) + 1.0
        if damping == 0.0:
            assert np.max(np.abs(energies - E0)) < 1e-3 * scale
        else:
            assert np.max(energies) < E0 + 1e-4 * scale
            assert energies[-1] < E0
