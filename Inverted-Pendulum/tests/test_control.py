import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from pendulum_rl.common.evaluate import rollout
from pendulum_rl.control import (
    EnergySwingUp,
    LQRController,
    controllability_matrix,
    linearize_upright,
    pbh_margins,
    single_pendulum_params,
)
from pendulum_rl.envs import ACTUATIONS, PendulumChainEnv


def test_single_pendulum_linearization_matches_analytic():
    """MuJoCo's finite-difference A, B vs. exp of the hand-derived continuous model.

    I_p theta'' = m g l_c theta - b theta' + tau   (linearized at theta = 0)
    => A_c = [[0, 1], [m g l_c / I_p, -b / I_p]],  B_c = [0, 1 / I_p]^T
    """
    env = PendulumChainEnv(n_links=1)
    p = single_pendulum_params(env)
    A_c = np.array([[0.0, 1.0], [p.m * p.g * p.l_c / p.I_p, -p.b / p.I_p]])
    B_c = np.array([[0.0], [1.0 / p.I_p]])

    dt = env.cfg.timestep
    A, B = linearize_upright(env.model)
    # Exact ZOH discretization via the augmented matrix exponential.
    M = expm(np.block([[A_c, B_c], [np.zeros((1, 3))]]) * dt)
    A_exact, B_exact = M[:2, :2], M[:2, 2:]
    assert np.allclose(A, A_exact, atol=1e-6)
    assert np.allclose(B, B_exact, atol=1e-6)
    # The unstable eigenvalue is exp(sqrt(m g l_c / I_p) dt) to first order in damping.
    assert np.isclose(np.max(np.abs(np.linalg.eigvals(A))), np.exp(np.sqrt(A_c[1, 0]) * dt), rtol=1e-3)


@pytest.mark.parametrize("n,actuation", itertools.product([1, 2, 3, 4], ACTUATIONS))
def test_upright_is_controllable(n, actuation):
    env = PendulumChainEnv(n_links=n, actuation=actuation)
    lqr = LQRController(env)
    margins = pbh_margins(lqr.A, lqr.B)
    assert margins, "upright should have unstable modes"
    assert all(s > 1e-8 for _, s in margins)
    assert np.linalg.matrix_rank(controllability_matrix(lqr.A, lqr.B)) == lqr.A.shape[0]


@pytest.mark.parametrize("n,actuation", itertools.product([1, 2, 3, 4], ACTUATIONS))
def test_lqr_holds_upright(n, actuation):
    env = PendulumChainEnv(n_links=n, actuation=actuation, init="up", init_noise=0.02, episode_seconds=5.0)
    lqr = LQRController(env)
    assert np.all(np.abs(lqr.closed_loop_eigs()) < 1.0)
    ro = rollout(env, lqr, seed=0)
    # Bounded, not tight: for n = 4 the lower links swing out ~10x the initial tilt to catch
    # the upper ones (the only way a single base input can move the passive joints).
    assert ro.frac_upright(tol=0.5) == 1.0
    assert np.max(np.abs(ro.link_angles[-50:])) < 1e-2


def test_energy_swingup_reaches_and_holds_upright():
    env = PendulumChainEnv(n_links=1, episode_seconds=15.0)
    ctrl = EnergySwingUp(env)
    for seed in range(3):
        ro = rollout(env, ctrl, seed=seed)
        assert ro.time_to_upright() < 10.0, f"seed {seed}"
