# Plan: Inverted pendulum chain swing-up with RL (MuJoCo)

## Context
A learning project: simulate an n-link inverted pendulum (n = 1…4) in MuJoCo, swing it up from hanging down and balance it, and use it to study RL methods mathematically and qualitatively. The policy state can optionally include an **orientation measure of the pendulum's tip** (end of the last link), read like an IMU at the tip. Actuation is a **torque at the base joint** (pendubot-style for n ≥ 2), with a **cart-on-rail fallback** switchable by config for n = 3/4, since one base torque on a triple or quadruple chain may be beyond what RL can do here.

Target folder: `/Users/abhilash/Projects/Personal_Projects/Control-Problems-with-Learned-Controllers/Inverted-Pendulum/` (new subfolder of an existing, freshly created git repo on `main`; nothing else in it yet).

Machine: Apple Silicon, miniconda (base is Python 3.14, which is too new for some wheels). MuJoCo, PyTorch and Gymnasium are not installed.

## Emphasis
This is a learning project, so math and understanding come before tooling. Every component gets a derivation in `notes/` and, where possible, an analytical check in `tests/` (hand-derived linearization vs. MuJoCo, energy balance, Riccati/LQR as the exactly solvable RL problem). RL results are read against the analytical baselines: does the policy pump energy, does it match the LQR gain near upright, does the critic match $x^\top P x$. Infrastructure (logging, sweeps, videos) stays minimal.

## Layout
```
Inverted-Pendulum/
  environment.yml           # conda env "pendulum-rl", python 3.12
  pyproject.toml            # editable install of pendulum_rl
  README.md
  pendulum_rl/
    envs/model.py           # MJCF generator: n links, actuation, tip site + sensors
    envs/chain_env.py       # Gymnasium env PendulumChainEnv
    control/lqr.py          # linearize at upright (mujoco.mjd_transitionFD) + discrete LQR
    control/energy_swingup.py   # energy pumping + LQR switch (n=1 baseline)
    algos/                  # single-file, CleanRL-style PyTorch implementations
      reinforce.py a2c.py ppo.py dqn.py ddpg.py td3.py sac.py
    common/                 # networks, replay buffer, running obs normalization, eval/video
  scripts/train.py          # --algo --n-links --actuation --obs-mode --seed ...
  scripts/evaluate.py       # metrics + mp4 rollout via mujoco.Renderer
  scripts/plot.py           # learning curves over seeds (mean ± spread)
  notes/                    # one markdown note per method: derivation + findings
  tests/
```

## Environment design (`PendulumChainEnv`)
- **Config**: `n_links ∈ {1,2,3,4}`, `actuation ∈ {"base_torque","cart"}`, `obs_mode ∈ {"joints","joints+tip"}`, torque/force limit, link length/mass, control rate.
- **Model**: generated MJCF string. Capsule links with hinge joints in a plane, light joint damping, a motor on the base hinge (or on a slider for the cart), and a `site` at the tip. Sim dt 2 ms with a frame skip to 50 Hz control. The n = 1 torque limit is set below `m·g·l` so the policy has to pump energy rather than lift straight up.
- **Tip orientation** is read from MuJoCo sensors on the tip site: `framezaxis` gives the absolute tilt angle φ_tip, encoded as (sin, cos), and `frameangvel` gives the tip's angular rate. So it's a measured quantity, not computed from the joints. For n = 1 it duplicates the joint angle; it carries new information in world frame from n ≥ 2.
- **Observation**: `joints` = sin/cos of each joint angle and the joint velocities (plus cart position and velocity for the cart). `joints+tip` = the same plus the tip (sin φ, cos φ, ω).
- **Start state**: hanging down plus small noise. Fixed-length episodes (~10 s), with no early termination for the base torque version. For the cart, leaving the rail is penalized and terminates the episode.
- **Reward**: an uprightness term (mean of cos of each link's absolute angle, or normalized tip height) minus small control and velocity penalties. It is kept in one function so you can experiment with reward shaping.

## Baselines (ground truth for comparisons)
- **LQR** around upright from MuJoCo's own finite-difference linearization. It works for every n and both actuations, and shows whether the system is controllable and what the best local controller is.
- **Energy-based swing-up and LQR switch** for n = 1, a classical full solution to compare the learned policies against.

## Algorithms (order follows the conceptual progression)
1. PPO and SAC first, as the reference implementations that get n = 1 working.
2. Then REINFORCE, A2C, DQN (with discretized torque, n = 1 only), DDPG and TD3, each added as its own file together with a `notes/` derivation.
3. Model-based RL (MPC on a learned model) is optional and comes later.

They share an obs normalizer, MLP networks and an evaluation and video helper. Training logs go to TensorBoard. Runs are keyed by (algo, n, actuation, obs_mode, seed).

## Milestones
- **M0, setup.** `environment.yml` (mujoco, gymnasium, torch, tensorboard, imageio[ffmpeg], matplotlib, pytest), conda env, editable install.
- **M1, n = 1 env and baselines.** Env, tests, LQR, energy swing-up, rendered video.
- **M2, first RL results on n = 1.** PPO and SAC on n = 1 in both obs modes, 5 seeds each, plus the learning-curve plot. This is the end of the first implementation pass.
- **M3, double pendulum.** n = 2 (pendubot), comparing obs modes and algorithms.
- **M4, triple and quadruple.** n = 3 and 4 with base torque, falling back to the cart if training stalls.
- The remaining algorithms are added during M2–M4.

## Verification
- `pytest`: Gymnasium `check_env` passes for every (n, actuation, obs_mode). The tip angle from the sensors equals the sum of the joint angles (up to wrapping). With zero torque, the model swings without gaining energy beyond damping. LQR holds n = 1…4 upright from a small perturbation in simulation.
- The energy swing-up baseline brings n = 1 upright and holds it, checked in a rendered mp4.
- `scripts/train.py --algo ppo --n-links 1` and `--algo sac` each reach a stable upright hold across ≥ 4 of 5 seeds. `scripts/evaluate.py` reports the fraction of time upright and writes a video. `scripts/plot.py` produces the comparison curves.
