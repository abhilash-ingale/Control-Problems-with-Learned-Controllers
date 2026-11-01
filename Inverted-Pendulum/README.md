# Inverted pendulum chain: swing-up with classical control and RL

An n-link (n = 1…4) planar pendulum in MuJoCo, swung up from hanging and balanced. The point is understanding: derive what can be derived, check it numerically, and use the analytical baselines (LQR, energy pumping) as ground truth when studying RL algorithms. See [PLAN.md](PLAN.md) for the full plan and milestones.

## Setup
```bash
conda env create -f environment.yml
conda activate pendulum-rl
pip install -e . --no-deps --no-build-isolation
pytest
```

## Layout
```
pendulum_rl/
  envs/model.py            MJCF generator (n links, base torque or cart, tip site + IMU-like sensors)
  envs/chain_env.py        PendulumChainEnv (Gymnasium), reward_fn
  control/lqr.py           MuJoCo linearization at upright, ZOH, DARE/LQR, PBH controllability
  control/energy_swingup.py  Åström–Furuta energy pumping + LQR catch (n = 1)
  common/evaluate.py       rollouts, metrics (fraction upright, time to upright), video
  algos/                   RL algorithms, one file each (M2+)
scripts/
  linear_analysis.py       open-loop poles, controllability margins, LQR gains for every config
  evaluate.py              run a controller over seeds, optional mp4
notes/                     the math: one note per topic / algorithm
tests/                     analytical checks (see below)
```

## What the tests check
- Gymnasium `check_env` for every (n, actuation, obs_mode).
- The tip sensor angle equals $\sum_i q_i$ and its rate equals $\sum_i \dot q_i$.
- Passive energy: conserved with zero damping, non-increasing with damping.
- MuJoCo's finite-difference linearization for n = 1 equals the exact discretization of the hand-derived model.
- Upright is controllable (PBH) and LQR holds it for every n and actuation.
- Energy swing-up brings n = 1 up and holds it.

## Try it
```bash
python scripts/linear_analysis.py
```
```bash
python scripts/evaluate.py --controller energy --n-links 1 --video runs/energy_n1.mp4
```
```bash
python scripts/evaluate.py --controller lqr --n-links 4 --actuation cart --init up
```

## Notes (reading order)
1. [Dynamics](notes/01_dynamics.md): manipulator equation, energy, linearization, controllability, what the tip observation adds.
2. [LQR](notes/02_lqr.md): Riccati by dynamic programming, and the LQR ↔ RL dictionary.
3. [Energy swing-up](notes/03_energy_swingup.md): Lyapunov/LaSalle derivation, saturation, switching.
4. [RL foundations](notes/04_rl_foundations.md): the MDP, discount vs. swing period, policy gradient theorems, algorithm map and the questions experiments should answer.
