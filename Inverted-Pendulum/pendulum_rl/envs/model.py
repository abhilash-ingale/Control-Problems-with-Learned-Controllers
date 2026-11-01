"""MJCF generator for an n-link planar pendulum chain.

Conventions (used everywhere else in the package):
- The chain lives in the world x-z plane; every hinge rotates about the world y axis.
- Links are defined pointing up (+z), so qpos = 0 is the upright equilibrium and
  hanging down is q_1 = pi, q_2..q_n = 0.
- Joint angles q_i are *relative* (link i w.r.t. link i-1). The absolute angle of
  link k from vertical is phi_k = q_1 + ... + q_k.
- Actuation: either a torque motor on the base hinge ("base_torque") or a force
  motor on a horizontal slider carrying the chain ("cart"). Only one actuator,
  so for n >= 2 the system is underactuated.
- A site "tip" sits at the free end of the last link and carries IMU-like
  sensors (framezaxis, frameangvel, framepos).
"""

from __future__ import annotations

from dataclasses import dataclass

ACTUATIONS = ("base_torque", "cart")

# Default actuator limits. For n = 1 the torque limit is well below the peak
# gravity torque m*g*l/2 (about 2.45 N m with the defaults), so the policy must
# pump energy instead of lifting the link straight up.
DEFAULT_MAX_TORQUE = {1: 1.0, 2: 6.0, 3: 12.0, 4: 20.0}
DEFAULT_MAX_FORCE = {1: 20.0, 2: 40.0, 3: 60.0, 4: 80.0}


@dataclass(frozen=True)
class ChainConfig:
    n_links: int = 1
    actuation: str = "base_torque"
    link_length: float = 0.5
    link_mass: float = 1.0
    link_radius: float = 0.02
    joint_damping: float = 0.01
    max_torque: float | None = None  # base_torque limit [N m]
    max_force: float | None = None  # cart force limit [N]
    cart_mass: float = 1.0
    rail_limit: float = 2.0  # |x| beyond this ends a cart episode
    timestep: float = 0.002
    gravity: float = 9.81

    def __post_init__(self):
        if not 1 <= self.n_links <= 4:
            raise ValueError("n_links must be in 1..4")
        if self.actuation not in ACTUATIONS:
            raise ValueError(f"actuation must be one of {ACTUATIONS}")

    @property
    def ctrl_limit(self) -> float:
        if self.actuation == "base_torque":
            return self.max_torque if self.max_torque is not None else DEFAULT_MAX_TORQUE[self.n_links]
        return self.max_force if self.max_force is not None else DEFAULT_MAX_FORCE[self.n_links]


def build_mjcf(cfg: ChainConfig) -> str:
    """Return the MJCF XML string for the configured chain."""
    L, r, m = cfg.link_length, cfg.link_radius, cfg.link_mass

    # Nested link bodies; body i is placed at the end of link i-1.
    links = ""
    for i in reversed(range(1, cfg.n_links + 1)):
        pos = "0 0 0" if i == 1 else f"0 0 {L}"
        tip = f'<site name="tip" pos="0 0 {L}" size="{1.5 * r}"/>' if i == cfg.n_links else ""
        links = f"""
        <body name="link{i}" pos="{pos}">
          <joint name="hinge{i}" type="hinge" axis="0 1 0" damping="{cfg.joint_damping}"/>
          <geom name="geom{i}" type="capsule" fromto="0 0 0 0 0 {L}" size="{r}" mass="{m}" rgba="{_color(i)}"/>
          {tip}{links}
        </body>"""

    u = cfg.ctrl_limit
    if cfg.actuation == "base_torque":
        chain = f'<body name="base" pos="0 0 0">{links}\n      </body>'
        actuator = f'<motor name="act" joint="hinge1" gear="1" ctrllimited="true" ctrlrange="{-u} {u}"/>'
    else:
        # Slider range is a bit wider than rail_limit; the env terminates at rail_limit.
        chain = f"""<body name="cart" pos="0 0 0">
        <joint name="slider" type="slide" axis="1 0 0" damping="{cfg.joint_damping}"
               limited="true" range="{-1.25 * cfg.rail_limit} {1.25 * cfg.rail_limit}"/>
        <geom name="cart" type="box" size="0.1 0.06 0.04" mass="{cfg.cart_mass}" rgba="0.3 0.3 0.3 1"/>{links}
      </body>"""
        actuator = f'<motor name="act" joint="slider" gear="1" ctrllimited="true" ctrlrange="{-u} {u}"/>'

    height = cfg.n_links * L
    # Camera distance so the full chain (and the rail, for the cart) fits a 45 deg field of view.
    cam_dist = max(1.0 + 2.2 * height, 3.8 if cfg.actuation == "cart" else 0.0)
    return f"""<mujoco model="pendulum_chain_n{cfg.n_links}_{cfg.actuation}">
  <compiler angle="radian"/>
  <option timestep="{cfg.timestep}" gravity="0 0 {-cfg.gravity}" integrator="RK4">
    <flag contact="disable" energy="enable"/>
  </option>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.95 0.95 0.97" rgb2="0.7 0.74 0.8" width="32" height="32"/>
  </asset>
  <worldbody>
    <light pos="0 -3 {2 * height + 1}" dir="0 1 -1"/>
    <geom name="rail" type="box" pos="0 0 -0.05" size="{cfg.rail_limit if cfg.actuation == 'cart' else 0.15} 0.02 0.01"
          rgba="0.6 0.6 0.6 1" contype="0" conaffinity="0"/>
    <camera name="side" pos="0 {-cam_dist} 0" xyaxes="1 0 0 0 0 1"/>
    {chain}
  </worldbody>
  <actuator>
    {actuator}
  </actuator>
  <sensor>
    <framezaxis name="tip_zaxis" objtype="site" objname="tip"/>
    <frameangvel name="tip_angvel" objtype="site" objname="tip"/>
    <framepos name="tip_pos" objtype="site" objname="tip"/>
  </sensor>
</mujoco>
"""


def _color(i: int) -> str:
    palette = ["0.85 0.35 0.2 1", "0.2 0.5 0.85 1", "0.3 0.7 0.3 1", "0.75 0.6 0.15 1"]
    return palette[(i - 1) % len(palette)]
