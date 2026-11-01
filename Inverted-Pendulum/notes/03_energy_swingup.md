# 03 · Energy-based swing-up (n = 1)

## The idea (Åström & Furuta, 2000)
Don't track a trajectory. Drive the **energy** to the value of the upright equilibrium, $E^* = m g l_c$. The homoclinic orbit $E = E^*$ passes through upright, so once the energy is right the pendulum coasts towards the top, where LQR catches it.

## Lyapunov derivation
$$E = \tfrac12 I_p\dot\theta^2 + m g l_c\cos\theta, \qquad \dot E = \dot\theta(\tau - b\dot\theta).$$

Take $V = \tfrac12 (E - E^*)^2$. Then $\dot V = (E-E^*)\,\dot\theta(\tau - b\dot\theta)$. Choosing

$$\tau = k\,(E^* - E)\,\dot\theta + b\dot\theta \quad\Rightarrow\quad \dot V = -k (E-E^*)^2\dot\theta^2 \le 0.$$

$\dot V = 0$ only where $\dot\theta = 0$ or $E = E^*$. By LaSalle, trajectories converge to the largest invariant set inside it, which (away from hanging at rest) is the orbit $E = E^*$. Hanging at rest is the one bad equilibrium: $\dot\theta = 0$ gives no pumping direction, which is why the controller gives a kick there.

## Saturation
With $|\tau|\le \tau_{\max}$ the law becomes $\tau = \operatorname{sat}(\cdot)$; the sign of $\dot V$ is unchanged because saturation keeps the sign of $\tau - b\dot\theta$ when $b$ is small. Far from $E^*$ it is bang-bang, $\tau = \tau_{\max}\operatorname{sign}((E^*-E)\dot\theta)$: push in the direction of motion. The energy gained per half swing is at most $2\tau_{\max}\cdot(\text{swing amplitude})$, which gives a lower bound on the number of swings (worth computing with the defaults).

## Switching
`EnergySwingUp` switches to LQR when $|\theta| < 0.3$ rad and back to pumping when $|\theta| > 0.8$ rad (hysteresis). A cleaner rule would use the LQR region of attraction estimate $\{x : x^\top P x < c\}$ with $c$ chosen so that $|Kx| \le u_{\max}$ inside it: try it.

## Why this matters for RL
- A learned policy should rediscover pumping. Plot the action against $(E^* - E)\dot\theta$ along a learned trajectory: a good policy will be roughly a saturated monotone function of it.
- For $n\ge2$ there is no equally simple law. $\dot E = \dot q_1\tau$ still holds, so energy can still be pumped, but reaching the right energy does not put the chain on a path to upright: the passive links also have to arrive in the right configuration. This is where learned controllers become interesting.
