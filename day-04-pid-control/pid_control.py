"""
Day 04: PID control of a mass-damper system.

Plant:  m * x'' + b * x' = F        (a mass sliding with friction)
Goal:   drive position x to a setpoint using a PID controller.

Covers:
  - what each of the three terms actually does on this plant
  - step response metrics (rise time, overshoot, settling, steady-state error)
  - integral windup when the actuator saturates, and how to stop it
  - disturbance rejection, which is where the integral term earns its place
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------- Plant and simulation ----------------
MASS = 1.0          # kg
DAMPING = 0.5       # N per m/s
DT = 0.01           # s
T_END = 12.0        # s
SETPOINT = 1.0      # m
FORCE_LIMIT = 20.0  # N, comfortably more than the controller needs

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


class PID:
    """Textbook PID with two practical details.

    1. Derivative on measurement, not on error. Differentiating a step
       change in the setpoint produces a huge spike ("derivative kick"),
       which real actuators hate.
    2. Optional anti-windup: stop integrating while the output is stuck
       against its limit, otherwise the integral keeps growing against a
       constraint it cannot beat and has to be "unwound" later.
    """

    def __init__(self, kp, ki, kd, limit=FORCE_LIMIT, anti_windup=True):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.anti_windup = anti_windup
        self.integral = 0.0
        self.prev_measurement = 0.0

    def step(self, setpoint, measurement, dt):
        error = setpoint - measurement

        derivative = -(measurement - self.prev_measurement) / dt
        self.prev_measurement = measurement

        unclamped = (self.kp * error
                     + self.ki * (self.integral + error * dt)
                     + self.kd * derivative)
        output = float(np.clip(unclamped, -self.limit, self.limit))

        pushing_further_into_limit = (unclamped != output
                                      and np.sign(error) == np.sign(unclamped))
        if not (self.anti_windup and pushing_further_into_limit):
            self.integral += error * dt

        return output


def simulate(kp, ki, kd, limit=FORCE_LIMIT, anti_windup=True,
             disturbance=0.0, disturbance_at=None):
    """Run the closed loop. Returns time, position, control force."""
    n = int(T_END / DT)
    t = np.arange(n) * DT
    x = np.zeros(n)
    v = 0.0
    u = np.zeros(n)

    pid = PID(kp, ki, kd, limit=limit, anti_windup=anti_windup)

    for i in range(1, n):
        force = pid.step(SETPOINT, x[i - 1], DT)
        u[i] = force

        applied = force
        if disturbance_at is not None and t[i] >= disturbance_at:
            applied += disturbance

        accel = (applied - DAMPING * v) / MASS
        v += accel * DT
        x[i] = x[i - 1] + v * DT

    return t, x, u


def metrics(t, x, setpoint=SETPOINT):
    """Rise time (10-90%), overshoot %, 2% settling time, steady-state error."""
    i10 = np.argmax(x >= 0.1 * setpoint) if (x >= 0.1 * setpoint).any() else 0
    i90 = np.argmax(x >= 0.9 * setpoint) if (x >= 0.9 * setpoint).any() else 0
    rise = t[i90] - t[i10] if i90 > i10 else np.nan

    overshoot = max(100 * (x.max() - setpoint) / setpoint, 0.0)

    outside = np.where(np.abs(x - setpoint) > 0.02 * setpoint)[0]
    settling = t[outside[-1]] if len(outside) and outside[-1] < len(t) - 1 else np.nan

    return rise, overshoot, settling, setpoint - x[-1]


def fmt(value, unit="s"):
    return "  n/a " if np.isnan(value) else f"{value:5.2f}{unit}"


def print_row(name, t, x):
    rise, over, settle, sse = metrics(t, x)
    print(f"  {name:<28} rise {fmt(rise)}   overshoot {over:6.1f}%   "
          f"settle {fmt(settle)}   final error {sse:+.3f} m")


def main():
    print(f"Plant: {MASS} kg mass, damping {DAMPING} N/(m/s), actuator limit ±{FORCE_LIMIT} N")
    print(f"Target: move to {SETPOINT} m\n")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # --- Panel 1: what each term contributes ---
    print("Controller comparison:")
    ax = axes[0, 0]
    for name, kp, ki, kd in (("P   (Kp=10)", 10, 0, 0),
                             ("PD  (Kp=10, Kd=6)", 10, 0, 6),
                             ("PID (Kp=10, Ki=8, Kd=6)", 10, 8, 6)):
        t, x, _ = simulate(kp, ki, kd)
        ax.plot(t, x, label=name)
        print_row(name, t, x)
    ax.axhline(SETPOINT, color="k", linestyle="--", linewidth=1, label="Setpoint")
    ax.set_title("P alone rings. Adding D damps it.")
    ax.legend(fontsize=8)

    # --- Panel 2: sweeping Kp on a PD controller ---
    print("\nRaising Kp with Kd=6, no integral:")
    ax = axes[0, 1]
    for kp in (2, 10, 40, 120):
        t, x, _ = simulate(kp, 0, 6)
        ax.plot(t, x, label=f"Kp={kp}")
        print_row(f"Kp={kp}", t, x)
    ax.axhline(SETPOINT, color="k", linestyle="--", linewidth=1)
    ax.set_title("Raising Kp: faster, then it starts to overshoot")
    ax.legend(fontsize=8)

    # --- Panel 3: integral windup against a tight actuator limit ---
    print("\nAggressive integral (Ki=20) against a tight ±3 N limit:")
    ax = axes[1, 0]
    for anti, name in ((False, "No anti-windup"), (True, "With anti-windup")):
        t, x, u = simulate(10, 20, 6, limit=3.0, anti_windup=anti)
        ax.plot(t, x, label=name)
        print_row(name, t, x)
    ax.axhline(SETPOINT, color="k", linestyle="--", linewidth=1)
    ax.set_title("Integral windup when the actuator cannot keep up")
    ax.legend(fontsize=8)

    # --- Panel 4: disturbance rejection ---
    print("\nConstant 4 N push applied at t=6s:")
    ax = axes[1, 1]
    for name, kp, ki, kd in (("PD  (no integral)", 10, 0, 6),
                             ("PID (Ki=8)", 10, 8, 6)):
        t, x, _ = simulate(kp, ki, kd, disturbance=-4.0, disturbance_at=6.0)
        ax.plot(t, x, label=name)
        print(f"  {name:<28} error 6 s after the push: {SETPOINT - x[-1]:+.3f} m")
    ax.axvline(6.0, color="tab:red", linestyle=":", label="Disturbance starts")
    ax.axhline(SETPOINT, color="k", linestyle="--", linewidth=1)
    ax.set_title("Only the integral term removes a steady offset")
    ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.grid(alpha=0.3)

    fig.suptitle("Day 04: PID control of a mass-damper system", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "pid_control.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
