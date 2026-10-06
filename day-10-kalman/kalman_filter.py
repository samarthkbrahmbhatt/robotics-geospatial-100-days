"""
Day 10: Kalman filter for 2D robot tracking.

A robot drives a curved path. Two sensors watch it, both bad in
different ways:

  GPS       position, accurate on average but noisy, and only 1 Hz
  Odometry  velocity, smooth and fast at 10 Hz, but with a bias that
            makes dead reckoning drift without limit

Neither is good enough alone. The filter fuses them, and the interesting
part is what happens during a GPS outage, when it has nothing left but
the drifting sensor and its own model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------- Setup ----------------
DT = 0.1
T_END = 60.0
GPS_EVERY = 10              # steps, so GPS arrives at 1 Hz
GPS_NOISE = 3.0             # metres, standard deviation
ODO_NOISE = 0.25            # m/s, standard deviation
ODO_BIAS = np.array([0.08, -0.05])   # m/s, constant and unknown to the filter
OUTAGE = (20.0, 32.0)       # seconds with no GPS at all
SEED = 3

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def true_path(t):
    """A smooth curving trajectory, so constant velocity is only ever
    approximately right. That mismatch is what process noise represents."""
    x = 0.35 * t ** 1.25
    y = 18 * np.sin(0.09 * t)
    vx = 0.35 * 1.25 * np.maximum(t, 1e-6) ** 0.25
    vy = 18 * 0.09 * np.cos(0.09 * t)
    return np.array([x, y]), np.array([vx, vy])


def run():
    rng = np.random.default_rng(SEED)
    steps = int(T_END / DT)
    times = np.arange(steps) * DT

    # State is [x, y, vx, vy]: constant velocity model
    F = np.eye(4)
    F[0, 2] = F[1, 3] = DT

    # Process noise: how much the constant velocity assumption is allowed to be wrong
    q = 0.35
    G = np.array([[DT ** 2 / 2, 0], [0, DT ** 2 / 2], [DT, 0], [0, DT]])
    Q = G @ (q ** 2 * np.eye(2)) @ G.T

    H_pos = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    H_vel = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    R_pos = GPS_NOISE ** 2 * np.eye(2)
    R_vel = ODO_NOISE ** 2 * np.eye(2)

    x = np.array([0.0, 0.0, 0.0, 1.6])          # initial guess
    P = np.diag([25.0, 25.0, 4.0, 4.0])

    truth, fused, dead_reckoned, gps_fixes = [], [], [], []
    sigmas, errors, gps_available = [], [], []
    dr = np.array([0.0, 0.0])

    for k, t in enumerate(times):
        pos_true, vel_true = true_path(t)

        # --- Sensors ---
        odo = vel_true + ODO_BIAS + rng.normal(0, ODO_NOISE, 2)
        dr = dr + odo * DT                       # dead reckoning: just integrate
        have_gps = (k % GPS_EVERY == 0) and not (OUTAGE[0] <= t < OUTAGE[1])

        # --- Predict ---
        x = F @ x
        P = F @ P @ F.T + Q

        # --- Update with odometry, every step ---
        for H, R, z in ((H_vel, R_vel, odo),):
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P

        # --- Update with GPS, when it arrives ---
        if have_gps:
            z = pos_true + rng.normal(0, GPS_NOISE, 2)
            gps_fixes.append((t, z))
            y = z - H_pos @ x
            S = H_pos @ P @ H_pos.T + R_pos
            K = P @ H_pos.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H_pos) @ P

        truth.append(pos_true)
        fused.append(x[:2].copy())
        dead_reckoned.append(dr.copy())
        sigmas.append(np.sqrt(np.trace(P[:2, :2]) / 2))
        errors.append(np.linalg.norm(x[:2] - pos_true))
        gps_available.append(have_gps)

    return (times, np.array(truth), np.array(fused), np.array(dead_reckoned),
            gps_fixes, np.array(sigmas), np.array(errors), np.array(gps_available))


def main():
    (times, truth, fused, dr, gps_fixes, sigmas, errors, had_gps) = run()

    gps_t = np.array([t for t, _ in gps_fixes])
    gps_xy = np.array([z for _, z in gps_fixes])
    gps_error = np.linalg.norm(gps_xy - np.array([true_path(t)[0] for t in gps_t]), axis=1)
    dr_error = np.linalg.norm(dr - truth, axis=1)

    print(f"Run: {T_END:.0f} s, GPS at {1 / (GPS_EVERY * DT):.0f} Hz "
          f"(sigma {GPS_NOISE} m), odometry at {1 / DT:.0f} Hz "
          f"(sigma {ODO_NOISE} m/s, unknown bias {ODO_BIAS})")
    print(f"GPS outage from {OUTAGE[0]:.0f} s to {OUTAGE[1]:.0f} s\n")

    print(f"{'Estimator':<28} {'RMSE':>8} {'Final error':>12}")
    print(f"{'GPS alone (at fix times)':<28} {np.sqrt(np.mean(gps_error ** 2)):8.2f} m "
          f"{gps_error[-1]:10.2f} m")
    print(f"{'Dead reckoning (odometry)':<28} {np.sqrt(np.mean(dr_error ** 2)):8.2f} m "
          f"{dr_error[-1]:10.2f} m")
    print(f"{'Kalman filter (both)':<28} {np.sqrt(np.mean(errors ** 2)):8.2f} m "
          f"{errors[-1]:10.2f} m")

    during = (times >= OUTAGE[0]) & (times < OUTAGE[1])
    print(f"\nDuring the {OUTAGE[1] - OUTAGE[0]:.0f} s GPS outage:")
    print(f"  Filter error grew from {errors[np.argmax(times >= OUTAGE[0])]:.2f} m "
          f"to {errors[during][-1]:.2f} m")
    print(f"  Its own reported uncertainty grew from "
          f"{sigmas[np.argmax(times >= OUTAGE[0])]:.2f} m to {sigmas[during][-1]:.2f} m")
    after = np.argmax(times >= OUTAGE[1])
    print(f"  Within 2 s of GPS returning: error {errors[after + 20]:.2f} m, "
          f"uncertainty {sigmas[after + 20]:.2f} m")

    # Consistency check: for a 2D Gaussian, 39.3% of samples should fall
    # inside the 1-sigma ellipse. Far from that means the filter is lying
    # about how confident it is.
    inside = np.mean(errors < sigmas * np.sqrt(2))
    print(f"\nConsistency: {100 * inside:.1f}% of errors fall inside the filter's "
          f"own 1-sigma ellipse (39.3% expected)")
    if inside > 0.7:
        print("  Higher than expected, so the filter is overstating its uncertainty.")
    elif inside < 0.2:
        print("  Lower than expected, so the filter is overconfident.")

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(truth[:, 0], truth[:, 1], "k-", linewidth=2, label="True path")
    ax.plot(dr[:, 0], dr[:, 1], color="tab:orange", linewidth=1.5, label="Dead reckoning")
    ax.scatter(gps_xy[:, 0], gps_xy[:, 1], s=14, color="tab:red", alpha=0.5, label="GPS fixes")
    ax.plot(fused[:, 0], fused[:, 1], color="tab:blue", linewidth=1.8, label="Kalman filter")
    ax.set_title("Trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, np.linalg.norm(dr - truth, axis=1), color="tab:orange", label="Dead reckoning")
    ax.plot(gps_t, gps_error, ".", color="tab:red", alpha=0.5, label="GPS fixes")
    ax.plot(times, errors, color="tab:blue", label="Kalman filter")
    ax.axvspan(*OUTAGE, color="gray", alpha=0.2, label="GPS outage")
    ax.set_title("Position error over time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(times, errors, color="tab:blue", label="Actual error")
    ax.plot(times, sigmas, color="tab:purple", linestyle="--", label="Filter's own 1-sigma")
    ax.axvspan(*OUTAGE, color="gray", alpha=0.2, label="GPS outage")
    ax.set_title("Does the filter know how wrong it is?")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Metres")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    window = (times > OUTAGE[0] - 4) & (times < OUTAGE[1] + 6)
    ax.plot(truth[window, 0], truth[window, 1], "k-", linewidth=2, label="True path")
    ax.plot(fused[window, 0], fused[window, 1], color="tab:blue", linewidth=2, label="Kalman filter")
    in_window = (gps_t > OUTAGE[0] - 4) & (gps_t < OUTAGE[1] + 6)
    ax.scatter(gps_xy[in_window, 0], gps_xy[in_window, 1], s=25, color="tab:red",
               alpha=0.6, label="GPS fixes")
    ax.set_title("Zoom on the outage and the recovery")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Day 10: Kalman filter fusing noisy GPS with drifting odometry", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "kalman_filter.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
