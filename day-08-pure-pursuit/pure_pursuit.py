"""
Day 08: Pure pursuit path following for a differential-drive robot.

Day 06 produced a path as a list of grid cells. A planner is allowed to
turn 45 degrees on the spot. A real robot is not: it has a forward speed,
a turn rate limit, and no way to teleport between cells.

This drives a unicycle model along a planner-style path with sharp
corners, using pure pursuit, and measures how far the robot actually
strays from the line it was told to follow.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------- Robot and simulation ----------------
V_MAX = 1.0          # m/s, constant cruise speed
OMEGA_MAX = 1.5      # rad/s, turn rate limit
DT = 0.05            # s
T_MAX = 120.0        # s, give up after this
GOAL_TOLERANCE = 0.25

LOOKAHEADS = [0.4, 1.0, 2.5]

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def reference_path():
    """A path shaped like a grid planner's output: straight runs and
    hard 90 and 45 degree corners, with no smoothing anywhere."""
    waypoints = [(0, 0), (6, 0), (6, 4), (10, 4), (10, 8),
                 (14, 8), (14, 4), (18, 4), (18, 0), (22, 0)]
    path = []
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        n = max(int(np.hypot(x1 - x0, y1 - y0) / 0.1), 1)
        for i in range(n):
            t = i / n
            path.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    path.append(waypoints[-1])
    return np.array(path)


def cross_track_error(point, path):
    """Shortest distance from the robot to the reference polyline.

    Measured against the line segments, not the sampled points, so the
    value does not depend on how finely the path happens to be sampled.
    """
    a = path[:-1]
    b = path[1:]
    ab = b - a
    ap = point - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.clip(np.einsum("ij,ij->i", ap, ab) / np.where(denom == 0, 1, denom), 0, 1)
    closest = a + t[:, None] * ab
    return np.min(np.linalg.norm(point - closest, axis=1))


def find_lookahead_point(pose, path, lookahead, start_idx):
    """First point on the path at least `lookahead` ahead of the robot.

    Searching forward from the last index stops the robot latching onto
    an earlier part of the path when the route doubles back on itself.
    """
    pos = pose[:2]
    for i in range(start_idx, len(path)):
        if np.linalg.norm(path[i] - pos) >= lookahead:
            return path[i], i
    return path[-1], len(path) - 1


def simulate(path, lookahead):
    pose = np.array([path[0][0], path[0][1], 0.0])     # x, y, heading
    trajectory, errors, omegas = [pose[:2].copy()], [0.0], [0.0]
    idx = 0
    steps = int(T_MAX / DT)

    for _ in range(steps):
        if np.linalg.norm(pose[:2] - path[-1]) < GOAL_TOLERANCE:
            break

        target, idx = find_lookahead_point(pose, path, lookahead, idx)

        # Target in the robot's own frame
        dx, dy = target - pose[:2]
        x_r = np.cos(-pose[2]) * dx - np.sin(-pose[2]) * dy
        y_r = np.sin(-pose[2]) * dx + np.cos(-pose[2]) * dy

        # Pure pursuit: the arc through the target has curvature 2y/L^2
        dist = max(np.hypot(x_r, y_r), 1e-6)
        curvature = 2 * y_r / (dist ** 2)
        omega = np.clip(V_MAX * curvature, -OMEGA_MAX, OMEGA_MAX)

        # Unicycle model
        pose[0] += V_MAX * np.cos(pose[2]) * DT
        pose[1] += V_MAX * np.sin(pose[2]) * DT
        pose[2] += omega * DT

        trajectory.append(pose[:2].copy())
        errors.append(cross_track_error(pose[:2], path))
        omegas.append(omega)

    reached = np.linalg.norm(pose[:2] - path[-1]) < GOAL_TOLERANCE
    return np.array(trajectory), np.array(errors), np.array(omegas), reached


def main():
    path = reference_path()
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    print(f"Reference path: {len(path)} points, {length:.1f} m long")
    print(f"Robot: v = {V_MAX} m/s, |omega| <= {OMEGA_MAX} rad/s\n")

    results = []
    for lookahead in LOOKAHEADS:
        traj, errors, omegas, reached = simulate(path, lookahead)
        results.append((lookahead, traj, errors, omegas, reached))
        saturated = 100 * np.mean(np.abs(omegas) >= OMEGA_MAX - 1e-9)
        print(f"Lookahead {lookahead:4.1f} m   "
              f"mean error {errors.mean():5.3f} m   max {errors.max():5.3f} m   "
              f"time {len(traj) * DT:5.1f} s   "
              f"mean |omega| {np.abs(omegas).mean():5.3f} rad/s   "
              f"at turn limit {saturated:4.1f}% of the time   "
              f"{'reached goal' if reached else 'DID NOT REACH GOAL'}")

    by_mean = min(results, key=lambda r: r[2].mean())
    by_max = min(results, key=lambda r: r[2].max())
    print(f"\nLowest mean error: L = {by_mean[0]} m")
    print(f"Lowest worst-case error: L = {by_max[0]} m")
    if by_mean[0] != by_max[0]:
        print("Those disagree, so there is no single best lookahead here: the short one "
              "hugs the straights but overshoots corners, the longer one is smoother "
              "but never gets tight to the path.")

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.5, label="Planned path")
    for lookahead, traj, _, _, _ in results:
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.8, label=f"L = {lookahead} m")
    ax.set_title("Planned path vs what the robot actually drove")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for lookahead, _, errors, _, _ in results:
        ax.plot(np.arange(len(errors)) * DT, errors, label=f"L = {lookahead} m")
    ax.set_title("Cross-track error over time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance from path (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for lookahead, _, _, omegas, _ in results:
        ax.plot(np.arange(len(omegas)) * DT, omegas, label=f"L = {lookahead} m")
    ax.axhline(OMEGA_MAX, color="red", linestyle=":", linewidth=1)
    ax.axhline(-OMEGA_MAX, color="red", linestyle=":", linewidth=1, label="Turn rate limit")
    ax.set_title("Commanded turn rate (red = actuator limit)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("omega (rad/s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    corner = (8, 20, 1, 10)                       # zoom on the tightest corners
    ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.5)
    for lookahead, traj, _, _, _ in results:
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, label=f"L = {lookahead} m")
    ax.set_xlim(corner[0], corner[1])
    ax.set_ylim(corner[2], corner[3])
    ax.set_title("Zoom: corner cutting grows with lookahead")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Day 08: Pure pursuit on a planner-style path", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "pure_pursuit.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
