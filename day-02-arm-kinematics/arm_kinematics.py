"""
Day 02: Forward and inverse kinematics of a 2-link planar robot arm.

Forward kinematics (FK): joint angles -> end effector position.
Inverse kinematics (IK): desired position -> joint angles.

Outputs:
  outputs/kinematics.png  - workspace, elbow up/down solutions, joint angles
  outputs/arm_tracing.gif - the arm tracing a circle
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Robot definition ----------------
L1 = 1.0   # length of link 1 (metres)
L2 = 0.8   # length of link 2 (metres)

# Circle the end effector will trace
CIRCLE_CENTRE = (0.9, 0.4)
CIRCLE_RADIUS = 0.35
N_FRAMES = 120

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def forward_kinematics(t1, t2, l1=L1, l2=L2):
    """Joint angles (radians) -> (elbow position, end effector position)."""
    elbow = np.array([l1 * np.cos(t1),
                      l1 * np.sin(t1)])
    end = elbow + np.array([l2 * np.cos(t1 + t2),
                            l2 * np.sin(t1 + t2)])
    return elbow, end


def inverse_kinematics(x, y, l1=L1, l2=L2, elbow_up=True):
    """Target position -> joint angles (radians).

    Uses the law of cosines. Two solutions exist for most reachable
    points: elbow up and elbow down. Raises if the point is outside
    the annulus the arm can reach.
    """
    r = np.hypot(x, y)
    if r > l1 + l2:
        raise ValueError(f"Target {x:.2f}, {y:.2f} is too far away (r={r:.2f} > {l1 + l2:.2f})")
    if r < abs(l1 - l2):
        raise ValueError(f"Target {x:.2f}, {y:.2f} is inside the dead zone (r={r:.2f} < {abs(l1 - l2):.2f})")

    # Angle at the elbow joint
    cos_t2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_t2 = np.clip(cos_t2, -1.0, 1.0)      # guard against tiny float overshoot
    t2 = -np.arccos(cos_t2) if elbow_up else np.arccos(cos_t2)

    # Angle at the shoulder joint
    t1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(t2), l1 + l2 * np.cos(t2))
    return t1, t2


def check_round_trip():
    """IK then FK should return the point we asked for."""
    targets = [(1.2, 0.5), (0.3, 1.1), (-0.6, 0.9), (0.9, -0.4)]
    worst = 0.0
    for x, y in targets:
        for elbow_up in (True, False):
            t1, t2 = inverse_kinematics(x, y, elbow_up=elbow_up)
            _, end = forward_kinematics(t1, t2)
            worst = max(worst, np.linalg.norm(end - np.array([x, y])))
    print(f"Worst round-trip error over {len(targets)} targets: {worst:.2e} m")
    return worst


def draw_arm(ax, t1, t2, colour="tab:blue", alpha=1.0, label=None):
    elbow, end = forward_kinematics(t1, t2)
    ax.plot([0, elbow[0], end[0]], [0, elbow[1], end[1]],
            "-o", color=colour, linewidth=3, markersize=7, alpha=alpha, label=label)
    return end


def plot_static():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # --- Panel 1: reachable workspace ---
    ax = axes[0]
    angles = np.linspace(0, 2 * np.pi, 300)
    ax.fill(np.cos(angles) * (L1 + L2), np.sin(angles) * (L1 + L2),
            color="tab:green", alpha=0.12, label="Reachable")
    ax.fill(np.cos(angles) * abs(L1 - L2), np.sin(angles) * abs(L1 - L2),
            color="white")
    ax.plot(np.cos(angles) * abs(L1 - L2), np.sin(angles) * abs(L1 - L2),
            "--", color="tab:red", label="Dead zone")
    for t1 in np.linspace(0, np.pi / 2, 4):
        draw_arm(ax, t1, -0.9, colour="tab:blue", alpha=0.45)
    ax.set_title(f"Workspace (L1={L1}, L2={L2})")
    ax.legend(loc="lower left", fontsize=8)

    # --- Panel 2: two IK solutions for one target ---
    ax = axes[1]
    target = (1.1, 0.6)
    for elbow_up, colour in ((True, "tab:blue"), (False, "tab:orange")):
        t1, t2 = inverse_kinematics(*target, elbow_up=elbow_up)
        draw_arm(ax, t1, t2, colour=colour,
                 label=f"{'Elbow up' if elbow_up else 'Elbow down'}: "
                       f"θ1={np.degrees(t1):.0f}°, θ2={np.degrees(t2):.0f}°")
    ax.plot(*target, "r*", markersize=16, label="Target", zorder=5)
    ax.set_title("Same target, two valid solutions")
    ax.legend(loc="lower left", fontsize=8)

    # --- Panel 3: joint angles while tracing the circle ---
    t1s, t2s = solve_circle()
    ax = axes[2]
    step = np.arange(N_FRAMES)
    ax.plot(step, np.degrees(t1s), label="θ1 (shoulder)")
    ax.plot(step, np.degrees(t2s), label="θ2 (elbow)")
    ax.set_xlabel("Step along circle")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Joint angles while tracing a circle")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    for ax in axes[:2]:
        ax.set_aspect("equal")
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-2.1, 2.1)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)

    fig.suptitle("Day 02: 2-link planar arm kinematics", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "kinematics.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


def solve_circle():
    """IK for every point on the circle. Returns arrays of joint angles."""
    phase = np.linspace(0, 2 * np.pi, N_FRAMES)
    xs = CIRCLE_CENTRE[0] + CIRCLE_RADIUS * np.cos(phase)
    ys = CIRCLE_CENTRE[1] + CIRCLE_RADIUS * np.sin(phase)

    t1s, t2s = [], []
    for x, y in zip(xs, ys):
        t1, t2 = inverse_kinematics(x, y, elbow_up=True)
        t1s.append(t1)
        t2s.append(t2)
    return np.array(t1s), np.array(t2s)


def animate_circle():
    t1s, t2s = solve_circle()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylim(-0.9, 1.7)
    ax.grid(alpha=0.3)
    ax.set_title("End effector tracing a circle")

    phase = np.linspace(0, 2 * np.pi, 200)
    ax.plot(CIRCLE_CENTRE[0] + CIRCLE_RADIUS * np.cos(phase),
            CIRCLE_CENTRE[1] + CIRCLE_RADIUS * np.sin(phase),
            "--", color="tab:red", linewidth=1, label="Target path")

    arm_line, = ax.plot([], [], "-o", color="tab:blue", linewidth=3, markersize=8)
    trail, = ax.plot([], [], ".", color="tab:green", markersize=3, label="Actual path")
    ax.legend(loc="upper left", fontsize=8)

    trail_x, trail_y = [], []

    def update(i):
        elbow, end = forward_kinematics(t1s[i], t2s[i])
        arm_line.set_data([0, elbow[0], end[0]], [0, elbow[1], end[1]])
        trail_x.append(end[0])
        trail_y.append(end[1])
        trail.set_data(trail_x, trail_y)
        return arm_line, trail

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=50, blit=True)
    out = OUT_DIR / "arm_tracing.gif"
    anim.save(out, writer=animation.PillowWriter(fps=20), dpi=90)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print(f"2-link arm: L1={L1} m, L2={L2} m")
    print(f"Reach: {abs(L1 - L2):.2f} m to {L1 + L2:.2f} m from the base")

    check_round_trip()

    # Show that unreachable targets are caught rather than silently wrong
    try:
        inverse_kinematics(2.5, 0.0)
    except ValueError as e:
        print(f"Correctly rejected: {e}")

    plot_static()
    animate_circle()


if __name__ == "__main__":
    main()
