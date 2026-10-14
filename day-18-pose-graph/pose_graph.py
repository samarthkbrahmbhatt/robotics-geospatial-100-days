"""
Day 18: Pose graph optimisation, and the one bad constraint that ruins it.

Day 16 ended by saying what was still missing. Scan matching gives the
motion between consecutive poses, but chaining those estimates means every
small error is carried forward forever, exactly like the dead reckoning in
Day 10. The trajectory drifts, and the map drawn from it does not close.

The fix is to notice when the robot returns somewhere it has been before,
and add a constraint saying "these two poses are the same place". That
single extra measurement is enough to correct the entire trajectory, not
just the end of it, because the whole graph is solved together.

The last section is the warning. One WRONG loop closure, saying two places
are the same when they are not, does more damage than all the odometry
drift it was meant to fix. That is why real systems use robust kernels
rather than trusting their place recognition.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------- Settings ----------------
ODO_TRANS_NOISE = 0.035           # metres per step
ODO_ROT_NOISE = 0.012             # radians per step
LOOP_TRANS_NOISE = 0.05
LOOP_ROT_NOISE = 0.02

MAX_ITERATIONS = 40
CONVERGENCE_TOLERANCE = 1e-8

SEED = 7
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# ---------------- SE(2) helpers ----------------
def wrap(angle):
    """Keep an angle in (-pi, pi]. Forgetting this is the classic way to
    make an optimiser take a 350 degree turn instead of a 10 degree one."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def v2t(pose):
    """(x, y, theta) to a 3x3 homogeneous transform."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([[c, -s, pose[0]],
                     [s,  c, pose[1]],
                     [0,  0, 1.0]])


def t2v(T):
    """3x3 homogeneous transform back to (x, y, theta)."""
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def relative_pose(a, b):
    """The motion from pose a to pose b, expressed in a's frame."""
    return t2v(np.linalg.inv(v2t(a)) @ v2t(b))


# ---------------- The error and its Jacobians ----------------
def constraint_error(xi, xj, z):
    """How far the current poses are from what the measurement says.

    The measurement z is a relative pose. The error is the difference
    between what the graph currently implies and what was measured,
    expressed as another relative pose.
    """
    e = t2v(np.linalg.inv(v2t(z)) @ np.linalg.inv(v2t(xi)) @ v2t(xj))
    e[2] = wrap(e[2])
    return e


def constraint_jacobians(xi, xj, z):
    """Derivatives of the error with respect to both poses.

    Written out analytically rather than by finite differences, because
    this runs for every constraint on every iteration. The self-test
    checks them against numerical derivatives, since a wrong Jacobian
    does not crash, it just quietly fails to converge.
    """
    ci, si = np.cos(xi[2]), np.sin(xi[2])
    cz, sz = np.cos(z[2]), np.sin(z[2])

    Ri = np.array([[ci, -si], [si, ci]])
    Rz = np.array([[cz, -sz], [sz, cz]])
    dRi = np.array([[-si, ci], [-ci, -si]])      # d(Ri^T)/d(theta_i)

    translation = (xj[:2] - xi[:2]).reshape(2, 1)

    A = np.zeros((3, 3))
    A[:2, :2] = -Rz.T @ Ri.T
    A[:2, 2:3] = Rz.T @ dRi @ translation
    A[2, 2] = -1.0

    B = np.zeros((3, 3))
    B[:2, :2] = Rz.T @ Ri.T
    B[2, 2] = 1.0

    return A, B


def optimise(poses, constraints, max_iterations=MAX_ITERATIONS,
             huber_delta=None):
    """Gauss-Newton on the whole graph.

    Every iteration builds the linear system H dx = -b from all the
    constraints and solves it in one go. That is why a single loop closure
    corrects the entire trajectory rather than just the poses near it: the
    correction propagates through H to every node.

    The first pose is held fixed. Without that the problem is
    under-determined, because sliding and rotating the whole map changes
    nothing about any relative measurement.

    huber_delta, if given, down-weights constraints with large errors,
    which is what stops one bad loop closure dominating the solution.
    """
    poses = poses.copy()
    n = len(poses)
    history = []

    for iteration in range(max_iterations):
        H = np.zeros((3 * n, 3 * n))
        b = np.zeros(3 * n)
        chi_squared = 0.0

        for i, j, z, information in constraints:
            e = constraint_error(poses[i], poses[j], z)
            A, B = constraint_jacobians(poses[i], poses[j], z)

            omega = information
            cost = float(e @ omega @ e)
            chi_squared += cost

            if huber_delta is not None and cost > huber_delta ** 2:
                # Huber: beyond the threshold, cost grows linearly rather
                # than quadratically, so an outlier stops dominating.
                omega = omega * (huber_delta / np.sqrt(cost))

            si, sj = 3 * i, 3 * j
            H[si:si + 3, si:si + 3] += A.T @ omega @ A
            H[si:si + 3, sj:sj + 3] += A.T @ omega @ B
            H[sj:sj + 3, si:si + 3] += B.T @ omega @ A
            H[sj:sj + 3, sj:sj + 3] += B.T @ omega @ B

            b[si:si + 3] += A.T @ omega @ e
            b[sj:sj + 3] += B.T @ omega @ e

        history.append(chi_squared)

        # Hold the first pose: the gauge freedom has to be removed or H
        # is singular and the solve is meaningless.
        H[:3, :3] += np.eye(3) * 1e6

        try:
            dx = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            print("  H is singular, stopping")
            break

        poses += dx.reshape(n, 3)
        poses[:, 2] = wrap(poses[:, 2])

        if np.linalg.norm(dx) < CONVERGENCE_TOLERANCE:
            break

    return poses, history


# ---------------- Building the problem ----------------
def true_trajectory():
    """A closed rectangular loop, so the robot ends where it started."""
    poses = [np.array([0.0, 0.0, 0.0])]
    legs = [(14, 0.6, 0.0), (1, 0.0, np.pi / 2),
            (10, 0.6, 0.0), (1, 0.0, np.pi / 2),
            (14, 0.6, 0.0), (1, 0.0, np.pi / 2),
            (10, 0.6, 0.0), (1, 0.0, np.pi / 2)]
    for count, step, turn in legs:
        for _ in range(count):
            current = poses[-1]
            theta = wrap(current[2] + turn)
            poses.append(np.array([current[0] + step * np.cos(theta),
                                   current[1] + step * np.sin(theta),
                                   theta]))
    return np.array(poses)


def build_graph(truth, rng, with_loop_closure=True, false_closure=None):
    """Noisy odometry constraints, plus an optional loop closure.

    Odometry is corrupted once and then integrated, so the drift is
    cumulative in exactly the way a real robot's is.
    """
    odometry_information = np.diag([1 / ODO_TRANS_NOISE ** 2,
                                    1 / ODO_TRANS_NOISE ** 2,
                                    1 / ODO_ROT_NOISE ** 2])
    loop_information = np.diag([1 / LOOP_TRANS_NOISE ** 2,
                                1 / LOOP_TRANS_NOISE ** 2,
                                1 / LOOP_ROT_NOISE ** 2])

    constraints = []
    estimate = [truth[0].copy()]

    for i in range(len(truth) - 1):
        z = relative_pose(truth[i], truth[i + 1])
        z = z + np.array([rng.normal(0, ODO_TRANS_NOISE),
                          rng.normal(0, ODO_TRANS_NOISE),
                          rng.normal(0, ODO_ROT_NOISE)])
        z[2] = wrap(z[2])
        constraints.append((i, i + 1, z, odometry_information))
        estimate.append(t2v(v2t(estimate[-1]) @ v2t(z)))

    if with_loop_closure:
        last = len(truth) - 1
        z = relative_pose(truth[last], truth[0])
        z = z + np.array([rng.normal(0, LOOP_TRANS_NOISE),
                          rng.normal(0, LOOP_TRANS_NOISE),
                          rng.normal(0, LOOP_ROT_NOISE)])
        z[2] = wrap(z[2])
        constraints.append((last, 0, z, loop_information))

    if false_closure is not None:
        i, j, offset = false_closure
        z = relative_pose(truth[i], truth[j]) + offset
        z[2] = wrap(z[2])
        constraints.append((i, j, z, loop_information))

    return np.array(estimate), constraints


def trajectory_error(estimate, truth):
    """Mean and final position error against ground truth."""
    d = np.linalg.norm(estimate[:, :2] - truth[:, :2], axis=1)
    return float(d.mean()), float(d[-1])


def self_test():
    """Check the pieces against answers that can be derived, not guessed."""
    rng = np.random.default_rng(0)

    # SE(2) round trip
    pose = np.array([1.3, -2.1, 0.7])
    assert np.allclose(t2v(v2t(pose)), pose), "v2t and t2v are not inverses"

    # Composing a relative pose must return the second pose
    a, b = np.array([1.0, 2.0, 0.4]), np.array([3.0, -1.0, -0.9])
    assert np.allclose(t2v(v2t(a) @ v2t(relative_pose(a, b))), b), \
        "relative_pose does not compose back"
    print("  SE(2) round trip and composition: exact")

    # Angle wrapping must take the short way round
    assert abs(wrap(np.deg2rad(350)) - np.deg2rad(-10)) < 1e-12
    print("  wrap(350 deg) = -10 deg, the short way round")

    # Analytic Jacobians against finite differences. A wrong Jacobian does
    # not crash, it just fails to converge, so this is the test that matters.
    xi = np.array([0.4, -0.2, 0.35])
    xj = np.array([1.9, 0.8, -0.6])
    z = np.array([1.4, 0.9, -0.95])
    A, B = constraint_jacobians(xi, xj, z)

    eps = 1e-7
    A_numeric = np.zeros((3, 3))
    B_numeric = np.zeros((3, 3))
    for k in range(3):
        step = np.zeros(3)
        step[k] = eps
        A_numeric[:, k] = (constraint_error(xi + step, xj, z)
                           - constraint_error(xi - step, xj, z)) / (2 * eps)
        B_numeric[:, k] = (constraint_error(xi, xj + step, z)
                           - constraint_error(xi, xj - step, z)) / (2 * eps)

    a_error = np.abs(A - A_numeric).max()
    b_error = np.abs(B - B_numeric).max()
    print(f"  analytic vs numerical Jacobians: max difference "
          f"{max(a_error, b_error):.2e}")
    assert a_error < 1e-5 and b_error < 1e-5, "the analytic Jacobians are wrong"

    # A graph with no noise at all must already be at zero cost
    truth = true_trajectory()
    information = np.eye(3)
    perfect = [(i, i + 1, relative_pose(truth[i], truth[i + 1]), information)
               for i in range(len(truth) - 1)]
    _, history = optimise(truth.copy(), perfect, max_iterations=3)
    print(f"  a graph built from perfect measurements starts at chi-squared "
          f"{history[0]:.2e}")
    assert history[0] < 1e-18, "a noiseless graph should already be solved"

    # And optimisation must never increase the cost
    noisy, constraints = build_graph(truth, rng)
    _, hist = optimise(noisy, constraints)
    assert hist[-1] <= hist[0], "optimisation made the cost worse"
    print(f"  on a noisy graph, chi-squared fell from {hist[0]:.1f} to "
          f"{hist[-1]:.1f}")


def main():
    print("Self-test:")
    self_test()

    truth = true_trajectory()
    rng = np.random.default_rng(SEED)

    print(f"\nTrajectory: {len(truth)} poses around a closed rectangular loop")

    # ---------- Odometry alone ----------
    odometry_only, odo_constraints = build_graph(truth, rng,
                                                 with_loop_closure=False)
    mean_o, final_o = trajectory_error(odometry_only, truth)
    gap = np.linalg.norm(odometry_only[-1, :2] - odometry_only[0, :2])

    print(f"\n--- Odometry alone ---")
    print(f"  Mean position error:  {mean_o:.3f} m")
    print(f"  Final position error: {final_o:.3f} m")
    print(f"  The robot returned to its start, but its estimate says it "
          f"ended {gap:.3f} m away.")

    # ---------- One loop closure ----------
    rng = np.random.default_rng(SEED)
    initial, constraints = build_graph(truth, rng, with_loop_closure=True)
    optimised, history = optimise(initial, constraints)
    mean_c, final_c = trajectory_error(optimised, truth)
    gap_after = np.linalg.norm(optimised[-1, :2] - optimised[0, :2])

    per_pose_odo = np.linalg.norm(odometry_only[:, :2] - truth[:, :2], axis=1)
    per_pose_opt = np.linalg.norm(optimised[:, :2] - truth[:, :2], axis=1)
    improved = int((per_pose_opt < per_pose_odo).sum())
    worsened = int((per_pose_opt > per_pose_odo).sum())

    print(f"\n--- With one loop closure ---")
    print(f"  Converged in {len(history)} iterations")
    print(f"  Chi-squared: {history[0]:.1f} -> {history[-1]:.1f}")
    print(f"  Gap at the loop:      {gap:.3f} m -> {gap_after:.3f} m "
          f"({100 * (1 - gap_after / gap):.0f}% closed)")
    print(f"  Final position error: {final_o:.3f} m -> {final_c:.3f} m")
    print(f"  Mean position error:  {mean_o:.3f} m -> {mean_c:.3f} m "
          f"(only {100 * (1 - mean_c / mean_o):.0f}% better)")

    print(f"\n  That last line is the one worth reading twice. A loop "
          f"closure does not\n  remove error, it REDISTRIBUTES it. "
          f"{improved} poses got closer to the truth and\n  {worsened} got "
          f"further away, which is why the mean barely moved while the\n"
          f"  loop gap collapsed.")
    print(f"\n  One extra measurement cannot undo "
          f"{len(truth) - 1} steps of accumulated noise. What it\n  "
          f"does is force the map to be CONSISTENT: the robot now agrees "
          f"with itself\n  about where it has been. For building a map "
          f"that is the thing that matters,\n  because an inconsistent map "
          f"has the same wall drawn twice in two places.")

    # ---------- One WRONG loop closure ----------
    rng = np.random.default_rng(SEED)
    bad_i, bad_j = 8, 30
    false_offset = np.array([2.5, -1.8, 0.6])
    initial_bad, bad_constraints = build_graph(
        truth, rng, with_loop_closure=True,
        false_closure=(bad_i, bad_j, false_offset))
    broken, broken_history = optimise(initial_bad, bad_constraints)
    mean_b, final_b = trajectory_error(broken, truth)

    print(f"\n--- The same graph, plus ONE wrong loop closure ---")
    print(f"  A false constraint says pose {bad_i} and pose {bad_j} are "
          f"the same place.")
    print(f"  Mean position error: {mean_c:.3f} m -> {mean_b:.3f} m")
    print(f"  That is {mean_b / mean_o:.1f}x WORSE than having no loop "
          f"closure at all ({mean_o:.3f} m).")
    print(f"  One bad constraint out of {len(bad_constraints)} undid every "
          f"correct one, because\n  least squares has no way to disbelieve "
          f"a measurement. It can only split\n  the difference between all "
          f"of them.")

    print(f"\n  But unlike Day 16's corridor, this failure IS detectable. "
          f"The good graph\n  settles at chi-squared {history[-1]:.1f}; "
          f"the poisoned one plateaus at "
          f"{broken_history[-1]:.0f}, "
          f"{broken_history[-1] / max(history[-1], 1e-9):.0f}x higher.\n  "
          f"A converged optimisation that still carries a large residual is "
          f"telling you\n  its constraints disagree with each other, and "
          f"that is worth checking before\n  the map is used for anything.")

    # ---------- The robust version ----------
    rng = np.random.default_rng(SEED)
    initial_robust, robust_constraints = build_graph(
        truth, rng, with_loop_closure=True,
        false_closure=(bad_i, bad_j, false_offset))
    repaired, robust_history = optimise(initial_robust, robust_constraints,
                                        huber_delta=2.0)
    mean_r, final_r = trajectory_error(repaired, truth)

    print(f"\n--- Same bad graph, with a Huber kernel ---")
    print(f"  Mean position error: {mean_b:.3f} m -> {mean_r:.3f} m")
    print(f"  Recovered {100 * (mean_b - mean_r) / max(mean_b - mean_c, 1e-9):.0f}% "
          f"of the damage.")
    print(f"  Beyond a threshold the cost grows linearly rather than "
          f"quadratically,\n  so a single large error stops dominating "
          f"everything else.")

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    def draw(ax, estimate, title, colour):
        ax.plot(truth[:, 0], truth[:, 1], "k--", linewidth=1.6,
                label="True path")
        ax.plot(estimate[:, 0], estimate[:, 1], color=colour, linewidth=1.9,
                label="Estimate")
        ax.plot(estimate[0, 0], estimate[0, 1], "o", color="tab:green",
                markersize=8)
        ax.plot(estimate[-1, 0], estimate[-1, 1], "s", color="tab:red",
                markersize=8)
        ax.set_title(title, fontsize=11)
        ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    draw(axes[0, 0], odometry_only,
         f"Odometry alone: {mean_o:.2f} m mean error", "tab:orange")
    draw(axes[0, 1], optimised,
         f"One loop closure: {mean_c:.2f} m", "tab:blue")
    draw(axes[0, 2], broken,
         f"One WRONG closure: {mean_b:.2f} m", "tab:red")

    ax = axes[1, 0]
    ax.plot(history, "o-", color="tab:blue", markersize=3, label="Good graph")
    ax.plot(broken_history, "o-", color="tab:red", markersize=3,
            label="With a false closure")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Chi-squared (log)")
    ax.set_title("Both converge. Only one is correct.", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    draw(axes[1, 1], repaired,
         f"Huber kernel on the bad graph: {mean_r:.2f} m", "tab:green")

    ax = axes[1, 2]
    ax.plot(per_pose_odo, color="tab:orange", linewidth=1.8,
            label=f"Odometry (mean {mean_o:.2f} m)")
    ax.plot(per_pose_opt, color="tab:blue", linewidth=1.8,
            label=f"Loop closed (mean {mean_c:.2f} m)")
    ax.fill_between(range(len(truth)), per_pose_odo, per_pose_opt,
                    where=per_pose_opt < per_pose_odo,
                    color="tab:green", alpha=0.25, label="Improved")
    ax.fill_between(range(len(truth)), per_pose_odo, per_pose_opt,
                    where=per_pose_opt > per_pose_odo,
                    color="tab:red", alpha=0.25, label="Made worse")
    ax.set_xlabel("Pose index"); ax.set_ylabel("Position error (m)")
    ax.set_title("A loop closure redistributes error, it does not remove it",
                 fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("Day 18: Pose graph optimisation, and the one bad "
                 "constraint that ruins it", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "pose_graph.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
