"""
Day 16: ICP scan matching, and the directions it cannot see.

Day 12 built a map assuming the pose was known. Day 14 found the pose
assuming the map was known. Neither is true on a robot that has just
been switched on, and the thing that bridges them is scan matching:
work out how far the robot moved by aligning the lidar scan it just
took against the previous one.

Iterative Closest Point does that with no features and no map:

  1. for every point in the new scan, find the nearest point in the old
  2. solve for the rigid transform that best lines up those pairs
  3. apply it, and repeat until it stops moving

Step 2 has an exact closed-form answer (Kabsch via SVD), which is why
ICP converges quickly. Step 1 is a guess that gets better as the
alignment improves, which is why ICP needs a decent starting point and
will confidently converge to the wrong answer without one.

The last section is the part worth knowing. In a long featureless
corridor, sliding along the corridor changes nothing about the scan, so
that direction is unobservable. ICP still returns an answer, with no
indication that one component of it is invented. The eigenvalues of the
information matrix say which directions were actually constrained.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# ---------------- Settings ----------------
N_BEAMS = 360
MAX_RANGE = 12.0
RANGE_NOISE = 0.02

ICP_MAX_ITERATIONS = 60
ICP_TOLERANCE = 1e-6
TRIM_FRACTION = 0.85          # keep the best 85% of correspondences

SEED = 4
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# ---------------- Worlds, as line segments ----------------
def room_world():
    """A room with furniture: plenty of structure in every direction."""
    walls = [((0, 0), (12, 0)), ((12, 0), (12, 9)),
             ((12, 9), (0, 9)), ((0, 9), (0, 0)),
             ((3, 0), (3, 3)), ((7, 9), (7, 6)),
             ((8, 2), (11, 2)), ((8, 2), (8, 4)),
             ((1.5, 5), (3.5, 5)), ((3.5, 5), (3.5, 7))]
    return [(np.array(a, dtype=float), np.array(b, dtype=float))
            for a, b in walls]


def corridor_world():
    """A long straight corridor and nothing else.

    Sliding along it produces an identical scan, so that direction
    carries no information at all.
    """
    walls = [((0, 0), (40, 0)), ((0, 2.4), (40, 2.4))]
    return [(np.array(a, dtype=float), np.array(b, dtype=float))
            for a, b in walls]


def ray_segment_distance(origin, direction, a, b):
    """Distance from origin along direction to a segment, or inf."""
    v1 = origin - a
    v2 = b - a
    v3 = np.array([-direction[1], direction[0]])

    denominator = np.dot(v2, v3)
    if abs(denominator) < 1e-12:
        return np.inf

    # 2D cross product written out: NumPy 2 removed np.cross for 2-vectors
    cross = v2[0] * v1[1] - v2[1] * v1[0]
    t1 = cross / denominator                 # distance along the ray
    t2 = np.dot(v1, v3) / denominator        # position along the segment
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return t1
    return np.inf


def take_scan(world, pose, rng, n_beams=N_BEAMS):
    """A 2D lidar scan from a pose, returned as points in the robot frame."""
    x, y, theta = pose
    origin = np.array([x, y])
    angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)

    points = []
    for a in angles:
        direction = np.array([np.cos(theta + a), np.sin(theta + a)])
        distance = min((ray_segment_distance(origin, direction, p, q)
                        for p, q in world), default=np.inf)
        if np.isfinite(distance) and distance < MAX_RANGE:
            measured = distance + rng.normal(0, RANGE_NOISE)
            points.append([measured * np.cos(a), measured * np.sin(a)])

    return np.array(points)


# ---------------- The maths ----------------
def best_rigid_transform(source, target):
    """The rotation and translation that best map source onto target.

    This is the Kabsch algorithm. Centre both clouds, take the SVD of the
    cross-covariance, and the rotation falls out. It is exact and
    non-iterative: given the correspondences, there is nothing to tune.

    The determinant correction stops the SVD returning a reflection,
    which fits the points just as well and is not a rotation.
    """
    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)

    H = (source - source_centre).T @ (target - target_centre)
    U, _, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T

    t = target_centre - R @ source_centre
    return R, t


def transform_points(points, R, t):
    return points @ R.T + t


def icp(source, target, initial_guess=None, max_iterations=ICP_MAX_ITERATIONS,
        trim=TRIM_FRACTION):
    """Align source onto target. Returns the transform and a history.

    Correspondences are re-found every iteration, because a better
    alignment makes nearest-neighbour pairing more likely to be right.
    The worst correspondences are trimmed each round, which is what stops
    points visible in one scan but not the other from dragging the fit.
    """
    tree = cKDTree(target)

    R = np.eye(2)
    t = np.zeros(2)
    if initial_guess is not None:
        R, t = initial_guess

    history = []
    previous_error = np.inf

    for iteration in range(max_iterations):
        moved = transform_points(source, R, t)
        distances, indices = tree.query(moved)

        keep = np.ones(len(moved), dtype=bool)
        if trim < 1.0:
            cutoff = np.quantile(distances, trim)
            keep = distances <= cutoff

        error = float(distances[keep].mean())
        history.append(error)

        R_step, t_step = best_rigid_transform(moved[keep], target[indices[keep]])
        R = R_step @ R
        t = R_step @ t + t_step

        if abs(previous_error - error) < ICP_TOLERANCE:
            break
        previous_error = error

    return R, t, history


def pose_from_transform(R, t):
    """Transform to (dx, dy, dtheta) in degrees, for printing."""
    return np.array([t[0], t[1], np.degrees(np.arctan2(R[1, 0], R[0, 0]))])


def information_matrix(source, target):
    """How strongly each direction of motion is constrained.

    Built from the point-to-plane Jacobian: for every matched point, the
    residual that matters is distance along the local surface normal, not
    the full displacement. Sliding a point along a wall changes nothing,
    so a wall constrains only the direction perpendicular to itself.

    The eigenvalues of J^T J say how much evidence each direction of
    (x, y, rotation) actually has. A near-zero eigenvalue means that
    direction is unobservable and whatever ICP reports for it is noise.
    """
    tree = cKDTree(target)
    _, indices = tree.query(source)
    matched = target[indices]

    # Estimate a local normal at each matched point from its neighbours
    _, neighbours = tree.query(matched, k=min(8, len(target)))
    J = []
    for i, point in enumerate(source):
        local = target[neighbours[i]]
        local = local - local.mean(axis=0)
        _, _, Vt = np.linalg.svd(local)
        normal = Vt[1]                       # smallest direction = the normal

        # d(residual)/d(tx, ty, dtheta) projected onto that normal
        J.append([normal[0], normal[1],
                  normal[1] * point[0] - normal[0] * point[1]])

    J = np.array(J)
    return J.T @ J


def self_test():
    """ICP has to recover a transform that was applied on purpose."""
    rng = np.random.default_rng(0)

    # Kabsch must be exact when the correspondences are exact
    cloud = rng.normal(size=(200, 2)) * 3
    angle = np.deg2rad(23.0)
    R_true = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    t_true = np.array([1.7, -0.9])
    moved = transform_points(cloud, R_true, t_true)

    R, t = best_rigid_transform(cloud, moved)
    assert np.allclose(R, R_true, atol=1e-9), "Kabsch rotation is wrong"
    assert np.allclose(t, t_true, atol=1e-9), "Kabsch translation is wrong"
    print(f"  Kabsch on exact correspondences: recovered "
          f"{np.degrees(angle):.1f} deg and {t_true} exactly")

    # A reflection must not be returned as a rotation
    flipped = cloud.copy()
    flipped[:, 1] *= -1
    R_flip, _ = best_rigid_transform(cloud, flipped)
    assert np.linalg.det(R_flip) > 0, "returned a reflection, not a rotation"
    print(f"  reflection guard: determinant stays +1 "
          f"({np.linalg.det(R_flip):.6f})")

    # Full ICP, correspondences unknown, on a shuffled and noisy cloud
    noisy = transform_points(cloud, R_true, t_true)
    noisy = noisy + rng.normal(0, 0.01, noisy.shape)
    noisy = noisy[rng.permutation(len(noisy))]
    R_icp, t_icp, _ = icp(cloud, noisy, trim=1.0)
    recovered = pose_from_transform(R_icp, t_icp)
    truth = pose_from_transform(R_true, t_true)
    error = np.abs(recovered - truth)
    print(f"  full ICP, correspondences unknown: error "
          f"{error[0]:.4f} m, {error[1]:.4f} m, {error[2]:.4f} deg")
    assert error[0] < 0.05 and error[1] < 0.05 and error[2] < 1.0, \
        "ICP failed to recover a transform it should have found"

    # The information matrix must flag an unobservable direction
    along = np.column_stack([np.linspace(0, 30, 400), np.zeros(400)])
    eigenvalues = np.linalg.eigvalsh(information_matrix(along, along))
    ratio = eigenvalues[0] / eigenvalues[-1]
    print(f"  information matrix on a single straight line: "
          f"smallest/largest eigenvalue = {ratio:.2e}")
    assert ratio < 1e-6, "a straight line should leave a direction unobservable"


def main():
    print("Self-test:")
    self_test()

    rng = np.random.default_rng(SEED)
    world = room_world()

    pose_a = np.array([4.0, 4.0, 0.3])
    true_motion = np.array([0.55, 0.32, np.deg2rad(9.0)])
    pose_b = pose_a + true_motion

    scan_a = take_scan(world, pose_a, rng)
    scan_b = take_scan(world, pose_b, rng)
    print(f"\nTwo scans from a furnished room: "
          f"{len(scan_a)} and {len(scan_b)} points")
    print(f"True motion between them: {true_motion[0]:.3f} m, "
          f"{true_motion[1]:.3f} m, {np.degrees(true_motion[2]):.1f} deg")

    # ---------- Does ICP recover the motion? ----------
    R, t, history = icp(scan_b, scan_a)
    estimated = pose_from_transform(R, t)
    truth_in_frame = np.array([true_motion[0], true_motion[1],
                               np.degrees(true_motion[2])])

    print(f"\n--- ICP from a cold start (identity) ---")
    print(f"  Converged in {len(history)} iterations, "
          f"mean residual {history[-1] * 100:.2f} cm")
    print(f"  Estimated rotation: {estimated[2]:+.2f} deg "
          f"(true {truth_in_frame[2]:+.2f})")
    print(f"  Rotation error: {abs(estimated[2] - truth_in_frame[2]):.3f} deg")

    # ---------- How good does the starting point have to be? ----------
    print(f"\n--- Sensitivity to the initial guess ---")
    print(f"  {'offset':>10}  {'final residual':>15}  {'rotation error':>15}")
    sensitivity = []
    for offset_deg in (0, 5, 15, 30, 60, 90, 150):
        angle = np.deg2rad(offset_deg)
        R0 = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
        R_o, t_o, hist = icp(scan_b, scan_a, initial_guess=(R0, np.zeros(2)))
        est = pose_from_transform(R_o, t_o)
        rot_error = abs(est[2] - truth_in_frame[2])
        sensitivity.append((offset_deg, hist[-1], rot_error))
        verdict = "converged" if rot_error < 2 else "WRONG ANSWER"
        print(f"  {offset_deg:8d} deg  {hist[-1] * 100:12.2f} cm  "
              f"{rot_error:12.2f} deg   {verdict}")

    print(f"\n  ICP only ever looks at the nearest neighbour, so it walks "
          f"downhill from\n  wherever it starts. Past a certain offset it "
          f"finds a different valley and\n  reports that answer with a "
          f"perfectly healthy-looking residual.")

    # ---------- The corridor ----------
    print(f"\n--- The same algorithm in a featureless corridor ---")
    corridor = corridor_world()
    c_pose_a = np.array([12.0, 1.2, 0.0])
    c_motion = np.array([0.8, 0.0, 0.0])          # slides along the corridor
    c_scan_a = take_scan(corridor, c_pose_a, rng)
    c_scan_b = take_scan(corridor, c_pose_a + c_motion, rng)

    R_c, t_c, hist_c = icp(c_scan_b, c_scan_a)
    est_c = pose_from_transform(R_c, t_c)
    print(f"  True motion:      {c_motion[0]:+.3f} m along, "
          f"{c_motion[1]:+.3f} m across")
    print(f"  ICP's answer:     {est_c[0]:+.3f} m along, "
          f"{est_c[1]:+.3f} m across")
    print(f"  Mean residual:    {hist_c[-1] * 100:.2f} cm  "
          f"(the fit looks excellent)")
    print(f"  Error along the corridor: {abs(est_c[0] - c_motion[0]):.3f} m")

    room_info = information_matrix(scan_b, scan_a)
    corridor_info = information_matrix(c_scan_b, c_scan_a)
    room_eig = np.linalg.eigvalsh(room_info)
    corridor_eig = np.linalg.eigvalsh(corridor_info)

    print(f"\n  Eigenvalues of the information matrix "
          f"(how constrained each direction is):")
    print(f"    furnished room: {np.array2string(room_eig, precision=1)}")
    print(f"    corridor:       {np.array2string(corridor_eig, precision=1)}")
    print(f"    condition number, room:     "
          f"{room_eig[-1] / max(room_eig[0], 1e-12):.1f}")
    print(f"    condition number, corridor: "
          f"{corridor_eig[-1] / max(corridor_eig[0], 1e-12):.1f}")
    print(f"\n  The corridor's smallest eigenvalue is the direction along it. "
          f"Nothing in\n  the scan constrains that, so ICP's answer for it is "
          f"invented. The residual\n  cannot tell you this, but the "
          f"eigenvalues can, before you trust the result.")

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    def draw_world(ax, w):
        for p, q in w:
            ax.plot([p[0], q[0]], [p[1], q[1]], color="k", linewidth=2)

    ax = axes[0, 0]
    draw_world(ax, world)
    ax.plot(*pose_a[:2], "o", color="tab:blue", markersize=9, label="Pose A")
    ax.plot(*pose_b[:2], "o", color="tab:orange", markersize=9, label="Pose B")
    ax.set_title("Two poses in a furnished room")
    ax.axis("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(scan_a[:, 0], scan_a[:, 1], s=4, color="tab:blue", label="Scan A")
    ax.scatter(scan_b[:, 0], scan_b[:, 1], s=4, color="tab:orange",
               alpha=0.7, label="Scan B (before)")
    ax.set_title("The two scans, unaligned")
    ax.axis("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    aligned = transform_points(scan_b, R, t)
    ax.scatter(scan_a[:, 0], scan_a[:, 1], s=4, color="tab:blue", label="Scan A")
    ax.scatter(aligned[:, 0], aligned[:, 1], s=4, color="tab:green",
               alpha=0.7, label="Scan B after ICP")
    ax.set_title(f"Aligned ({history[-1] * 100:.1f} cm residual)")
    ax.axis("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(np.array(history) * 100, "o-", color="tab:purple", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean residual (cm, log)")
    ax.set_title("ICP converging")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    offsets = [s[0] for s in sensitivity]
    errors = [s[2] for s in sensitivity]
    colours = ["tab:green" if e < 2 else "tab:red" for e in errors]
    ax.bar(range(len(offsets)), errors, color=colours)
    ax.set_xticks(range(len(offsets)), [f"{o}" for o in offsets])
    ax.axhline(2, color="k", linestyle=":", label="2 deg")
    ax.set_xlabel("Initial guess offset (deg)")
    ax.set_ylabel("Final rotation error (deg)")
    ax.set_title("ICP is local: a bad start finds a wrong valley")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 2]
    labels = ["x", "y", "rotation"]
    width = 0.38
    idx = np.arange(3)
    ax.bar(idx - width / 2, room_eig / room_eig[-1], width,
           label="Furnished room", color="tab:blue")
    ax.bar(idx + width / 2, corridor_eig / corridor_eig[-1], width,
           label="Corridor", color="tab:red")
    ax.set_yscale("log")
    ax.set_xticks(idx, ["weakest", "middle", "strongest"])
    ax.set_ylabel("Eigenvalue, relative to largest (log)")
    ax.set_title("Which directions are actually constrained")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Day 16: ICP scan matching, and the directions it cannot see",
                 fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "icp_scan_matching.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
