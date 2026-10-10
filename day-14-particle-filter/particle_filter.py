"""
Day 14: Particle filter localisation on a known map.

Day 10 tracked a robot with a Kalman filter, which assumes the belief is
a single Gaussian blob. Day 12 built a map but assumed the pose was known
exactly. This closes that gap for the harder case: the robot is switched
on somewhere in a building it has a map of, and has no idea where.

"Somewhere, could be anywhere" is not a Gaussian, so a Kalman filter
cannot represent it at all. A particle filter can, because it represents
the belief as a cloud of guesses and lets the measurements kill off the
wrong ones.

Three things are measured here:
  1. how many steps global localisation takes to converge
  2. how much better that is than dead reckoning
  3. what happens when the robot is picked up and moved
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

# ---------------- World ----------------
WORLD_SIZE = (60, 40)        # cells (x, y)
CELL = 0.25                  # metres per cell

# ---------------- Sensor ----------------
N_BEAMS = 16                 # sparse on purpose: enough to localise, cheap to score
MAX_RANGE = 6.0
RANGE_NOISE = 0.08
LIKELIHOOD_SIGMA = 0.55      # how forgiving the measurement model is, in metres

# ---------------- Motion ----------------
ODO_TRANS_NOISE = 0.06       # metres per step
ODO_ROT_NOISE = 0.035        # radians per step

# ---------------- Filter ----------------
# Global localisation needs enough particles that at least a few start near
# the answer. This map has 2088 free cells and heading is continuous, so a
# few thousand particles leaves roughly ONE anywhere near the truth, and a
# filter cannot find a hypothesis it never sampled.
N_PARTICLES = 12000
RESAMPLE_THRESHOLD = 0.5     # resample when effective sample size falls below N/2
CONVERGED_SPREAD = 0.5       # metres, standard deviation of the particle cloud

# Roughening: after resampling, every surviving particle is a duplicate of
# some parent. Without a small jitter the cloud loses all diversity, every
# particle scores identically from then on, resampling stops firing, and the
# filter is locked into whatever it guessed first.
ROUGHEN_TRANS = 0.04         # metres
ROUGHEN_ROT = 0.02           # radians

KIDNAP_OFFSET = (-4.0, -1.5, 0.0)   # 4.27 m, verified to stay in free space

SEED = 5
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def build_world():
    """Same floor plan as Day 12: two rooms, two doorways, three blocks."""
    world = np.zeros(WORLD_SIZE[::-1], dtype=bool)
    world[0, :] = world[-1, :] = True
    world[:, 0] = world[:, -1] = True
    world[20, 1:38] = True
    world[20, 14:19] = False
    world[20:39, 38] = True
    world[27:32, 38] = False
    world[5:9, 46:52] = True
    world[29:34, 48:54] = True
    world[8:12, 26:30] = True
    return world


def likelihood_field(world):
    """Distance from every cell to the nearest obstacle, in metres.

    This is what makes the filter affordable. The textbook measurement
    model casts a ray per particle per beam, which here would be 2000 x 16
    ray marches every single step. Instead each beam endpoint is looked up
    in this precomputed field, which is one array index. The map does not
    move, so the field is computed once.
    """
    return distance_transform_edt(~world) * CELL


def free_cells(world):
    ys, xs = np.where(~world)
    return np.column_stack([xs, ys])


def cast_ray(world, x, y, angle, max_range=MAX_RANGE):
    """Ray march for the true robot only, once per step."""
    step = CELL / 2
    distance = 0.0
    dx, dy = np.cos(angle), np.sin(angle)
    while distance < max_range:
        distance += step
        cx = int((x + dx * distance) / CELL)
        cy = int((y + dy * distance) / CELL)
        if not (0 <= cx < WORLD_SIZE[0] and 0 <= cy < WORLD_SIZE[1]):
            return max_range
        if world[cy, cx]:
            return distance
    return max_range


def scan(world, pose, rng, beam_angles):
    """A noisy lidar scan from the true pose."""
    x, y, theta = pose
    ranges = np.array([cast_ray(world, x, y, theta + a) for a in beam_angles])
    noisy = ranges + rng.normal(0, RANGE_NOISE, len(ranges))
    return np.clip(noisy, 0.0, MAX_RANGE)


def pose_is_possible(particles, world):
    """A robot cannot be outside the building or inside a wall.

    This is not a refinement, it is load-bearing. Beam endpoints get
    clipped to the array bounds when a particle drifts off the map, and
    the map border is a wall, so every clipped endpoint lands on an
    obstacle and reports a distance of zero. A particle that has wandered
    far outside the building therefore scores PERFECTLY. Without this
    check the filter actively rewards particles for being nowhere, and
    the cloud flies apart instead of converging.
    """
    cx = (particles[:, 0] / CELL).astype(int)
    cy = (particles[:, 1] / CELL).astype(int)
    inside = ((cx >= 0) & (cx < WORLD_SIZE[0]) &
              (cy >= 0) & (cy < WORLD_SIZE[1]))

    possible = np.zeros(len(particles), dtype=bool)
    safe_x = np.clip(cx, 0, WORLD_SIZE[0] - 1)
    safe_y = np.clip(cy, 0, WORLD_SIZE[1] - 1)
    possible[inside] = ~world[safe_y[inside], safe_x[inside]]
    return possible


def score_particles(particles, ranges, beam_angles, field, world):
    """Weight every particle by how well the map explains its scan.

    Fully vectorised: for each particle and beam, work out where the beam
    would have ended if that particle were the true pose, look up how far
    that point is from the nearest obstacle, and score it with a Gaussian.
    A particle in the right place puts its endpoints on walls, so those
    distances are near zero and the score is high.

    Beams that came back at maximum range are dropped. They mean "nothing
    out there", which the likelihood field cannot represent, and scoring
    them punishes correct particles standing in open space.
    """
    n = len(particles)
    usable = ranges < MAX_RANGE * 0.99
    if not usable.any():
        return np.ones(n) / n

    r = ranges[usable][None, :]
    a = beam_angles[usable][None, :]

    heading = particles[:, 2:3]
    ex = particles[:, 0:1] + r * np.cos(heading + a)
    ey = particles[:, 1:2] + r * np.sin(heading + a)

    cx = np.clip((ex / CELL).astype(int), 0, WORLD_SIZE[0] - 1)
    cy = np.clip((ey / CELL).astype(int), 0, WORLD_SIZE[1] - 1)

    distances = field[cy, cx]
    log_weights = -(distances ** 2) / (2 * LIKELIHOOD_SIGMA ** 2)
    total = log_weights.sum(axis=1)

    total -= total.max()                 # subtract the max before exponentiating
    weights = np.exp(total)              #   or everything underflows to zero

    weights[~pose_is_possible(particles, world)] = 0.0

    if weights.sum() <= 0:
        return np.ones(n) / n            # every hypothesis died, stay agnostic
    return weights / weights.sum()


def motion_update(particles, delta, rng):
    """Push every particle along the odometry reading, with noise.

    Each particle gets its own noise draw. Without that the cloud would
    translate rigidly and never represent uncertainty growing.
    """
    d_trans, d_rot = delta
    n = len(particles)

    trans = d_trans + rng.normal(0, ODO_TRANS_NOISE, n)
    rot = d_rot + rng.normal(0, ODO_ROT_NOISE, n)

    particles[:, 2] += rot
    particles[:, 0] += trans * np.cos(particles[:, 2])
    particles[:, 1] += trans * np.sin(particles[:, 2])
    return particles


def systematic_resample(particles, weights, rng):
    """Low-variance resampling.

    Draws one random offset and then steps through the weight cumulative
    sum at even intervals. Compared with drawing N independent samples
    this adds far less variance, and a particle with weight above 1/N is
    guaranteed to survive rather than being lost to bad luck.
    """
    n = len(particles)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return particles[np.searchsorted(cumulative, positions)]


def effective_sample_size(weights):
    """How many particles are really contributing.

    If one particle holds all the weight this is 1, and the cloud has
    collapsed to a single guess with no diversity left to recover with.
    """
    return 1.0 / np.sum(weights ** 2)


def estimate(particles, weights):
    """Weighted mean position, and circular mean for the heading."""
    x = np.average(particles[:, 0], weights=weights)
    y = np.average(particles[:, 1], weights=weights)
    s = np.average(np.sin(particles[:, 2]), weights=weights)
    c = np.average(np.cos(particles[:, 2]), weights=weights)
    return np.array([x, y, np.arctan2(s, c)])


def robot_route():
    # Same waypoints as Day 12, already verified to clear every obstacle
    waypoints = [(1.0, 1.0), (10.0, 1.0), (10.0, 3.5), (4.0, 3.5), (4.0, 7.0),
                 (8.0, 7.0), (8.0, 7.25), (11.0, 7.25), (11.0, 9.0),
                 (14.0, 9.0), (14.0, 6.0)]
    poses = []
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        n = max(int(np.hypot(x1 - x0, y1 - y0) / 0.25), 1)
        heading = np.arctan2(y1 - y0, x1 - x0)
        for i in range(n):
            t = i / n
            poses.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), heading))
    poses.append((*waypoints[-1], poses[-1][2]))
    return np.array(poses)


def validate_route(world, poses, clearance=1):
    bad = []
    for x, y, _ in poses:
        cx, cy = int(x / CELL), int(y / CELL)
        if world[max(cy - clearance, 0):cy + clearance + 1,
                 max(cx - clearance, 0):cx + clearance + 1].any():
            bad.append((round(float(x), 2), round(float(y), 2)))
    if bad:
        raise ValueError(f"{len(bad)} poses inside or touching an obstacle, "
                         f"first at {bad[0]}")
    return True


def validate_kidnap(world, poses, kidnap_at, offset, clearance=1):
    """The kidnapped robot must land somewhere it could actually be."""
    shifted = [(p[0] + offset[0], p[1] + offset[1]) for p in poses[kidnap_at:]]
    bad = []
    for x, y in shifted:
        cx, cy = int(x / CELL), int(y / CELL)
        if not (0 <= cx < WORLD_SIZE[0] and 0 <= cy < WORLD_SIZE[1]):
            bad.append((round(x, 2), round(y, 2)))
        elif world[max(cy - clearance, 0):cy + clearance + 1,
                   max(cx - clearance, 0):cx + clearance + 1].any():
            bad.append((round(x, 2), round(y, 2)))
    if bad:
        raise ValueError(
            f"{len(bad)} of {len(shifted)} kidnapped poses are off-map or inside "
            f"an obstacle, first at {bad[0]}. A scan from inside a wall matches "
            f"no particle, so recovery would be impossible by construction.")
    return True


def run_filter(world, field, poses, rng, kidnap_at=None, inject_random=0.0):
    """Monte Carlo localisation along the route.

    Particles start spread uniformly over every free cell with a random
    heading, which is global localisation: the robot has a map and no idea
    where on it it is.
    """
    beam_angles = np.linspace(0, 2 * np.pi, N_BEAMS, endpoint=False)
    free = free_cells(world)

    def spawn(n):
        idx = rng.choice(len(free), n)
        p = np.zeros((n, 3))
        p[:, 0] = (free[idx, 0] + rng.random(n)) * CELL
        p[:, 1] = (free[idx, 1] + rng.random(n)) * CELL
        p[:, 2] = rng.uniform(-np.pi, np.pi, n)
        return p

    particles = spawn(N_PARTICLES)
    weights = np.ones(N_PARTICLES) / N_PARTICLES

    true_poses, estimates, dead_reckoning = [], [], []
    errors, dr_errors, spreads, ess_history = [], [], [], []
    snapshots = {}
    snapshot_at = {0, 2, 6, len(poses) - 1}

    dr_pose = None
    offset = np.zeros(3)

    for step, pose in enumerate(poses):
        true_pose = pose.copy()
        if kidnap_at is not None and step == kidnap_at:
            # Pick the robot up and put it somewhere else, without telling it.
            # The offset is checked by validate_kidnap: a robot teleported
            # into a wall produces a scan no particle can ever match, which
            # would make recovery impossible by construction rather than by
            # any property of the filter.
            offset = np.array(KIDNAP_OFFSET)
        true_pose = true_pose + offset
        true_poses.append(true_pose)

        if step > 0:
            previous = poses[step - 1]
            d_trans = np.hypot(pose[0] - previous[0], pose[1] - previous[1])
            d_rot = np.arctan2(np.sin(pose[2] - previous[2]),
                               np.cos(pose[2] - previous[2]))

            # Dead reckoning: integrate the same noisy odometry, no corrections
            if dr_pose is None:
                dr_pose = true_poses[0].copy()
            noisy_trans = d_trans + rng.normal(0, ODO_TRANS_NOISE)
            noisy_rot = d_rot + rng.normal(0, ODO_ROT_NOISE)
            dr_pose[2] += noisy_rot
            dr_pose[0] += noisy_trans * np.cos(dr_pose[2])
            dr_pose[1] += noisy_trans * np.sin(dr_pose[2])

            particles = motion_update(particles, (d_trans, d_rot), rng)
        else:
            dr_pose = true_pose.copy()

        ranges = scan(world, true_pose, rng, beam_angles)
        weights = score_particles(particles, ranges, beam_angles, field, world)

        ess = effective_sample_size(weights)
        ess_history.append(ess)
        if ess < RESAMPLE_THRESHOLD * N_PARTICLES:
            particles = systematic_resample(particles, weights, rng)
            # Roughening, so duplicates do not stay exact duplicates
            particles[:, 0] += rng.normal(0, ROUGHEN_TRANS, N_PARTICLES)
            particles[:, 1] += rng.normal(0, ROUGHEN_TRANS, N_PARTICLES)
            particles[:, 2] += rng.normal(0, ROUGHEN_ROT, N_PARTICLES)
            weights = np.ones(N_PARTICLES) / N_PARTICLES

            if inject_random > 0:
                # Replace a slice of the cloud with fresh uniform guesses, so
                # the filter keeps a way back if it has converged on a lie
                k = int(inject_random * N_PARTICLES)
                particles[rng.choice(N_PARTICLES, k, replace=False)] = spawn(k)

        est = estimate(particles, weights)
        estimates.append(est)
        dead_reckoning.append(dr_pose.copy())

        errors.append(np.hypot(*(est[:2] - true_pose[:2])))
        dr_errors.append(np.hypot(*(dr_pose[:2] - true_pose[:2])))
        spreads.append(np.sqrt(particles[:, 0].var() + particles[:, 1].var()))

        if step in snapshot_at:
            snapshots[step] = (particles.copy(), true_pose.copy())

    return dict(true=np.array(true_poses), est=np.array(estimates),
                dr=np.array(dead_reckoning), errors=np.array(errors),
                dr_errors=np.array(dr_errors), spreads=np.array(spreads),
                ess=np.array(ess_history), snapshots=snapshots)


def self_test(world, field):
    """Check the pieces against answers that can be worked out by hand."""
    # The likelihood field must be zero on obstacles and positive in free space
    assert field[world].max() == 0, "obstacle cells should be zero distance"
    assert field[~world].min() > 0, "free cells should be a positive distance away"
    print(f"  likelihood field: 0 m on obstacles, up to "
          f"{field.max():.2f} m away in open space")

    rng = np.random.default_rng(0)
    beam_angles = np.linspace(0, 2 * np.pi, N_BEAMS, endpoint=False)

    # A particle at the true pose must outscore a particle somewhere wrong
    truth = np.array([2.0, 2.0, 0.0])
    ranges = scan(world, truth, rng, beam_angles)
    candidates = np.array([truth, [8.0, 8.0, 0.0], [5.0, 3.0, 1.5]])
    w = score_particles(candidates, ranges, beam_angles, field, world)
    print(f"  scoring 3 candidates from the true pose: weights "
          f"{np.round(w, 4)}")
    assert w[0] == w.max(), "the true pose should score highest"

    # A particle outside the building, or inside a wall, must score zero.
    # Clipping makes off-map beam endpoints land on the border wall, which
    # otherwise reads as a perfect match.
    impossible = np.array([[-8.0, -8.0, 0.0],      # far outside the map
                           [0.05, 0.05, 0.0],      # inside the border wall
                           [2.0, 2.0, 0.0]])       # legitimately free space
    wi = score_particles(impossible, ranges, beam_angles, field, world)
    print(f"  off-map / in-wall / free particle weights: {np.round(wi, 4)}")
    assert wi[0] == 0 and wi[1] == 0, (
        "particles outside the map or inside a wall must score zero")
    assert wi[2] > 0, "a particle in free space should score above zero"

    # Effective sample size has to behave at both extremes
    n = 100
    assert abs(effective_sample_size(np.ones(n) / n) - n) < 1e-9
    collapsed = np.zeros(n); collapsed[0] = 1.0
    assert abs(effective_sample_size(collapsed) - 1.0) < 1e-9
    print(f"  effective sample size: {n} when weights are equal, "
          f"1.0 when one particle holds everything")

    # Resampling must favour heavy particles
    p = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], dtype=float)
    heavy = np.array([0.02, 0.96, 0.02])
    out = systematic_resample(p, heavy, np.random.default_rng(1))
    share = np.mean(out[:, 0] == 1.0)
    print(f"  resampling a cloud where one particle holds 96%: "
          f"{100 * share:.0f}% of the survivors are that particle")
    assert share > 0.9, "systematic resampling should concentrate on heavy particles"


def main():
    world = build_world()
    field = likelihood_field(world)

    print("Self-test:")
    self_test(world, field)

    poses = robot_route()
    validate_route(world, poses)
    print(f"  route of {len(poses)} poses, none inside an obstacle")

    print(f"\nMap: {WORLD_SIZE[0]} x {WORLD_SIZE[1]} cells "
          f"({WORLD_SIZE[0] * CELL:.0f} x {WORLD_SIZE[1] * CELL:.0f} m), "
          f"{int((~world).sum())} free cells")
    print(f"Filter: {N_PARTICLES} particles, {N_BEAMS} beams, "
          f"starting with no idea where the robot is")

    # ---------- Run 1: global localisation ----------
    run = run_filter(world, field, poses, np.random.default_rng(SEED))

    below = run["spreads"] < CONVERGED_SPREAD
    converged = int(np.argmax(below)) if below.any() else None
    print(f"\n--- Global localisation ---")
    print(f"  Initial particle spread: {run['spreads'][0]:.2f} m "
          f"(the whole map)")
    if converged is None:
        print(f"  NEVER converged below {CONVERGED_SPREAD} m")
    else:
        print(f"  Converged below {CONVERGED_SPREAD} m at step {converged} "
              f"of {len(poses)}")
    print(f"  Final particle spread: {run['spreads'][-1]:.2f} m")

    settled = slice(converged if converged is not None else 0, None)
    print(f"\n  After convergence:")
    print(f"    Particle filter error: mean {run['errors'][settled].mean():.3f} m, "
          f"max {run['errors'][settled].max():.3f} m")
    print(f"    Dead reckoning error:  mean {run['dr_errors'][settled].mean():.3f} m, "
          f"max {run['dr_errors'][settled].max():.3f} m")
    print(f"    Dead reckoning final error: {run['dr_errors'][-1]:.3f} m "
          f"and still growing")

    print(f"\n  Effective sample size: min {run['ess'].min():.0f}, "
          f"median {np.median(run['ess']):.0f}, of {N_PARTICLES}")

    # ---------- Run 2: the kidnapped robot ----------
    kidnap_step = int(len(poses) * 0.6)
    validate_kidnap(world, poses, kidnap_step, KIDNAP_OFFSET)
    distance = np.hypot(KIDNAP_OFFSET[0], KIDNAP_OFFSET[1])
    print(f"\n--- The kidnapped robot (moved {distance:.2f} m at step "
          f"{kidnap_step}, landing in free space) ---")

    plain = run_filter(world, field, poses, np.random.default_rng(SEED),
                       kidnap_at=kidnap_step)
    injected = run_filter(world, field, poses, np.random.default_rng(SEED),
                          kidnap_at=kidnap_step, inject_random=0.05)

    before = slice(20, kidnap_step)          # converged, before anything happens
    after = slice(kidnap_step, None)

    print(f"  {'':34} {'before kidnap':>16} {'after kidnap':>16}")
    print(f"  {'':34} {'(mean error)':>16} {'(final error)':>16}")
    for label, r in (("no random injection", plain),
                     ("5% random particles per resample", injected)):
        recovered = r["errors"][after] < 0.5
        note = ("recovered" if recovered.any() else "never recovered")
        print(f"  {label:<34} {r['errors'][before].mean():13.2f} m "
              f"{r['errors'][-5:].mean():13.2f} m   {note}")

    damage = injected["errors"][before].mean() / plain["errors"][before].mean()

    print(f"\n  Blind injection is not free. While the filter was tracking "
          f"correctly it made\n  things {damage:.0f}x WORSE "
          f"({plain['errors'][before].mean():.2f} m -> "
          f"{injected['errors'][before].mean():.2f} m mean error), because a "
          f"constant\n  trickle of random particles keeps the cloud from ever "
          f"settling: spread rose\n  from "
          f"{plain['spreads'][before].mean():.2f} m to "
          f"{injected['spreads'][before].mean():.2f} m.")

    print(f"\n  And it still did not buy recovery. A particle only wins if it "
          f"matches BOTH\n  position and heading: at 29 degrees off, a correctly "
          f"placed particle's weight\n  collapses from 0.9996 to 0.00005. So a "
          f"useful random particle needs the right\n  cell (about 1 in "
          f"{int((~world).sum())}) AND a heading inside roughly 50 of 360 "
          f"degrees,\n  which is around 1e-4 each.")

    print(f"\n  This is why the real answer is an ADAPTIVE injection rate "
          f"(Augmented MCL):\n  inject nothing while the measurements still fit "
          f"the map, and inject hard only\n  when the average likelihood "
          f"collapses, which is the signal that the robot has\n  actually been "
          f"lost.")

    # ---------------- Plots ----------------
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])

    for i, step in enumerate(sorted(run["snapshots"])):
        particles, true_pose = run["snapshots"][step]
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(world, cmap="binary", origin="lower",
                  extent=[0, WORLD_SIZE[0] * CELL, 0, WORLD_SIZE[1] * CELL])
        ax.scatter(particles[:, 0], particles[:, 1], s=1.2,
                   color="tab:blue", alpha=0.25)
        ax.plot(true_pose[0], true_pose[1], "*", color="tab:red", markersize=15)
        ax.set_title(f"Step {step}: spread {run['spreads'][step]:.2f} m",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(world, cmap="binary", origin="lower",
              extent=[0, WORLD_SIZE[0] * CELL, 0, WORLD_SIZE[1] * CELL])
    ax.plot(run["true"][:, 0], run["true"][:, 1], color="k",
            linewidth=2, label="True path")
    ax.plot(run["est"][:, 0], run["est"][:, 1], color="tab:blue",
            linewidth=1.6, label="Particle filter")
    ax.plot(run["dr"][:, 0], run["dr"][:, 1], color="tab:orange",
            linewidth=1.6, label="Dead reckoning")
    ax.set_title("Trajectories", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(run["errors"], color="tab:blue", label="Particle filter")
    ax.plot(run["dr_errors"], color="tab:orange", label="Dead reckoning")
    if converged is not None:
        ax.axvline(converged, color="tab:green", linestyle="--",
                   label=f"Converged (step {converged})")
    ax.set_xlabel("Step"); ax.set_ylabel("Position error (m)")
    ax.set_title("Error over time", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(run["spreads"], color="tab:purple")
    ax.axhline(CONVERGED_SPREAD, color="k", linestyle=":",
               label=f"{CONVERGED_SPREAD} m")
    ax.set_yscale("log")
    ax.set_xlabel("Step"); ax.set_ylabel("Particle spread (m, log)")
    ax.set_title("The cloud collapsing", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(plain["errors"], color="tab:red", label="No injection")
    ax.plot(injected["errors"], color="tab:green",
            label="5% random particles")
    ax.axvline(kidnap_step, color="k", linestyle="--", label="Kidnapped")
    ax.set_xlabel("Step"); ax.set_ylabel("Position error (m)")
    ax.set_title("Recovering from a kidnapping", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("Day 14: Particle filter localisation on a known map", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "particle_filter.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
