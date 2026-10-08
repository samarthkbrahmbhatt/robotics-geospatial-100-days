"""
Day 12: Occupancy grid mapping from simulated lidar scans.

Day 06 planned a path across a map that was simply handed to it. A real
robot has to build that map itself, from range measurements that are
noisy, sparse, and taken from wherever it happened to be standing.

This drives a robot through two rooms joined by doorways, casts simulated
lidar rays at each pose, and accumulates the scans into an occupancy grid
using an inverse sensor model in log odds.

The interesting part is what the map does NOT contain: space the robot
never saw, and surfaces it only ever saw edge-on.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------- World and robot ----------------
WORLD_SIZE = (60, 40)        # cells (x, y)
CELL = 0.25                  # metres per cell

N_BEAMS = 180                # rays per scan
FOV = 2 * np.pi              # full 360 degree lidar
MAX_RANGE = 8.0              # metres
RANGE_NOISE = 0.03           # metres, standard deviation
HIT_RATE = 0.98              # fraction of beams that return at all

# Log odds. These encode how much one measurement is worth.
L_OCCUPIED = 0.85
L_FREE = -0.4
L_PRIOR = 0.0
L_CLAMP = 6.0                # stop any cell becoming infinitely certain

SEED = 11
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def build_world():
    """The ground truth the robot cannot see: two rooms and an open area.

    Each interior wall has a doorway, so everywhere is reachable. The
    robot has to drive through those doorways to see what is beyond
    them, which is the whole point.
    """
    world = np.zeros(WORLD_SIZE[::-1], dtype=bool)   # (rows, cols) = (y, x)
    world[0, :] = world[-1, :] = True                # outer walls
    world[:, 0] = world[:, -1] = True

    world[20, 1:38] = True                           # horizontal wall
    world[20, 14:19] = False                         #   with a doorway
    world[20:39, 38] = True                          # vertical wall
    world[27:32, 38] = False                         #   with a doorway

    world[5:9, 46:52] = True                         # free-standing blocks
    world[29:34, 48:54] = True
    world[8:12, 26:30] = True
    return world


def robot_route():
    """Waypoints the robot drives through, in metres.

    The route has to pass through the doorways rather than through the
    walls. validate_route is what actually enforces that.
    """
    waypoints = [(1.0, 1.0), (10.0, 1.0), (10.0, 3.5), (4.0, 3.5), (4.0, 7.0),
                 (8.0, 7.0), (8.0, 7.25), (11.0, 7.25), (11.0, 9.0),
                 (14.0, 9.0), (14.0, 6.0)]
    poses = []
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        n = max(int(np.hypot(x1 - x0, y1 - y0) / 0.3), 1)
        for i in range(n):
            t = i / n
            poses.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    poses.append(waypoints[-1])
    return np.array(poses)


def validate_route(world, poses, clearance=1):
    """Refuse to map from a pose inside or touching an obstacle.

    A robot cannot stand inside a wall, and a scan taken from inside one
    reports that wall's own cells as empty, quietly eating obstacles out
    of the finished map. Without this check the route still looks fine
    on the plot and the error shows up only as a slightly worse score,
    which is the hardest kind of bug to notice.
    """
    violations = []
    for x, y in poses:
        cx, cy = int(x / CELL), int(y / CELL)
        patch = world[max(cy - clearance, 0):cy + clearance + 1,
                      max(cx - clearance, 0):cx + clearance + 1]
        if patch.any():
            violations.append((round(float(x), 2), round(float(y), 2)))

    if violations:
        raise ValueError(
            f"{len(violations)} of {len(poses)} poses are inside or touching an "
            f"obstacle, starting at {violations[0]}. Fix the waypoints: a scan "
            f"taken from inside a wall erases that wall from the map."
        )
    return True


def cast_ray(world, x, y, angle, rng):
    """March along a ray until it hits something or runs out of range.

    Returns the measured range and whether anything was actually hit.
    A real lidar misses sometimes, on dark or glancing surfaces, which
    is what HIT_RATE represents.
    """
    step = CELL / 3
    distance = 0.0
    dx, dy = np.cos(angle), np.sin(angle)

    while distance < MAX_RANGE:
        distance += step
        cx = int((x + dx * distance) / CELL)
        cy = int((y + dy * distance) / CELL)
        if not (0 <= cx < WORLD_SIZE[0] and 0 <= cy < WORLD_SIZE[1]):
            return MAX_RANGE, False
        if world[cy, cx]:
            if rng.random() > HIT_RATE:
                return MAX_RANGE, False        # the beam missed this surface
            measured = distance + rng.normal(0, RANGE_NOISE)
            return float(np.clip(measured, 0.0, MAX_RANGE)), True

    return MAX_RANGE, False


def bresenham(x0, y0, x1, y1):
    """Integer cells along a line, so every cell the beam passed through
    is marked free, not only the ones near the endpoints."""
    cells = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def update_map(log_odds, x, y, ranges, angles, hits, trust_dropouts=True):
    """The inverse sensor model, in log odds.

    Every cell along a beam before the endpoint is evidence of free
    space. The endpoint is evidence of an obstacle, but only if the beam
    actually returned: a beam that reached max range without hitting
    anything says the space is free and says nothing about what lies
    beyond it.

    Log odds is used because combining measurements then becomes
    addition rather than repeated application of Bayes' rule.

    trust_dropouts decides what to do with a beam that returned nothing.
    A real lidar cannot tell "nothing out there" from "that surface did
    not reflect", so both arrive as an empty return. Trusting them means
    clearing the whole ray, which carves straight through any surface
    that failed to reflect. Discarding them means throwing away real
    evidence about genuinely empty space. Neither is free.
    """
    cx0, cy0 = int(x / CELL), int(y / CELL)

    for r, angle, hit in zip(ranges, angles, hits):
        if not hit and not trust_dropouts:
            continue
        ex = x + np.cos(angle) * r
        ey = y + np.sin(angle) * r
        cx1 = int(np.clip(ex / CELL, 0, WORLD_SIZE[0] - 1))
        cy1 = int(np.clip(ey / CELL, 0, WORLD_SIZE[1] - 1))

        cells = bresenham(cx0, cy0, cx1, cy1)
        for cx, cy in cells[:-1]:
            log_odds[cy, cx] += L_FREE
        if hit:
            log_odds[cy1, cx1] += L_OCCUPIED
        else:
            log_odds[cy1, cx1] += L_FREE      # ran out of range, still free

    np.clip(log_odds, -L_CLAMP, L_CLAMP, out=log_odds)


def self_test():
    """Check the bookkeeping against hand-computable answers."""
    assert abs(1 / (1 + np.exp(-0.0)) - 0.5) < 1e-12   # log odds 0 is "no idea"

    occupied_once = 1 / (1 + np.exp(-L_OCCUPIED))
    free_once = 1 / (1 + np.exp(-L_FREE))
    assert occupied_once > 0.5, "an occupied reading must raise the probability"
    assert free_once < 0.5, "a free reading must lower the probability"
    print(f"  one occupied reading -> p = {occupied_once:.3f}")
    print(f"  one free reading     -> p = {free_once:.3f}")

    for n in (1, 3, 10):
        p = 1 / (1 + np.exp(-min(n * L_OCCUPIED, L_CLAMP)))
        print(f"  {n:2d} occupied readings -> p = {p:.4f}")

    p_max = 1 / (1 + np.exp(-L_CLAMP))
    assert p_max < 1.0, "clamped probability must stay below certainty"
    print(f"  clamp at +-{L_CLAMP} -> p never exceeds {p_max:.4f}")

    line = bresenham(0, 0, 5, 3)
    assert line[0] == (0, 0) and line[-1] == (5, 3)
    assert bresenham(5, 3, 0, 0)[0] == (5, 3)
    print(f"  bresenham (0,0)->(5,3) covers {len(line)} cells, endpoints included")

    # The route validator has to actually reject a bad route, or it is decoration
    toy = np.zeros((10, 10), dtype=bool)
    toy[5, 5] = True
    try:
        validate_route(toy, np.array([[5 * CELL, 5 * CELL]]), clearance=0)
    except ValueError:
        print("  route validator rejects a pose inside an obstacle, as it should")
    else:
        raise AssertionError("the route validator failed to catch a bad pose")


def run_mapping(world, poses, angles, trust_dropouts, seed=SEED, snapshot_at=()):
    """Build a map from the whole route. Returns log odds and snapshots."""
    rng = np.random.default_rng(seed)
    log_odds = np.full(world.shape, L_PRIOR)
    snapshots, total_hits = {}, 0

    for i, (x, y) in enumerate(poses):
        results = [cast_ray(world, x, y, a, rng) for a in angles]
        ranges = np.array([r for r, _ in results])
        hits = np.array([h for _, h in results])
        total_hits += int(hits.sum())
        update_map(log_odds, x, y, ranges, angles, hits, trust_dropouts)
        if i in snapshot_at:
            snapshots[i] = log_odds.copy()

    return log_odds, snapshots, total_hits


def score(log_odds, world):
    """Precision and recall of the finished map against ground truth."""
    p = 1 / (1 + np.exp(-log_odds))
    occupied = p > 0.65
    free = p < 0.35
    correct = int((occupied & world).sum())
    phantom = int((occupied & ~world).sum())
    missed = int((free & world).sum())
    precision = 100 * correct / max(int(occupied.sum()), 1)
    recall = 100 * correct / int(world.sum())
    unknown = 100 * (1 - (np.abs(log_odds) > 0.5).mean())
    return dict(p=p, occupied=occupied, free=free, correct=correct,
                phantom=phantom, missed=missed, precision=precision,
                recall=recall, unknown=unknown)


def main():
    print("Self-test:")
    self_test()

    world = build_world()
    poses = robot_route()
    validate_route(world, poses)
    print(f"  route of {len(poses)} poses, none inside or touching an obstacle")

    angles = np.linspace(0, FOV, N_BEAMS, endpoint=False)

    snapshot_at = {len(poses) // 10, len(poses) // 3, len(poses) - 1}

    print(f"\nWorld: {WORLD_SIZE[0]} x {WORLD_SIZE[1]} cells at {CELL} m "
          f"({WORLD_SIZE[0] * CELL:.0f} x {WORLD_SIZE[1] * CELL:.0f} m)")
    print(f"Lidar: {N_BEAMS} beams per scan, max range {MAX_RANGE} m, "
          f"{100 * (1 - HIT_RATE):.0f}% of beams return nothing")

    log_odds, snapshots, total_hits = run_mapping(
        world, poses, angles, trust_dropouts=True, snapshot_at=snapshot_at)
    trusting = score(log_odds, world)

    discarding_log_odds, _, _ = run_mapping(
        world, poses, angles, trust_dropouts=False)
    discarding = score(discarding_log_odds, world)

    probability = trusting["p"]
    believed_occupied = trusting["occupied"]
    believed_free = trusting["free"]

    print(f"\nAfter {len(poses)} scans ({total_hits} beam returns):")
    print(f"  Cells with no opinion either way: {trusting['unknown']:.1f}%")
    print(f"  Called occupied: {int(believed_occupied.sum()):5d} "
          f"({trusting['correct']} correct, {trusting['phantom']} phantom)")
    print(f"  Called free:     {int(believed_free.sum()):5d} "
          f"({trusting['missed']} of them were really obstacles)")

    print(f"\nWhat to do with the {100 * (1 - HIT_RATE):.0f}% of beams "
          f"that return nothing:")
    print(f"{'':28} {'precision':>10} {'recall':>9} {'missed':>8} {'unknown':>9}")
    for label, r in (("trust them as free space", trusting),
                     ("discard them entirely", discarding)):
        print(f"  {label:<26} {r['precision']:9.1f}% {r['recall']:8.1f}% "
              f"{r['missed']:8d} {r['unknown']:8.1f}%")

    print(f"\n  A beam that returns nothing is ambiguous: the space really is "
          f"empty,\n  or a surface failed to reflect. Trusting them clears a "
          f"line straight\n  through anything that did not reflect, which is "
          f"what hollows out the\n  blocks. Discarding them keeps the obstacles "
          f"but leaves more of the\n  map unexplored.")

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    axes[0, 0].imshow(world, cmap="binary", origin="lower")
    axes[0, 0].plot(poses[:, 0] / CELL, poses[:, 1] / CELL,
                    color="tab:red", linewidth=2, label="Robot path")
    axes[0, 0].plot(poses[0, 0] / CELL, poses[0, 1] / CELL, "o",
                    color="tab:green", markersize=8, label="Start")
    axes[0, 0].set_title("Ground truth (the robot never sees this)")
    axes[0, 0].legend(fontsize=8, loc="upper left")

    for ax, (step, snap) in zip([axes[0, 1], axes[0, 2], axes[1, 0]],
                                sorted(snapshots.items())):
        ax.imshow(1 / (1 + np.exp(-snap)), cmap="gray_r",
                  origin="lower", vmin=0, vmax=1)
        ax.plot(poses[:step + 1, 0] / CELL, poses[:step + 1, 1] / CELL,
                color="tab:red", linewidth=1.5)
        ax.set_title(f"After {step + 1} scans")

    im = axes[1, 1].imshow(probability, cmap="gray_r", origin="lower", vmin=0, vmax=1)
    axes[1, 1].set_title("Final map (white = free, black = occupied)")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.035, label="P(occupied)")

    comparison = np.full(world.shape + (3,), 0.85)            # unknown: grey
    comparison[believed_free & ~world] = (1, 1, 1)            # correct free
    comparison[believed_occupied & world] = (0.1, 0.1, 0.1)   # correct obstacle
    comparison[believed_occupied & ~world] = (0.9, 0.2, 0.2)  # phantom obstacle
    comparison[believed_free & world] = (0.2, 0.4, 0.95)      # missed obstacle
    axes[1, 2].imshow(comparison, origin="lower")
    axes[1, 2].set_title("Grey = no opinion, red = phantom, blue = missed")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Day 12: Occupancy grid mapping from simulated lidar", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "occupancy_grid.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
