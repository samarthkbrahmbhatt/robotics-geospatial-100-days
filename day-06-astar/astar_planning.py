"""
Day 06: A* path planning on a grid map.

Three searches over the same map, so the difference is the strategy
and nothing else:

  Dijkstra          explores by distance travelled so far
  A*                adds a heuristic estimate of distance remaining
  Greedy best-first uses the heuristic alone

Movement is 8-connected. Diagonal steps cost sqrt(2), straight steps
cost 1, and cutting a corner between two obstacles is not allowed.
"""

import heapq
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------- Map ----------------
WIDTH, HEIGHT = 60, 40
START = (2, 20)
GOAL = (57, 20)
SEED = 7

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SQRT2 = np.sqrt(2)
NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]


def build_map():
    """Scattered blocks first, then vertical walls with a gap in each.

    Order matters: the walls are carved last so a random block can
    never seal a gap and make the goal unreachable.
    """
    grid = np.zeros((HEIGHT, WIDTH), dtype=bool)   # True = obstacle

    rng = np.random.default_rng(SEED)
    for _ in range(18):
        y = rng.integers(0, HEIGHT - 4)
        x = rng.integers(0, WIDTH - 4)
        grid[y:y + rng.integers(2, 5), x:x + rng.integers(2, 5)] = True

    for x, gap_y in ((14, 6), (28, 31), (42, 12)):
        grid[:, x] = True
        grid[gap_y:gap_y + 7, x] = False
        grid[gap_y:gap_y + 7, x - 1] = False       # keep the approach clear
        grid[gap_y:gap_y + 7, x + 1] = False

    for cx, cy in (START, GOAL):                   # clear space to start and finish
        grid[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3] = False

    return grid


def octile(a, b):
    """Shortest possible distance ignoring obstacles, for 8-connected moves.

    Never overestimates, which is what makes A* return an optimal path.
    """
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2) * min(dx, dy)


def search(grid, start, goal, weight=1.0, greedy=False):
    """One search function, three behaviours.

    weight=0  -> Dijkstra (heuristic ignored)
    weight=1  -> A*
    greedy    -> priority is the heuristic alone
    """
    open_heap = [(0.0, start)]
    came_from = {}
    cost_so_far = {start: 0.0}
    expanded = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in expanded:
            continue
        expanded.add(current)

        if current == goal:
            break

        cx, cy = current
        for dx, dy in NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < WIDTH and 0 <= ny < HEIGHT) or grid[ny, nx]:
                continue
            # No squeezing diagonally between two obstacles
            if dx and dy and (grid[cy, nx] or grid[ny, cx]):
                continue

            step = SQRT2 if dx and dy else 1.0
            new_cost = cost_so_far[current] + step
            neighbour = (nx, ny)
            if new_cost < cost_so_far.get(neighbour, float("inf")):
                cost_so_far[neighbour] = new_cost
                h = octile(neighbour, goal)
                priority = h if greedy else new_cost + weight * h
                heapq.heappush(open_heap, (priority, neighbour))
                came_from[neighbour] = current

    if goal not in came_from and goal != start:
        return None, expanded, float("inf")

    path, node = [goal], goal
    while node != start:
        node = came_from[node]
        path.append(node)
    return path[::-1], expanded, cost_so_far[goal]


def draw(ax, grid, path, expanded, title):
    canvas = np.zeros((HEIGHT, WIDTH, 3))
    canvas[:] = 1.0
    for x, y in expanded:
        canvas[y, x] = (0.85, 0.90, 1.0)       # searched
    canvas[grid] = (0.25, 0.25, 0.3)           # obstacles

    ax.imshow(canvas, origin="lower")
    if path:
        ax.plot([p[0] for p in path], [p[1] for p in path],
                color="tab:red", linewidth=2)
    ax.plot(*START, "o", color="tab:green", markersize=9)
    ax.plot(*GOAL, "*", color="tab:orange", markersize=15)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    grid = build_map()
    free = int((~grid).sum())
    print(f"Map: {WIDTH} x {HEIGHT}, {free} free cells, start {START}, goal {GOAL}\n")

    runs = [
        ("Dijkstra (no heuristic)", dict(weight=0.0)),
        ("A* (octile heuristic)", dict(weight=1.0)),
        ("Greedy best-first", dict(greedy=True)),
        ("Weighted A* (w=2.5)", dict(weight=2.5)),
    ]

    results = []
    for name, kwargs in runs:
        path, expanded, cost = search(grid, START, GOAL, **kwargs)
        results.append((name, path, expanded, cost))
        print(f"{name:<26} cost {cost:7.2f}   cells searched {len(expanded):5d}   "
              f"path length {len(path) if path else 0:4d}")

    optimal = results[0][3]
    print(f"\nOptimal cost (Dijkstra): {optimal:.2f}")
    for name, _, expanded, cost in results[1:]:
        gap = 100 * (cost - optimal) / optimal
        print(f"  {name:<24} {gap:+5.1f}% longer, "
              f"searched {100 * len(expanded) / len(results[0][2]):5.1f}% as many cells")

    assert abs(results[1][3] - optimal) < 1e-9, "A* did not find the optimal path"
    print("\nCheck passed: A* matches Dijkstra's cost exactly, as an admissible heuristic guarantees.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, path, expanded, cost) in zip(axes.ravel(), results):
        draw(ax, grid, path, expanded,
             f"{name}\ncost {cost:.1f}, searched {len(expanded)} cells")

    fig.suptitle("Day 06: A* and friends on a grid map", fontsize=15)
    fig.tight_layout()
    out = OUT_DIR / "astar_planning.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
