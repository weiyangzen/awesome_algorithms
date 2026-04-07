"""D* Lite minimal runnable MVP on a 2D grid with dynamic obstacle replanning."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

Point = Tuple[int, int]
Key = Tuple[float, float]


@dataclass
class PriorityQueue:
    """Min-heap with lazy deletion for D* Lite keys."""

    heap: List[Tuple[float, float, Point]] = field(default_factory=list)
    active_keys: Dict[Point, Key] = field(default_factory=dict)

    def push(self, node: Point, key: Key) -> None:
        self.active_keys[node] = key
        heapq.heappush(self.heap, (key[0], key[1], node))

    def remove(self, node: Point) -> None:
        self.active_keys.pop(node, None)

    def _discard_stale_top(self) -> None:
        while self.heap:
            k1, k2, node = self.heap[0]
            active = self.active_keys.get(node)
            if active is None or active != (k1, k2):
                heapq.heappop(self.heap)
                continue
            break

    def top_key(self) -> Key:
        self._discard_stale_top()
        if not self.heap:
            return (float("inf"), float("inf"))
        k1, k2, _ = self.heap[0]
        return (k1, k2)

    def pop(self) -> Tuple[Key, Optional[Point]]:
        self._discard_stale_top()
        if not self.heap:
            return (float("inf"), float("inf")), None

        k1, k2, node = heapq.heappop(self.heap)
        self.active_keys.pop(node, None)
        return (k1, k2), node


class DStarLite:
    """Small, explicit D* Lite implementation for 4-neighbor grid worlds."""

    def __init__(self, grid: np.ndarray, start: Point, goal: Point) -> None:
        self.grid = self._validate_grid(grid)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.goal = goal
        self.last = start
        self._validate_endpoint("start", self.start)
        self._validate_endpoint("goal", self.goal)

        self.g = np.full((self.rows, self.cols), float("inf"), dtype=float)
        self.rhs = np.full((self.rows, self.cols), float("inf"), dtype=float)
        self.rhs[self.goal] = 0.0

        self.km = 0.0
        self.open = PriorityQueue()
        self.open.push(self.goal, self.calculate_key(self.goal))
        self.expansions = 0

    @staticmethod
    def _validate_grid(grid: np.ndarray) -> np.ndarray:
        mat = np.asarray(grid, dtype=int)
        if mat.ndim != 2:
            raise ValueError(f"grid must be 2D, got ndim={mat.ndim}")
        if mat.size == 0:
            raise ValueError("grid must be non-empty")
        if not np.all((mat == 0) | (mat == 1)):
            raise ValueError("grid values must be 0 (free) or 1 (blocked)")
        return mat

    def _validate_endpoint(self, name: str, p: Point) -> None:
        r, c = p
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"{name} {p} is out of bounds for grid shape {self.grid.shape}")
        if self.grid[p] == 1:
            raise ValueError(f"{name} {p} is blocked")

    def heuristic(self, a: Point, b: Point) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def in_bounds(self, p: Point) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def adjacent_cells(self, p: Point) -> List[Point]:
        r, c = p
        out: List[Point] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            q = (r + dr, c + dc)
            if self.in_bounds(q):
                out.append(q)
        return out

    def cost(self, u: Point, v: Point) -> float:
        if not self.in_bounds(u) or not self.in_bounds(v):
            return float("inf")
        if self.heuristic(u, v) != 1.0:
            return float("inf")
        if self.grid[u] == 1 or self.grid[v] == 1:
            return float("inf")
        return 1.0

    @staticmethod
    def key_less(a: Key, b: Key) -> bool:
        return a < b

    def calculate_key(self, s: Point) -> Key:
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + self.heuristic(self.start, s) + self.km, g_rhs)

    def update_vertex(self, u: Point) -> None:
        if u != self.goal:
            candidates = [self.cost(u, s) + self.g[s] for s in self.adjacent_cells(u)]
            self.rhs[u] = min(candidates) if candidates else float("inf")

        if u in self.open.active_keys:
            self.open.remove(u)

        if self.g[u] != self.rhs[u]:
            self.open.push(u, self.calculate_key(u))

    def compute_shortest_path(self, max_iterations: int = 1_000_000) -> int:
        iters = 0
        while self.key_less(self.open.top_key(), self.calculate_key(self.start)) or self.rhs[self.start] != self.g[self.start]:
            if iters >= max_iterations:
                raise RuntimeError("compute_shortest_path reached max_iterations")

            k_old, u = self.open.pop()
            if u is None:
                break

            k_new = self.calculate_key(u)
            if self.key_less(k_old, k_new):
                self.open.push(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.expansions += 1
                for p in self.adjacent_cells(u):
                    self.update_vertex(p)
            else:
                self.g[u] = float("inf")
                self.expansions += 1
                self.update_vertex(u)
                for p in self.adjacent_cells(u):
                    self.update_vertex(p)
            iters += 1
        return iters

    def current_path(self, max_steps: int = 10_000) -> Optional[List[Point]]:
        if np.isinf(self.g[self.start]):
            return None

        path = [self.start]
        cur = self.start
        visited = {cur}

        for _ in range(max_steps):
            if cur == self.goal:
                return path

            best_next: Optional[Point] = None
            best_score = float("inf")
            for nxt in self.adjacent_cells(cur):
                score = self.cost(cur, nxt) + self.g[nxt]
                if score < best_score:
                    best_score = score
                    best_next = nxt

            if best_next is None or np.isinf(best_score):
                return None
            if best_next in visited:
                return None

            path.append(best_next)
            visited.add(best_next)
            cur = best_next

        return None

    def move_start(self, new_start: Point) -> None:
        self._validate_endpoint("new_start", new_start)
        self.last = self.start
        self.start = new_start
        self.km += self.heuristic(self.last, self.start)

    def set_obstacle(self, cell: Point, blocked: bool) -> None:
        if not self.in_bounds(cell):
            raise ValueError(f"cell {cell} is out of bounds")
        if cell == self.start or cell == self.goal:
            raise ValueError("cannot block current start or goal")

        old = bool(self.grid[cell])
        if old == blocked:
            return

        self.grid[cell] = 1 if blocked else 0
        affected = [cell] + self.adjacent_cells(cell)
        for u in affected:
            self.update_vertex(u)


def render_grid(grid: np.ndarray, start: Point, goal: Point, path: Optional[List[Point]] = None) -> str:
    """Text grid for deterministic demo output."""
    mat = np.asarray(grid, dtype=int)
    canvas = np.full(mat.shape, ".", dtype="<U1")
    canvas[mat == 1] = "#"

    if path is not None:
        for r, c in path:
            if (r, c) not in (start, goal):
                canvas[r, c] = "*"

    canvas[start[0], start[1]] = "S"
    canvas[goal[0], goal[1]] = "G"
    return "\n".join(" ".join(row) for row in canvas)


def path_cost(path: Optional[List[Point]]) -> float:
    if path is None:
        return float("inf")
    return float(len(path) - 1)


def main() -> None:
    # Two gaps in the middle wall. Initially the left gap is open and preferred.
    # During execution we discover that gap and block it, forcing a replan.
    grid = np.zeros((7, 10), dtype=int)
    grid[3, 1:9] = 1
    grid[3, 3] = 0
    grid[3, 7] = 0

    start = (6, 1)
    goal = (0, 8)

    planner = DStarLite(grid=grid, start=start, goal=goal)

    init_iters = planner.compute_shortest_path()
    init_path = planner.current_path()

    print("=== Initial Plan ===")
    print(f"start={planner.start}, goal={planner.goal}")
    print(f"compute iterations : {init_iters}")
    print(f"expansions         : {planner.expansions}")
    print(f"path cost          : {path_cost(init_path)}")
    print(render_grid(planner.grid, planner.start, planner.goal, init_path))
    print(f"path sequence      : {init_path}")

    if init_path is None or len(init_path) < 4:
        raise RuntimeError("initial path generation failed or path too short for demo")

    # Robot moves two steps along the current policy.
    moved_to = init_path[2]
    planner.move_start(moved_to)
    move_iters = planner.compute_shortest_path()
    move_path = planner.current_path()

    print("\n=== After Robot Moves ===")
    print(f"new start          : {planner.start}")
    print(f"km                 : {planner.km}")
    print(f"compute iterations : {move_iters}")
    print(f"path cost          : {path_cost(move_path)}")
    print(render_grid(planner.grid, planner.start, planner.goal, move_path))
    print(f"path sequence      : {move_path}")

    # Dynamic change: the near gap is now discovered blocked.
    planner.set_obstacle((3, 3), blocked=True)
    repl_iters = planner.compute_shortest_path()
    repl_path = planner.current_path()

    print("\n=== Replan After New Obstacle at (3, 3) ===")
    print(f"compute iterations : {repl_iters}")
    print(f"total expansions   : {planner.expansions}")
    print(f"path cost          : {path_cost(repl_path)}")
    print(render_grid(planner.grid, planner.start, planner.goal, repl_path))
    print(f"path sequence      : {repl_path}")

    if repl_path is None:
        raise RuntimeError("replanning failed unexpectedly")


if __name__ == "__main__":
    main()
