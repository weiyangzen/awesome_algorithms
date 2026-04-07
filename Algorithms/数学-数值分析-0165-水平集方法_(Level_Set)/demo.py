"""Level Set MVP: 2D interface propagation by constant normal speed.

We evolve the level set function phi(x, y, t) with:
    phi_t + F * |grad(phi)| = 0
where F is a constant normal speed.

Demo setup:
- Initial interface: a circle represented by signed distance function.
- Numerical method: first-order explicit Euler in time + Godunov upwind spatial derivative.
- Verification: compare interface-equivalent radius against analytic radius R(t)=R0+F*t.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np


@dataclass
class Snapshot:
    step: int
    time: float
    radius_numeric: float
    radius_exact: float
    radius_abs_error: float
    enclosed_area: float


def make_grid(
    nx: int,
    ny: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Create Cartesian mesh and return X, Y, dx, dy."""
    if nx < 8 or ny < 8:
        raise ValueError("nx and ny must be >= 8")
    x0, x1 = x_range
    y0, y1 = y_range
    if not (x1 > x0 and y1 > y0):
        raise ValueError("invalid coordinate ranges")

    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (ny - 1)
    return xg, yg, dx, dy


def signed_distance_circle(
    xg: np.ndarray,
    yg: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
) -> np.ndarray:
    """Signed distance: negative inside, zero on interface, positive outside."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    dist = np.sqrt((xg - center_x) ** 2 + (yg - center_y) ** 2)
    return dist - radius


def apply_neumann_boundary(phi: np.ndarray) -> None:
    """Apply zero-normal-gradient boundary by edge replication in-place."""
    phi[0, :] = phi[1, :]
    phi[-1, :] = phi[-2, :]
    phi[:, 0] = phi[:, 1]
    phi[:, -1] = phi[:, -2]


def godunov_grad_norm(phi: np.ndarray, dx: float, dy: float, speed: float) -> np.ndarray:
    """Compute upwind |grad(phi)| for Hamiltonian H = speed * |grad(phi)|."""
    p = np.pad(phi, 1, mode="edge")
    c = p[1:-1, 1:-1]
    left = p[1:-1, :-2]
    right = p[1:-1, 2:]
    down = p[:-2, 1:-1]
    up = p[2:, 1:-1]

    dmx = (c - left) / dx
    dpx = (right - c) / dx
    dmy = (c - down) / dy
    dpy = (up - c) / dy

    if speed >= 0.0:
        gx2 = np.maximum(dmx, 0.0) ** 2 + np.minimum(dpx, 0.0) ** 2
        gy2 = np.maximum(dmy, 0.0) ** 2 + np.minimum(dpy, 0.0) ** 2
    else:
        gx2 = np.minimum(dmx, 0.0) ** 2 + np.maximum(dpx, 0.0) ** 2
        gy2 = np.minimum(dmy, 0.0) ** 2 + np.maximum(dpy, 0.0) ** 2

    return np.sqrt(gx2 + gy2)


def estimate_radius_from_area(phi: np.ndarray, dx: float, dy: float) -> Tuple[float, float]:
    """Estimate interface radius by converting enclosed area to equivalent circle."""
    area = float(np.count_nonzero(phi <= 0.0) * dx * dy)
    radius = math.sqrt(max(area, 0.0) / math.pi)
    return radius, area


def evolve_level_set(
    phi0: np.ndarray,
    speed: float,
    dx: float,
    dy: float,
    final_time: float,
    initial_radius: float,
    cfl: float = 0.45,
    snapshot_count: int = 8,
) -> Tuple[np.ndarray, List[Snapshot], float, int]:
    """Run explicit level set evolution and collect snapshots."""
    if not np.all(np.isfinite(phi0)):
        raise ValueError("phi0 contains non-finite values")
    if final_time < 0.0:
        raise ValueError("final_time must be non-negative")
    if not (0.0 < cfl <= 0.95):
        raise ValueError("cfl must be in (0, 0.95]")
    if snapshot_count < 1:
        raise ValueError("snapshot_count must be >= 1")

    if abs(speed) < 1e-14 or final_time == 0.0:
        steps = 1
        dt = final_time
    else:
        dt_cfl = cfl * min(dx, dy) / abs(speed)
        steps = max(1, int(math.ceil(final_time / dt_cfl)))
        dt = final_time / steps

    phi = phi0.copy()
    apply_neumann_boundary(phi)

    snapshots: List[Snapshot] = []
    r0_num, area0 = estimate_radius_from_area(phi, dx, dy)
    snapshots.append(
        Snapshot(
            step=0,
            time=0.0,
            radius_numeric=r0_num,
            radius_exact=initial_radius,
            radius_abs_error=abs(r0_num - initial_radius),
            enclosed_area=area0,
        )
    )

    stride = max(1, steps // snapshot_count)
    for step in range(1, steps + 1):
        grad_norm = godunov_grad_norm(phi, dx, dy, speed)
        phi = phi - dt * speed * grad_norm
        apply_neumann_boundary(phi)

        if not np.all(np.isfinite(phi)):
            raise RuntimeError(f"Non-finite phi encountered at step={step}")

        if (step % stride == 0) or (step == steps):
            t = step * dt
            radius_num, area = estimate_radius_from_area(phi, dx, dy)
            radius_exact = initial_radius + speed * t
            snapshots.append(
                Snapshot(
                    step=step,
                    time=t,
                    radius_numeric=radius_num,
                    radius_exact=radius_exact,
                    radius_abs_error=abs(radius_num - radius_exact),
                    enclosed_area=area,
                )
            )

    return phi, snapshots, dt, steps


def print_report(
    snapshots: List[Snapshot],
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    speed: float,
    dt: float,
    steps: int,
) -> None:
    """Print deterministic, non-interactive diagnostics."""
    print("Level Set MVP (2D constant-speed front propagation)")
    print(
        f"grid={nx}x{ny}, dx={dx:.6f}, dy={dy:.6f}, "
        f"speed={speed:.6f}, dt={dt:.6f}, steps={steps}"
    )
    print(
        "{:<8s} {:<10s} {:<14s} {:<14s} {:<14s} {:<14s}".format(
            "step", "time", "R_numeric", "R_exact", "abs_error", "area_inside"
        )
    )
    for s in snapshots:
        print(
            f"{s.step:<8d} {s.time:<10.5f} {s.radius_numeric:<14.6f} "
            f"{s.radius_exact:<14.6f} {s.radius_abs_error:<14.6f} {s.enclosed_area:<14.6f}"
        )

    final_err = snapshots[-1].radius_abs_error
    max_err = max(s.radius_abs_error for s in snapshots)
    print(f"Final radius abs error: {final_err:.6e}")
    print(f"Max radius abs error over snapshots: {max_err:.6e}")


def main() -> None:
    # Problem setup
    nx, ny = 161, 161
    x_range = (-1.0, 1.0)
    y_range = (-1.0, 1.0)
    center_x, center_y = 0.0, 0.0
    radius0 = 0.30
    speed = 0.35
    final_time = 0.60
    cfl = 0.45

    xg, yg, dx, dy = make_grid(nx, ny, x_range, y_range)
    phi0 = signed_distance_circle(xg, yg, center_x, center_y, radius0)

    _, snapshots, dt, steps = evolve_level_set(
        phi0=phi0,
        speed=speed,
        dx=dx,
        dy=dy,
        final_time=final_time,
        initial_radius=radius0,
        cfl=cfl,
        snapshot_count=8,
    )

    print_report(
        snapshots=snapshots,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        speed=speed,
        dt=dt,
        steps=steps,
    )


if __name__ == "__main__":
    main()
