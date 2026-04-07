"""Bezier surface MVP.

This script implements a tensor-product Bezier surface patch with two
independent evaluation paths:
1) Bernstein polynomial tensor form
2) de Casteljau recursive evaluation

It also validates boundary consistency, partial derivatives, normal vectors,
and convex-hull related properties.
"""

from __future__ import annotations

from math import comb
from typing import Tuple

import numpy as np


Array = np.ndarray


def _as_control_net(control_net: Array) -> Array:
    net = np.asarray(control_net, dtype=float)
    if net.ndim != 3 or net.shape[0] < 1 or net.shape[1] < 1:
        raise ValueError("control_net must have shape (m+1, n+1, dim)")
    return net


def bernstein_basis(n: int, t_values: Array) -> Array:
    t = np.asarray(t_values, dtype=float).reshape(-1)
    omt = 1.0 - t
    basis = np.empty((t.size, n + 1), dtype=float)
    for i in range(n + 1):
        basis[:, i] = comb(n, i) * (t**i) * (omt ** (n - i))
    return basis


def bezier_curve_bernstein(control_points: Array, t_values: Array) -> Array:
    cp = np.asarray(control_points, dtype=float)
    if cp.ndim != 2 or cp.shape[0] < 1:
        raise ValueError("control_points must have shape (k, dim), k >= 1")
    basis = bernstein_basis(cp.shape[0] - 1, t_values)
    return basis @ cp


def bezier_surface_bernstein(control_net: Array, u_values: Array, v_values: Array) -> Array:
    net = _as_control_net(control_net)
    u = np.asarray(u_values, dtype=float).reshape(-1)
    v = np.asarray(v_values, dtype=float).reshape(-1)

    bu = bernstein_basis(net.shape[0] - 1, u)
    bv = bernstein_basis(net.shape[1] - 1, v)
    return np.einsum("ui,vj,ijd->uvd", bu, bv, net, optimize=True)


def de_casteljau_curve_point(control_points: Array, t: float) -> Array:
    cp = np.asarray(control_points, dtype=float)
    if cp.ndim != 2 or cp.shape[0] < 1:
        raise ValueError("control_points must have shape (k, dim), k >= 1")

    work = cp.copy()
    count = work.shape[0]
    for level in range(1, count):
        work[: count - level] = (1.0 - t) * work[: count - level] + t * work[1 : count - level + 1]
    return work[0]


def de_casteljau_surface_point(control_net: Array, u: float, v: float) -> Array:
    net = _as_control_net(control_net)
    rows = np.empty((net.shape[0], net.shape[2]), dtype=float)
    for i in range(net.shape[0]):
        rows[i] = de_casteljau_curve_point(net[i], v)
    return de_casteljau_curve_point(rows, u)


def bezier_surface_de_casteljau(control_net: Array, u_values: Array, v_values: Array) -> Array:
    net = _as_control_net(control_net)
    u = np.asarray(u_values, dtype=float).reshape(-1)
    v = np.asarray(v_values, dtype=float).reshape(-1)
    out = np.empty((u.size, v.size, net.shape[2]), dtype=float)
    for ui, uu in enumerate(u):
        for vi, vv in enumerate(v):
            out[ui, vi] = de_casteljau_surface_point(net, float(uu), float(vv))
    return out


def derivative_control_nets(control_net: Array) -> Tuple[Array, Array]:
    net = _as_control_net(control_net)
    if net.shape[0] < 2 or net.shape[1] < 2:
        raise ValueError("need at least 2x2 control points to compute partial derivatives")
    du = (net.shape[0] - 1) * (net[1:, :, :] - net[:-1, :, :])
    dv = (net.shape[1] - 1) * (net[:, 1:, :] - net[:, :-1, :])
    return du, dv


def surface_partials(control_net: Array, u_values: Array, v_values: Array) -> Tuple[Array, Array]:
    du_net, dv_net = derivative_control_nets(control_net)
    su = bezier_surface_bernstein(du_net, u_values, v_values)
    sv = bezier_surface_bernstein(dv_net, u_values, v_values)
    return su, sv


def surface_point_bernstein(control_net: Array, u: float, v: float) -> Array:
    return bezier_surface_bernstein(control_net, np.array([u]), np.array([v]))[0, 0]


def inside_axis_aligned_hull(points: Array, control_net: Array, tol: float = 1e-12) -> bool:
    pts = np.asarray(points, dtype=float)
    net = _as_control_net(control_net)
    flat = net.reshape(-1, net.shape[2])
    lo = flat.min(axis=0) - tol
    hi = flat.max(axis=0) + tol
    return bool(np.all((pts >= lo) & (pts <= hi)))


def approximate_surface_area(surface_grid: Array) -> float:
    grid = np.asarray(surface_grid, dtype=float)
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise ValueError("surface_grid must have shape (nu, nv, 3)")

    a = grid[:-1, :-1, :]
    b = grid[1:, :-1, :]
    c = grid[:-1, 1:, :]
    d = grid[1:, 1:, :]

    tri1 = np.cross(b - a, c - a)
    tri2 = np.cross(d - b, d - c)
    area = 0.5 * (np.linalg.norm(tri1, axis=2).sum() + np.linalg.norm(tri2, axis=2).sum())
    return float(area)


def build_demo_control_net() -> Array:
    return np.array(
        [
            [[0.0, 0.0, 0.2], [0.0, 1.0, 0.5], [0.0, 2.0, 0.3], [0.0, 3.0, 0.0]],
            [[1.0, 0.0, 0.8], [1.0, 1.0, 1.1], [1.0, 2.0, 0.9], [1.0, 3.0, 0.4]],
            [[2.0, 0.0, 0.7], [2.0, 1.0, 1.2], [2.0, 2.0, 1.0], [2.0, 3.0, 0.6]],
            [[3.0, 0.0, 0.1], [3.0, 1.0, 0.6], [3.0, 2.0, 0.5], [3.0, 3.0, 0.2]],
        ],
        dtype=float,
    )


def main() -> None:
    control_net = build_demo_control_net()
    u_values = np.linspace(0.0, 1.0, 41)
    v_values = np.linspace(0.0, 1.0, 37)

    surface_b = bezier_surface_bernstein(control_net, u_values, v_values)
    surface_d = bezier_surface_de_casteljau(control_net, u_values, v_values)
    method_diff = float(np.max(np.linalg.norm(surface_b - surface_d, axis=2)))

    bu = bernstein_basis(control_net.shape[0] - 1, u_values)
    bv = bernstein_basis(control_net.shape[1] - 1, v_values)
    partition_err = max(
        float(np.max(np.abs(np.sum(bu, axis=1) - 1.0))),
        float(np.max(np.abs(np.sum(bv, axis=1) - 1.0))),
    )

    edge_v = np.linspace(0.0, 1.0, 101)
    edge_u = np.linspace(0.0, 1.0, 101)

    s_u0 = bezier_surface_bernstein(control_net, np.array([0.0]), edge_v)[0]
    s_u1 = bezier_surface_bernstein(control_net, np.array([1.0]), edge_v)[0]
    s_v0 = bezier_surface_bernstein(control_net, edge_u, np.array([0.0]))[:, 0]
    s_v1 = bezier_surface_bernstein(control_net, edge_u, np.array([1.0]))[:, 0]

    c_u0 = bezier_curve_bernstein(control_net[0, :, :], edge_v)
    c_u1 = bezier_curve_bernstein(control_net[-1, :, :], edge_v)
    c_v0 = bezier_curve_bernstein(control_net[:, 0, :], edge_u)
    c_v1 = bezier_curve_bernstein(control_net[:, -1, :], edge_u)

    edge_err = max(
        float(np.max(np.linalg.norm(s_u0 - c_u0, axis=1))),
        float(np.max(np.linalg.norm(s_u1 - c_u1, axis=1))),
        float(np.max(np.linalg.norm(s_v0 - c_v0, axis=1))),
        float(np.max(np.linalg.norm(s_v1 - c_v1, axis=1))),
    )

    corners = np.array(
        [
            surface_point_bernstein(control_net, 0.0, 0.0) - control_net[0, 0],
            surface_point_bernstein(control_net, 0.0, 1.0) - control_net[0, -1],
            surface_point_bernstein(control_net, 1.0, 0.0) - control_net[-1, 0],
            surface_point_bernstein(control_net, 1.0, 1.0) - control_net[-1, -1],
        ]
    )
    corner_err = float(np.max(np.linalg.norm(corners, axis=1)))

    su, sv = surface_partials(control_net, u_values, v_values)
    normals = np.cross(su, sv)
    normal_norm = np.linalg.norm(normals, axis=2)
    center_normal_norm = float(normal_norm[normal_norm.shape[0] // 2, normal_norm.shape[1] // 2])

    eps = 1e-6
    u0, v0 = 0.37, 0.58
    du_fd = (
        surface_point_bernstein(control_net, u0 + eps, v0) - surface_point_bernstein(control_net, u0 - eps, v0)
    ) / (2.0 * eps)
    dv_fd = (
        surface_point_bernstein(control_net, u0, v0 + eps) - surface_point_bernstein(control_net, u0, v0 - eps)
    ) / (2.0 * eps)

    du_net, dv_net = derivative_control_nets(control_net)
    du_exact = surface_point_bernstein(du_net, u0, v0)
    dv_exact = surface_point_bernstein(dv_net, u0, v0)
    du_err = float(np.linalg.norm(du_fd - du_exact))
    dv_err = float(np.linalg.norm(dv_fd - dv_exact))

    hull_ok = inside_axis_aligned_hull(surface_b.reshape(-1, surface_b.shape[2]), control_net, tol=1e-10)
    area_est = approximate_surface_area(surface_b)

    print("Bezier Surface MVP (CS-0279)")
    print(f"Control net: {control_net.shape[0]} x {control_net.shape[1]} (degree {control_net.shape[0]-1}, {control_net.shape[1]-1})")
    print(f"Samples: {u_values.size} x {v_values.size}")
    print(f"Max(Bernstein - de Casteljau): {method_diff:.3e}")
    print(f"Partition of unity error: {partition_err:.3e}")
    print(f"Boundary curve consistency error: {edge_err:.3e}")
    print(f"Corner interpolation error: {corner_err:.3e}")
    print(f"Finite-difference du error: {du_err:.3e}")
    print(f"Finite-difference dv error: {dv_err:.3e}")
    print(f"Center normal magnitude: {center_normal_norm:.6f}")
    print(f"Axis-aligned hull check: {hull_ok}")
    print(f"Approximate surface area: {area_est:.6f}")

    assert method_diff < 1e-11, "Bernstein and de Casteljau surface mismatch"
    assert partition_err < 1e-12, "Bernstein basis partition-of-unity failed"
    assert edge_err < 1e-12, "Boundary curves inconsistent with surface edges"
    assert corner_err < 1e-12, "Surface failed corner interpolation"
    assert du_err < 5e-6 and dv_err < 5e-6, "Partial derivative formula mismatch"
    assert center_normal_norm > 1e-6, "Surface normal degenerated at center"
    assert hull_ok, "Surface violated control-net axis-aligned hull"
    assert area_est > 1e-6, "Estimated surface area should be positive"

    print("All checks passed.")


if __name__ == "__main__":
    main()
