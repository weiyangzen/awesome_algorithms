"""Mortar finite element MVP on a decomposed 2D Poisson problem.

This demo solves:
    -Δu = f  in Ω = (0,1) x (0,1),
with homogeneous Dirichlet boundary conditions and exact solution:
    u(x, y) = sin(pi x) sin(pi y).

The domain is split into two non-overlapping subdomains with nonmatching
meshes at the interface x = 0.5. A mortar-style weak continuity constraint is
imposed via Lagrange multipliers (piecewise-constant basis on interface
segments).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np


@dataclass(frozen=True)
class StructuredSubdomain:
    """Rectangular structured mesh for one subdomain."""

    xa: float
    xb: float
    nx: int
    ny: int

    def node_id(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def hx(self) -> float:
        return (self.xb - self.xa) / self.nx

    @property
    def hy(self) -> float:
        return 1.0 / self.ny

    def node_coord(self, i: int, j: int) -> tuple[float, float]:
        return (self.xa + i * self.hx, j * self.hy)

    def is_dirichlet(self, i: int, j: int) -> bool:
        # Top/bottom are always outer boundaries.
        if j == 0 or j == self.ny:
            return True
        # Left physical boundary only for left subdomain.
        if abs(self.xa - 0.0) < 1e-14 and i == 0:
            return True
        # Right physical boundary only for right subdomain.
        if abs(self.xb - 1.0) < 1e-14 and i == self.nx:
            return True
        return False


def exact_u(x: float, y: float) -> float:
    return np.sin(pi * x) * np.sin(pi * y)


def rhs_f(x: float, y: float) -> float:
    return 2.0 * (pi**2) * np.sin(pi * x) * np.sin(pi * y)


def build_free_dof_map(sub: StructuredSubdomain) -> dict[int, int]:
    dof_map: dict[int, int] = {}
    counter = 0
    for j in range(sub.ny + 1):
        for i in range(sub.nx + 1):
            if sub.is_dirichlet(i, j):
                continue
            dof_map[sub.node_id(i, j)] = counter
            counter += 1
    return dof_map


def q1_shape_and_grads(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    dndxi = 0.25 * np.array(
        [
            -(1.0 - eta),
            +(1.0 - eta),
            +(1.0 + eta),
            -(1.0 + eta),
        ]
    )
    dndeta = 0.25 * np.array(
        [
            -(1.0 - xi),
            -(1.0 + xi),
            +(1.0 + xi),
            +(1.0 - xi),
        ]
    )
    return n, dndxi, dndeta


def assemble_subdomain_system(
    sub: StructuredSubdomain,
    f_callable,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Assemble local FEM matrix and load vector with homogeneous Dirichlet BC."""

    dof_map = build_free_dof_map(sub)
    ndof = len(dof_map)
    a = np.zeros((ndof, ndof), dtype=float)
    b = np.zeros(ndof, dtype=float)

    gp = [-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)]

    for j in range(sub.ny):
        for i in range(sub.nx):
            n00 = sub.node_id(i, j)
            n10 = sub.node_id(i + 1, j)
            n11 = sub.node_id(i + 1, j + 1)
            n01 = sub.node_id(i, j + 1)
            elem_nodes = [n00, n10, n11, n01]

            coords = np.array(
                [
                    sub.node_coord(i, j),
                    sub.node_coord(i + 1, j),
                    sub.node_coord(i + 1, j + 1),
                    sub.node_coord(i, j + 1),
                ],
                dtype=float,
            )

            ke = np.zeros((4, 4), dtype=float)
            fe = np.zeros(4, dtype=float)

            for xi in gp:
                for eta in gp:
                    n, dndxi, dndeta = q1_shape_and_grads(xi, eta)

                    jac = np.zeros((2, 2), dtype=float)
                    jac[0, 0] = np.dot(dndxi, coords[:, 0])
                    jac[0, 1] = np.dot(dndeta, coords[:, 0])
                    jac[1, 0] = np.dot(dndxi, coords[:, 1])
                    jac[1, 1] = np.dot(dndeta, coords[:, 1])

                    det_j = np.linalg.det(jac)
                    if det_j <= 0.0:
                        raise ValueError("Invalid element Jacobian detected.")

                    grad_ref = np.vstack((dndxi, dndeta))
                    grad_phys = np.linalg.solve(jac.T, grad_ref)

                    ke += (grad_phys.T @ grad_phys) * det_j

                    xq = float(np.dot(n, coords[:, 0]))
                    yq = float(np.dot(n, coords[:, 1]))
                    fe += n * f_callable(xq, yq) * det_j

            for a_local, node_a in enumerate(elem_nodes):
                ia = dof_map.get(node_a)
                if ia is None:
                    continue
                b[ia] += fe[a_local]
                for b_local, node_b in enumerate(elem_nodes):
                    ib = dof_map.get(node_b)
                    if ib is None:
                        continue
                    a[ia, ib] += ke[a_local, b_local]

    return a, b, dof_map


def interface_node_id(sub: StructuredSubdomain, j: int, side: str) -> int:
    i = sub.nx if side == "right" else 0
    return sub.node_id(i, j)


def assemble_mortar_matrix(
    sub: StructuredSubdomain,
    dof_map: dict[int, int],
    side: str,
    n_lambda: int,
    quad_order: int = 3,
) -> np.ndarray:
    """Build B matrix for piecewise-constant mortar multipliers."""

    bmat_local = np.zeros((n_lambda, len(dof_map)), dtype=float)
    qp, qw = np.polynomial.legendre.leggauss(quad_order)
    h_side = 1.0 / sub.ny

    for k in range(n_lambda):
        y0 = k / n_lambda
        y1 = (k + 1) / n_lambda
        mid = 0.5 * (y0 + y1)
        half = 0.5 * (y1 - y0)

        for t, w in zip(qp, qw):
            y = mid + half * t
            weight = half * w

            j = min(int(y / h_side), sub.ny - 1)
            yj = j * h_side
            xi = (y - yj) / h_side

            n0 = interface_node_id(sub, j, side)
            n1 = interface_node_id(sub, j + 1, side)
            phi0 = 1.0 - xi
            phi1 = xi

            i0 = dof_map.get(n0)
            i1 = dof_map.get(n1)
            if i0 is not None:
                bmat_local[k, i0] += weight * phi0
            if i1 is not None:
                bmat_local[k, i1] += weight * phi1

    return bmat_local


def expand_to_full(sub: StructuredSubdomain, dof_map: dict[int, int], u_free: np.ndarray) -> np.ndarray:
    u = np.zeros(sub.n_nodes, dtype=float)
    for node_id, idx in dof_map.items():
        u[node_id] = u_free[idx]
    return u


def trace_value(sub: StructuredSubdomain, u_full: np.ndarray, side: str, y: float) -> float:
    y_clip = min(max(y, 0.0), 1.0)
    h_side = 1.0 / sub.ny
    j = min(int(y_clip / h_side), sub.ny - 1)
    yj = j * h_side
    xi = (y_clip - yj) / h_side
    n0 = interface_node_id(sub, j, side)
    n1 = interface_node_id(sub, j + 1, side)
    return (1.0 - xi) * u_full[n0] + xi * u_full[n1]


def compute_l2_error(sub: StructuredSubdomain, u_full: np.ndarray) -> float:
    gp = [-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)]
    err2 = 0.0

    for j in range(sub.ny):
        for i in range(sub.nx):
            elem_nodes = [
                sub.node_id(i, j),
                sub.node_id(i + 1, j),
                sub.node_id(i + 1, j + 1),
                sub.node_id(i, j + 1),
            ]
            coords = np.array(
                [
                    sub.node_coord(i, j),
                    sub.node_coord(i + 1, j),
                    sub.node_coord(i + 1, j + 1),
                    sub.node_coord(i, j + 1),
                ],
                dtype=float,
            )
            u_elem = u_full[elem_nodes]

            for xi in gp:
                for eta in gp:
                    n, dndxi, dndeta = q1_shape_and_grads(xi, eta)
                    jac = np.zeros((2, 2), dtype=float)
                    jac[0, 0] = np.dot(dndxi, coords[:, 0])
                    jac[0, 1] = np.dot(dndeta, coords[:, 0])
                    jac[1, 0] = np.dot(dndxi, coords[:, 1])
                    jac[1, 1] = np.dot(dndeta, coords[:, 1])
                    det_j = np.linalg.det(jac)

                    xq = float(np.dot(n, coords[:, 0]))
                    yq = float(np.dot(n, coords[:, 1]))
                    uh = float(np.dot(n, u_elem))
                    ue = exact_u(xq, yq)
                    err2 += (uh - ue) ** 2 * det_j

    return sqrt(err2)


def compute_interface_jump_l2(
    left: StructuredSubdomain,
    u_left: np.ndarray,
    right: StructuredSubdomain,
    u_right: np.ndarray,
    n_segments: int = 20,
    quad_order: int = 4,
) -> tuple[float, float]:
    qp, qw = np.polynomial.legendre.leggauss(quad_order)
    jump_l2 = 0.0
    max_abs_jump = 0.0

    for k in range(n_segments):
        y0 = k / n_segments
        y1 = (k + 1) / n_segments
        mid = 0.5 * (y0 + y1)
        half = 0.5 * (y1 - y0)

        for t, w in zip(qp, qw):
            y = mid + half * t
            weight = half * w
            jump = trace_value(left, u_left, "right", y) - trace_value(right, u_right, "left", y)
            jump_l2 += jump * jump * weight
            max_abs_jump = max(max_abs_jump, abs(jump))

    return sqrt(jump_l2), max_abs_jump


def main() -> None:
    # Nonmatching interface grids in y-direction (ny differs).
    left = StructuredSubdomain(xa=0.0, xb=0.5, nx=10, ny=8)
    right = StructuredSubdomain(xa=0.5, xb=1.0, nx=7, ny=5)

    a_l, b_l, map_l = assemble_subdomain_system(left, rhs_f)
    a_r, b_r, map_r = assemble_subdomain_system(right, rhs_f)

    # Piecewise-constant multipliers on the coarser interface partition.
    n_lambda = min(left.ny, right.ny)
    b_lm = assemble_mortar_matrix(left, map_l, side="right", n_lambda=n_lambda)
    b_rm = assemble_mortar_matrix(right, map_r, side="left", n_lambda=n_lambda)

    n_l = len(map_l)
    n_r = len(map_r)

    # Dense KKT assembly for portability in minimal environments.
    total = n_l + n_r + n_lambda
    kkt = np.zeros((total, total), dtype=float)

    kkt[:n_l, :n_l] = a_l
    kkt[n_l : n_l + n_r, n_l : n_l + n_r] = a_r

    kkt[:n_l, n_l + n_r :] = b_lm.T
    kkt[n_l : n_l + n_r, n_l + n_r :] = -b_rm.T

    kkt[n_l + n_r :, :n_l] = b_lm
    kkt[n_l + n_r :, n_l : n_l + n_r] = -b_rm

    rhs = np.concatenate([b_l, b_r, np.zeros(n_lambda, dtype=float)])
    sol = np.linalg.solve(kkt, rhs)

    u_l_free = sol[:n_l]
    u_r_free = sol[n_l : n_l + n_r]
    lam = sol[n_l + n_r :]

    u_l = expand_to_full(left, map_l, u_l_free)
    u_r = expand_to_full(right, map_r, u_r_free)

    l2_left = compute_l2_error(left, u_l)
    l2_right = compute_l2_error(right, u_r)
    jump_l2, jump_max = compute_interface_jump_l2(left, u_l, right, u_r)

    print("Mortar FEM MVP (2D Poisson, nonmatching interface grids)")
    print(f"left mesh  : nx={left.nx}, ny={left.ny}, free dof={n_l}")
    print(f"right mesh : nx={right.nx}, ny={right.ny}, free dof={n_r}")
    print(f"lambda dof : {n_lambda}")
    print(f"L2 error (left)  = {l2_left:.6e}")
    print(f"L2 error (right) = {l2_right:.6e}")
    print(f"interface jump L2  = {jump_l2:.6e}")
    print(f"interface jump max = {jump_max:.6e}")
    print(f"||lambda||_2       = {np.linalg.norm(lam):.6e}")


if __name__ == "__main__":
    main()
