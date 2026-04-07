"""Minimal runnable MVP for 边界元方法 (BEM).

Problem solved in this demo:
- PDE: 2D Laplace equation inside the unit disk, Δu = 0
- Boundary condition (Dirichlet): u|Γ = x^2 - y^2 = cos(2θ)
- Goal: recover boundary flux q = ∂u/∂n and evaluate interior potential

Discretization choices:
- Constant boundary elements on polygonal circle approximation
- Collocation at element midpoints
- Dense linear system G q = (0.5 I + H) u
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fundamental_solution(p: np.ndarray, q: np.ndarray) -> float:
    """2D Laplace fundamental solution G(p,q) = -(1/(2π)) ln|p-q|."""
    r = np.linalg.norm(p - q)
    return -(1.0 / (2.0 * np.pi)) * np.log(r)


def dG_dn_source(p: np.ndarray, q: np.ndarray, n_q: np.ndarray) -> float:
    """Normal derivative wrt source point q: ∂G(p,q)/∂n_q."""
    d = q - p
    r2 = float(np.dot(d, d))
    return -(1.0 / (2.0 * np.pi)) * float(np.dot(d, n_q)) / r2


@dataclass
class BoundaryMesh:
    vertices: np.ndarray
    collocation: np.ndarray
    normals: np.ndarray
    lengths: np.ndarray


@dataclass
class BEMResult:
    mesh: BoundaryMesh
    u_boundary: np.ndarray
    q_boundary: np.ndarray
    q_exact_boundary: np.ndarray
    boundary_flux_rel_l2: float
    interior_points: np.ndarray
    interior_u_bem: np.ndarray
    interior_u_exact: np.ndarray
    interior_abs_linf: float


def build_unit_circle_mesh(num_elements: int) -> BoundaryMesh:
    if num_elements < 8:
        raise ValueError("num_elements must be >= 8 for a meaningful circle approximation")

    theta = np.linspace(0.0, 2.0 * np.pi, num_elements + 1)
    vertices = np.column_stack((np.cos(theta), np.sin(theta)))

    edge_vec = vertices[1:] - vertices[:-1]
    lengths = np.linalg.norm(edge_vec, axis=1)
    collocation = 0.5 * (vertices[:-1] + vertices[1:])

    # Unit circle outward normal equals radial direction.
    normals = collocation / np.linalg.norm(collocation, axis=1, keepdims=True)

    return BoundaryMesh(
        vertices=vertices,
        collocation=collocation,
        normals=normals,
        lengths=lengths,
    )


def boundary_dirichlet_u(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return x * x - y * y


def boundary_flux_exact(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    grad_u = np.column_stack((2.0 * x, -2.0 * y))
    return np.sum(grad_u * normals, axis=1)


def exact_u(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return x * x - y * y


def assemble_boundary_operators(mesh: BoundaryMesh) -> tuple[np.ndarray, np.ndarray]:
    n = mesh.collocation.shape[0]
    g_mat = np.zeros((n, n), dtype=float)
    h_mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        p_i = mesh.collocation[i]
        for j in range(n):
            q_j = mesh.collocation[j]
            l_j = mesh.lengths[j]

            if i == j:
                # Exact integral of logarithmic singularity over straight element [-L/2, L/2].
                g_mat[i, j] = l_j / (2.0 * np.pi) * (1.0 - np.log(l_j / 2.0))
                # Principal value part for the double-layer term on constant element.
                h_mat[i, j] = 0.0
                continue

            g_ij = fundamental_solution(p_i, q_j)
            h_ij = dG_dn_source(p_i, q_j, mesh.normals[j])
            g_mat[i, j] = g_ij * l_j
            h_mat[i, j] = h_ij * l_j

    return g_mat, h_mat


def solve_dirichlet_bem(mesh: BoundaryMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_bnd = boundary_dirichlet_u(mesh.collocation)
    g_mat, h_mat = assemble_boundary_operators(mesh)

    # 0.5*u + H*u = G*q  (interior Dirichlet problem on smooth boundary)
    rhs = (0.5 * np.eye(mesh.collocation.shape[0]) + h_mat) @ u_bnd
    q_bnd = np.linalg.solve(g_mat, rhs)
    q_exact = boundary_flux_exact(mesh.collocation, mesh.normals)
    return u_bnd, q_bnd, q_exact


def evaluate_interior_u(mesh: BoundaryMesh, u_bnd: np.ndarray, q_bnd: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.zeros(points.shape[0], dtype=float)

    for k, p in enumerate(points):
        g_row = np.zeros(mesh.collocation.shape[0], dtype=float)
        h_row = np.zeros(mesh.collocation.shape[0], dtype=float)

        for j in range(mesh.collocation.shape[0]):
            q_j = mesh.collocation[j]
            l_j = mesh.lengths[j]
            g_row[j] = fundamental_solution(p, q_j) * l_j
            h_row[j] = dG_dn_source(p, q_j, mesh.normals[j]) * l_j

        # Interior representation: u = ∫Γ G q dΓ - ∫Γ (∂G/∂n) u dΓ
        values[k] = float(np.dot(g_row, q_bnd) - np.dot(h_row, u_bnd))

    return values


def run_demo(num_elements: int = 80) -> BEMResult:
    mesh = build_unit_circle_mesh(num_elements=num_elements)
    u_bnd, q_bnd, q_exact_bnd = solve_dirichlet_bem(mesh)

    flux_rel_l2 = float(np.linalg.norm(q_bnd - q_exact_bnd) / np.linalg.norm(q_exact_bnd))

    interior_points = np.array(
        [
            [0.0, 0.0],
            [0.30, 0.20],
            [-0.45, 0.10],
            [0.10, -0.60],
            [0.62, 0.05],
        ],
        dtype=float,
    )
    radii = np.linalg.norm(interior_points, axis=1)
    if np.any(radii >= 1.0):
        raise ValueError("All interior_points must satisfy r < 1")

    u_in_bem = evaluate_interior_u(mesh, u_bnd=u_bnd, q_bnd=q_bnd, points=interior_points)
    u_in_exact = exact_u(interior_points)
    interior_abs_linf = float(np.max(np.abs(u_in_bem - u_in_exact)))

    return BEMResult(
        mesh=mesh,
        u_boundary=u_bnd,
        q_boundary=q_bnd,
        q_exact_boundary=q_exact_bnd,
        boundary_flux_rel_l2=flux_rel_l2,
        interior_points=interior_points,
        interior_u_bem=u_in_bem,
        interior_u_exact=u_in_exact,
        interior_abs_linf=interior_abs_linf,
    )


def main() -> None:
    result = run_demo(num_elements=80)

    print("=== Boundary Element Method (2D Laplace, interior Dirichlet) ===")
    print(f"elements                 : {result.mesh.collocation.shape[0]}")
    print(f"boundary flux rel-L2 err : {result.boundary_flux_rel_l2:.6e}")
    print(f"interior abs-Linf err    : {result.interior_abs_linf:.6e}")
    print()
    print("Interior point comparison (x, y, u_bem, u_exact, abs_err):")

    abs_err = np.abs(result.interior_u_bem - result.interior_u_exact)
    for p, ub, ue, e in zip(
        result.interior_points,
        result.interior_u_bem,
        result.interior_u_exact,
        abs_err,
    ):
        print(f"({p[0]: .2f}, {p[1]: .2f})  {ub: .8f}  {ue: .8f}  {e:.3e}")

    # Minimal quality gate for this MVP discretization.
    if result.boundary_flux_rel_l2 > 8.0e-2:
        raise RuntimeError(
            "Boundary flux relative L2 error is too large for the configured mesh. "
            f"Got {result.boundary_flux_rel_l2:.3e}."
        )
    if result.interior_abs_linf > 1.5e-2:
        raise RuntimeError(
            "Interior potential Linf error is too large for the configured mesh. "
            f"Got {result.interior_abs_linf:.3e}."
        )


if __name__ == "__main__":
    main()
