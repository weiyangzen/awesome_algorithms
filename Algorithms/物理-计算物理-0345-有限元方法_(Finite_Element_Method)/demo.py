"""Minimal runnable MVP for Finite Element Method (PHYS-0338).

This script solves a 1D Poisson boundary value problem on [0, 1]
using linear finite elements:

    -u''(x) = f(x),  u(0)=u(1)=0,

with manufactured exact solution u(x) = sin(pi x), therefore
f(x) = pi^2 sin(pi x).

The implementation explicitly includes:
- mesh generation
- element-level stiffness/load computation
- global sparse assembly
- Dirichlet boundary condition treatment
- sparse linear solve
- convergence/error report
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class FEMConfig:
    """Configuration for the FEM demonstration."""

    n_elements: int = 80
    boundary_left: float = 0.0
    boundary_right: float = 0.0
    convergence_levels: tuple[int, ...] = (10, 20, 40, 80)

    def validate(self) -> None:
        if self.n_elements < 2:
            raise ValueError("n_elements must be >= 2")
        if len(self.convergence_levels) < 2:
            raise ValueError("Need at least two convergence levels")
        if any(n < 2 for n in self.convergence_levels):
            raise ValueError("Every convergence level must be >= 2")


@dataclass(frozen=True)
class FEMResult:
    """Container for a single FEM solve result."""

    x: np.ndarray
    u_numeric: np.ndarray
    u_exact: np.ndarray
    l2_error: float
    h1_semi_error: float
    residual_inf: float


def forcing(x: np.ndarray) -> np.ndarray:
    """Right-hand side f(x)=pi^2 sin(pi x) for exact solution sin(pi x)."""
    return (np.pi**2) * np.sin(np.pi * x)


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Exact solution used for validation."""
    return np.sin(np.pi * x)


def exact_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of exact solution."""
    return np.pi * np.cos(np.pi * x)


def compute_l2_error(x: np.ndarray, u_numeric: np.ndarray) -> float:
    """Compute integral L2 error ||u_h-u||_{L2} via 3-point Gauss quadrature."""
    gp = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)], dtype=np.float64)
    gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)

    err_sq = 0.0
    for i in range(x.size - 1):
        x_left = x[i]
        x_right = x[i + 1]
        h = x_right - x_left
        jacobian = 0.5 * h
        ui = u_numeric[i]
        uj = u_numeric[i + 1]

        for q in range(3):
            xi = gp[q]
            wq = gw[q]
            n1 = 0.5 * (1.0 - xi)
            n2 = 0.5 * (1.0 + xi)
            xq = 0.5 * (x_left + x_right) + jacobian * xi
            uh_q = n1 * ui + n2 * uj
            ue_q = np.sin(np.pi * xq)
            err_sq += wq * (uh_q - ue_q) ** 2 * jacobian

    return float(np.sqrt(err_sq))


def generate_uniform_mesh(n_elements: int) -> tuple[np.ndarray, np.ndarray]:
    """Return node coordinates and 2-node connectivity for a uniform 1D mesh."""
    x = np.linspace(0.0, 1.0, n_elements + 1, dtype=np.float64)
    connectivity = np.column_stack(
        [np.arange(0, n_elements, dtype=np.int64), np.arange(1, n_elements + 1, dtype=np.int64)]
    )
    return x, connectivity


def local_stiffness(h: float) -> np.ndarray:
    """Local stiffness matrix for linear element on length h."""
    return np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64) / h


def local_load_vector(x_left: float, x_right: float) -> np.ndarray:
    """Compute local load vector by 2-point Gauss quadrature."""
    h = x_right - x_left
    jacobian = 0.5 * h

    # 2-point Gauss points/weights on [-1,1]
    gp = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)
    gw = np.array([1.0, 1.0], dtype=np.float64)

    fe = np.zeros(2, dtype=np.float64)
    for q in range(2):
        xi = gp[q]
        wq = gw[q]
        # Reference linear basis on [-1,1]
        n1 = 0.5 * (1.0 - xi)
        n2 = 0.5 * (1.0 + xi)
        xq = 0.5 * (x_left + x_right) + jacobian * xi
        fq = forcing(np.array([xq]))[0]

        fe[0] += wq * fq * n1 * jacobian
        fe[1] += wq * fq * n2 * jacobian
    return fe


def assemble_system(
    x: np.ndarray,
    connectivity: np.ndarray,
    u0: float,
    u1: float,
) -> tuple[csr_matrix, np.ndarray]:
    """Assemble global sparse matrix K and load vector F, including Dirichlet shifts."""
    n_nodes = x.size
    k_global = lil_matrix((n_nodes, n_nodes), dtype=np.float64)
    f_global = np.zeros(n_nodes, dtype=np.float64)

    for e in range(connectivity.shape[0]):
        i, j = int(connectivity[e, 0]), int(connectivity[e, 1])
        x_i, x_j = float(x[i]), float(x[j])
        h = x_j - x_i

        ke = local_stiffness(h)
        fe = local_load_vector(x_i, x_j)

        dofs = [i, j]
        for a in range(2):
            f_global[dofs[a]] += fe[a]
            for b in range(2):
                k_global[dofs[a], dofs[b]] += ke[a, b]

    # Apply non-zero Dirichlet by shifting known values to RHS of interior equations.
    boundary_values = {0: u0, n_nodes - 1: u1}
    for b_idx, b_val in boundary_values.items():
        if abs(b_val) > 0.0:
            f_global -= k_global[:, b_idx].toarray().ravel() * b_val

    return k_global.tocsr(), f_global


def solve_poisson_1d_fem(config: FEMConfig, n_elements: int) -> FEMResult:
    """Solve the Poisson problem on a mesh with n_elements linear elements."""
    x, connectivity = generate_uniform_mesh(n_elements)
    k_global, f_global = assemble_system(
        x=x,
        connectivity=connectivity,
        u0=config.boundary_left,
        u1=config.boundary_right,
    )

    n_nodes = x.size
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    boundary_mask[0] = True
    boundary_mask[-1] = True
    interior = np.where(~boundary_mask)[0]

    k_ii = k_global[interior][:, interior]
    f_i = f_global[interior]

    u = np.zeros(n_nodes, dtype=np.float64)
    u[0] = config.boundary_left
    u[-1] = config.boundary_right
    u[interior] = spsolve(k_ii, f_i)

    u_exact = exact_solution(x)

    l2_error = compute_l2_error(x, u)

    # Piecewise-constant derivative from linear FEM on each element
    h = np.diff(x)
    du_num = np.diff(u) / h
    x_mid = 0.5 * (x[:-1] + x[1:])
    du_exact = exact_derivative(x_mid)
    h1_semi_error = float(np.sqrt(np.sum(((du_num - du_exact) ** 2) * h)))

    # Residual on interior equations
    residual = k_global @ u - f_global
    residual_inf = float(np.max(np.abs(residual[interior])))

    return FEMResult(
        x=x,
        u_numeric=u,
        u_exact=u_exact,
        l2_error=l2_error,
        h1_semi_error=h1_semi_error,
        residual_inf=residual_inf,
    )


def run_convergence_study(config: FEMConfig) -> pd.DataFrame:
    """Run multiple mesh levels and estimate observed error order."""
    rows: list[dict[str, float | int]] = []
    prev_l2: float | None = None

    for n_elements in config.convergence_levels:
        result = solve_poisson_1d_fem(config, n_elements)
        order = np.nan
        if prev_l2 is not None:
            order = np.log(prev_l2 / result.l2_error) / np.log(2.0)

        rows.append(
            {
                "n_elements": n_elements,
                "h": 1.0 / n_elements,
                "l2_error": result.l2_error,
                "h1_semi_error": result.h1_semi_error,
                "residual_inf": result.residual_inf,
                "observed_order_l2": float(order),
            }
        )
        prev_l2 = result.l2_error

    return pd.DataFrame(rows)


def main() -> None:
    config = FEMConfig()
    config.validate()

    primary = solve_poisson_1d_fem(config, config.n_elements)
    study = run_convergence_study(config)

    max_abs_error = float(np.max(np.abs(primary.u_numeric - primary.u_exact)))

    summary = pd.DataFrame(
        {
            "metric": [
                "n_elements",
                "n_nodes",
                "l2_error",
                "h1_semi_error",
                "max_abs_error",
                "residual_inf",
                "u(0), u(1)",
            ],
            "value": [
                str(config.n_elements),
                str(config.n_elements + 1),
                f"{primary.l2_error:.6e}",
                f"{primary.h1_semi_error:.6e}",
                f"{max_abs_error:.6e}",
                f"{primary.residual_inf:.6e}",
                f"({primary.u_numeric[0]:.3e}, {primary.u_numeric[-1]:.3e})",
            ],
        }
    )

    print("Finite Element Method MVP (1D Poisson, linear elements)")
    print("\nConvergence study")
    print(study.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print("\nPrimary run summary")
    print(summary.to_string(index=False))

    # Validation checks: conservative thresholds for a robust MVP.
    finest = study.iloc[-1]
    observed_order = float(finest["observed_order_l2"])

    assert primary.residual_inf < 1e-10, (
        f"Residual too large: residual_inf={primary.residual_inf:.3e}"
    )
    assert primary.l2_error < 2.0e-4, (
        f"L2 error too large: {primary.l2_error:.3e}"
    )
    assert observed_order > 1.8, (
        f"Observed L2 order too low: {observed_order:.3f}"
    )

    print("\nValidation: PASS")


if __name__ == "__main__":
    main()
