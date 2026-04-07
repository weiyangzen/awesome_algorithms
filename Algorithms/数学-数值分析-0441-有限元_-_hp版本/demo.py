"""Minimal runnable MVP for a 1D hp-FEM solver.

Problem:
    -u''(x) = f(x), x in (0, 1)
    u(0) = u(1) = 0

This script demonstrates:
- high-order finite elements (p-refinement),
- element splitting (h-refinement),
- a simple hp-adaptation heuristic based on residual + smoothness.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.polynomial import polyval

try:
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    HAVE_SCIPY = True
except Exception:
    lil_matrix = None
    spsolve = None
    HAVE_SCIPY = False


PI = np.pi
A_LAYER = 80.0
C_LAYER = 0.30
BETA_LAYER = 0.20


@dataclass(frozen=True)
class Element:
    left: float
    right: float
    p: int

    @property
    def h(self) -> float:
        return self.right - self.left


@dataclass
class Mesh:
    elements: List[Element]


@dataclass(frozen=True)
class CycleStats:
    cycle: int
    n_elements: int
    n_dofs: int
    l2_error: float
    h1_semi_error: float
    max_indicator: float
    mean_p: float
    h_refined: int
    p_refined: int


def layer_parts(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return t, t', t'' for t(x)=beta*x*(1-x)*exp(-a*(x-c)^2)."""
    g = x * (1.0 - x)
    g1 = 1.0 - 2.0 * x
    g2 = -2.0 * np.ones_like(x)

    dx = x - C_LAYER
    expv = np.exp(-A_LAYER * dx * dx)
    exp1 = -2.0 * A_LAYER * dx * expv
    exp2 = (-2.0 * A_LAYER + 4.0 * A_LAYER * A_LAYER * dx * dx) * expv

    t = BETA_LAYER * g * expv
    t1 = BETA_LAYER * (g1 * expv + g * exp1)
    t2 = BETA_LAYER * (g2 * expv + 2.0 * g1 * exp1 + g * exp2)
    return t, t1, t2


def exact_u(x: np.ndarray) -> np.ndarray:
    t, _, _ = layer_parts(x)
    return np.sin(PI * x) + t


def exact_du(x: np.ndarray) -> np.ndarray:
    _, t1, _ = layer_parts(x)
    return PI * np.cos(PI * x) + t1


def rhs_f(x: np.ndarray) -> np.ndarray:
    """f = -u'' for the manufactured exact solution."""
    _, _, t2 = layer_parts(x)
    return PI * PI * np.sin(PI * x) - t2


@lru_cache(maxsize=None)
def reference_quadrature(nq: int) -> Tuple[np.ndarray, np.ndarray]:
    xi, w = leggauss(nq)
    return xi, w


@lru_cache(maxsize=None)
def lagrange_nodes(p: int) -> np.ndarray:
    if p < 1:
        raise ValueError("Polynomial order p must be >= 1.")
    if p == 1:
        return np.array([-1.0, 1.0])
    idx = np.arange(p + 1, dtype=float)
    return -np.cos(np.pi * idx / p)


@lru_cache(maxsize=None)
def lagrange_coefficients(p: int) -> np.ndarray:
    """Rows are monomial coefficients for each Lagrange basis on [-1,1]."""
    nodes = lagrange_nodes(p)
    vand = np.vander(nodes, N=p + 1, increasing=True)
    coeff_cols = np.linalg.solve(vand, np.eye(p + 1))
    return coeff_cols.T.copy()


def polyder_matrix(coeffs: np.ndarray, order: int) -> np.ndarray:
    out = coeffs.copy()
    for _ in range(order):
        if out.shape[1] <= 1:
            return np.zeros((out.shape[0], 1), dtype=out.dtype)
        powers = np.arange(1, out.shape[1], dtype=out.dtype)
        out = out[:, 1:] * powers[None, :]
    return out


@lru_cache(maxsize=None)
def basis_tables(p: int, nq: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi, w = reference_quadrature(nq)
    coeff = lagrange_coefficients(p)
    d1 = polyder_matrix(coeff, order=1)
    d2 = polyder_matrix(coeff, order=2)

    phi = np.vstack([polyval(xi, row) for row in coeff])
    dphi = np.vstack([polyval(xi, row) for row in d1])
    d2phi = np.vstack([polyval(xi, row) for row in d2])
    return xi, w, phi, dphi, d2phi


def validate_mesh(mesh: Mesh) -> None:
    if not mesh.elements:
        raise ValueError("Mesh must contain at least one element.")

    elems = mesh.elements
    tol = 1e-12
    if abs(elems[0].left - 0.0) > tol or abs(elems[-1].right - 1.0) > tol:
        raise ValueError("Mesh must cover [0, 1].")

    for i, e in enumerate(elems):
        if e.h <= 0.0:
            raise ValueError("Element size must be positive.")
        if e.p < 1:
            raise ValueError("Element polynomial order must be >= 1.")
        if i > 0 and abs(elems[i - 1].right - e.left) > tol:
            raise ValueError("Mesh must be contiguous without gaps/overlaps.")


def build_dof_map(mesh: Mesh) -> Tuple[List[np.ndarray], int, int, int]:
    verts = sorted({round(e.left, 14) for e in mesh.elements} | {round(e.right, 14) for e in mesh.elements})
    vert_to_dof = {v: i for i, v in enumerate(verts)}

    next_dof = len(verts)
    elem_dofs: List[np.ndarray] = []

    for e in mesh.elements:
        p = e.p
        local = np.empty(p + 1, dtype=int)
        left_key = round(e.left, 14)
        right_key = round(e.right, 14)
        local[0] = vert_to_dof[left_key]
        local[-1] = vert_to_dof[right_key]
        for j in range(1, p):
            local[j] = next_dof
            next_dof += 1
        elem_dofs.append(local)

    left_bc = vert_to_dof[min(verts)]
    right_bc = vert_to_dof[max(verts)]
    return elem_dofs, next_dof, left_bc, right_bc


def solve_fem(mesh: Mesh) -> Tuple[np.ndarray, List[np.ndarray], int]:
    validate_mesh(mesh)
    elem_dofs, n_dofs, left_bc, right_bc = build_dof_map(mesh)

    stiff = lil_matrix((n_dofs, n_dofs), dtype=float) if HAVE_SCIPY else np.zeros((n_dofs, n_dofs), dtype=float)
    load = np.zeros(n_dofs, dtype=float)

    for e, l2g in zip(mesh.elements, elem_dofs):
        p = e.p
        nq = max(2 * p + 4, 8)
        xi, w, phi, dphi, _ = basis_tables(p, nq)

        h = e.h
        jac = 0.5 * h
        xq = 0.5 * (e.left + e.right) + jac * xi
        fq = rhs_f(xq)

        dphi_dx = dphi * (2.0 / h)
        weight = w * jac

        ke = (dphi_dx * weight[None, :]) @ dphi_dx.T
        fe = (phi * (fq * weight)[None, :]).sum(axis=1)

        nloc = p + 1
        for a in range(nloc):
            ga = l2g[a]
            load[ga] += fe[a]
            for b in range(nloc):
                gb = l2g[b]
                stiff[ga, gb] += ke[a, b]

    fixed = np.array([left_bc, right_bc], dtype=int)
    all_dofs = np.arange(n_dofs, dtype=int)
    free = np.setdiff1d(all_dofs, fixed, assume_unique=True)

    if HAVE_SCIPY:
        kff = stiff[free][:, free].tocsr()
    else:
        kff = stiff[np.ix_(free, free)]
    ff = load[free]

    u = np.zeros(n_dofs, dtype=float)
    if HAVE_SCIPY:
        u[free] = spsolve(kff, ff)
    else:
        u[free] = np.linalg.solve(kff, ff)

    if not np.all(np.isfinite(u)):
        raise FloatingPointError("Linear solve produced non-finite values.")

    return u, elem_dofs, n_dofs


def local_smoothness_indicator(u_local: np.ndarray, p: int) -> float:
    """Estimate whether local solution is smooth enough for p-enrichment."""
    if p <= 1:
        return 1.0

    nodes = lagrange_nodes(p)
    vand_low = np.vander(nodes, N=p, increasing=True)
    coeff_low, *_ = np.linalg.lstsq(vand_low, u_local, rcond=None)
    u_low = vand_low @ coeff_low

    num = np.linalg.norm(u_local - u_low)
    den = np.linalg.norm(u_local) + 1e-14
    return float(num / den)


def compute_errors_and_indicators(
    mesh: Mesh,
    u: np.ndarray,
    elem_dofs: Sequence[np.ndarray],
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    l2_sq = 0.0
    h1_sq = 0.0
    indicators: List[float] = []
    smoothness: List[float] = []

    for e, l2g in zip(mesh.elements, elem_dofs):
        p = e.p
        nq = max(2 * p + 6, 10)
        xi, w, phi, dphi, d2phi = basis_tables(p, nq)

        h = e.h
        jac = 0.5 * h
        xq = 0.5 * (e.left + e.right) + jac * xi

        u_local = u[l2g]
        uh = u_local @ phi
        duh = (u_local @ dphi) * (2.0 / h)
        d2uh = (u_local @ d2phi) * ((2.0 / h) ** 2)

        ue = exact_u(xq)
        due = exact_du(xq)

        l2_sq += np.sum(w * (uh - ue) ** 2 * jac)
        h1_sq += np.sum(w * (duh - due) ** 2 * jac)

        residual = rhs_f(xq) + d2uh
        eta_sq = (h * h) * np.sum(w * residual * residual * jac)
        indicators.append(float(np.sqrt(max(eta_sq, 0.0))))

        smoothness.append(local_smoothness_indicator(u_local=u_local, p=p))

    return float(np.sqrt(l2_sq)), float(np.sqrt(h1_sq)), np.array(indicators), np.array(smoothness)


def hp_adapt(
    mesh: Mesh,
    indicators: np.ndarray,
    smoothness: np.ndarray,
    *,
    theta: float = 0.55,
    sigma_thresh: float = 0.12,
    p_max: int = 6,
    h_min: float = 1.0 / 256.0,
) -> Tuple[Mesh, int, int]:
    max_eta = float(np.max(indicators))
    marked = indicators >= theta * max_eta

    p_possible = np.array([e.p < p_max for e in mesh.elements], dtype=bool)
    force_p_idx = -1

    if not np.any(marked & (smoothness < sigma_thresh) & p_possible):
        candidates = np.where((~marked) & (smoothness < 0.5 * sigma_thresh) & p_possible)[0]
        if candidates.size > 0:
            force_p_idx = int(candidates[np.argmax(indicators[candidates])])

    new_elements: List[Element] = []
    h_count = 0
    p_count = 0

    for idx, e in enumerate(mesh.elements):
        force_p = idx == force_p_idx

        if force_p:
            new_elements.append(Element(e.left, e.right, e.p + 1))
            p_count += 1
            continue

        if not marked[idx]:
            new_elements.append(e)
            continue

        if smoothness[idx] < sigma_thresh and e.p < p_max:
            new_elements.append(Element(e.left, e.right, e.p + 1))
            p_count += 1
            continue

        if 0.5 * e.h >= h_min:
            mid = 0.5 * (e.left + e.right)
            new_elements.append(Element(e.left, mid, e.p))
            new_elements.append(Element(mid, e.right, e.p))
            h_count += 1
        elif e.p < p_max:
            new_elements.append(Element(e.left, e.right, e.p + 1))
            p_count += 1
        else:
            new_elements.append(e)

    adapted = Mesh(new_elements)
    validate_mesh(adapted)
    return adapted, h_count, p_count


def initial_mesh() -> Mesh:
    xs = np.linspace(0.0, 1.0, 7)
    elems: List[Element] = []
    for i in range(len(xs) - 1):
        # Mixed initial polynomial orders improve chance of both h and p actions.
        p = 2 if i % 2 == 0 else 1
        elems.append(Element(float(xs[i]), float(xs[i + 1]), p))
    mesh = Mesh(elems)
    validate_mesh(mesh)
    return mesh


def run_hp_fem_demo(n_cycles: int = 5) -> List[CycleStats]:
    mesh = initial_mesh()
    stats: List[CycleStats] = []

    for cycle in range(n_cycles):
        u, elem_dofs, n_dofs = solve_fem(mesh)
        l2_err, h1_err, eta, sigma = compute_errors_and_indicators(mesh, u, elem_dofs)

        h_count = 0
        p_count = 0
        if cycle < n_cycles - 1:
            mesh, h_count, p_count = hp_adapt(mesh, eta, sigma)

        mean_p = float(np.mean([e.p for e in mesh.elements]))
        stats.append(
            CycleStats(
                cycle=cycle,
                n_elements=len(mesh.elements),
                n_dofs=n_dofs,
                l2_error=l2_err,
                h1_semi_error=h1_err,
                max_indicator=float(np.max(eta)),
                mean_p=mean_p,
                h_refined=h_count,
                p_refined=p_count,
            )
        )

    return stats


def run_checks(stats: Sequence[CycleStats]) -> None:
    if not stats:
        raise AssertionError("No cycle statistics produced.")

    for item in stats:
        if not np.isfinite(item.l2_error) or not np.isfinite(item.h1_semi_error):
            raise AssertionError("Detected non-finite error metric.")

    if not (stats[-1].l2_error < stats[0].l2_error):
        raise AssertionError("Final L2 error did not improve over initial cycle.")

    if not any(s.h_refined > 0 for s in stats[:-1]):
        raise AssertionError("No h-refinement occurred; hp behavior incomplete.")

    if not any(s.p_refined > 0 for s in stats[:-1]):
        raise AssertionError("No p-refinement occurred; hp behavior incomplete.")


def print_report(stats: Sequence[CycleStats]) -> None:
    print("hp-FEM MVP: -u''=f on [0,1], u(0)=u(1)=0")
    print(
        "cycle | elements | dofs |      L2 error |   H1-semi error | max indicator | "
        "mean p | h-ref | p-ref"
    )
    print("-" * 104)
    for s in stats:
        print(
            f"{s.cycle:5d} | {s.n_elements:8d} | {s.n_dofs:4d} | "
            f"{s.l2_error:13.6e} | {s.h1_semi_error:15.6e} | {s.max_indicator:13.6e} | "
            f"{s.mean_p:6.2f} | {s.h_refined:5d} | {s.p_refined:5d}"
        )


def main() -> None:
    stats = run_hp_fem_demo(n_cycles=5)
    print_report(stats)
    run_checks(stats)
    print("All checks passed.")


if __name__ == "__main__":
    main()
