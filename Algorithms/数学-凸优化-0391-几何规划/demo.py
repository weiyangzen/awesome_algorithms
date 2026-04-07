"""Geometric Programming (GP) MVP.

This demo solves a small geometric program by the standard log-transform:
  x > 0, y = log(x)
which converts posynomial constraints into convex log-sum-exp constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


def stable_logsumexp(z: np.ndarray) -> float:
    """Numerically stable log(sum(exp(z)))."""
    z_max = float(np.max(z))
    return z_max + float(np.log(np.sum(np.exp(z - z_max))))


@dataclass(frozen=True)
class Monomial:
    """Monomial: c * prod_i x_i ** a_i, where c > 0 and x_i > 0."""

    coefficient: float
    exponents: np.ndarray

    def __post_init__(self) -> None:
        if self.coefficient <= 0:
            raise ValueError("Monomial coefficient must be > 0 for GP.")
        if self.exponents.ndim != 1:
            raise ValueError("Monomial exponents must be a 1D vector.")

    def log_value(self, y: np.ndarray) -> float:
        # In log-space: log(c * exp(a^T y)) = log(c) + a^T y
        return float(np.log(self.coefficient) + np.dot(self.exponents, y))

    def value(self, x: np.ndarray) -> float:
        return float(self.coefficient * np.prod(x**self.exponents))


@dataclass(frozen=True)
class Posynomial:
    """Posynomial: sum_k c_k * prod_i x_i ** a_ki."""

    terms: Sequence[Monomial]

    def __post_init__(self) -> None:
        if len(self.terms) == 0:
            raise ValueError("Posynomial must contain at least one monomial.")
        dim = self.terms[0].exponents.shape[0]
        if any(term.exponents.shape[0] != dim for term in self.terms):
            raise ValueError("All monomials in a posynomial must have same dimension.")

    @property
    def dim(self) -> int:
        return self.terms[0].exponents.shape[0]

    def value(self, x: np.ndarray) -> float:
        return float(sum(term.value(x) for term in self.terms))

    def log_value(self, y: np.ndarray) -> float:
        z = np.array([term.log_value(y) for term in self.terms], dtype=float)
        return stable_logsumexp(z)

    def grad_log_value(self, y: np.ndarray) -> np.ndarray:
        z = np.array([term.log_value(y) for term in self.terms], dtype=float)
        z_max = np.max(z)
        w = np.exp(z - z_max)
        w /= np.sum(w)
        exps = np.vstack([term.exponents for term in self.terms])
        return w @ exps


@dataclass
class GeometricProgram:
    """A small GP container solved in log-space with SLSQP."""

    objective: Posynomial
    ineq_posynomials: Sequence[Posynomial]  # fi(x) <= 1
    eq_monomials: Sequence[Monomial]  # hj(x) == 1

    def __post_init__(self) -> None:
        dim = self.objective.dim
        for p in self.ineq_posynomials:
            if p.dim != dim:
                raise ValueError("Objective and inequality posynomials dimension mismatch.")
        for m in self.eq_monomials:
            if m.exponents.shape[0] != dim:
                raise ValueError("Objective and equality monomials dimension mismatch.")
        self.dim = dim

    def solve(self, x0: np.ndarray, maxiter: int = 200):
        if x0.shape != (self.dim,):
            raise ValueError(f"x0 shape must be ({self.dim},), got {x0.shape}.")
        if np.any(x0 <= 0):
            raise ValueError("x0 must be strictly positive.")

        y0 = np.log(x0)

        def objective_fun(y: np.ndarray) -> float:
            return self.objective.log_value(y)

        def objective_jac(y: np.ndarray) -> np.ndarray:
            return self.objective.grad_log_value(y)

        constraints = []
        for p in self.ineq_posynomials:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda y, posy=p: -posy.log_value(y),
                    "jac": lambda y, posy=p: -posy.grad_log_value(y),
                }
            )

        for m in self.eq_monomials:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda y, mono=m: mono.log_value(y),
                    "jac": lambda y, mono=m: mono.exponents.copy(),
                }
            )

        result = minimize(
            fun=objective_fun,
            x0=y0,
            jac=objective_jac,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": maxiter, "ftol": 1e-12, "disp": False},
        )

        if not result.success:
            raise RuntimeError(f"SLSQP failed: {result.message}")

        x_star = np.exp(result.x)
        return result, x_star


def main() -> None:
    # Example GP:
    # minimize   x^(-1) * y^(-1/2)
    # s.t.       (x + y) / 10 <= 1
    #            2 / x <= 1
    #            3 / y <= 1
    #            x, y > 0
    objective = Posynomial(
        terms=[Monomial(coefficient=1.0, exponents=np.array([-1.0, -0.5]))]
    )
    constraints = [
        Posynomial(
            terms=[
                Monomial(coefficient=0.1, exponents=np.array([1.0, 0.0])),
                Monomial(coefficient=0.1, exponents=np.array([0.0, 1.0])),
            ]
        ),
        Posynomial(terms=[Monomial(coefficient=2.0, exponents=np.array([-1.0, 0.0]))]),
        Posynomial(terms=[Monomial(coefficient=3.0, exponents=np.array([0.0, -1.0]))]),
    ]

    gp = GeometricProgram(objective=objective, ineq_posynomials=constraints, eq_monomials=[])

    x0 = np.array([5.0, 5.0], dtype=float)
    result, x_star = gp.solve(x0=x0)

    obj_star = objective.value(x_star)
    expected_x = np.array([20.0 / 3.0, 10.0 / 3.0], dtype=float)
    expected_obj = objective.value(expected_x)

    ineq_values = np.array([p.value(x_star) - 1.0 for p in constraints], dtype=float)

    print("=== Geometric Programming MVP ===")
    print(f"Solver status: {result.message}")
    print(f"x* = {x_star}")
    print(f"objective(x*) = {obj_star:.12f}")
    print(f"analytic x* ≈ {expected_x}")
    print(f"analytic objective ≈ {expected_obj:.12f}")
    print(f"||x* - x*_analytic||_2 = {np.linalg.norm(x_star - expected_x):.6e}")
    print("Constraint residuals fi(x*) - 1 <= 0:")
    for i, res in enumerate(ineq_values, start=1):
        print(f"  c{i}: {res:+.6e}")

    tol = 1e-7
    if np.any(ineq_values > tol):
        raise AssertionError("Inequality constraints violated.")
    if np.linalg.norm(x_star - expected_x) > 1e-5:
        raise AssertionError("Solution deviates from analytic optimum unexpectedly.")

    print("All checks passed.")


if __name__ == "__main__":
    main()
