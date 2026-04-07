"""Minimal runnable MVP for Thermodynamic Integration (TI).

The script computes free-energy difference DeltaF between two 1D harmonic systems:
    A: U_A(x) = 0.5 * k_A * (x - mu_A)^2 + c_A
    B: U_B(x) = 0.5 * k_B * (x - mu_B)^2 + c_B

Using TI:
    DeltaF = integral_0^1 <dU/dlambda>_lambda d lambda
           = integral_0^1 <U_B - U_A>_lambda d lambda

where ensemble averages are evaluated under p_lambda(x) ~ exp(-beta U_lambda(x)),
U_lambda = (1-lambda) U_A + lambda U_B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import simpson


@dataclass(frozen=True)
class HarmonicState:
    k: float
    mu: float
    c: float

    def validate(self) -> None:
        if self.k <= 0.0:
            raise ValueError(f"Spring constant k must be positive, got {self.k}")


@dataclass(frozen=True)
class TIProblem:
    beta: float
    state_a: HarmonicState
    state_b: HarmonicState
    x_grid: np.ndarray
    lambda_grid: np.ndarray

    def validate(self) -> None:
        if self.beta <= 0.0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        self.state_a.validate()
        self.state_b.validate()
        if self.x_grid.ndim != 1 or self.x_grid.size < 3:
            raise ValueError("x_grid must be a 1D array with at least 3 points")
        if self.lambda_grid.ndim != 1 or self.lambda_grid.size < 3:
            raise ValueError("lambda_grid must be a 1D array with at least 3 points")
        if np.any(np.diff(self.x_grid) <= 0.0):
            raise ValueError("x_grid must be strictly increasing")
        if np.any(np.diff(self.lambda_grid) <= 0.0):
            raise ValueError("lambda_grid must be strictly increasing")
        if self.lambda_grid[0] < 0.0 or self.lambda_grid[-1] > 1.0:
            raise ValueError("lambda_grid must stay inside [0, 1]")


def harmonic_potential(x: np.ndarray, state: HarmonicState) -> np.ndarray:
    return 0.5 * state.k * (x - state.mu) ** 2 + state.c


def interpolate_potential(u_a: np.ndarray, u_b: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - lam) * u_a + lam * u_b


def boltzmann_average(observable: np.ndarray, energy: np.ndarray, beta: float, x: np.ndarray) -> float:
    """Compute <observable> under p(x) ~ exp(-beta * energy(x)) with a stable shift."""
    scaled = beta * energy
    shift = float(np.min(scaled))
    weights = np.exp(-(scaled - shift))
    denominator = float(simpson(weights, x=x))
    if denominator <= 0.0:
        raise FloatingPointError("Boltzmann normalization became non-positive")
    numerator = float(simpson(observable * weights, x=x))
    return numerator / denominator


def log_partition(energy: np.ndarray, beta: float, x: np.ndarray) -> float:
    """Compute log Z = log int exp(-beta U(x)) dx using a stable shift."""
    scaled = beta * energy
    shift = float(np.min(scaled))
    reduced = float(simpson(np.exp(-(scaled - shift)), x=x))
    if reduced <= 0.0:
        raise FloatingPointError("Reduced partition integral became non-positive")
    return -shift + np.log(reduced)


def free_energy_from_partition(energy: np.ndarray, beta: float, x: np.ndarray) -> float:
    return -log_partition(energy=energy, beta=beta, x=x) / beta


def analytic_harmonic_free_energy(state: HarmonicState, beta: float) -> float:
    """Exact classical free energy for 1D harmonic potential with offset c."""
    return state.c + 0.5 * np.log(beta * state.k / (2.0 * np.pi)) / beta


def ti_profile(problem: TIProblem) -> pd.DataFrame:
    x = problem.x_grid
    beta = problem.beta
    u_a = harmonic_potential(x, problem.state_a)
    u_b = harmonic_potential(x, problem.state_b)
    d_u_d_lambda = u_b - u_a

    means = np.empty(problem.lambda_grid.size, dtype=np.float64)
    for i, lam in enumerate(problem.lambda_grid):
        u_lam = interpolate_potential(u_a, u_b, lam=float(lam))
        means[i] = boltzmann_average(
            observable=d_u_d_lambda,
            energy=u_lam,
            beta=beta,
            x=x,
        )

    return pd.DataFrame(
        {
            "lambda": problem.lambda_grid,
            "mean_dU_dlambda": means,
        }
    )


def solve(problem: TIProblem) -> dict[str, float]:
    problem.validate()

    x = problem.x_grid
    beta = problem.beta
    u_a = harmonic_potential(x, problem.state_a)
    u_b = harmonic_potential(x, problem.state_b)

    profile = ti_profile(problem)
    delta_f_ti = float(simpson(profile["mean_dU_dlambda"].to_numpy(), x=profile["lambda"].to_numpy()))

    f_a_num = free_energy_from_partition(u_a, beta=beta, x=x)
    f_b_num = free_energy_from_partition(u_b, beta=beta, x=x)
    delta_f_partition = f_b_num - f_a_num

    f_a_exact = analytic_harmonic_free_energy(problem.state_a, beta=beta)
    f_b_exact = analytic_harmonic_free_energy(problem.state_b, beta=beta)
    delta_f_exact = f_b_exact - f_a_exact

    return {
        "delta_f_ti": delta_f_ti,
        "delta_f_partition": delta_f_partition,
        "delta_f_exact": delta_f_exact,
        "err_vs_partition": abs(delta_f_ti - delta_f_partition),
        "err_vs_exact": abs(delta_f_ti - delta_f_exact),
        "profile_min": float(profile["mean_dU_dlambda"].min()),
        "profile_max": float(profile["mean_dU_dlambda"].max()),
    }


def main() -> None:
    problem = TIProblem(
        beta=1.7,
        state_a=HarmonicState(k=1.3, mu=-1.6, c=0.2),
        state_b=HarmonicState(k=3.5, mu=1.4, c=1.0),
        x_grid=np.linspace(-8.0, 8.0, 8001, dtype=np.float64),
        lambda_grid=np.linspace(0.0, 1.0, 81, dtype=np.float64),
    )

    result = solve(problem)

    print("Thermodynamic Integration MVP")
    print(f"beta={problem.beta:.3f}")
    print(
        "state_a="
        f"(k={problem.state_a.k:.3f}, mu={problem.state_a.mu:.3f}, c={problem.state_a.c:.3f})"
    )
    print(
        "state_b="
        f"(k={problem.state_b.k:.3f}, mu={problem.state_b.mu:.3f}, c={problem.state_b.c:.3f})"
    )
    print(f"grid_x={problem.x_grid.size}, grid_lambda={problem.lambda_grid.size}")
    print(f"delta_f_ti={result['delta_f_ti']:.8f}")
    print(f"delta_f_partition={result['delta_f_partition']:.8f}")
    print(f"delta_f_exact={result['delta_f_exact']:.8f}")
    print(f"err_vs_partition={result['err_vs_partition']:.3e}")
    print(f"err_vs_exact={result['err_vs_exact']:.3e}")
    print(
        "mean_dU_dlambda_range="
        f"[{result['profile_min']:.8f}, {result['profile_max']:.8f}]"
    )

    assert result["err_vs_partition"] < 8.0e-4, (
        "TI mismatch against partition-function reference: "
        f"{result['err_vs_partition']}"
    )
    assert result["err_vs_exact"] < 8.0e-4, (
        "TI mismatch against analytic harmonic reference: "
        f"{result['err_vs_exact']}"
    )
    assert result["profile_max"] >= result["profile_min"], "Profile range is inconsistent"

    print("All checks passed.")


if __name__ == "__main__":
    main()
