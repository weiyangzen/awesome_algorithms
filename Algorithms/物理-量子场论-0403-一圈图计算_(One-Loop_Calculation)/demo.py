r"""One-loop calculation MVP: scalar equal-mass bubble master integral.

We evaluate the renormalized finite Euclidean one-loop bubble function

    B_hat(Q^2, m) = (1/(16*pi^2)) * \int_0^1 dx ln(1 + x(1-x) Q^2 / m^2)

which is obtained after Feynman parameterization and subtraction at Q^2=0.
The script compares:
1) numerical quadrature,
2) analytic closed form,
3) small-Q^2 series expansion,
then prints a table and runs non-interactive checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad


@dataclass(frozen=True)
class OneLoopConfig:
    mass: float
    coupling: float
    q2_values: tuple[float, ...]
    epsabs: float = 1e-12
    epsrel: float = 1e-12


def _check_nonnegative_scalar(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def _check_positive_scalar(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def _check_grid(q2_values: np.ndarray) -> None:
    if q2_values.ndim != 1 or q2_values.size < 2:
        raise ValueError("q2_values must be a 1D array with at least two entries.")
    if not np.all(np.isfinite(q2_values)):
        raise ValueError("q2_values must contain finite values only.")
    if np.any(q2_values < 0.0):
        raise ValueError("q2_values must be non-negative.")
    if not np.all(np.diff(q2_values) >= 0.0):
        raise ValueError("q2_values must be sorted in nondecreasing order.")


def bubble_integrand_feynman_x(x: float, q2: float, mass: float) -> float:
    """Integrand of the finite Feynman-parameter representation."""
    a = q2 / (mass * mass)
    return np.log1p(a * x * (1.0 - x))


def bubble_master_numeric(q2: float, mass: float, epsabs: float, epsrel: float) -> float:
    """Numerical one-loop master integral via 1D quadrature over Feynman x."""
    _check_nonnegative_scalar("q2", q2)
    _check_positive_scalar("mass", mass)
    _check_positive_scalar("epsabs", epsabs)
    _check_positive_scalar("epsrel", epsrel)

    if q2 == 0.0:
        return 0.0

    integral, _ = quad(
        lambda x: bubble_integrand_feynman_x(x, q2=q2, mass=mass),
        0.0,
        1.0,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=200,
    )
    return integral / (16.0 * np.pi * np.pi)


def _j_analytic_dimensionless(a: float) -> float:
    """Closed form for J(a)=int_0^1 ln(1+a x(1-x)) dx, with a>=0.

    For a > 0:
      J(a) = 2 * ( sqrt((a+4)/a) * artanh(sqrt(a/(a+4))) - 1 )
    For tiny a we switch to series to avoid cancellation.
    """
    _check_nonnegative_scalar("a", a)

    if a == 0.0:
        return 0.0
    if a < 1e-8:
        return a / 6.0 - (a * a) / 60.0 + (a * a * a) / 420.0

    u = np.sqrt(a / (a + 4.0))
    prefactor = np.sqrt((a + 4.0) / a)
    return 2.0 * (prefactor * np.arctanh(u) - 1.0)


def bubble_master_analytic(q2: float, mass: float) -> float:
    """Analytic closed form of the finite one-loop master integral."""
    _check_nonnegative_scalar("q2", q2)
    _check_positive_scalar("mass", mass)

    a = q2 / (mass * mass)
    j_value = _j_analytic_dimensionless(a)
    return j_value / (16.0 * np.pi * np.pi)


def bubble_master_series_q2(q2: float, mass: float) -> float:
    """Small-Q^2 expansion up to O(Q^4):

    B_hat(Q^2,m) = (1/(16*pi^2)) * [a/6 - a^2/60 + O(a^3)], a=Q^2/m^2.
    """
    _check_nonnegative_scalar("q2", q2)
    _check_positive_scalar("mass", mass)

    a = q2 / (mass * mass)
    j_series = a / 6.0 - (a * a) / 60.0
    return j_series / (16.0 * np.pi * np.pi)


def one_loop_coupling_shift(q2: float, mass: float, coupling: float, use_analytic: bool = True) -> float:
    """Toy one-loop correction scale for phi^4-like coupling channel.

    We use: delta_lambda(Q^2) = 3 * lambda^2 * B_hat(Q^2, m).
    """
    _check_nonnegative_scalar("q2", q2)
    _check_positive_scalar("mass", mass)
    _check_positive_scalar("coupling", coupling)

    b_hat = bubble_master_analytic(q2, mass) if use_analytic else bubble_master_series_q2(q2, mass)
    return 3.0 * coupling * coupling * b_hat


def build_report(config: OneLoopConfig) -> pd.DataFrame:
    _check_positive_scalar("mass", config.mass)
    _check_positive_scalar("coupling", config.coupling)
    _check_positive_scalar("epsabs", config.epsabs)
    _check_positive_scalar("epsrel", config.epsrel)

    q2_grid = np.asarray(config.q2_values, dtype=float)
    _check_grid(q2_grid)

    rows: list[dict[str, float]] = []
    for q2 in q2_grid:
        numeric = bubble_master_numeric(
            q2=float(q2),
            mass=config.mass,
            epsabs=config.epsabs,
            epsrel=config.epsrel,
        )
        analytic = bubble_master_analytic(q2=float(q2), mass=config.mass)
        series = bubble_master_series_q2(q2=float(q2), mass=config.mass)

        abs_err_num_ana = abs(numeric - analytic)
        abs_err_series_ana = abs(series - analytic)
        rel_err_series_ana = abs_err_series_ana / max(1e-16, abs(analytic))

        rows.append(
            {
                "q2": float(q2),
                "a=q2/m^2": float(q2 / (config.mass * config.mass)),
                "B_numeric": numeric,
                "B_analytic": analytic,
                "abs_err_num_vs_ana": abs_err_num_ana,
                "B_series_O(q4)": series,
                "abs_err_series_vs_ana": abs_err_series_ana,
                "rel_err_series_vs_ana": rel_err_series_ana,
                "delta_lambda": one_loop_coupling_shift(
                    q2=float(q2),
                    mass=config.mass,
                    coupling=config.coupling,
                    use_analytic=True,
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    config = OneLoopConfig(
        mass=0.7,
        coupling=1.2,
        q2_values=(0.0, 0.02, 0.1, 0.3, 0.8, 1.5, 3.0, 6.0),
    )

    report = build_report(config)

    print("=== One-Loop Calculation MVP: Finite Bubble Master Integral ===")
    print(f"mass m = {config.mass:.6f}")
    print(f"coupling lambda = {config.coupling:.6f}")
    print("definition: B_hat(Q^2,m) = (1/16pi^2) * int_0^1 dx ln(1 + x(1-x)Q^2/m^2)")
    print()
    print(report.to_string(index=False, float_format=lambda x: f"{x:.12e}"))

    num_err = report["abs_err_num_vs_ana"].to_numpy()
    b_values = report["B_analytic"].to_numpy()
    q2_values = report["q2"].to_numpy()
    series_rel = report["rel_err_series_vs_ana"].to_numpy()

    # 1) Numerical quadrature must reproduce the analytic one-loop formula.
    assert np.max(num_err) < 1e-12, "Numeric and analytic one-loop values should match very closely."

    # 2) B_hat(0,m)=0 and B_hat should be nondecreasing in Euclidean Q^2.
    assert abs(float(b_values[0])) < 1e-15, "B_hat(Q^2=0,m) must be zero by subtraction definition."
    assert np.all(np.diff(b_values) >= -1e-14), "B_hat should be nondecreasing for nonnegative Q^2."

    # 3) Small-Q^2 series should be accurate on the low-energy points.
    low_energy_mask = q2_values <= 0.1
    low_energy_rel = series_rel[low_energy_mask]
    assert np.max(low_energy_rel) < 5e-3, "Low-energy expansion should be accurate at small Q^2."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
