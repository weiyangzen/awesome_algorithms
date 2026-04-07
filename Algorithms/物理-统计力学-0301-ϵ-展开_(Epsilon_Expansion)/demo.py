"""Minimal runnable MVP for epsilon expansion in O(N) phi^4 criticality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar


@dataclass(frozen=True)
class EpsilonExpansionConfig:
    """Configuration for the epsilon-expansion MVP experiment."""

    n_values: tuple[int, ...] = (1, 2, 3)
    epsilon_values: tuple[float, ...] = (0.1, 0.5, 1.0)
    root_rtol: float = 1e-12
    root_xtol: float = 1e-14


def beta_phi4_one_loop(g: float, epsilon: float, n: int) -> float:
    """One-loop beta function for dimensionless phi^4 coupling."""

    return -epsilon * g + ((n + 8.0) / 6.0) * g * g


def validate_inputs(n: int, epsilon: float) -> None:
    """Validate physical domain used by the low-order epsilon expansion."""

    if n <= -8:
        raise ValueError("n must be > -8 to avoid singular denominators in low-order formulas")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0 to target the nontrivial Wilson-Fisher fixed point")


def find_wilson_fisher_fixed_point(
    n: int,
    epsilon: float,
    *,
    rtol: float = 1e-12,
    xtol: float = 1e-14,
) -> tuple[float, float, float]:
    """Solve for the nontrivial fixed point g*>0 and compare with analytic one-loop value."""

    validate_inputs(n=n, epsilon=epsilon)

    slope = (n + 8.0) / 6.0

    def reduced_beta(g: float) -> float:
        # Solves beta(g)/g = 0 for g>0, avoiding the trivial Gaussian root g=0.
        return -epsilon + slope * g

    g_star_analytic = 6.0 * epsilon / (n + 8.0)
    upper = max(1.0, 2.0 * g_star_analytic)

    sol = root_scalar(reduced_beta, bracket=(0.0, upper), method="brentq", rtol=rtol, xtol=xtol)
    if not sol.converged:
        raise RuntimeError(f"root solve did not converge for n={n}, epsilon={epsilon}")

    g_star_numeric = float(sol.root)
    rel_err = abs(g_star_numeric - g_star_analytic) / max(1e-15, abs(g_star_analytic))
    return g_star_numeric, g_star_analytic, rel_err


def critical_exponents_low_order(n: int, epsilon: float) -> dict[str, float]:
    """Compute low-order epsilon-expansion exponents for O(N) universality class."""

    validate_inputs(n=n, epsilon=epsilon)

    d = 4.0 - epsilon
    nu_inv = 2.0 - ((n + 2.0) / (n + 8.0)) * epsilon
    if nu_inv <= 0.0:
        raise ValueError(f"nu_inv <= 0 encountered: n={n}, epsilon={epsilon}, nu_inv={nu_inv}")

    nu = 1.0 / nu_inv
    # In epsilon expansion, eta first appears at O(epsilon^2).
    eta = ((n + 2.0) / (2.0 * (n + 8.0) ** 2)) * (epsilon**2)

    gamma = nu * (2.0 - eta)
    beta_mag = 0.5 * nu * (d - 2.0 + eta)
    alpha = 2.0 - d * nu
    delta = (d + 2.0 - eta) / (d - 2.0 + eta)

    return {
        "nu": float(nu),
        "eta": float(eta),
        "gamma": float(gamma),
        "beta": float(beta_mag),
        "alpha": float(alpha),
        "delta": float(delta),
    }


def run_mvp(config: EpsilonExpansionConfig) -> pd.DataFrame:
    """Run the epsilon-expansion MVP over configured (N, epsilon) grid."""

    rows: list[dict[str, float | int]] = []
    for n in config.n_values:
        for epsilon in config.epsilon_values:
            g_star_numeric, g_star_analytic, rel_err = find_wilson_fisher_fixed_point(
                n=n,
                epsilon=epsilon,
                rtol=config.root_rtol,
                xtol=config.root_xtol,
            )
            exponents = critical_exponents_low_order(n=n, epsilon=epsilon)

            rows.append(
                {
                    "N": n,
                    "epsilon": float(epsilon),
                    "dimension_d": 4.0 - float(epsilon),
                    "g_star_numeric": g_star_numeric,
                    "g_star_analytic": g_star_analytic,
                    "g_star_rel_err": rel_err,
                    **exponents,
                }
            )

    return pd.DataFrame(rows)


def compare_3d_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Compare epsilon=1 predictions with common 3D benchmark values (rough references)."""

    references = {
        1: {"nu_ref": 0.62997, "eta_ref": 0.03630, "gamma_ref": 1.23708},
        2: {"nu_ref": 0.67170, "eta_ref": 0.03810, "gamma_ref": 1.31770},
        3: {"nu_ref": 0.71120, "eta_ref": 0.03750, "gamma_ref": 1.39600},
    }

    subset = df[np.isclose(df["epsilon"], 1.0)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["N", "nu", "eta", "gamma", "nu_ref", "eta_ref", "gamma_ref"])

    rows: list[dict[str, float | int]] = []
    for row in subset.itertuples(index=False):
        n = int(row.N)
        ref = references.get(n)
        if ref is None:
            continue

        rows.append(
            {
                "N": n,
                "nu": float(row.nu),
                "eta": float(row.eta),
                "gamma": float(row.gamma),
                "nu_ref": ref["nu_ref"],
                "eta_ref": ref["eta_ref"],
                "gamma_ref": ref["gamma_ref"],
                "abs_err_nu": abs(float(row.nu) - ref["nu_ref"]),
                "abs_err_eta": abs(float(row.eta) - ref["eta_ref"]),
                "abs_err_gamma": abs(float(row.gamma) - ref["gamma_ref"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    config = EpsilonExpansionConfig()
    result_df = run_mvp(config)
    compare_df = compare_3d_reference(result_df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print("=== Epsilon Expansion MVP (O(N) phi^4, one-loop focused) ===")
    print(
        "config:",
        {
            "N_values": config.n_values,
            "epsilon_values": config.epsilon_values,
            "root_rtol": config.root_rtol,
            "root_xtol": config.root_xtol,
        },
    )
    print()
    print("Main table:")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()
    print("3D (epsilon=1) rough-reference comparison:")
    if compare_df.empty:
        print("<empty>")
    else:
        print(compare_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    # Minimal quality gates.
    if not (result_df["g_star_rel_err"].to_numpy() < 1e-10).all():
        raise AssertionError("numeric fixed points should match one-loop analytic values tightly")
    if not (result_df["nu"].to_numpy() > 0).all():
        raise AssertionError("nu should stay positive in configured range")
    if not (result_df["g_star_numeric"].to_numpy() > 0).all():
        raise AssertionError("nontrivial Wilson-Fisher fixed point should be positive for epsilon>0")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
