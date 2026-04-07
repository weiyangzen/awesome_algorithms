"""Minimal runnable MVP for the Ginzburg criterion in statistical mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class LandauParams:
    """Dimensionless Landau parameters for a 3D phi^4 model near Tc."""

    a0: float
    b: float
    c: float
    kbt: float


def validate_params(params: LandauParams) -> None:
    """Basic physical sanity checks."""
    if params.a0 <= 0:
        raise ValueError("a0 must be > 0")
    if params.b <= 0:
        raise ValueError("b must be > 0")
    if params.c <= 0:
        raise ValueError("c must be > 0")
    if params.kbt <= 0:
        raise ValueError("kbt must be > 0")


def correlation_length(params: LandauParams, tau: float) -> float:
    """Mean-field correlation length xi ~ sqrt(c / (a0 * tau))."""
    if tau <= 0:
        raise ValueError("tau must be > 0")
    return float(np.sqrt(params.c / (params.a0 * tau)))


def mean_field_order_parameter_sq(params: LandauParams, tau: float) -> float:
    """Below Tc: phi0^2 = a0 * tau / b, where tau = |T-Tc|/Tc."""
    if tau <= 0:
        raise ValueError("tau must be > 0")
    return float(params.a0 * tau / params.b)


def fluctuation_variance_within_xi(params: LandauParams, tau: float, n_k: int = 4000) -> float:
    """Compute <(delta phi)^2> by integrating k-modes up to k_max = 1/xi.

    3D isotropic integral:
    <(delta phi)^2> = (kBT / 2pi^2) * int_0^{k_max} k^2/(a0*tau + c*k^2) dk
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    if n_k < 16:
        raise ValueError("n_k must be >= 16")

    xi = correlation_length(params, tau)
    k_max = 1.0 / xi
    k = np.linspace(0.0, k_max, n_k)
    integrand = (k * k) / (params.a0 * tau + params.c * k * k)
    integral = np.trapezoid(integrand, k)
    prefactor = params.kbt / (2.0 * np.pi * np.pi)
    return float(prefactor * integral)


def ginzburg_ratio(params: LandauParams, tau: float, n_k: int = 4000) -> float:
    """R(tau) = fluctuation variance / mean-field order parameter squared."""
    fluct = fluctuation_variance_within_xi(params, tau, n_k=n_k)
    phi_sq = mean_field_order_parameter_sq(params, tau)
    return float(fluct / max(phi_sq, EPS))


def estimate_ginzburg_tau(
    params: LandauParams,
    tau_low: float = 1e-6,
    tau_high: float = 2e-1,
    tol: float = 1e-7,
    max_iter: int = 80,
) -> float:
    """Solve R(tau)=1 by bisection, where R is the Ginzburg ratio."""
    if tau_low <= 0 or tau_high <= 0 or tau_low >= tau_high:
        raise ValueError("require 0 < tau_low < tau_high")

    f_low = ginzburg_ratio(params, tau_low) - 1.0
    f_high = ginzburg_ratio(params, tau_high) - 1.0
    if not (f_low > 0 and f_high < 0):
        raise ValueError(
            "Bisection bracket invalid: need ratio(tau_low)>1 and ratio(tau_high)<1. "
            f"Got f_low={f_low:.3e}, f_high={f_high:.3e}"
        )

    left = tau_low
    right = tau_high
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = ginzburg_ratio(params, mid) - 1.0

        if abs(f_mid) < tol or (right - left) < tol:
            return float(mid)

        if f_mid > 0:
            left = mid
        else:
            right = mid

    return float(0.5 * (left + right))


def asymptotic_tau_g(params: LandauParams) -> float:
    """Closed-form scaling estimate under the same k_max=1/xi truncation.

    For 3D with this truncation:
    R(tau) ~ C * tau^{-1/2}, C = ((1-pi/4)/(2*pi^2)) * kBT*b/(c^(3/2)*sqrt(a0)).
    So tau_G ~ C^2 from R=1.
    """
    c_pref = (1.0 - np.pi / 4.0) / (2.0 * np.pi * np.pi)
    coeff = c_pref * params.kbt * params.b / (params.c ** 1.5 * np.sqrt(params.a0))
    return float(coeff * coeff)


def classify_regime(ratio: float) -> str:
    """Interpretation helper for printed table."""
    if ratio < 0.1:
        return "Mean-field valid"
    if ratio < 1.0:
        return "Crossover"
    return "Fluctuation-dominated"


def build_report_table(params: LandauParams, tau_grid: np.ndarray) -> pd.DataFrame:
    """Assemble a compact diagnostic table across tau."""
    records: list[dict[str, float | str]] = []
    for tau in tau_grid:
        tau_f = float(tau)
        xi = correlation_length(params, tau_f)
        phi_sq = mean_field_order_parameter_sq(params, tau_f)
        fluct = fluctuation_variance_within_xi(params, tau_f)
        ratio = fluct / max(phi_sq, EPS)
        records.append(
            {
                "tau=|T-Tc|/Tc": tau_f,
                "xi": xi,
                "phi0^2": phi_sq,
                "<delta_phi^2>_xi": fluct,
                "ginzburg_ratio": ratio,
                "regime": classify_regime(ratio),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    params = LandauParams(a0=1.0, b=9.0, c=1.2, kbt=1.0)
    validate_params(params)

    tau_grid = np.logspace(-4, -1, 16)
    table = build_report_table(params, tau_grid)

    tau_g_num = estimate_ginzburg_tau(params, tau_low=1e-5, tau_high=2e-1)
    tau_g_asym = asymptotic_tau_g(params)
    ratio_at_tau_g = ginzburg_ratio(params, tau_g_num)

    ratios = table["ginzburg_ratio"].to_numpy(dtype=float)
    monotonic_decreasing = bool(np.all(np.diff(ratios) < 0))

    checks = {
        "ratio(tau) decreases with tau": monotonic_decreasing,
        "ratio(tau_G) close to 1": abs(ratio_at_tau_g - 1.0) < 2e-3,
        "tau_G in practical window [1e-4, 5e-2]": 1e-4 <= tau_g_num <= 5e-2,
        "numerical vs asymptotic tau_G relerr < 8%": abs(tau_g_num - tau_g_asym) / tau_g_asym < 0.08,
    }

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("=== Ginzburg Criterion MVP (PHYS-0301) ===")
    print("Landau params:", params)
    print("\nDiagnostic table:")
    print(table.to_string(index=False))

    print("\nEstimated Ginzburg reduced temperature:")
    print(f"tau_G (numerical, R=1): {tau_g_num:.6e}")
    print(f"tau_G (asymptotic):      {tau_g_asym:.6e}")
    print(f"R(tau_G):                {ratio_at_tau_g:.6e}")

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
