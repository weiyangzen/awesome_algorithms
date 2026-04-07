"""Minimal runnable MVP for unitarity-bound checking in a toy QFT model."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import eval_legendre


@dataclass(frozen=True)
class ModelParams:
    """Parameters for a simple 2->2 scalar scattering toy model."""

    lam: float
    g: float
    m_med: float
    m_ext: float = 0.0


def mandelstam_t(s: float, cos_theta: np.ndarray, m_ext: float) -> np.ndarray:
    """Compute t(s, cos(theta)) for identical external masses in COM frame."""
    return -0.5 * (s - 4.0 * m_ext * m_ext) * (1.0 - cos_theta)


def tree_level_amplitude(s: float, cos_theta: np.ndarray, params: ModelParams) -> np.ndarray:
    """Toy tree-level amplitude: M = lam + g^2 / (m_med^2 - t)."""
    t = mandelstam_t(s=s, cos_theta=cos_theta, m_ext=params.m_ext)
    denom = params.m_med * params.m_med - t
    return params.lam + (params.g * params.g) / denom


def partial_wave_a_l(s: float, ell: int, params: ModelParams, n_angle: int = 4097) -> float:
    """Numerically compute a_ell(s) = (1/32pi) int_{-1}^{1} P_ell(c) M(s,c) dc."""
    cos_theta = np.linspace(-1.0, 1.0, n_angle)
    legendre = eval_legendre(ell, cos_theta)
    amp = tree_level_amplitude(s=s, cos_theta=cos_theta, params=params)
    integral = np.trapezoid(legendre * amp, cos_theta)
    return float(integral / (32.0 * pi))


def scan_unitarity(s_values: np.ndarray, ell: int, params: ModelParams) -> pd.DataFrame:
    """Scan partial-wave amplitudes and unitarity margins over an energy grid."""
    a_values = np.array([partial_wave_a_l(s=s, ell=ell, params=params) for s in s_values])
    bound = 0.5  # perturbative tree-level bound: |Re a_ell| <= 1/2
    df = pd.DataFrame(
        {
            "s": s_values,
            "sqrt_s": np.sqrt(s_values),
            "a_l": a_values,
            "abs_re_a_l": np.abs(np.real(a_values)),
        }
    )
    df["margin_to_bound"] = bound - df["abs_re_a_l"]
    df["violates_bound"] = df["margin_to_bound"] < 0.0
    return df


def contact_closed_form_a0(lam: float) -> float:
    """For M=lam constant, a0 = lam / (16pi)."""
    return lam / (16.0 * pi)


def verify_contact_formula(lam: float) -> float:
    """Return absolute error between numeric and closed-form contact a0."""
    params = ModelParams(lam=lam, g=0.0, m_med=1.0, m_ext=0.0)
    numeric = partial_wave_a_l(s=10.0, ell=0, params=params)
    closed_form = contact_closed_form_a0(lam)
    return abs(numeric - closed_form)


def max_abs_re_a0_for_lambda(
    lam: float, s_values: np.ndarray, g: float, m_med: float, m_ext: float
) -> float:
    """Helper for root solving lambda at the perturbative unitarity edge."""
    params = ModelParams(lam=lam, g=g, m_med=m_med, m_ext=m_ext)
    df = scan_unitarity(s_values=s_values, ell=0, params=params)
    return float(df["abs_re_a_l"].max())


def find_critical_lambda(s_values: np.ndarray, g: float, m_med: float, m_ext: float = 0.0) -> float:
    """Solve max_s |Re a0(s)| = 1/2 via Brent's method."""

    def objective(lam: float) -> float:
        return max_abs_re_a0_for_lambda(
            lam=lam,
            s_values=s_values,
            g=g,
            m_med=m_med,
            m_ext=m_ext,
        ) - 0.5

    lower = 0.0
    upper = 8.0 * pi
    f_lower = objective(lower)
    f_upper = objective(upper)

    # Expand bracket conservatively if the default upper bound is not enough.
    while f_upper <= 0.0:
        upper *= 1.5
        f_upper = objective(upper)
        if upper > 1e4:
            raise RuntimeError("Failed to bracket critical lambda for unitarity bound.")

    if f_lower >= 0.0:
        raise RuntimeError("Unitarity already violated at lambda=0; check model parameters.")

    return float(brentq(objective, lower, upper, xtol=1e-10, rtol=1e-10))


def summarize_case(label: str, s_values: np.ndarray, params: ModelParams) -> tuple[float, pd.DataFrame]:
    """Run one scenario and print concise diagnostics."""
    df = scan_unitarity(s_values=s_values, ell=0, params=params)
    max_idx = df["abs_re_a_l"].idxmax()
    max_row = df.loc[max_idx]

    print(f"\n[{label}]")
    print(f"lambda={params.lam:.6f}, g={params.g:.3f}, m_med={params.m_med:.3f}")
    print(
        "max |Re a0|={:.6f} at sqrt(s)={:.6f}, margin={:.6f}, violated={}".format(
            max_row["abs_re_a_l"],
            max_row["sqrt_s"],
            max_row["margin_to_bound"],
            bool(max_row["violates_bound"]),
        )
    )
    preview = df[["sqrt_s", "a_l", "abs_re_a_l", "margin_to_bound", "violates_bound"]].head(5)
    print("preview:")
    print(preview.to_string(index=False))
    return float(max_row["abs_re_a_l"]), df


def main() -> None:
    # Energy scan in natural units: s = E_cm^2.
    s_values = np.linspace(0.1, 36.0, 24)

    # 1) Basic numerical-consistency check against closed-form contact result.
    contact_err = verify_contact_formula(lam=7.5)
    print(f"contact-model check |a0_numeric - a0_closed| = {contact_err:.3e}")

    # 2) Solve critical lambda where perturbative unitarity is saturated.
    g = 1.2
    m_med = 3.0
    lambda_crit = find_critical_lambda(s_values=s_values, g=g, m_med=m_med, m_ext=0.0)
    print(f"critical lambda (max_s |Re a0| = 1/2): {lambda_crit:.6f}")

    # 3) Compare a safe point and an intentionally violating point.
    safe_params = ModelParams(lam=0.6 * lambda_crit, g=g, m_med=m_med)
    risky_params = ModelParams(lam=1.05 * lambda_crit, g=g, m_med=m_med)

    safe_max, _ = summarize_case("safe", s_values=s_values, params=safe_params)
    risky_max, _ = summarize_case("risky", s_values=s_values, params=risky_params)

    # 4) MVP assertions.
    assert contact_err < 1e-9, "Contact-formula numerical check failed."
    assert safe_max <= 0.5 + 1e-10, "Safe case should satisfy perturbative unitarity."
    assert risky_max > 0.5, "Risky case should violate perturbative unitarity."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
