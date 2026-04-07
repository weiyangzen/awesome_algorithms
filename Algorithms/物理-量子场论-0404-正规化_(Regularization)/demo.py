r"""Regularization MVP for a one-loop QFT tadpole integral.

We demonstrate ultraviolet regularization via a hard momentum cutoff:

    I(\Lambda, m) = \int_{|k|<\Lambda} d^4k_E/(2\pi)^4 * 1/(k^2 + m^2)

and compare:
1) numerical radial quadrature,
2) analytic closed form,
3) large-cutoff asymptotic expansion,
4) subtraction-based renormalized finite part.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad


@dataclass(frozen=True)
class RegularizationConfig:
    mass: float
    mu_ren: float
    cutoffs: tuple[float, ...]
    epsabs: float = 1e-11
    epsrel: float = 1e-11


def _check_positive_scalar(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def _check_cutoffs(cutoffs: np.ndarray) -> None:
    if cutoffs.ndim != 1 or cutoffs.size < 2:
        raise ValueError("cutoffs must be a 1D array with at least two entries.")
    if not np.all(np.isfinite(cutoffs)) or not np.all(cutoffs > 0.0):
        raise ValueError("cutoffs must contain only positive finite values.")
    if not np.all(np.diff(cutoffs) > 0.0):
        raise ValueError("cutoffs must be strictly increasing.")


def tadpole_integrand_radial(k: float, mass: float) -> float:
    """Radial integrand for Euclidean 4D tadpole before angular prefactor."""
    return k**3 / (k**2 + mass**2)


def tadpole_cutoff_numeric(mass: float, cutoff: float, epsabs: float, epsrel: float) -> float:
    """Compute I(Λ,m) by numerical quadrature in radial coordinates.

    d^4k = 2*pi^2 k^3 dk, and d^4k/(2*pi)^4 gives prefactor 1/(8*pi^2).
    """
    _check_positive_scalar("mass", mass)
    _check_positive_scalar("cutoff", cutoff)
    _check_positive_scalar("epsabs", epsabs)
    _check_positive_scalar("epsrel", epsrel)

    radial_value, _ = quad(
        lambda k: tadpole_integrand_radial(k, mass),
        0.0,
        cutoff,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=200,
    )
    return radial_value / (8.0 * np.pi**2)


def tadpole_cutoff_analytic(mass: float, cutoff: float) -> float:
    """Closed form of the hard-cutoff tadpole integral.

    I(Λ,m) = [Λ^2 - m^2 ln(1 + Λ^2/m^2)] / (16*pi^2)
    """
    _check_positive_scalar("mass", mass)
    _check_positive_scalar("cutoff", cutoff)
    ratio_sq = (cutoff / mass) ** 2
    return (cutoff**2 - mass**2 * np.log1p(ratio_sq)) / (16.0 * np.pi**2)


def tadpole_cutoff_asymptotic(mass: float, cutoff: float) -> float:
    """Leading large-Λ asymptotic approximation.

    I(Λ,m) ~ [Λ^2 - m^2 ln(Λ^2/m^2)] / (16*pi^2)
    """
    _check_positive_scalar("mass", mass)
    _check_positive_scalar("cutoff", cutoff)
    return (cutoff**2 - mass**2 * np.log((cutoff**2) / (mass**2))) / (16.0 * np.pi**2)


def renormalized_tadpole_subtracted(mass: float, cutoff: float, mu_ren: float) -> float:
    """Subtraction-based finite quantity in a cutoff scheme.

    I_R(Λ,m;μ) = I(Λ,m) - [Λ^2 - m^2 ln(Λ^2/μ^2)]/(16*pi^2)
    This removes power/log divergences explicitly in this toy setup.
    """
    _check_positive_scalar("mass", mass)
    _check_positive_scalar("cutoff", cutoff)
    _check_positive_scalar("mu_ren", mu_ren)

    raw = tadpole_cutoff_analytic(mass=mass, cutoff=cutoff)
    counterterm = (cutoff**2 - mass**2 * np.log((cutoff**2) / (mu_ren**2))) / (16.0 * np.pi**2)
    return raw - counterterm


def renormalized_limit(mass: float, mu_ren: float) -> float:
    """Λ->∞ limit of the subtraction-based finite part."""
    _check_positive_scalar("mass", mass)
    _check_positive_scalar("mu_ren", mu_ren)
    return (mass**2 * np.log((mass**2) / (mu_ren**2))) / (16.0 * np.pi**2)


def build_report(config: RegularizationConfig) -> pd.DataFrame:
    mass = float(config.mass)
    mu_ren = float(config.mu_ren)
    _check_positive_scalar("mass", mass)
    _check_positive_scalar("mu_ren", mu_ren)

    cutoffs = np.asarray(config.cutoffs, dtype=float)
    _check_cutoffs(cutoffs)

    rows: list[dict[str, float]] = []
    target_limit = renormalized_limit(mass=mass, mu_ren=mu_ren)

    for lam in cutoffs:
        numeric = tadpole_cutoff_numeric(
            mass=mass,
            cutoff=float(lam),
            epsabs=config.epsabs,
            epsrel=config.epsrel,
        )
        analytic = tadpole_cutoff_analytic(mass=mass, cutoff=float(lam))
        asymptotic = tadpole_cutoff_asymptotic(mass=mass, cutoff=float(lam))
        renorm = renormalized_tadpole_subtracted(mass=mass, cutoff=float(lam), mu_ren=mu_ren)

        rows.append(
            {
                "cutoff": float(lam),
                "I_numeric": numeric,
                "I_analytic": analytic,
                "abs_err_numeric_vs_analytic": abs(numeric - analytic),
                "I_asymptotic": asymptotic,
                "abs_err_asymptotic_vs_analytic": abs(asymptotic - analytic),
                "I_ren_subtracted": renorm,
                "abs_err_ren_vs_limit": abs(renorm - target_limit),
            }
        )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    config = RegularizationConfig(
        mass=0.7,
        mu_ren=1.0,
        cutoffs=(2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
    )

    report = build_report(config)
    limit_value = renormalized_limit(mass=config.mass, mu_ren=config.mu_ren)

    print("=== QFT Regularization MVP: Euclidean Tadpole with Hard Cutoff ===")
    print(f"mass m = {config.mass:.6f}")
    print(f"renormalization scale mu = {config.mu_ren:.6f}")
    print(f"predicted finite limit (Lambda->inf): {limit_value:.12f}")
    print()
    print(report.to_string(index=False, float_format=lambda x: f"{x:.12e}"))

    raw_values = report["I_analytic"].to_numpy()
    ren_errors = report["abs_err_ren_vs_limit"].to_numpy()
    num_errors = report["abs_err_numeric_vs_analytic"].to_numpy()

    # Assertions for correctness and behavior checks.
    assert np.max(num_errors) < 1e-10, "Numerical quadrature should match analytic formula very closely."
    assert np.all(np.diff(raw_values) > 0.0), "Raw cutoff integral should grow with cutoff (UV divergence)."
    assert ren_errors[-1] < 5e-7, "Renormalized value at largest cutoff should be near its finite limit."
    assert ren_errors[-1] < ren_errors[0], "Renormalized error should decrease as cutoff increases."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
