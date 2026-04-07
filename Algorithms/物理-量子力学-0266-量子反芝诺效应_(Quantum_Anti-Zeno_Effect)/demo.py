"""Quantum Anti-Zeno Effect (QAZE) minimal runnable MVP.

Model idea:
- Measurement interval tau broadens the transition line by a sinc^2 filter.
- Effective decay rate under repeated projective measurements:
    Gamma_eff(tau) = tau * integral J(omega) * sinc^2((omega-omega_0)*tau/2) d omega
- Golden-rule no-measurement reference:
    Gamma_0 = 2*pi*J(omega_0)

If Gamma_eff(tau) < Gamma_0: Zeno suppression.
If Gamma_eff(tau) > Gamma_0: Anti-Zeno acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import simpson


@dataclass(frozen=True)
class QAZEConfig:
    """Configuration for the anti-Zeno numerical demonstration."""

    omega_0: float = 1.0
    omega_peak: float = 1.35
    spectral_width: float = 0.12
    coupling_scale: float = 0.22

    omega_max: float = 5.0
    n_omega: int = 20001

    tau_min: float = 0.01
    tau_max: float = 12.0
    n_tau: int = 80

    total_time: float = 10.0
    sample_taus: tuple[float, ...] = (0.02, 0.05, 0.10, 0.30, 0.60, 1.0, 2.0, 4.0, 8.0, 12.0)


def validate_config(config: QAZEConfig) -> None:
    """Validate configuration ranges for numerical stability."""
    if config.omega_0 <= 0.0:
        raise ValueError("omega_0 must be positive.")
    if config.spectral_width <= 0.0:
        raise ValueError("spectral_width must be positive.")
    if config.coupling_scale <= 0.0:
        raise ValueError("coupling_scale must be positive.")
    if config.omega_max <= config.omega_0:
        raise ValueError("omega_max must be larger than omega_0.")
    if config.n_omega < 1001:
        raise ValueError("n_omega should be >= 1001.")
    if config.tau_min <= 0.0 or config.tau_max <= config.tau_min:
        raise ValueError("tau range is invalid.")
    if config.n_tau < 10:
        raise ValueError("n_tau should be >= 10.")
    if config.total_time <= 0.0:
        raise ValueError("total_time must be positive.")


def spectral_density(omega: np.ndarray | float, config: QAZEConfig) -> np.ndarray:
    """Gaussian-shaped reservoir spectral density J(omega)."""
    omega_arr = np.asarray(omega, dtype=np.float64)
    centered = (omega_arr - config.omega_peak) / config.spectral_width
    return config.coupling_scale * np.exp(-0.5 * centered * centered)


def measurement_filter(omega: np.ndarray, tau: float, omega_0: float) -> np.ndarray:
    """Measurement broadening filter sinc^2((omega-omega_0) * tau / 2)."""
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    arg = ((omega - omega_0) * tau / 2.0) / np.pi
    return np.sinc(arg) ** 2


def golden_rule_rate(config: QAZEConfig) -> float:
    """No-measurement reference decay rate Gamma_0 = 2*pi*J(omega_0)."""
    j0 = float(spectral_density(np.array([config.omega_0]), config)[0])
    return float(2.0 * np.pi * j0)


def effective_decay_rate(tau: float, omega_grid: np.ndarray, j_omega: np.ndarray, omega_0: float) -> float:
    """Compute Gamma_eff(tau) via spectral-overlap integral."""
    filt = measurement_filter(omega_grid, tau=tau, omega_0=omega_0)
    integrand = j_omega * filt
    return float(tau * simpson(integrand, x=omega_grid))


def repeated_measurement_survival(gamma_eff: float, total_time: float) -> float:
    """Return survival probability under repeated measurements for total_time."""
    return float(np.exp(-gamma_eff * total_time))


def locate_crossovers(taus: np.ndarray, ratios: np.ndarray) -> list[float]:
    """Locate approximate tau where ratio crosses 1 by linear interpolation."""
    crossings: list[float] = []
    shifted = ratios - 1.0

    for i in range(len(taus) - 1):
        y1 = shifted[i]
        y2 = shifted[i + 1]
        if y1 == 0.0:
            crossings.append(float(taus[i]))
            continue
        if y1 * y2 < 0.0:
            t1 = taus[i]
            t2 = taus[i + 1]
            t_cross = t1 + (0.0 - y1) * (t2 - t1) / (y2 - y1)
            crossings.append(float(t_cross))
    return crossings


def run_qaze_scan(config: QAZEConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | bool | int | list[float]]]:
    """Run QAZE sweep and return full report, sampled report, and diagnostics."""
    validate_config(config)

    omega = np.linspace(0.0, config.omega_max, config.n_omega)
    j_omega = spectral_density(omega, config)

    gamma_0 = golden_rule_rate(config)
    if gamma_0 <= 0.0:
        raise RuntimeError("Gamma_0 must be positive.")

    taus = np.logspace(np.log10(config.tau_min), np.log10(config.tau_max), config.n_tau)
    gamma_eff = np.array(
        [effective_decay_rate(tau, omega, j_omega, config.omega_0) for tau in taus],
        dtype=np.float64,
    )
    ratio = gamma_eff / gamma_0

    p_measured = np.exp(-gamma_eff * config.total_time)
    p_free = float(np.exp(-gamma_0 * config.total_time))

    regime = np.where(ratio < 1.0, "Zeno suppression", "Anti-Zeno acceleration")

    report = pd.DataFrame(
        {
            "tau": taus,
            "gamma_eff": gamma_eff,
            "ratio_to_gamma0": ratio,
            "survival_measured": p_measured,
            "survival_free_ref": np.full_like(taus, p_free),
            "regime": regime,
        }
    )

    sample_rows: list[dict[str, float | str]] = []
    for tau in config.sample_taus:
        g = effective_decay_rate(tau, omega, j_omega, config.omega_0)
        r = g / gamma_0
        sample_rows.append(
            {
                "tau": float(tau),
                "gamma_eff": float(g),
                "ratio_to_gamma0": float(r),
                "survival_measured": repeated_measurement_survival(g, config.total_time),
                "survival_free_ref": p_free,
                "regime": "Zeno suppression" if r < 1.0 else "Anti-Zeno acceleration",
            }
        )
    sample_report = pd.DataFrame(sample_rows)

    crossings = locate_crossovers(taus, ratio)
    i_min = int(np.argmin(ratio))
    i_max = int(np.argmax(ratio))

    diagnostics: dict[str, float | bool | int | list[float]] = {
        "gamma_0": float(gamma_0),
        "min_ratio": float(ratio[i_min]),
        "tau_at_min_ratio": float(taus[i_min]),
        "max_ratio": float(ratio[i_max]),
        "tau_at_max_ratio": float(taus[i_max]),
        "zeno_exists": bool(np.any(ratio < 1.0)),
        "anti_zeno_exists": bool(np.any(ratio > 1.0)),
        "crossover_count": int(len(crossings)),
        "crossover_taus": crossings,
    }
    return report, sample_report, diagnostics


def run_checks(report: pd.DataFrame, sample_report: pd.DataFrame, diagnostics: dict[str, float | bool | int | list[float]]) -> None:
    """Assert key QAZE properties for this MVP."""
    if report.empty:
        raise AssertionError("Full report is empty.")
    if sample_report.empty:
        raise AssertionError("Sample report is empty.")

    assert float(diagnostics["gamma_0"]) > 0.0, "Gamma_0 must be positive."
    assert bool(diagnostics["zeno_exists"]), "Expected at least one Zeno regime point (ratio < 1)."
    assert bool(diagnostics["anti_zeno_exists"]), "Expected at least one anti-Zeno point (ratio > 1)."
    assert float(diagnostics["min_ratio"]) < 0.2, "Zeno suppression is too weak for this configuration."
    assert float(diagnostics["max_ratio"]) > 1.5, "Anti-Zeno acceleration is too weak for this configuration."
    assert int(diagnostics["crossover_count"]) >= 1, "No Zeno/anti-Zeno crossover detected."


def main() -> None:
    config = QAZEConfig()
    report, sample_report, diagnostics = run_qaze_scan(config)
    run_checks(report, sample_report, diagnostics)

    print("Quantum Anti-Zeno Effect MVP (spectral-overlap model)")
    print(
        "J(omega) = coupling_scale * exp(-0.5*((omega-omega_peak)/spectral_width)^2), "
        "Gamma_eff(tau) = tau * integral J(omega) * sinc^2((omega-omega0)tau/2) d omega"
    )
    print(
        f"omega0={config.omega_0:.3f}, omega_peak={config.omega_peak:.3f}, "
        f"spectral_width={config.spectral_width:.3f}, coupling_scale={config.coupling_scale:.3f}"
    )
    print(
        f"Gamma_0={float(diagnostics['gamma_0']):.6e}, "
        f"total_time={config.total_time:.2f}"
    )
    print()
    print("Sampled tau points:")
    print(
        sample_report.to_string(
            index=False,
            float_format=lambda v: f"{v: .6e}",
        )
    )
    print()
    print("Diagnostics:")
    print(f"  min_ratio        : {float(diagnostics['min_ratio']):.6f} at tau={float(diagnostics['tau_at_min_ratio']):.6f}")
    print(f"  max_ratio        : {float(diagnostics['max_ratio']):.6f} at tau={float(diagnostics['tau_at_max_ratio']):.6f}")
    print(f"  crossover_count  : {int(diagnostics['crossover_count'])}")
    print(f"  crossover_taus   : {diagnostics['crossover_taus']}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
