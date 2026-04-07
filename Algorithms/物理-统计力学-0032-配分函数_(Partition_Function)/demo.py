"""Minimal MVP for the partition function in canonical ensemble.

Model:
- Quantum 1D harmonic oscillator with energy levels E_n = hbar*omega*(n + 1/2).
- Partition function is evaluated numerically by truncated level summation.
- Thermodynamic observables are derived from the level probabilities.

Validation:
- Compare numerical results against analytic formulas over a temperature grid.
- Keep assertions so `uv run python demo.py` is self-checking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp


@dataclass(frozen=True)
class PartitionConfig:
    hbar: float = 1.0
    omega: float = 1.7
    n_max: int = 180
    temperatures: tuple[float, ...] = (0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.8, 4.0)


def build_qho_spectrum(cfg: PartitionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return energies and degeneracies for truncated QHO spectrum."""
    n = np.arange(cfg.n_max + 1, dtype=float)
    energies = cfg.hbar * cfg.omega * (n + 0.5)
    degeneracies = np.ones_like(energies)
    return energies, degeneracies


def thermodynamics_from_spectrum(
    energies: np.ndarray,
    degeneracies: np.ndarray,
    temperature: float,
) -> dict[str, float]:
    """Compute Z, F, U, S, C from a discrete spectrum at one temperature.

    We use reduced units k_B = 1, so beta = 1 / T.
    Heat capacity is computed from fluctuations:
        C = Var(E) / T^2.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    beta = 1.0 / temperature
    log_weights = np.log(degeneracies) - beta * energies
    log_z = float(logsumexp(log_weights))

    probs = np.exp(log_weights - log_z)
    internal_energy = float(np.dot(probs, energies))
    energy_var = float(np.dot(probs, (energies - internal_energy) ** 2))

    z = float(np.exp(log_z))
    free_energy = -temperature * log_z
    entropy = (internal_energy - free_energy) / temperature
    heat_capacity = energy_var / (temperature * temperature)

    return {
        "Z": z,
        "logZ": log_z,
        "U": internal_energy,
        "F": free_energy,
        "S": entropy,
        "C": heat_capacity,
    }


def analytic_qho_thermodynamics(temperature: float, hbar: float, omega: float) -> dict[str, float]:
    """Analytic canonical thermodynamics for the 1D quantum harmonic oscillator."""
    x = (hbar * omega) / temperature
    y = np.exp(-x)

    # logZ = -x/2 - log(1 - e^{-x})
    log_z = float(-0.5 * x - np.log1p(-y))
    z = float(np.exp(log_z))

    internal_energy = float(hbar * omega * (0.5 + y / (1.0 - y)))
    free_energy = -temperature * log_z
    entropy = (internal_energy - free_energy) / temperature
    heat_capacity = float((x * x * y) / ((1.0 - y) ** 2))

    return {
        "Z": z,
        "logZ": log_z,
        "U": internal_energy,
        "F": free_energy,
        "S": entropy,
        "C": heat_capacity,
    }


def run_partition_function_demo(cfg: PartitionConfig) -> pd.DataFrame:
    energies, degeneracies = build_qho_spectrum(cfg)

    rows: list[dict[str, float]] = []
    for temp in cfg.temperatures:
        numeric = thermodynamics_from_spectrum(energies, degeneracies, temp)
        analytic = analytic_qho_thermodynamics(temp, cfg.hbar, cfg.omega)

        rows.append(
            {
                "T": temp,
                "logZ_numeric": numeric["logZ"],
                "logZ_analytic": analytic["logZ"],
                "U_numeric": numeric["U"],
                "U_analytic": analytic["U"],
                "C_numeric": numeric["C"],
                "C_analytic": analytic["C"],
                "S_numeric": numeric["S"],
                "S_analytic": analytic["S"],
            }
        )

    df = pd.DataFrame(rows)

    for key in ("logZ", "U", "C", "S"):
        num_col = f"{key}_numeric"
        ana_col = f"{key}_analytic"
        rel_col = f"relerr_{key}"
        denom = np.maximum(np.abs(df[ana_col].to_numpy()), 1e-12)
        df[rel_col] = np.abs(df[num_col] - df[ana_col]) / denom

    return df


def main() -> None:
    print("Partition Function MVP: canonical thermodynamics from discrete spectrum")

    cfg = PartitionConfig()
    result = run_partition_function_demo(cfg)

    view_cols = [
        "T",
        "logZ_numeric",
        "logZ_analytic",
        "U_numeric",
        "U_analytic",
        "C_numeric",
        "C_analytic",
        "relerr_logZ",
        "relerr_U",
        "relerr_C",
    ]
    with pd.option_context("display.precision", 7, "display.width", 180):
        print(result[view_cols].to_string(index=False))

    max_logz_err = float(result["relerr_logZ"].max())
    max_u_err = float(result["relerr_U"].max())
    max_c_err = float(result["relerr_C"].max())
    max_s_err = float(result["relerr_S"].max())

    print(
        "max relative errors -> "
        f"logZ={max_logz_err:.3e}, U={max_u_err:.3e}, C={max_c_err:.3e}, S={max_s_err:.3e}"
    )

    # Self-validation checks for this MVP.
    assert len(result) == len(cfg.temperatures)
    assert np.all(np.diff(result["U_numeric"].to_numpy()) > 0.0)
    assert max_logz_err < 2e-8
    assert max_u_err < 2e-6
    assert max_c_err < 3e-4
    assert max_s_err < 2e-6

    print("Checks passed: numerical partition-function sums match analytic thermodynamics.")


if __name__ == "__main__":
    main()
