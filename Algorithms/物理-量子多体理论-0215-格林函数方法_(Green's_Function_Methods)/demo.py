"""Minimal runnable MVP for Green's Function Methods (PHYS-0214).

This demo builds a pedagogical interacting single-band Green's-function workflow:
1) define a 1D lattice dispersion epsilon(k)
2) define a causal retarded self-energy Sigma^R(omega)
3) solve Dyson equation for G^R(k, omega)
4) compute spectral function A(k, omega), DOS and occupations
5) compare interacting and non-interacting spectral peaks near k_F
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import find_peaks


@dataclass(frozen=True)
class GreenConfig:
    """Configuration for a minimal many-body Green's-function model."""

    nk: int = 81
    k_max: float = float(np.pi)

    nw: int = 2201
    omega_min: float = -8.0
    omega_max: float = 8.0

    hopping: float = 1.0
    chemical_potential: float = 0.0

    eta: float = 0.04

    interaction_u: float = 1.45
    self_energy_pole: float = 1.20
    self_energy_damping: float = 0.35

    temperature: float = 0.10

    n_report_k: int = 7
    peak_window: float = 2.5


def build_k_grid(cfg: GreenConfig) -> np.ndarray:
    if cfg.nk < 11:
        raise ValueError("nk must be >= 11")
    if cfg.nk % 2 == 0:
        raise ValueError("nk must be odd for symmetric sampling around k=0")
    if cfg.k_max <= 0.0:
        raise ValueError("k_max must be positive")
    return np.linspace(-cfg.k_max, cfg.k_max, cfg.nk, dtype=float)


def build_omega_grid(cfg: GreenConfig) -> np.ndarray:
    if cfg.nw < 401:
        raise ValueError("nw must be >= 401")
    if cfg.omega_max <= cfg.omega_min:
        raise ValueError("omega_max must be larger than omega_min")
    return np.linspace(cfg.omega_min, cfg.omega_max, cfg.nw, dtype=float)


def dispersion(k_grid: np.ndarray, cfg: GreenConfig) -> np.ndarray:
    if cfg.hopping <= 0.0:
        raise ValueError("hopping must be positive")
    return -2.0 * cfg.hopping * np.cos(k_grid) - cfg.chemical_potential


def retarded_self_energy(omega_grid: np.ndarray, cfg: GreenConfig) -> np.ndarray:
    if cfg.interaction_u <= 0.0:
        raise ValueError("interaction_u must be positive")
    if cfg.self_energy_pole <= 0.0:
        raise ValueError("self_energy_pole must be positive")
    if cfg.self_energy_damping <= 0.0:
        raise ValueError("self_energy_damping must be positive")

    coupling2 = cfg.interaction_u * cfg.interaction_u
    denom = omega_grid + cfg.self_energy_pole + 1j * cfg.self_energy_damping
    return coupling2 / denom


def retarded_green(
    omega_grid: np.ndarray,
    eps_k: np.ndarray,
    eta: float,
    sigma_omega: np.ndarray | None = None,
) -> np.ndarray:
    if eta <= 0.0:
        raise ValueError("eta must be positive")

    base = omega_grid[:, None] + 1j * eta - eps_k[None, :]
    if sigma_omega is not None:
        base = base - sigma_omega[:, None]
    return 1.0 / base


def spectral_function(green_wk: np.ndarray) -> np.ndarray:
    return -np.imag(green_wk) / np.pi


def fermi_dirac(omega_grid: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    beta = 1.0 / temperature
    x = np.clip(beta * omega_grid, -200.0, 200.0)
    return 1.0 / (np.exp(x) + 1.0)


def dominant_peak(omega_grid: np.ndarray, spectrum: np.ndarray) -> tuple[float, float]:
    max_val = float(np.max(spectrum))
    threshold = max(0.05 * max_val, 1.0e-10)
    peaks, info = find_peaks(spectrum, height=threshold)

    if peaks.size == 0:
        idx = int(np.argmax(spectrum))
        return float(omega_grid[idx]), float(spectrum[idx])

    best_local = int(np.argmax(info["peak_heights"]))
    best_idx = int(peaks[best_local])
    return float(omega_grid[best_idx]), float(spectrum[best_idx])


def run_green_function_mvp(cfg: GreenConfig) -> dict[str, object]:
    k_grid = build_k_grid(cfg)
    omega_grid = build_omega_grid(cfg)

    eps_k = dispersion(k_grid, cfg)
    sigma = retarded_self_energy(omega_grid, cfg)

    g_interacting = retarded_green(omega_grid, eps_k, cfg.eta, sigma)
    g_noninteracting = retarded_green(omega_grid, eps_k, cfg.eta, None)

    a_interacting = spectral_function(g_interacting)
    a_noninteracting = spectral_function(g_noninteracting)

    dos_interacting = np.mean(a_interacting, axis=1)
    dos_noninteracting = np.mean(a_noninteracting, axis=1)

    sum_rule_k = simpson(a_interacting, x=omega_grid, axis=0)
    sum_rule_error_k = np.abs(sum_rule_k - 1.0)

    fdist = fermi_dirac(omega_grid, cfg.temperature)
    occupation_k = simpson(a_interacting * fdist[:, None], x=omega_grid, axis=0)
    total_occupation = float(np.mean(occupation_k))

    dos_integral = float(simpson(dos_interacting, x=omega_grid))

    kf_target = 0.5 * np.pi
    kf_index = int(np.argmin(np.abs(k_grid - kf_target)))

    win_mask = np.abs(omega_grid) <= cfg.peak_window
    omega_win = omega_grid[win_mask]
    line_int = a_interacting[win_mask, kf_index]
    line_non = a_noninteracting[win_mask, kf_index]

    int_peak_energy, int_peak_height = dominant_peak(omega_win, line_int)
    non_peak_energy, non_peak_height = dominant_peak(omega_win, line_non)
    peak_shift = float(int_peak_energy - non_peak_energy)

    idx0 = int(np.argmin(np.abs(omega_grid)))
    if idx0 == 0 or idx0 == omega_grid.size - 1:
        raise RuntimeError("omega grid must include interior point near zero")

    re_sigma = np.real(sigma)
    d_re_sigma = (re_sigma[idx0 + 1] - re_sigma[idx0 - 1]) / (omega_grid[idx0 + 1] - omega_grid[idx0 - 1])
    z_factor = float(1.0 / (1.0 - d_re_sigma))

    im_sigma_fermi = float(np.imag(sigma[idx0]))
    scattering_rate = float(-2.0 * z_factor * im_sigma_fermi)

    report_indices = np.linspace(0, cfg.nk - 1, cfg.n_report_k, dtype=int)
    k_rows = []
    for idx in report_indices:
        k_rows.append(
            {
                "k_index": int(idx),
                "k_value": float(k_grid[idx]),
                "epsilon_k": float(eps_k[idx]),
                "occupation_nk": float(occupation_k[idx]),
                "sum_rule": float(sum_rule_k[idx]),
                "sum_rule_error": float(sum_rule_error_k[idx]),
            }
        )
    k_table = pd.DataFrame(k_rows)

    summary = {
        "fermi_k_index": kf_index,
        "fermi_k_value": float(k_grid[kf_index]),
        "peak_energy_interacting": int_peak_energy,
        "peak_energy_noninteracting": non_peak_energy,
        "peak_shift_interacting_minus_noninteracting": peak_shift,
        "peak_height_interacting": int_peak_height,
        "peak_height_noninteracting": non_peak_height,
        "quasiparticle_residue_Z": z_factor,
        "im_sigma_at_fermi": im_sigma_fermi,
        "fermi_scattering_rate": scattering_rate,
        "mean_sum_rule_error": float(np.mean(sum_rule_error_k)),
        "max_sum_rule_error": float(np.max(sum_rule_error_k)),
        "dos_integral": dos_integral,
        "total_occupation": total_occupation,
        "occupation_min": float(np.min(occupation_k)),
        "occupation_max": float(np.max(occupation_k)),
        "max_imag_green": float(np.max(np.imag(g_interacting))),
        "min_spectral_value": float(np.min(a_interacting)),
    }

    return {
        "cfg": cfg,
        "k_grid": k_grid,
        "omega_grid": omega_grid,
        "eps_k": eps_k,
        "sigma": sigma,
        "g_interacting": g_interacting,
        "g_noninteracting": g_noninteracting,
        "a_interacting": a_interacting,
        "a_noninteracting": a_noninteracting,
        "dos_interacting": dos_interacting,
        "dos_noninteracting": dos_noninteracting,
        "sum_rule_k": sum_rule_k,
        "occupation_k": occupation_k,
        "k_table": k_table,
        "summary": summary,
    }


def main() -> None:
    cfg = GreenConfig()
    result = run_green_function_mvp(cfg)
    summary = result["summary"]
    k_table = result["k_table"]

    checks = {
        "All computed arrays are finite": bool(
            np.isfinite(result["a_interacting"]).all()
            and np.isfinite(result["dos_interacting"]).all()
            and np.isfinite(result["occupation_k"]).all()
        ),
        "Spectral function is non-negative up to tolerance": summary["min_spectral_value"] > -1.0e-7,
        "Retarded causality Im G^R <= 0": summary["max_imag_green"] <= 1.0e-8,
        "Mean k-resolved sum-rule error < 5e-2": summary["mean_sum_rule_error"] < 5.0e-2,
        "Max k-resolved sum-rule error < 1.2e-1": summary["max_sum_rule_error"] < 1.2e-1,
        "DOS integral within [0.90, 1.10]": 0.90 < summary["dos_integral"] < 1.10,
        "Occupation stays in [0, 1]": -1.0e-6 <= summary["occupation_min"] and summary["occupation_max"] <= 1.0 + 1.0e-6,
        "Finite positive scattering rate at Fermi": summary["fermi_scattering_rate"] > 0.0,
        "Interacting peak shifts from non-interacting": abs(summary["peak_shift_interacting_minus_noninteracting"]) > 2.0e-2,
    }

    pd.set_option("display.float_format", lambda x: f"{x:.8f}")

    print("=== Green's Function Methods MVP (PHYS-0214) ===")
    print(
        f"nk={cfg.nk}, nw={cfg.nw}, k in [-{cfg.k_max:.4f}, {cfg.k_max:.4f}], "
        f"omega in [{cfg.omega_min:.2f}, {cfg.omega_max:.2f}]"
    )
    print(
        f"hopping={cfg.hopping:.4f}, mu={cfg.chemical_potential:.4f}, "
        f"U={cfg.interaction_u:.4f}, pole={cfg.self_energy_pole:.4f}, gamma={cfg.self_energy_damping:.4f}"
    )

    print("\nRepresentative k-point diagnostics:")
    print(k_table.to_string(index=False))

    summary_table = pd.DataFrame(
        {
            "quantity": [
                "k_F sample index",
                "k_F sample value",
                "interacting peak energy",
                "noninteracting peak energy",
                "peak shift (interacting - noninteracting)",
                "interacting peak height",
                "noninteracting peak height",
                "quasiparticle residue Z",
                "Im Sigma(omega=0)",
                "scattering rate at Fermi",
                "mean sum-rule error",
                "max sum-rule error",
                "DOS integral",
                "total occupation",
                "occupation min",
                "occupation max",
                "max Im G",
                "min A(k,omega)",
            ],
            "value": [
                float(summary["fermi_k_index"]),
                summary["fermi_k_value"],
                summary["peak_energy_interacting"],
                summary["peak_energy_noninteracting"],
                summary["peak_shift_interacting_minus_noninteracting"],
                summary["peak_height_interacting"],
                summary["peak_height_noninteracting"],
                summary["quasiparticle_residue_Z"],
                summary["im_sigma_at_fermi"],
                summary["fermi_scattering_rate"],
                summary["mean_sum_rule_error"],
                summary["max_sum_rule_error"],
                summary["dos_integral"],
                summary["total_occupation"],
                summary["occupation_min"],
                summary["occupation_max"],
                summary["max_imag_green"],
                summary["min_spectral_value"],
            ],
        }
    )

    print("\nGreen-function summary:")
    print(summary_table.to_string(index=False))

    print("\nThreshold checks:")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(checks.values())
    print(f"\nValidation: {'PASS' if all_passed else 'FAIL'}")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
