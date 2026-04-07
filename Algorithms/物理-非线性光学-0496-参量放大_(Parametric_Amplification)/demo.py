"""Minimal runnable MVP for parametric amplification (OPA, undepleted pump)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd
from scipy.constants import c


@dataclass(frozen=True)
class OPAConfig:
    """Configuration for a 1D chi(2) optical parametric amplifier MVP."""

    wavelength_p_m: float = 532e-9
    wavelength_s_m: float = 1064e-9
    crystal_length_m: float = 0.02
    pump_intensity_w_m2: float = 5.0e11
    reference_intensity_w_m2: float = 1.0e12
    kappa_ref_per_m: float = 120.0
    delta_k_rad_per_m: float = 0.0
    signal_seed_amp: complex = 1.0 + 0.0j
    idler_seed_amp: complex = 0.0 + 0.0j
    z_steps: int = 4000


def _validate_config(cfg: OPAConfig) -> None:
    if cfg.wavelength_p_m <= 0:
        raise ValueError("wavelength_p_m must be positive")
    if cfg.wavelength_s_m <= 0:
        raise ValueError("wavelength_s_m must be positive")
    if cfg.crystal_length_m <= 0:
        raise ValueError("crystal_length_m must be positive")
    if cfg.pump_intensity_w_m2 <= 0:
        raise ValueError("pump_intensity_w_m2 must be positive")
    if cfg.reference_intensity_w_m2 <= 0:
        raise ValueError("reference_intensity_w_m2 must be positive")
    if cfg.kappa_ref_per_m <= 0:
        raise ValueError("kappa_ref_per_m must be positive")
    if cfg.z_steps < 200:
        raise ValueError("z_steps must be >= 200")
    if abs(cfg.signal_seed_amp) == 0.0:
        raise ValueError("signal_seed_amp magnitude must be > 0")


def _idler_wavelength_from_energy_conservation(
    wavelength_p_m: float,
    wavelength_s_m: float,
) -> float:
    freq_p = c / wavelength_p_m
    freq_s = c / wavelength_s_m
    freq_i = freq_p - freq_s
    if freq_i <= 0:
        raise ValueError("Invalid pump/signal frequencies: freq_i <= 0")
    return c / freq_i


def _coupling_from_pump(cfg: OPAConfig) -> float:
    return cfg.kappa_ref_per_m * math.sqrt(cfg.pump_intensity_w_m2 / cfg.reference_intensity_w_m2)


def _rhs(z_m: float, state: np.ndarray, kappa_per_m: float, delta_k_rad_per_m: float) -> np.ndarray:
    """Coupled-wave RHS for [A_s, conj(A_i)] under undepleted pump approximation."""

    a_s = state[0]
    a_i_conj = state[1]

    phase = np.exp(1j * delta_k_rad_per_m * z_m)
    d_a_s = 1j * kappa_per_m * a_i_conj * phase
    d_a_i_conj = -1j * kappa_per_m * a_s * np.conj(phase)

    return np.array([d_a_s, d_a_i_conj], dtype=np.complex128)


def _rk4_propagate(cfg: OPAConfig) -> dict[str, object]:
    _validate_config(cfg)

    kappa = _coupling_from_pump(cfg)
    z_m = np.linspace(0.0, cfg.crystal_length_m, cfg.z_steps + 1)
    dz = float(z_m[1] - z_m[0])

    state = np.zeros((cfg.z_steps + 1, 2), dtype=np.complex128)
    state[0, 0] = cfg.signal_seed_amp
    state[0, 1] = np.conj(cfg.idler_seed_amp)

    for idx in range(cfg.z_steps):
        z_now = z_m[idx]
        y_now = state[idx]

        k1 = _rhs(z_now, y_now, kappa, cfg.delta_k_rad_per_m)
        k2 = _rhs(z_now + 0.5 * dz, y_now + 0.5 * dz * k1, kappa, cfg.delta_k_rad_per_m)
        k3 = _rhs(z_now + 0.5 * dz, y_now + 0.5 * dz * k2, kappa, cfg.delta_k_rad_per_m)
        k4 = _rhs(z_now + dz, y_now + dz * k3, kappa, cfg.delta_k_rad_per_m)

        state[idx + 1] = y_now + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    a_s = state[:, 0]
    a_i = np.conj(state[:, 1])

    n_s = np.abs(a_s) ** 2
    n_i = np.abs(a_i) ** 2

    seed_signal = float(n_s[0])
    signal_gain_linear = float(n_s[-1] / seed_signal)
    idler_conversion_linear = float(n_i[-1] / seed_signal)

    invariant = n_s - n_i
    baseline = max(abs(float(invariant[0])), 1e-15)
    invariant_drift = np.abs(invariant - invariant[0]) / baseline

    return {
        "config": cfg,
        "z_m": z_m,
        "a_s": a_s,
        "a_i": a_i,
        "n_s": n_s,
        "n_i": n_i,
        "kappa_per_m": kappa,
        "signal_gain_linear": signal_gain_linear,
        "signal_gain_db": 10.0 * math.log10(signal_gain_linear),
        "idler_conversion_linear": idler_conversion_linear,
        "invariant_max_rel_drift": float(np.max(invariant_drift)),
    }


def _analytic_signal_gain(kappa_per_m: float, delta_k_rad_per_m: float, length_m: float) -> float:
    """Small-signal analytical gain for seeded signal with zero idler input."""

    g_sq = kappa_per_m * kappa_per_m - (0.5 * delta_k_rad_per_m) ** 2

    if g_sq > 0.0:
        g = math.sqrt(g_sq)
        return 1.0 + (kappa_per_m * kappa_per_m / (g * g)) * (math.sinh(g * length_m) ** 2)

    if g_sq < 0.0:
        q = math.sqrt(-g_sq)
        return 1.0 + (kappa_per_m * kappa_per_m / (q * q)) * (math.sin(q * length_m) ** 2)

    return 1.0 + (kappa_per_m * length_m) ** 2


def _pump_scan(base_cfg: OPAConfig, factors: list[float]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for factor in factors:
        cfg = replace(base_cfg, pump_intensity_w_m2=base_cfg.pump_intensity_w_m2 * factor)
        result = _rk4_propagate(cfg)
        rows.append(
            {
                "pump_GW_per_cm2": cfg.pump_intensity_w_m2 * 1e-13,
                "kappa_per_m": float(result["kappa_per_m"]),
                "signal_gain_linear": float(result["signal_gain_linear"]),
                "signal_gain_db": float(result["signal_gain_db"]),
            }
        )
    return pd.DataFrame(rows)


def _phase_mismatch_scan(base_cfg: OPAConfig, multipliers: list[float]) -> pd.DataFrame:
    kappa_ref = _coupling_from_pump(base_cfg)

    rows: list[dict[str, float]] = []
    for mul in multipliers:
        delta_k = mul * kappa_ref
        cfg = replace(base_cfg, delta_k_rad_per_m=delta_k)
        result = _rk4_propagate(cfg)
        analytic = _analytic_signal_gain(
            kappa_per_m=float(result["kappa_per_m"]),
            delta_k_rad_per_m=delta_k,
            length_m=cfg.crystal_length_m,
        )
        numeric = float(result["signal_gain_linear"])
        rel_err = abs(numeric - analytic) / max(analytic, 1e-15)

        rows.append(
            {
                "delta_k_per_m": delta_k,
                "signal_gain_linear": numeric,
                "analytic_gain": analytic,
                "rel_error": rel_err,
            }
        )

    return pd.DataFrame(rows)


def _build_z_samples(result: dict[str, object], count: int = 9) -> pd.DataFrame:
    z_m = result["z_m"]
    n_s = result["n_s"]
    n_i = result["n_i"]

    idx = np.linspace(0, len(z_m) - 1, count, dtype=int)
    return pd.DataFrame(
        {
            "z_mm": z_m[idx] * 1e3,
            "signal_power_norm": n_s[idx],
            "idler_power_norm": n_i[idx],
            "gain_vs_input": n_s[idx] / n_s[0],
        }
    )


def main() -> None:
    cfg = OPAConfig()

    wavelength_i_m = _idler_wavelength_from_energy_conservation(
        wavelength_p_m=cfg.wavelength_p_m,
        wavelength_s_m=cfg.wavelength_s_m,
    )

    freq_p = c / cfg.wavelength_p_m
    freq_s = c / cfg.wavelength_s_m
    freq_i = c / wavelength_i_m
    closure = abs(freq_p - (freq_s + freq_i)) / freq_p

    base = _rk4_propagate(cfg)

    analytic_base = _analytic_signal_gain(
        kappa_per_m=float(base["kappa_per_m"]),
        delta_k_rad_per_m=cfg.delta_k_rad_per_m,
        length_m=cfg.crystal_length_m,
    )
    rel_err_base = abs(float(base["signal_gain_linear"]) - analytic_base) / max(analytic_base, 1e-15)

    pump_scan = _pump_scan(cfg, factors=[0.5, 1.0, 1.5, 2.0])
    mismatch_scan = _phase_mismatch_scan(cfg, multipliers=[-2.5, -1.5, -0.75, 0.0, 0.75, 1.5, 2.5])
    z_samples = _build_z_samples(base)

    gains = pump_scan["signal_gain_linear"].to_numpy()
    pump_monotonic = bool(np.all(np.diff(gains) > 0.0))

    mismatch_gains = mismatch_scan["signal_gain_linear"].to_numpy()
    mismatch_delta = mismatch_scan["delta_k_per_m"].to_numpy()
    max_idx = int(np.argmax(mismatch_gains))
    peak_near_zero = abs(float(mismatch_delta[max_idx])) < 1e-12

    invariant_pass = float(base["invariant_max_rel_drift"]) < 5e-4
    analytic_pass = rel_err_base < 2e-3

    validation_pass = pump_monotonic and peak_near_zero and invariant_pass and analytic_pass and closure < 1e-15

    print("=== Parametric Amplification MVP (undepleted pump, chi(2)) ===")
    print(
        "Core metrics:\n"
        f"  lambda_p = {cfg.wavelength_p_m * 1e9:.2f} nm\n"
        f"  lambda_s = {cfg.wavelength_s_m * 1e9:.2f} nm\n"
        f"  lambda_i = {wavelength_i_m * 1e9:.2f} nm\n"
        f"  kappa = {float(base['kappa_per_m']):.6f} 1/m\n"
        f"  delta_k = {cfg.delta_k_rad_per_m:.6f} 1/m\n"
        f"  signal_gain_linear = {float(base['signal_gain_linear']):.6f}\n"
        f"  signal_gain_db = {float(base['signal_gain_db']):.6f} dB\n"
        f"  idler_conversion_linear = {float(base['idler_conversion_linear']):.6f}\n"
        f"  invariant_max_rel_drift = {float(base['invariant_max_rel_drift']):.3e}\n"
        f"  base_numeric_vs_analytic_rel_err = {rel_err_base:.3e}"
    )

    print("\nPump intensity scan:")
    print(pump_scan.to_string(index=False, justify="center", float_format=lambda x: f"{x:.6e}"))

    print("\nPhase mismatch scan:")
    print(mismatch_scan.to_string(index=False, justify="center", float_format=lambda x: f"{x:.6e}"))

    print("\nPropagation sample points:")
    print(z_samples.to_string(index=False, justify="center", float_format=lambda x: f"{x:.6e}"))

    print("\nValidation checks:")
    print(f"  energy_closure = {closure:.3e} (threshold < 1e-15)")
    print(f"  pump_gain_monotonic = {pump_monotonic}")
    print(f"  mismatch_peak_at_delta_k_0 = {peak_near_zero}")
    print(f"  invariant_drift_ok = {invariant_pass}")
    print(f"  analytic_agreement_ok = {analytic_pass}")
    print(f"Validation: {'PASS' if validation_pass else 'FAIL'}")

    if not validation_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
