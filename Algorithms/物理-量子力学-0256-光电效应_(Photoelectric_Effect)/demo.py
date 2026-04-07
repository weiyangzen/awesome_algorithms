"""Photoelectric Effect MVP.

This script builds a small, auditable pipeline for the Einstein photoelectric law:
    e * V_s = h * f - phi

Pipeline:
1) Generate synthetic stopping-potential measurements with noise.
2) Estimate h and phi from linear regression on emitting data.
3) Fit a constrained piecewise model on all points (including non-emission points).
4) Validate recovered parameters against ground truth and print PASS/FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# SI constants
PLANCK_J_S = 6.626_070_15e-34
E_CHARGE_C = 1.602_176_634e-19


@dataclass(frozen=True)
class PhotoelectricConfig:
    work_function_ev_true: float = 2.28
    freq_min_thz: float = 380.0
    freq_max_thz: float = 1300.0
    n_points: int = 48
    noise_std_v: float = 0.03
    emission_detect_v: float = 0.04
    random_seed: int = 20260407
    h_rel_tol: float = 0.05
    phi_rel_tol: float = 0.08
    f0_rel_tol: float = 0.08
    fit_rmse_tol_v: float = 0.07


@dataclass(frozen=True)
class FitResult:
    slope_v_per_hz: float
    intercept_v: float
    h_est_j_s: float
    phi_est_ev: float
    f0_est_hz: float
    rmse_v: float
    r2: float


@dataclass(frozen=True)
class PiecewiseResult:
    slope_v_per_hz: float
    f0_est_hz: float
    h_est_j_s: float
    phi_est_ev: float
    rmse_v: float


@dataclass(frozen=True)
class PhotoelectricResult:
    config: PhotoelectricConfig
    freq_hz: np.ndarray
    photon_energy_ev: np.ndarray
    v_stop_true: np.ndarray
    v_stop_obs: np.ndarray
    linear: FitResult
    piecewise: PiecewiseResult


def planck_ev_s() -> float:
    """Planck constant in eV*s."""
    return PLANCK_J_S / E_CHARGE_C


def photon_energy_ev(freq_hz: np.ndarray) -> np.ndarray:
    """Photon energy in eV from frequency in Hz."""
    return planck_ev_s() * freq_hz


def stopping_potential_ideal(freq_hz: np.ndarray, work_function_ev: float) -> np.ndarray:
    """Ideal stopping potential (V): V_s = max(0, h f / e - phi/e)."""
    return np.maximum(0.0, photon_energy_ev(freq_hz) - work_function_ev)


def simulate_dataset(cfg: PhotoelectricConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic experiment data with additive Gaussian voltage noise."""
    if cfg.n_points < 8:
        raise ValueError("n_points must be >= 8 for stable estimation.")
    if cfg.freq_min_thz <= 0.0 or cfg.freq_max_thz <= cfg.freq_min_thz:
        raise ValueError("frequency range is invalid.")

    freq_hz = np.linspace(cfg.freq_min_thz, cfg.freq_max_thz, cfg.n_points, dtype=float) * 1e12
    e_ev = photon_energy_ev(freq_hz)
    v_true = stopping_potential_ideal(freq_hz, cfg.work_function_ev_true)

    rng = np.random.default_rng(cfg.random_seed)
    v_obs = v_true + rng.normal(loc=0.0, scale=cfg.noise_std_v, size=v_true.shape)
    v_obs = np.clip(v_obs, 0.0, None)
    return freq_hz, e_ev, v_true, v_obs


def fit_linear_emission_region(
    freq_hz: np.ndarray,
    v_obs: np.ndarray,
    emission_detect_v: float,
) -> tuple[FitResult, np.ndarray]:
    """Fit V_s = slope * f + intercept on detected emitting points only."""
    emit_mask = v_obs > emission_detect_v
    if int(np.sum(emit_mask)) < 6:
        raise RuntimeError("Insufficient emitting points for linear fit.")

    x = freq_hz[emit_mask].reshape(-1, 1)
    y = v_obs[emit_mask]

    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    if slope <= 0.0:
        raise RuntimeError("Linear fit produced non-physical slope <= 0.")

    y_fit = model.predict(x)
    residual = y - y_fit
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)

    h_est = slope * E_CHARGE_C
    phi_est_ev = -intercept
    f0_est = -intercept / slope

    return (
        FitResult(
            slope_v_per_hz=slope,
            intercept_v=intercept,
            h_est_j_s=h_est,
            phi_est_ev=phi_est_ev,
            f0_est_hz=f0_est,
            rmse_v=rmse,
            r2=r2,
        ),
        emit_mask,
    )


def piecewise_photoelectric_model(freq_hz: np.ndarray, slope_v_per_hz: float, f0_hz: float) -> np.ndarray:
    """Physically constrained model: V_s = max(0, slope * (f - f0))."""
    return np.maximum(0.0, slope_v_per_hz * (freq_hz - f0_hz))


def fit_piecewise_all_points(
    freq_hz: np.ndarray,
    v_obs: np.ndarray,
    init_f0_hz: float,
) -> PiecewiseResult:
    """Fit constrained photoelectric threshold model on all points."""
    slope_guess = PLANCK_J_S / E_CHARGE_C
    p0 = np.array([slope_guess, init_f0_hz], dtype=float)

    f_min = float(np.min(freq_hz))
    f_max = float(np.max(freq_hz))
    lower = np.array([1e-18, 0.5 * f_min], dtype=float)
    upper = np.array([5e-14, 1.5 * f_max], dtype=float)

    params, _ = curve_fit(
        f=piecewise_photoelectric_model,
        xdata=freq_hz,
        ydata=v_obs,
        p0=p0,
        bounds=(lower, upper),
        maxfev=20_000,
    )

    slope, f0 = float(params[0]), float(params[1])
    pred = piecewise_photoelectric_model(freq_hz, slope, f0)
    rmse = float(np.sqrt(np.mean((pred - v_obs) ** 2)))

    h_est = slope * E_CHARGE_C
    phi_est_ev = slope * f0
    return PiecewiseResult(
        slope_v_per_hz=slope,
        f0_est_hz=f0,
        h_est_j_s=h_est,
        phi_est_ev=phi_est_ev,
        rmse_v=rmse,
    )


def run_mvp(cfg: PhotoelectricConfig) -> PhotoelectricResult:
    freq_hz, e_ev, v_true, v_obs = simulate_dataset(cfg)
    linear_fit, _emit_mask = fit_linear_emission_region(
        freq_hz=freq_hz,
        v_obs=v_obs,
        emission_detect_v=cfg.emission_detect_v,
    )
    piecewise_fit = fit_piecewise_all_points(
        freq_hz=freq_hz,
        v_obs=v_obs,
        init_f0_hz=linear_fit.f0_est_hz,
    )
    return PhotoelectricResult(
        config=cfg,
        freq_hz=freq_hz,
        photon_energy_ev=e_ev,
        v_stop_true=v_true,
        v_stop_obs=v_obs,
        linear=linear_fit,
        piecewise=piecewise_fit,
    )


def build_report_table(result: PhotoelectricResult) -> pd.DataFrame:
    linear_pred = np.maximum(
        0.0,
        result.linear.slope_v_per_hz * result.freq_hz + result.linear.intercept_v,
    )
    piecewise_pred = piecewise_photoelectric_model(
        result.freq_hz,
        result.piecewise.slope_v_per_hz,
        result.piecewise.f0_est_hz,
    )
    return pd.DataFrame(
        {
            "freq_thz": result.freq_hz / 1e12,
            "photon_energy_ev": result.photon_energy_ev,
            "v_stop_true_v": result.v_stop_true,
            "v_stop_obs_v": result.v_stop_obs,
            "v_stop_linear_pred_v": linear_pred,
            "v_stop_piecewise_pred_v": piecewise_pred,
        }
    )


def validate(result: PhotoelectricResult) -> tuple[bool, dict[str, float]]:
    cfg = result.config
    f0_true = cfg.work_function_ev_true / planck_ev_s()

    rel_h_linear = abs(result.linear.h_est_j_s - PLANCK_J_S) / PLANCK_J_S
    rel_h_piecewise = abs(result.piecewise.h_est_j_s - PLANCK_J_S) / PLANCK_J_S
    rel_phi_linear = abs(result.linear.phi_est_ev - cfg.work_function_ev_true) / cfg.work_function_ev_true
    rel_phi_piecewise = abs(result.piecewise.phi_est_ev - cfg.work_function_ev_true) / cfg.work_function_ev_true
    rel_f0_piecewise = abs(result.piecewise.f0_est_hz - f0_true) / f0_true

    pred_emit = piecewise_photoelectric_model(
        result.freq_hz,
        result.piecewise.slope_v_per_hz,
        result.piecewise.f0_est_hz,
    ) > cfg.emission_detect_v
    true_emit = result.v_stop_true > 0.0
    emit_acc = float(np.mean(pred_emit == true_emit))

    above_f0 = result.freq_hz >= result.piecewise.f0_est_hz
    monotonic_ok = True
    if np.sum(above_f0) >= 2:
        seq = piecewise_photoelectric_model(
            result.freq_hz[above_f0],
            result.piecewise.slope_v_per_hz,
            result.piecewise.f0_est_hz,
        )
        monotonic_ok = bool(np.all(np.diff(seq) >= -1e-12))

    finite_ok = bool(
        np.isfinite(result.freq_hz).all()
        and np.isfinite(result.v_stop_obs).all()
        and np.isfinite(result.v_stop_true).all()
        and np.isfinite(result.linear.h_est_j_s)
        and np.isfinite(result.piecewise.h_est_j_s)
    )

    passed = (
        finite_ok
        and monotonic_ok
        and rel_h_linear <= cfg.h_rel_tol
        and rel_h_piecewise <= cfg.h_rel_tol
        and rel_phi_piecewise <= cfg.phi_rel_tol
        and rel_f0_piecewise <= cfg.f0_rel_tol
        and result.piecewise.rmse_v <= cfg.fit_rmse_tol_v
        and emit_acc >= 0.90
    )

    metrics = {
        "rel_h_linear": float(rel_h_linear),
        "rel_h_piecewise": float(rel_h_piecewise),
        "rel_phi_linear": float(rel_phi_linear),
        "rel_phi_piecewise": float(rel_phi_piecewise),
        "rel_f0_piecewise": float(rel_f0_piecewise),
        "linear_rmse_v": float(result.linear.rmse_v),
        "piecewise_rmse_v": float(result.piecewise.rmse_v),
        "linear_r2_emit": float(result.linear.r2),
        "emission_class_acc": float(emit_acc),
        "finite_ok": float(finite_ok),
        "monotonic_ok": float(monotonic_ok),
        "f0_true_hz": float(f0_true),
    }
    return passed, metrics


def main() -> None:
    cfg = PhotoelectricConfig()
    result = run_mvp(cfg)
    table = build_report_table(result)
    passed, metrics = validate(result)

    print("=== Photoelectric Effect MVP ===")
    print(
        f"True work function phi = {cfg.work_function_ev_true:.4f} eV, "
        f"threshold f0(true) = {metrics['f0_true_hz'] / 1e12:.4f} THz"
    )
    print(
        f"Linear fit:    h = {result.linear.h_est_j_s:.6e} J*s, "
        f"phi = {result.linear.phi_est_ev:.6f} eV, "
        f"f0 = {result.linear.f0_est_hz / 1e12:.4f} THz"
    )
    print(
        f"Piecewise fit: h = {result.piecewise.h_est_j_s:.6e} J*s, "
        f"phi = {result.piecewise.phi_est_ev:.6f} eV, "
        f"f0 = {result.piecewise.f0_est_hz / 1e12:.4f} THz"
    )
    print()
    print("Validation metrics:")
    print(f"rel_h_linear        = {metrics['rel_h_linear']:.6e} (tol={cfg.h_rel_tol:.2f})")
    print(f"rel_h_piecewise     = {metrics['rel_h_piecewise']:.6e} (tol={cfg.h_rel_tol:.2f})")
    print(f"rel_phi_piecewise   = {metrics['rel_phi_piecewise']:.6e} (tol={cfg.phi_rel_tol:.2f})")
    print(f"rel_f0_piecewise    = {metrics['rel_f0_piecewise']:.6e} (tol={cfg.f0_rel_tol:.2f})")
    print(f"piecewise_rmse_v    = {metrics['piecewise_rmse_v']:.6e} (tol={cfg.fit_rmse_tol_v:.2f})")
    print(f"emission_class_acc  = {metrics['emission_class_acc']:.6f} (tol>=0.90)")
    print(f"linear_r2_emit      = {metrics['linear_r2_emit']:.6f}")
    print(f"finite_ok           = {bool(metrics['finite_ok'])}")
    print(f"monotonic_ok        = {bool(metrics['monotonic_ok'])}")
    print()

    sample_idx = np.linspace(0, len(table) - 1, 12, dtype=int)
    print("Sampled dataset rows:")
    print(table.iloc[sample_idx].to_string(index=False, float_format=lambda x: f'{x:.6e}'))
    print()
    print(f"Validation: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
