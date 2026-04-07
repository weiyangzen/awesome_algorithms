"""Minimal runnable MVP for Fresnel diffraction (PHYS-0092).

Model: one-dimensional single-slit Fresnel diffraction.
We compare two implementations at finite propagation distance z:
1) Direct numerical Fresnel integral on a discrete aperture grid.
2) Analytical single-slit solution expressed with Fresnel integrals C/S.

The script reports quantitative agreement metrics and exits non-zero on failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import fresnel


@dataclass
class FresnelConfig:
    wavelength: float = 532e-9  # meter
    slit_width: float = 120e-6  # meter
    propagation_distance: float = 0.03  # meter
    aperture_half_span: float = 0.6e-3  # meter
    observation_half_span: float = 0.8e-3  # meter
    num_aperture_samples: int = 1400
    num_observation_samples: int = 900
    compare_x_limit: float = 0.35e-3  # meter, central window for metric
    rmse_threshold: float = 0.025
    corr_threshold: float = 0.995
    max_abs_error_threshold: float = 0.08


@dataclass
class ValidationReport:
    rmse: float
    correlation: float
    max_abs_error_central: float


@dataclass
class FresnelResult:
    x_aperture: np.ndarray
    x_observation: np.ndarray
    aperture: np.ndarray
    field_numeric: np.ndarray
    field_theory: np.ndarray
    intensity_numeric: np.ndarray
    intensity_theory: np.ndarray


def build_grids(cfg: FresnelConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build aperture and observation grids, and rectangular slit transmission."""
    x_ap = np.linspace(
        -cfg.aperture_half_span,
        cfg.aperture_half_span,
        cfg.num_aperture_samples,
        endpoint=False,
    )
    x_obs = np.linspace(
        -cfg.observation_half_span,
        cfg.observation_half_span,
        cfg.num_observation_samples,
        endpoint=False,
    )
    aperture = (np.abs(x_ap) <= 0.5 * cfg.slit_width).astype(float)

    if x_ap.size < 2:
        raise RuntimeError("Aperture grid must have at least two points.")
    dx = float(x_ap[1] - x_ap[0])
    return x_ap, x_obs, aperture, dx


def compute_numeric_fresnel_field(
    cfg: FresnelConfig,
    x_ap: np.ndarray,
    x_obs: np.ndarray,
    aperture: np.ndarray,
    dx: float,
) -> np.ndarray:
    """Directly evaluate the Fresnel diffraction integral on a discrete grid.

    U(x') ~ integral A(x) * exp(i*pi/(lambda*z) * (x' - x)^2) dx
    A global constant factor is omitted since we only compare normalized intensity.
    """
    phase_scale = np.pi / (cfg.wavelength * cfg.propagation_distance)
    kernel = np.exp(1j * phase_scale * (x_obs[:, None] - x_ap[None, :]) ** 2)
    return (kernel @ aperture) * dx


def compute_theory_field_single_slit(cfg: FresnelConfig, x_obs: np.ndarray) -> np.ndarray:
    """Analytical single-slit Fresnel field via C/S Fresnel integrals.

    For slit limits [-a/2, a/2], define
      u1 = sqrt(2/(lambda*z)) * (-a/2 - x')
      u2 = sqrt(2/(lambda*z)) * ( a/2 - x')
    Then field shape is proportional to
      (C(u2)-C(u1)) + i*(S(u2)-S(u1)).
    """
    scale = np.sqrt(2.0 / (cfg.wavelength * cfg.propagation_distance))
    u1 = scale * (-0.5 * cfg.slit_width - x_obs)
    u2 = scale * (0.5 * cfg.slit_width - x_obs)

    s1, c1 = fresnel(u1)
    s2, c2 = fresnel(u2)

    return (c2 - c1) + 1j * (s2 - s1)


def normalize_intensity(field: np.ndarray) -> np.ndarray:
    intensity = np.abs(field) ** 2
    peak = float(np.max(intensity))
    if peak <= 0.0:
        raise RuntimeError("Peak intensity is non-positive; normalization failed.")
    return intensity / peak


def run_simulation(cfg: FresnelConfig) -> FresnelResult:
    x_ap, x_obs, aperture, dx = build_grids(cfg)

    field_num = compute_numeric_fresnel_field(
        cfg=cfg,
        x_ap=x_ap,
        x_obs=x_obs,
        aperture=aperture,
        dx=dx,
    )
    field_theory = compute_theory_field_single_slit(cfg=cfg, x_obs=x_obs)

    intensity_num = normalize_intensity(field_num)
    intensity_theory = normalize_intensity(field_theory)

    return FresnelResult(
        x_aperture=x_ap,
        x_observation=x_obs,
        aperture=aperture,
        field_numeric=field_num,
        field_theory=field_theory,
        intensity_numeric=intensity_num,
        intensity_theory=intensity_theory,
    )


def build_validation_report(
    cfg: FresnelConfig,
    x_obs: np.ndarray,
    intensity_num: np.ndarray,
    intensity_theory: np.ndarray,
) -> ValidationReport:
    mask = np.abs(x_obs) <= cfg.compare_x_limit
    if np.count_nonzero(mask) < 80:
        raise RuntimeError("Not enough central samples for validation.")

    diff = intensity_num[mask] - intensity_theory[mask]
    rmse = float(np.sqrt(np.mean(diff * diff)))

    corr_matrix = np.corrcoef(intensity_num[mask], intensity_theory[mask])
    corr = float(corr_matrix[0, 1])
    max_abs_error = float(np.max(np.abs(diff)))

    return ValidationReport(
        rmse=rmse,
        correlation=corr,
        max_abs_error_central=max_abs_error,
    )


def build_profile_table(
    x_obs: np.ndarray,
    intensity_num: np.ndarray,
    intensity_theory: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x_mm": x_obs * 1e3,
            "intensity_numeric": intensity_num,
            "intensity_theory": intensity_theory,
            "abs_error": np.abs(intensity_num - intensity_theory),
        }
    )


def preview_rows(df: pd.DataFrame, target_x_mm: float, n: int = 4) -> pd.DataFrame:
    arr = df["x_mm"].to_numpy()
    idx = np.argsort(np.abs(arr - target_x_mm))[:n]
    return df.iloc[np.sort(idx)]


def print_profile_preview(df: pd.DataFrame) -> None:
    left = preview_rows(df, target_x_mm=-0.30, n=4)
    center = preview_rows(df, target_x_mm=0.0, n=5)
    right = preview_rows(df, target_x_mm=0.30, n=4)

    preview = pd.concat([left, center, right], axis=0)
    preview = preview.drop_duplicates().sort_values("x_mm")
    print(preview.to_string(index=False))


def main() -> None:
    cfg = FresnelConfig()
    result = run_simulation(cfg)

    report = build_validation_report(
        cfg=cfg,
        x_obs=result.x_observation,
        intensity_num=result.intensity_numeric,
        intensity_theory=result.intensity_theory,
    )

    df = build_profile_table(
        x_obs=result.x_observation,
        intensity_num=result.intensity_numeric,
        intensity_theory=result.intensity_theory,
    )

    fresnel_number = (cfg.slit_width**2) / (cfg.wavelength * cfg.propagation_distance)

    checks = {
        "central RMSE < 0.025": report.rmse < cfg.rmse_threshold,
        "central correlation > 0.995": report.correlation > cfg.corr_threshold,
        "central max abs error < 0.08": report.max_abs_error_central < cfg.max_abs_error_threshold,
    }

    print("=== Fresnel Diffraction MVP (PHYS-0092) ===")
    print("Model: 1D single slit, direct Fresnel integral vs Fresnel C/S analytical form")

    print("\n[Parameters]")
    print(
        "wavelength = {lam_nm:.1f} nm, slit_width = {a_um:.1f} um, z = {z_cm:.2f} cm".format(
            lam_nm=cfg.wavelength * 1e9,
            a_um=cfg.slit_width * 1e6,
            z_cm=cfg.propagation_distance * 1e2,
        )
    )
    print(
        "aperture_samples = {na}, observation_samples = {no}, aperture_span = +/-{xa:.2f} mm, obs_span = +/-{xo:.2f} mm".format(
            na=cfg.num_aperture_samples,
            no=cfg.num_observation_samples,
            xa=cfg.aperture_half_span * 1e3,
            xo=cfg.observation_half_span * 1e3,
        )
    )
    print(f"Fresnel number N_F = a^2/(lambda*z) = {fresnel_number:.4f}")

    print("\n[Validation metrics in central window]")
    print(f"RMSE = {report.rmse:.4e}")
    print(f"Correlation = {report.correlation:.6f}")
    print(f"Max abs error = {report.max_abs_error_central:.4e}")

    print("\n[Profile preview around x = -0.30, 0, +0.30 mm]")
    print_profile_preview(df)

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
        return

    print("\nValidation: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
