"""Cosmological Redshift MVP.

This script builds a compact, transparent pipeline around cosmological redshift:
1) Flat-LCDM background expansion E(z), H(z), and scale factor a(z).
2) Distance ladder from redshift: chi(z), d_L(z), d_A(z), lookback time.
3) Spectral-line redshift conversion lambda_obs = (1+z) * lambda_emit.
4) Low-z Hubble-law regression with scikit-learn.
5) Omega_m0 fitting from synthetic distance-modulus data with PyTorch.
6) Redshift inversion from luminosity distance with scipy.root_scalar.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root_scalar
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class FlatLambdaCDM:
    """Minimal flat-LCDM parameter set for the redshift demo."""

    h: float = 0.674
    omega_m0: float = 0.315
    omega_lambda0: float = 0.685
    c_km_s: float = 299792.458
    mpc_in_km: float = 3.085677581491367e19

    @property
    def h0_km_s_mpc(self) -> float:
        """Hubble constant H0 in km/s/Mpc."""

        return 100.0 * self.h


def e_of_z(z: np.ndarray | float, cosmo: FlatLambdaCDM) -> np.ndarray | float:
    """Dimensionless expansion rate E(z)=H(z)/H0 for flat LCDM."""

    z_arr = np.asarray(z)
    return np.sqrt(cosmo.omega_m0 * (1.0 + z_arr) ** 3 + cosmo.omega_lambda0)


def hubble_of_z(z: np.ndarray | float, cosmo: FlatLambdaCDM) -> np.ndarray | float:
    """Hubble expansion rate H(z) in km/s/Mpc."""

    return cosmo.h0_km_s_mpc * e_of_z(z, cosmo)


def scale_factor_from_redshift(z: np.ndarray | float) -> np.ndarray | float:
    """Scale factor a(z)=1/(1+z)."""

    z_arr = np.asarray(z)
    return 1.0 / (1.0 + z_arr)


def redshift_from_scale_factors(a_emit: float, a_obs: float = 1.0) -> float:
    """Redshift from scale-factor ratio: 1+z = a_obs / a_emit."""

    if a_emit <= 0.0 or a_obs <= 0.0:
        raise ValueError("Scale factors must be strictly positive")
    return a_obs / a_emit - 1.0


def _as_output_shape(values: np.ndarray, scalar_input: bool) -> np.ndarray | float:
    """Return scalar when input was scalar, otherwise ndarray."""

    if scalar_input:
        return float(values[0])
    return values


def comoving_distance_mpc(
    z: np.ndarray | float,
    cosmo: FlatLambdaCDM,
    n_grid: int = 4096,
) -> np.ndarray | float:
    """Comoving radial distance chi(z)=c/H0 * integral_0^z dz'/E(z')."""

    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    if np.any(z_arr < 0.0):
        raise ValueError("Redshift must be non-negative")

    z_max = float(np.max(z_arr))
    if z_max == 0.0:
        return _as_output_shape(np.zeros_like(z_arr), np.isscalar(z))

    z_grid = np.linspace(0.0, z_max, n_grid)
    inv_e = 1.0 / e_of_z(z_grid, cosmo)
    integral = cumulative_trapezoid(inv_e, z_grid, initial=0.0)
    chi_grid = (cosmo.c_km_s / cosmo.h0_km_s_mpc) * integral
    chi = np.interp(z_arr, z_grid, chi_grid)
    return _as_output_shape(chi, np.isscalar(z))


def lookback_time_gyr(
    z: np.ndarray | float,
    cosmo: FlatLambdaCDM,
    n_grid: int = 4096,
) -> np.ndarray | float:
    """Lookback time in Gyr: integral_0^z dz'/[(1+z')H(z')]."""

    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    if np.any(z_arr < 0.0):
        raise ValueError("Redshift must be non-negative")

    z_max = float(np.max(z_arr))
    if z_max == 0.0:
        return _as_output_shape(np.zeros_like(z_arr), np.isscalar(z))

    z_grid = np.linspace(0.0, z_max, n_grid)
    integrand = 1.0 / ((1.0 + z_grid) * e_of_z(z_grid, cosmo))
    integral = cumulative_trapezoid(integrand, z_grid, initial=0.0)

    # Convert 1/H0 from seconds to Gyr.
    h0_per_s = cosmo.h0_km_s_mpc / cosmo.mpc_in_km
    sec_per_gyr = 3.15576e16
    hubble_time_gyr = (1.0 / h0_per_s) / sec_per_gyr

    lookback_grid = hubble_time_gyr * integral
    lookback = np.interp(z_arr, z_grid, lookback_grid)
    return _as_output_shape(lookback, np.isscalar(z))


def luminosity_distance_mpc(z: np.ndarray | float, cosmo: FlatLambdaCDM) -> np.ndarray | float:
    """Luminosity distance d_L=(1+z)*chi."""

    scalar_input = np.isscalar(z)
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    chi = np.atleast_1d(np.asarray(comoving_distance_mpc(z_arr, cosmo), dtype=float))
    dl = (1.0 + z_arr) * chi
    return _as_output_shape(dl, scalar_input)


def angular_diameter_distance_mpc(z: np.ndarray | float, cosmo: FlatLambdaCDM) -> np.ndarray | float:
    """Angular diameter distance d_A=chi/(1+z)."""

    scalar_input = np.isscalar(z)
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    chi = np.atleast_1d(np.asarray(comoving_distance_mpc(z_arr, cosmo), dtype=float))
    da = chi / (1.0 + z_arr)
    return _as_output_shape(da, scalar_input)


def distance_modulus(z: np.ndarray | float, cosmo: FlatLambdaCDM) -> np.ndarray | float:
    """Distance modulus mu=5*log10(d_L/Mpc)+25 for z>0."""

    scalar_input = np.isscalar(z)
    dl = np.atleast_1d(np.asarray(luminosity_distance_mpc(z, cosmo), dtype=float))
    out = np.full_like(dl, np.nan, dtype=float)
    mask = dl > 0.0
    out[mask] = 5.0 * np.log10(dl[mask]) + 25.0
    return _as_output_shape(out, scalar_input)


def infer_redshift_from_luminosity_distance(dl_target_mpc: float, cosmo: FlatLambdaCDM) -> float:
    """Infer z by solving d_L(z)-d_L,target=0 with Brent root-finding."""

    if dl_target_mpc <= 0.0:
        raise ValueError("Target luminosity distance must be positive")

    def residual(z: float) -> float:
        return float(luminosity_distance_mpc(z, cosmo) - dl_target_mpc)

    z_hi = 10.0
    if residual(z_hi) < 0.0:
        raise RuntimeError("Target distance is outside the bracketing redshift range [0,10]")

    sol = root_scalar(residual, bracket=[1e-12, z_hi], method="brentq")
    if not sol.converged:
        raise RuntimeError("Redshift inversion did not converge")
    return float(sol.root)


def build_redshift_report(cosmo: FlatLambdaCDM) -> pd.DataFrame:
    """Create a compact report table for selected redshifts."""

    z = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0], dtype=float)
    a = scale_factor_from_redshift(z)
    chi = np.asarray(comoving_distance_mpc(z, cosmo), dtype=float)
    dl = np.asarray(luminosity_distance_mpc(z, cosmo), dtype=float)
    da = np.asarray(angular_diameter_distance_mpc(z, cosmo), dtype=float)
    t_lb = np.asarray(lookback_time_gyr(z, cosmo), dtype=float)

    lambda_emit_nm = 121.567  # Lyman-alpha
    lambda_obs_nm = lambda_emit_nm * (1.0 + z)
    z_from_line = lambda_obs_nm / lambda_emit_nm - 1.0

    return pd.DataFrame(
        {
            "z": z,
            "a(z)": a,
            "H(z) [km/s/Mpc]": hubble_of_z(z, cosmo),
            "chi(z) [Mpc]": chi,
            "d_L(z) [Mpc]": dl,
            "d_A(z) [Mpc]": da,
            "lookback [Gyr]": t_lb,
            "lambda_emit [nm]": np.full_like(z, lambda_emit_nm),
            "lambda_obs [nm]": lambda_obs_nm,
            "z_from_line": z_from_line,
        }
    )


def low_z_hubble_fit(cosmo: FlatLambdaCDM, rng: np.random.Generator) -> tuple[float, float, pd.DataFrame]:
    """Estimate H0 from low-z synthetic data using v~H0*d_L and LinearRegression."""

    z = np.linspace(0.005, 0.08, 40)
    d_l = np.asarray(luminosity_distance_mpc(z, cosmo), dtype=float)
    v_true = cosmo.c_km_s * z
    v_obs = v_true + rng.normal(0.0, 120.0, size=z.size)

    reg = LinearRegression().fit(d_l.reshape(-1, 1), v_obs)
    h0_est = float(reg.coef_[0])
    r2 = float(reg.score(d_l.reshape(-1, 1), v_obs))

    sample = pd.DataFrame(
        {
            "z": z,
            "d_L [Mpc]": d_l,
            "v_obs [km/s]": v_obs,
            "v_fit [km/s]": reg.predict(d_l.reshape(-1, 1)),
        }
    ).iloc[[0, 8, 16, 24, 32, 39]]

    return h0_est, r2, sample


def torch_distance_modulus_model(
    z: torch.Tensor,
    omega_m0: torch.Tensor,
    h: float,
    n_int: int = 300,
) -> torch.Tensor:
    """Differentiable mu(z; Omega_m0,h) for flat LCDM via torch.trapz."""

    u = torch.linspace(0.0, 1.0, n_int, dtype=z.dtype, device=z.device)
    z_eval = z.unsqueeze(1) * u.unsqueeze(0)
    omega_l0 = 1.0 - omega_m0

    e = torch.sqrt(omega_m0 * (1.0 + z_eval) ** 3 + omega_l0)
    integral = z * torch.trapz(1.0 / e, u, dim=1)

    c_over_h0 = 299792.458 / (100.0 * h)
    chi = c_over_h0 * integral
    d_l = (1.0 + z) * chi
    return 5.0 * torch.log10(d_l) + 25.0


def fit_omega_m0_with_torch(
    cosmo: FlatLambdaCDM,
    rng: np.random.Generator,
) -> tuple[float, float, pd.DataFrame]:
    """Fit Omega_m0 from synthetic SN-like distance-modulus data."""

    z = np.linspace(0.01, 1.2, 90)
    mu_true = np.asarray(distance_modulus(z, cosmo), dtype=float)
    mu_obs = mu_true + rng.normal(0.0, 0.12, size=z.size)

    z_t = torch.tensor(z, dtype=torch.float64)
    mu_obs_t = torch.tensor(mu_obs, dtype=torch.float64)

    raw = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.08)

    for _ in range(700):
        optimizer.zero_grad()
        omega_m0 = 0.05 + 0.55 * torch.sigmoid(raw)
        mu_pred = torch_distance_modulus_model(z_t, omega_m0, cosmo.h)
        loss = torch.mean((mu_pred - mu_obs_t) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        omega_fit = float((0.05 + 0.55 * torch.sigmoid(raw)).item())
        mu_fit = torch_distance_modulus_model(z_t, torch.tensor(omega_fit, dtype=torch.float64), cosmo.h)
        rmse = float(torch.sqrt(torch.mean((mu_fit - mu_obs_t) ** 2)).item())

    preview = pd.DataFrame(
        {
            "z": z,
            "mu_obs": mu_obs,
            "mu_pred_fit": mu_fit.detach().cpu().numpy(),
            "residual": mu_obs - mu_fit.detach().cpu().numpy(),
        }
    ).iloc[[0, 15, 30, 45, 60, 75, 89]]

    return omega_fit, rmse, preview


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    cosmo = FlatLambdaCDM()
    if not np.isclose(cosmo.omega_m0 + cosmo.omega_lambda0, 1.0, atol=1e-8):
        raise ValueError("This MVP assumes flat LCDM with omega_m0 + omega_lambda0 = 1")

    redshift_df = build_redshift_report(cosmo)

    rng = np.random.default_rng(42)
    h0_est, h0_r2, hubble_sample = low_z_hubble_fit(cosmo, rng)

    omega_fit, sn_rmse, sn_preview = fit_omega_m0_with_torch(cosmo, rng)

    z_ref = 1.0
    dl_ref = float(luminosity_distance_mpc(z_ref, cosmo))
    z_inverted = infer_redshift_from_luminosity_distance(dl_ref, cosmo)

    a_emit = 1.0 / 3.0
    z_from_scale = redshift_from_scale_factors(a_emit=a_emit, a_obs=1.0)

    print("=== Cosmological Redshift MVP (PHYS-0344) ===")
    print(
        f"h={cosmo.h:.3f}, Omega_m0={cosmo.omega_m0:.3f}, "
        f"Omega_Lambda0={cosmo.omega_lambda0:.3f}, H0={cosmo.h0_km_s_mpc:.3f} km/s/Mpc"
    )
    print()

    print("[Redshift-Distance-Time Report]")
    print(redshift_df.to_string(index=False, float_format=lambda x: f"{x: .6e}"))
    print()

    print("[Low-z Hubble Regression (scikit-learn)]")
    print(f"Estimated H0 from synthetic low-z sample: {h0_est:.3f} km/s/Mpc")
    print(f"Regression R^2: {h0_r2:.6f}")
    print(hubble_sample.to_string(index=False, float_format=lambda x: f"{x: .6e}"))
    print()

    print("[Omega_m0 Fit from Synthetic mu(z) (PyTorch)]")
    print(f"True Omega_m0: {cosmo.omega_m0:.4f}")
    print(f"Fitted Omega_m0: {omega_fit:.4f}")
    print(f"Distance-modulus RMSE: {sn_rmse:.4f} mag")
    print(sn_preview.to_string(index=False, float_format=lambda x: f"{x: .6e}"))
    print()

    print("[Inverse Checks]")
    print(f"Scale-factor check: a_emit={a_emit:.6f} -> z={z_from_scale:.6f}")
    print(f"Distance inversion check: z_ref={z_ref:.6f} -> d_L={dl_ref:.3f} Mpc -> z_inv={z_inverted:.6f}")


if __name__ == "__main__":
    main()
