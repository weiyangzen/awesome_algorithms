"""Structure Formation MVP.

This demo implements a compact, transparent cosmology pipeline:
1) Solve linear growth D(a) in flat LCDM with an ODE.
2) Build a toy linear matter power spectrum P(k, z) = D(z)^2 * P0(k).
3) Compute Press-Schechter halo mass function dn/dlnM at z=0.
4) Use sklearn to estimate an effective spectral slope.
5) Use PyTorch to fit growth index gamma in f(a) ~= Omega_m(a)^gamma.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.integrate import simpson, solve_ivp
from scipy.optimize import root_scalar
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class FlatLambdaCDM:
    """Minimal flat-LCDM parameter set for this MVP."""

    h: float = 0.67
    omega_m0: float = 0.315
    omega_lambda0: float = 0.685
    n_s: float = 0.965
    k_damp: float = 0.35  # h / Mpc, toy high-k damping scale
    delta_c: float = 1.686
    sigma8_target: float = 0.81


def e_of_a(a: np.ndarray | float, cosmo: FlatLambdaCDM) -> np.ndarray | float:
    """Dimensionless expansion rate E(a)=H(a)/H0."""

    return np.sqrt(cosmo.omega_m0 * np.asarray(a) ** -3 + cosmo.omega_lambda0)


def omega_m_of_a(a: np.ndarray | float, cosmo: FlatLambdaCDM) -> np.ndarray | float:
    """Matter density fraction as a function of scale factor."""

    a_arr = np.asarray(a)
    return cosmo.omega_m0 * a_arr**-3 / e_of_a(a_arr, cosmo) ** 2


def growth_ode(lna: float, y: np.ndarray, cosmo: FlatLambdaCDM) -> np.ndarray:
    """Linear growth ODE in ln(a): D'' + [2 + dlnH/dlna]D' -1.5 Om(a)D = 0."""

    a = np.exp(lna)
    d, d_prime = y
    omega_m_a = float(omega_m_of_a(a, cosmo))
    dlnh_dlna = -1.5 * omega_m_a  # flat LCDM (matter + Lambda)
    d2 = -(2.0 + dlnh_dlna) * d_prime + 1.5 * omega_m_a * d
    return np.array([d_prime, d2], dtype=float)


def solve_growth_history(
    cosmo: FlatLambdaCDM,
    a_min: float = 1e-3,
    n_grid: int = 500,
) -> pd.DataFrame:
    """Solve and return growth history table with columns a,z,D,f."""

    lna_grid = np.linspace(np.log(a_min), 0.0, n_grid)
    y0 = np.array([a_min, a_min], dtype=float)  # matter era: D(a) ~ a

    sol = solve_ivp(
        fun=lambda x, y: growth_ode(x, y, cosmo),
        t_span=(lna_grid[0], lna_grid[-1]),
        y0=y0,
        t_eval=lna_grid,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(f"Growth ODE failed: {sol.message}")

    a = np.exp(sol.t)
    d_raw = sol.y[0]
    d_prime = sol.y[1]
    d = d_raw / d_raw[-1]
    f = d_prime / d_raw  # dlnD/dlna; invariant to global D normalization
    z = 1.0 / a - 1.0

    return pd.DataFrame({"a": a, "z": z, "D": d, "f": f})


def toy_primordial_power(k: np.ndarray, cosmo: FlatLambdaCDM) -> np.ndarray:
    """Toy linear spectrum shape (amplitude-free): k^ns * exp[-(k/kd)^2]."""

    return k**cosmo.n_s * np.exp(-(k / cosmo.k_damp) ** 2)


def top_hat_window(x: np.ndarray) -> np.ndarray:
    """Real-space top-hat window W(x)=3(sin x - x cos x)/x^3 with x->0 limit."""

    x = np.asarray(x)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    out[small] = 1.0 - x[small] ** 2 / 10.0
    xs = x[~small]
    out[~small] = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / (xs**3)
    return out


def sigma_r(k: np.ndarray, pk: np.ndarray, radius: float) -> float:
    """Variance sigma(R) from linear P(k) with a top-hat filter."""

    w = top_hat_window(k * radius)
    integrand = (k**2) * pk * (w**2) / (2.0 * np.pi**2)
    sigma2 = float(simpson(integrand, x=k))
    return float(np.sqrt(max(sigma2, 0.0)))


def linear_power_at_z(k: np.ndarray, p0: np.ndarray, growth: pd.DataFrame, z: float) -> np.ndarray:
    """Linear evolution P(k,z)=D(z)^2 * P0(k)."""

    z_grid = growth["z"].to_numpy()
    d_grid = growth["D"].to_numpy()
    d_z = float(np.interp(z, z_grid[::-1], d_grid[::-1]))
    return p0 * d_z**2


def press_schechter_mass_function(
    k: np.ndarray,
    pz: np.ndarray,
    cosmo: FlatLambdaCDM,
    masses: np.ndarray,
) -> pd.DataFrame:
    """Compute Press-Schechter dn/dlnM on a mass grid."""

    rho_crit0 = 2.775e11 * cosmo.h**2  # Msun / Mpc^3
    rho_m0 = cosmo.omega_m0 * rho_crit0

    radii = (3.0 * masses / (4.0 * np.pi * rho_m0)) ** (1.0 / 3.0)
    sigmas = np.array([sigma_r(k, pz, r) for r in radii], dtype=float)
    sigmas = np.clip(sigmas, 1e-10, None)

    ln_sigma = np.log(sigmas)
    ln_m = np.log(masses)
    dlnsigma_dlnm = np.gradient(ln_sigma, ln_m)

    nu = cosmo.delta_c / sigmas
    pref = np.sqrt(2.0 / np.pi) * (rho_m0 / masses) * nu
    dndlnm = pref * np.abs(dlnsigma_dlnm) * np.exp(-0.5 * nu**2)

    return pd.DataFrame(
        {
            "M_msun": masses,
            "R_mpc": radii,
            "sigma_M": sigmas,
            "nu": nu,
            "dn_dlnM": dndlnm,
        }
    )


def fit_growth_index_torch(growth: pd.DataFrame, cosmo: FlatLambdaCDM) -> tuple[float, float]:
    """Fit gamma in f(a) ~ Omega_m(a)^gamma via PyTorch autograd."""

    a = growth["a"].to_numpy()
    f_true = growth["f"].to_numpy()
    omega_m = omega_m_of_a(a, cosmo)

    omega_t = torch.tensor(omega_m, dtype=torch.float32)
    f_t = torch.tensor(f_true, dtype=torch.float32)

    gamma = torch.tensor(0.55, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([gamma], lr=0.05)

    for _ in range(400):
        optimizer.zero_grad()
        pred = omega_t**gamma
        loss = torch.mean((pred - f_t) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        rmse = float(torch.sqrt(torch.mean((omega_t**gamma - f_t) ** 2)).item())

    return float(gamma.item()), rmse


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    cosmo = FlatLambdaCDM()

    # 1) Solve growth history D(a)
    growth = solve_growth_history(cosmo)

    # 2) Build k-grid and amplitude-free P0(k) shape
    k = np.logspace(-3, 1, 1024)  # h / Mpc
    p_shape = toy_primordial_power(k, cosmo)

    # 3) Calibrate amplitude so sigma8(z=0) matches target
    r8 = 8.0 / cosmo.h  # Mpc (for k in h/Mpc)

    def sigma8_minus_target(amplitude: float) -> float:
        return sigma_r(k, amplitude * p_shape, r8) - cosmo.sigma8_target

    amp_solution = root_scalar(
        sigma8_minus_target,
        bracket=[1e-6, 1e6],
        method="brentq",
    )
    if not amp_solution.converged:
        raise RuntimeError("Amplitude calibration for sigma8 did not converge")

    amplitude = float(amp_solution.root)
    p0 = amplitude * p_shape

    # 4) Evaluate linear P(k,z) at sample redshifts
    z_samples = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    d_samples = np.array(
        [float(np.interp(z, growth["z"].to_numpy()[::-1], growth["D"].to_numpy()[::-1])) for z in z_samples]
    )
    sigma8_samples = d_samples * cosmo.sigma8_target
    p_k01_samples = [
        float(np.interp(0.1, k, linear_power_at_z(k, p0, growth, float(z)))) for z in z_samples
    ]
    summary_df = pd.DataFrame(
        {
            "z": z_samples,
            "D(z)": d_samples,
            "sigma8(z)": sigma8_samples,
            "P(k=0.1,z)": p_k01_samples,
        }
    )

    # 5) Effective slope estimation around quasi-linear scales using sklearn
    mask = (k >= 0.02) & (k <= 0.2)
    reg = LinearRegression().fit(np.log10(k[mask]).reshape(-1, 1), np.log10(p0[mask]))
    n_eff = float(reg.coef_[0])

    # 6) Fit growth index gamma with PyTorch
    gamma_fit, gamma_rmse = fit_growth_index_torch(growth, cosmo)

    # 7) Press-Schechter mass function at z=0
    masses = np.logspace(11, 15, 60)  # Msun
    pz0 = linear_power_at_z(k, p0, growth, z=0.0)
    mf_df = press_schechter_mass_function(k, pz0, cosmo, masses)
    mf_preview = mf_df.iloc[[0, 12, 24, 36, 48, 59]].copy()

    # Console report
    print("=== Structure Formation MVP (PHYS-0356) ===")
    print(
        f"h={cosmo.h:.3f}, Omega_m0={cosmo.omega_m0:.3f}, "
        f"Omega_Lambda0={cosmo.omega_lambda0:.3f}"
    )
    print(f"Calibrated toy amplitude A={amplitude:.6e} (sigma8 target={cosmo.sigma8_target:.3f})")
    print()

    print("[Growth & Linear Power Summary]")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x: .6e}"))
    print()

    print("[Diagnostics]")
    print(f"Effective spectral slope n_eff (0.02<k<0.2 h/Mpc): {n_eff:.5f}")
    print(f"Fitted growth index gamma via PyTorch: {gamma_fit:.5f}, RMSE={gamma_rmse:.3e}")
    print()

    print("[Press-Schechter Mass Function Sample at z=0]")
    print(mf_preview.to_string(index=False, float_format=lambda x: f"{x: .6e}"))


if __name__ == "__main__":
    main()
