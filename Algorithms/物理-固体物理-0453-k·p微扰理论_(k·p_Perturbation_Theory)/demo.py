"""Minimal runnable MVP for k·p perturbation theory (PHYS-0433).

The demo uses a 3-band toy crystal model as synthetic "reference data" and
builds a 2-band k·p effective Hamiltonian around Gamma (k=0):
1) Solve full 3x3 Hamiltonian bands E_full(k).
2) Derive 2x2 effective parameters with second-order perturbation (remote band elimination).
3) Compare k·p bands against full bands near Gamma.
4) Fit effective parameters with Torch to verify the perturbative initialization is sensible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


HBAR2_OVER_2M0_EV_A2 = 3.80998212  # eV * Angstrom^2


@dataclass(frozen=True)
class KPConfig:
    """Configuration for a minimal k·p perturbation MVP."""

    # Band-edge energies at Gamma (eV): Ev < Ec < Er.
    ev0: float = -0.20
    ec0: float = 1.40
    er0: float = 4.80

    # Bare diagonal k^2 coefficients (eV * Angstrom^2).
    av: float = -2.80
    ac: float = 1.20
    ar: float = 0.60

    # Linear momentum-coupling coefficients (eV * Angstrom).
    p_cv: float = 7.50
    p_cr: float = 2.40
    p_vr: float = 1.80

    # Sampling and fitting windows.
    k_max: float = 0.12
    n_k: int = 161
    fit_k_max: float = 0.06

    # Torch optimizer settings.
    torch_steps: int = 900
    torch_lr: float = 3e-2
    torch_seed: int = 7

    # Validation tolerances.
    rmse_near_gamma_max: float = 0.030
    hermitian_tol: float = 1e-12


def validate_config(cfg: KPConfig) -> None:
    if not (cfg.ev0 < cfg.ec0 < cfg.er0):
        raise ValueError("Require ev0 < ec0 < er0")
    if cfg.k_max <= 0.0:
        raise ValueError("k_max must be positive")
    if cfg.n_k < 31:
        raise ValueError("n_k is too small")
    if cfg.n_k % 2 == 0:
        raise ValueError("n_k must be odd so k=0 is included")
    if not (0.0 < cfg.fit_k_max <= cfg.k_max):
        raise ValueError("fit_k_max must satisfy 0 < fit_k_max <= k_max")
    if cfg.torch_steps < 1:
        raise ValueError("torch_steps must be >= 1")


def full_three_band_hamiltonian(k: float, cfg: KPConfig) -> np.ndarray:
    """3x3 toy Hamiltonian in basis (v, c, r)."""
    k2 = k * k
    h = np.array(
        [
            [cfg.ev0 + cfg.av * k2, cfg.p_cv * k, cfg.p_vr * k],
            [cfg.p_cv * k, cfg.ec0 + cfg.ac * k2, cfg.p_cr * k],
            [cfg.p_vr * k, cfg.p_cr * k, cfg.er0 + cfg.ar * k2],
        ],
        dtype=float,
    )
    return h


def solve_full_bands(k_grid: np.ndarray, cfg: KPConfig) -> tuple[np.ndarray, float]:
    """Return full model eigenvalues [n_k, 3] and max Hermitian residual."""
    eigvals = np.zeros((k_grid.size, 3), dtype=float)
    hermitian_residual = 0.0

    for i, k in enumerate(k_grid):
        h = full_three_band_hamiltonian(float(k), cfg)
        hermitian_residual = max(hermitian_residual, float(np.max(np.abs(h - h.T))))
        w, _ = eigh(h, overwrite_a=False, check_finite=True)
        eigvals[i] = w

    return eigvals, hermitian_residual


def perturbative_two_band_params(cfg: KPConfig) -> dict[str, float]:
    """Second-order elimination of remote band r -> effective 2-band parameters."""
    # Loewdin-like second-order diagonal corrections.
    delta_av = cfg.p_vr**2 / (cfg.ev0 - cfg.er0)
    delta_ac = cfg.p_cr**2 / (cfg.ec0 - cfg.er0)

    return {
        "ev0": cfg.ev0,
        "ec0": cfg.ec0,
        "av_eff": cfg.av + delta_av,
        "ac_eff": cfg.ac + delta_ac,
        "p_eff": cfg.p_cv,
        "delta_av_remote": delta_av,
        "delta_ac_remote": delta_ac,
    }


def two_band_hamiltonian(k: float, params: dict[str, float]) -> np.ndarray:
    k2 = k * k
    return np.array(
        [
            [params["ev0"] + params["av_eff"] * k2, params["p_eff"] * k],
            [params["p_eff"] * k, params["ec0"] + params["ac_eff"] * k2],
        ],
        dtype=float,
    )


def solve_two_band(k_grid: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Return [n_k, 2] = (valence, conduction) from 2x2 effective model."""
    out = np.zeros((k_grid.size, 2), dtype=float)
    for i, k in enumerate(k_grid):
        w, _ = eigh(two_band_hamiltonian(float(k), params), overwrite_a=False, check_finite=True)
        out[i] = w
    return out


def solve_diagonal_baseline(k_grid: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """No interband mixing baseline: keeps only diagonal k^2 terms."""
    k2 = k_grid * k_grid
    valence = params["ev0"] + params["av_eff"] * k2
    conduction = params["ec0"] + params["ac_eff"] * k2
    out = np.column_stack([valence, conduction])
    # keep ordering consistent (ev <= ec)
    out.sort(axis=1)
    return out


def fit_effective_masses(k_grid: np.ndarray, bands: np.ndarray, fit_mask: np.ndarray) -> pd.DataFrame:
    """Fit E(k)=E0+slope*k^2 near Gamma and convert slope to m*/m0."""
    x = (k_grid[fit_mask] ** 2).reshape(-1, 1)

    rows: list[dict[str, float | str]] = []
    for name, idx in (("valence", 0), ("conduction", 1)):
        y = bands[fit_mask, idx] - bands[np.argmin(np.abs(k_grid)), idx]
        reg = LinearRegression(fit_intercept=False)
        reg.fit(x, y)
        slope = float(reg.coef_[0])

        if name == "valence":
            mass_ratio = HBAR2_OVER_2M0_EV_A2 / max(abs(slope), 1e-12)
            curvature_sign = -1.0
        else:
            mass_ratio = HBAR2_OVER_2M0_EV_A2 / max(slope, 1e-12)
            curvature_sign = 1.0

        rows.append(
            {
                "band": name,
                "slope_eV_A2": slope,
                "curvature_sign_expected": curvature_sign,
                "effective_mass_over_m0": mass_ratio,
            }
        )

    return pd.DataFrame(rows)


def fit_two_band_with_torch(
    k_grid: np.ndarray,
    target_bands: np.ndarray,
    init_params: dict[str, float],
    fit_mask: np.ndarray,
    cfg: KPConfig,
) -> dict[str, float]:
    """Refine (av_eff, ac_eff, p_eff) via gradient-based least squares."""
    torch.manual_seed(cfg.torch_seed)

    dtype = torch.float64
    k = torch.tensor(k_grid[fit_mask], dtype=dtype)
    target = torch.tensor(target_bands[fit_mask], dtype=dtype)

    ev0 = torch.tensor(init_params["ev0"], dtype=dtype)
    ec0 = torch.tensor(init_params["ec0"], dtype=dtype)

    theta = torch.tensor(
        [init_params["av_eff"], init_params["ac_eff"], init_params["p_eff"]],
        dtype=dtype,
        requires_grad=True,
    )
    theta0 = theta.detach().clone()

    optimizer = torch.optim.Adam([theta], lr=cfg.torch_lr)

    for _ in range(cfg.torch_steps):
        optimizer.zero_grad()

        av_eff, ac_eff, p_eff = theta[0], theta[1], theta[2]
        k2 = k * k
        h11 = ev0 + av_eff * k2
        h22 = ec0 + ac_eff * k2
        h12 = p_eff * k

        # Closed-form eigenvalues for symmetric 2x2 matrix.
        tr = 0.5 * (h11 + h22)
        rad = torch.sqrt(((h11 - h22) * 0.5) ** 2 + h12 * h12 + 1e-18)
        e1 = tr - rad
        e2 = tr + rad
        pred = torch.stack([e1, e2], dim=1)

        loss_main = torch.mean((pred - target) ** 2)
        reg = 1e-4 * torch.sum((theta - theta0) ** 2)
        loss = loss_main + reg

        loss.backward()
        optimizer.step()

    av_eff, ac_eff, p_eff = theta.detach().cpu().numpy()
    return {
        "ev0": init_params["ev0"],
        "ec0": init_params["ec0"],
        "av_eff": float(av_eff),
        "ac_eff": float(ac_eff),
        "p_eff": float(p_eff),
    }


def rmse_for_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true[mask].reshape(-1), y_pred[mask].reshape(-1))))


def main() -> None:
    cfg = KPConfig()
    validate_config(cfg)

    k_grid = np.linspace(-cfg.k_max, cfg.k_max, cfg.n_k, dtype=float)
    fit_mask = np.abs(k_grid) <= cfg.fit_k_max

    full_all, hermitian_residual = solve_full_bands(k_grid, cfg)
    full_cv = full_all[:, :2]  # valence + conduction

    kp_params = perturbative_two_band_params(cfg)
    kp_bands = solve_two_band(k_grid, kp_params)
    diag_bands = solve_diagonal_baseline(k_grid, kp_params)

    torch_params = fit_two_band_with_torch(k_grid, full_cv, kp_params, fit_mask, cfg)
    torch_bands = solve_two_band(k_grid, torch_params)

    metrics = pd.DataFrame(
        {
            "model": ["diag_baseline", "kp_perturbative", "kp_torch_refined"],
            "rmse_near_gamma_eV": [
                rmse_for_mask(full_cv, diag_bands, fit_mask),
                rmse_for_mask(full_cv, kp_bands, fit_mask),
                rmse_for_mask(full_cv, torch_bands, fit_mask),
            ],
            "rmse_full_range_eV": [
                rmse_for_mask(full_cv, diag_bands, np.ones_like(fit_mask, dtype=bool)),
                rmse_for_mask(full_cv, kp_bands, np.ones_like(fit_mask, dtype=bool)),
                rmse_for_mask(full_cv, torch_bands, np.ones_like(fit_mask, dtype=bool)),
            ],
        }
    )

    mass_table = fit_effective_masses(k_grid, kp_bands, fit_mask)

    sample_idx = np.linspace(0, cfg.n_k - 1, 9, dtype=int)
    sample_table = pd.DataFrame(
        {
            "k(1/A)": k_grid[sample_idx],
            "E_v_full(eV)": full_cv[sample_idx, 0],
            "E_v_kp(eV)": kp_bands[sample_idx, 0],
            "E_v_torch(eV)": torch_bands[sample_idx, 0],
            "E_c_full(eV)": full_cv[sample_idx, 1],
            "E_c_kp(eV)": kp_bands[sample_idx, 1],
            "E_c_torch(eV)": torch_bands[sample_idx, 1],
        }
    )

    checks = {
        "Hamiltonian Hermitian residual <= tol": hermitian_residual <= cfg.hermitian_tol,
        "k·p near-Gamma RMSE <= threshold": (
            float(metrics.loc[metrics["model"] == "kp_perturbative", "rmse_near_gamma_eV"].iloc[0])
            <= cfg.rmse_near_gamma_max
        ),
        "k·p beats diagonal baseline near Gamma": (
            float(metrics.loc[metrics["model"] == "kp_perturbative", "rmse_near_gamma_eV"].iloc[0])
            < float(metrics.loc[metrics["model"] == "diag_baseline", "rmse_near_gamma_eV"].iloc[0])
        ),
        "Torch refinement not worse than perturbative (near Gamma)": (
            float(metrics.loc[metrics["model"] == "kp_torch_refined", "rmse_near_gamma_eV"].iloc[0])
            <= float(metrics.loc[metrics["model"] == "kp_perturbative", "rmse_near_gamma_eV"].iloc[0])
            + 1e-10
        ),
        "Conduction effective mass in (0.02, 3.0)": (
            0.02
            < float(mass_table.loc[mass_table["band"] == "conduction", "effective_mass_over_m0"].iloc[0])
            < 3.0
        ),
    }

    pd.set_option("display.float_format", lambda v: f"{v:.8f}")

    params_df = pd.DataFrame(
        [
            {
                "ev0": kp_params["ev0"],
                "ec0": kp_params["ec0"],
                "av_eff": kp_params["av_eff"],
                "ac_eff": kp_params["ac_eff"],
                "p_eff": kp_params["p_eff"],
                "delta_av_remote": kp_params["delta_av_remote"],
                "delta_ac_remote": kp_params["delta_ac_remote"],
            },
            {
                "ev0": torch_params["ev0"],
                "ec0": torch_params["ec0"],
                "av_eff": torch_params["av_eff"],
                "ac_eff": torch_params["ac_eff"],
                "p_eff": torch_params["p_eff"],
                "delta_av_remote": np.nan,
                "delta_ac_remote": np.nan,
            },
        ],
        index=["perturbative_init", "torch_refined"],
    )

    print("=== k·p Perturbation Theory MVP (PHYS-0433) ===")
    print(
        f"k-grid: [{-cfg.k_max:.3f}, {cfg.k_max:.3f}] 1/A with {cfg.n_k} points; "
        f"fit window |k| <= {cfg.fit_k_max:.3f} 1/A"
    )
    print(
        f"band edges (eV): Ev0={cfg.ev0:.3f}, Ec0={cfg.ec0:.3f}, Er0={cfg.er0:.3f}; "
        f"bare couplings: Pcv={cfg.p_cv:.3f}, Pcr={cfg.p_cr:.3f}, Pvr={cfg.p_vr:.3f}"
    )

    print("\nEffective-parameter table:")
    print(params_df.to_string())

    print("\nError metrics:")
    print(metrics.to_string(index=False))

    print("\nEffective masses from k·p bands (fit near Gamma):")
    print(mass_table.to_string(index=False))

    print("\nBand samples:")
    print(sample_table.to_string(index=False))

    print("\nChecks:")
    for name, ok in checks.items():
        print(f"- [{ 'PASS' if ok else 'FAIL' }] {name}")

    all_ok = all(checks.values())
    print(f"\nValidation: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
