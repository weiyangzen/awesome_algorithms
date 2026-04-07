"""Minimal runnable MVP for Neutron Scattering in condensed matter.

This script builds a synthetic inelastic neutron scattering map I(Q, omega),
fits each Q-cut with SciPy, regresses dispersion with scikit-learn,
and performs a global consistency refinement with PyTorch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

KB_MEV_PER_K = 0.08617333262


@dataclass(frozen=True)
class NeutronScatteringConfig:
    """Physical and numerical settings for the neutron-scattering MVP."""

    seed: int = 20260407
    temperature_K: float = 140.0

    q_min_Ainv: float = 0.25
    q_max_Ainv: float = 2.55
    n_q: int = 22

    omega_min_meV: float = -25.0
    omega_max_meV: float = 25.0
    n_omega: int = 281

    lattice_a_A: float = 3.35
    resolution_sigma_meV: float = 0.75
    elastic_sigma_meV: float = 0.24

    c_true_meV: float = 6.8
    gap_true_meV: float = 3.2
    gamma0_true_meV: float = 0.45
    gamma1_true_meV: float = 0.30
    amp_scale_true: float = 165.0
    elastic_scale_true: float = 34.0
    background_true: float = 0.90

    torch_epochs: int = 420
    torch_lr: float = 0.05
    torch_weight_decay: float = 2.0e-4


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    return float(np.log(np.expm1(x)))


def check_config(cfg: NeutronScatteringConfig) -> None:
    if cfg.n_q < 10:
        raise ValueError("n_q must be >= 10")
    if cfg.n_omega < 120:
        raise ValueError("n_omega must be >= 120")
    if cfg.q_max_Ainv <= cfg.q_min_Ainv:
        raise ValueError("invalid q range")
    if cfg.omega_max_meV <= cfg.omega_min_meV:
        raise ValueError("invalid omega range")
    if cfg.temperature_K <= 1.0:
        raise ValueError("temperature_K must be > 1 K")
    if cfg.lattice_a_A <= 0.0:
        raise ValueError("lattice_a_A must be positive")
    if cfg.resolution_sigma_meV <= 0.0 or cfg.elastic_sigma_meV <= 0.0:
        raise ValueError("sigma values must be positive")
    if min(cfg.c_true_meV, cfg.gap_true_meV, cfg.gamma0_true_meV, cfg.amp_scale_true) <= 0.0:
        raise ValueError("true parameters must be positive where required")
    if cfg.gamma1_true_meV < 0.0:
        raise ValueError("gamma1_true_meV must be non-negative")


def q_to_xi(q_Ainv: np.ndarray, lattice_a_A: float) -> np.ndarray:
    """Dimensionless reduced wavevector proxy xi=2 sin(Qa/2)."""

    return 2.0 * np.sin(0.5 * q_Ainv * lattice_a_A)


def dispersion_meV(c_meV: float, gap_meV: float, xi: np.ndarray) -> np.ndarray:
    """Gapped acoustic-like dispersion omega(Q)=sqrt(gap^2 + c^2*xi^2)."""

    return np.sqrt(np.maximum(gap_meV * gap_meV + (c_meV * xi) ** 2, 1e-12))


def damping_meV(gamma0_meV: float, gamma1_meV: float, xi: np.ndarray) -> np.ndarray:
    """Phenomenological linewidth gamma(Q)=gamma0+gamma1*xi^2."""

    return np.maximum(gamma0_meV + gamma1_meV * xi * xi, 1e-5)


def bose_factor(energy_meV: np.ndarray | float, temperature_K: float) -> np.ndarray:
    """Bose occupation number n(E)=1/(exp(E/kBT)-1)."""

    x = np.asarray(energy_meV, dtype=np.float64) / (KB_MEV_PER_K * temperature_K)
    x = np.clip(x, 1e-8, 80.0)
    return 1.0 / np.expm1(x)


def lorentzian(omega: np.ndarray, center: float | np.ndarray, gamma: float | np.ndarray) -> np.ndarray:
    """Normalized Lorentzian line shape."""

    return np.asarray(gamma, dtype=np.float64) / np.pi / (
        (omega - np.asarray(center, dtype=np.float64)) ** 2 + np.asarray(gamma, dtype=np.float64) ** 2
    )


def gaussian(omega: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """Normalized Gaussian line shape."""

    sigma = max(float(sigma), 1e-8)
    z = (omega - center) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def q_form_factor(q_Ainv: np.ndarray, lattice_a_A: float) -> np.ndarray:
    """Simple magnetic form-factor-like envelope used by this MVP."""

    q_Ainv = np.asarray(q_Ainv, dtype=np.float64)
    qa_half = 0.5 * q_Ainv * lattice_a_A
    return np.exp(-(0.45 * q_Ainv) ** 2) * (0.20 + np.sin(qa_half) ** 2)


def intensity_slice(
    omega_meV: np.ndarray,
    q_Ainv: float,
    cfg: NeutronScatteringConfig,
    amp_scale: float,
    c_meV: float,
    gap_meV: float,
    gamma0_meV: float,
    gamma1_meV: float,
    elastic_scale: float,
    background: float,
) -> np.ndarray:
    """Forward model for one Q-cut of INS intensity."""

    xi = q_to_xi(np.array([q_Ainv], dtype=np.float64), cfg.lattice_a_A)[0]
    omega0 = float(dispersion_meV(c_meV, gap_meV, np.array([xi], dtype=np.float64))[0])
    gamma = float(damping_meV(gamma0_meV, gamma1_meV, np.array([xi], dtype=np.float64))[0])

    # Approximate instrumental broadening as additive linewidth.
    gamma_eff = gamma + cfg.resolution_sigma_meV

    n_bose = float(bose_factor(omega0, cfg.temperature_K))
    amp_q = amp_scale * float(q_form_factor(np.array([q_Ainv]), cfg.lattice_a_A)[0])
    elastic_q = elastic_scale * float(np.exp(-0.5 * (q_Ainv / 1.8) ** 2))

    inelastic = amp_q * (
        (n_bose + 1.0) * lorentzian(omega_meV, omega0, gamma_eff)
        + n_bose * lorentzian(omega_meV, -omega0, gamma_eff)
    )
    elastic = elastic_q * gaussian(
        omega_meV,
        center=0.0,
        sigma=cfg.elastic_sigma_meV + 0.55 * cfg.resolution_sigma_meV,
    )

    total = inelastic + elastic + background
    return np.clip(total, 1e-12, None)


def simulate_dataset(cfg: NeutronScatteringConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate clean and noisy intensity maps I(Q, omega)."""

    rng = np.random.default_rng(cfg.seed)
    q_grid = np.linspace(cfg.q_min_Ainv, cfg.q_max_Ainv, cfg.n_q)
    omega_grid = np.linspace(cfg.omega_min_meV, cfg.omega_max_meV, cfg.n_omega)

    clean_map = np.zeros((cfg.n_q, cfg.n_omega), dtype=np.float64)
    for iq, q_val in enumerate(q_grid):
        clean_map[iq] = intensity_slice(
            omega_grid,
            q_val,
            cfg,
            cfg.amp_scale_true,
            cfg.c_true_meV,
            cfg.gap_true_meV,
            cfg.gamma0_true_meV,
            cfg.gamma1_true_meV,
            cfg.elastic_scale_true,
            cfg.background_true,
        )

    noisy_map = rng.poisson(lam=np.clip(clean_map, 1e-8, None)).astype(np.float64)
    return q_grid, omega_grid, clean_map, noisy_map


def fit_single_q_cut(
    omega_grid: np.ndarray,
    intensity_obs: np.ndarray,
    q_value: float,
    cfg: NeutronScatteringConfig,
) -> dict[str, float]:
    """Fit one Q-cut with SciPy curve_fit."""

    omega = np.asarray(omega_grid, dtype=np.float64)
    y = np.asarray(intensity_obs, dtype=np.float64)

    mask_pos = omega > 1.0
    if not np.any(mask_pos):
        return {"success": 0.0}

    y_pos = y[mask_pos]
    omega_pos = omega[mask_pos]

    omega0_guess = float(omega_pos[np.argmax(y_pos)])
    bg_guess = float(np.percentile(y, 8.0))
    elastic_guess = float(max(np.max(y[np.abs(omega) < 1.0]) - bg_guess, 0.0))
    amp_guess = float(max(np.max(y_pos) - bg_guess, 1.0))
    gamma_guess = 0.9

    def fit_func(w: np.ndarray, amp: float, omega0: float, gamma: float, elastic_amp: float, bg: float) -> np.ndarray:
        gamma_eff = gamma + cfg.resolution_sigma_meV
        n_bose = float(bose_factor(omega0, cfg.temperature_K))
        inelastic = amp * (
            (n_bose + 1.0) * lorentzian(w, omega0, gamma_eff) + n_bose * lorentzian(w, -omega0, gamma_eff)
        )
        elastic = elastic_amp * gaussian(
            w,
            center=0.0,
            sigma=cfg.elastic_sigma_meV + 0.55 * cfg.resolution_sigma_meV,
        )
        return inelastic + elastic + bg

    lower = np.array([1e-4, 0.35, 0.05, 0.0, 0.0], dtype=np.float64)
    upper = np.array([2e4, cfg.omega_max_meV - 0.2, 8.0, 2e4, 250.0], dtype=np.float64)
    x0 = np.array([amp_guess, omega0_guess, gamma_guess, elastic_guess, bg_guess], dtype=np.float64)
    x0 = np.clip(x0, lower + 1e-6, upper - 1e-6)

    try:
        popt, _ = curve_fit(
            f=fit_func,
            xdata=omega,
            ydata=y,
            p0=x0,
            bounds=(lower, upper),
            method="trf",
            maxfev=8000,
        )
    except Exception:
        return {"success": 0.0}

    pred = fit_func(omega, *popt)
    return {
        "q_Ainv": float(q_value),
        "amp_fit": float(popt[0]),
        "omega0_fit_meV": float(popt[1]),
        "gamma_fit_meV": float(popt[2]),
        "elastic_fit": float(popt[3]),
        "background_fit": float(popt[4]),
        "slice_r2": float(r2_score(y, pred)),
        "slice_mae": float(mean_absolute_error(y, pred)),
        "success": 1.0,
    }


def regress_dispersion(fit_df: pd.DataFrame, cfg: NeutronScatteringConfig) -> dict[str, float]:
    """Fit omega(Q)^2 = c^2 * xi(Q)^2 + gap^2 with LinearRegression."""

    good = fit_df[fit_df["success"] > 0.5].copy()
    if good.empty:
        raise RuntimeError("No successful Q-cut fits for dispersion regression")

    xi = q_to_xi(good["q_Ainv"].to_numpy(), cfg.lattice_a_A)
    x = (xi * xi).reshape(-1, 1)
    y = good["omega0_fit_meV"].to_numpy() ** 2

    reg = LinearRegression()
    reg.fit(x, y)
    y_pred = reg.predict(x)

    slope = float(max(reg.coef_[0], 0.0))
    intercept = float(max(reg.intercept_, 0.0))

    c_fit = float(np.sqrt(slope))
    gap_fit = float(np.sqrt(intercept))

    return {
        "c_fit_meV": c_fit,
        "gap_fit_meV": gap_fit,
        "dispersion_r2": float(r2_score(y, y_pred)),
        "dispersion_mae_sq": float(mean_absolute_error(y, y_pred)),
        "n_reg_samples": float(len(good)),
    }


def torch_model_map(
    q_grid: torch.Tensor,
    omega_grid: torch.Tensor,
    cfg: NeutronScatteringConfig,
    c_meV: torch.Tensor,
    gap_meV: torch.Tensor,
    gamma0_meV: torch.Tensor,
    gamma1_meV: torch.Tensor,
    amp_scale: torch.Tensor,
    elastic_scale: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    """Differentiable map model used in global PyTorch refinement."""

    dtype = q_grid.dtype
    device = q_grid.device

    qa_half = 0.5 * q_grid * cfg.lattice_a_A
    xi = 2.0 * torch.sin(qa_half)

    omega0 = torch.sqrt(torch.clamp(gap_meV * gap_meV + (c_meV * xi) ** 2, min=1e-10))
    gamma = torch.clamp(gamma0_meV + gamma1_meV * xi * xi, min=1e-6)
    gamma_eff = gamma + cfg.resolution_sigma_meV

    # n(E)=1/(exp(E/kBT)-1)
    x = torch.clamp(omega0 / (KB_MEV_PER_K * cfg.temperature_K), min=1e-8, max=80.0)
    n_bose = 1.0 / torch.expm1(x)

    form_factor = torch.exp(-(0.45 * q_grid) ** 2) * (0.20 + torch.sin(qa_half) ** 2)
    amp_q = amp_scale * form_factor
    elastic_q = elastic_scale * torch.exp(-0.5 * (q_grid / 1.8) ** 2)

    w = omega_grid[None, :]
    omega0_col = omega0[:, None]
    gamma_col = gamma_eff[:, None]
    nb_col = n_bose[:, None]
    amp_col = amp_q[:, None]
    elastic_col = elastic_q[:, None]

    lor_pos = gamma_col / np.pi / ((w - omega0_col) ** 2 + gamma_col * gamma_col)
    lor_neg = gamma_col / np.pi / ((w + omega0_col) ** 2 + gamma_col * gamma_col)

    sigma_el = cfg.elastic_sigma_meV + 0.55 * cfg.resolution_sigma_meV
    gauss_el = torch.exp(-0.5 * (w / sigma_el) ** 2) / (sigma_el * np.sqrt(2.0 * np.pi))

    signal = amp_col * ((nb_col + 1.0) * lor_pos + nb_col * lor_neg)
    elastic = elastic_col * gauss_el

    return signal + elastic + background.to(dtype=dtype, device=device)


def torch_global_refinement(
    q_grid: np.ndarray,
    omega_grid: np.ndarray,
    observed_map: np.ndarray,
    cfg: NeutronScatteringConfig,
) -> dict[str, float]:
    """Global parameter refinement with PyTorch on the full Q-omega map."""

    torch.manual_seed(cfg.seed)

    q_t = torch.tensor(q_grid, dtype=torch.float64)
    w_t = torch.tensor(omega_grid, dtype=torch.float64)
    obs_t = torch.tensor(observed_map, dtype=torch.float64)

    raw_c = torch.nn.Parameter(torch.tensor(_inv_softplus(5.0), dtype=torch.float64))
    raw_gap = torch.nn.Parameter(torch.tensor(_inv_softplus(2.5), dtype=torch.float64))
    raw_g0 = torch.nn.Parameter(torch.tensor(_inv_softplus(0.6), dtype=torch.float64))
    raw_g1 = torch.nn.Parameter(torch.tensor(_inv_softplus(0.2), dtype=torch.float64))
    raw_amp = torch.nn.Parameter(torch.tensor(_inv_softplus(120.0), dtype=torch.float64))
    raw_elastic = torch.nn.Parameter(torch.tensor(_inv_softplus(20.0), dtype=torch.float64))
    raw_bg = torch.nn.Parameter(torch.tensor(_inv_softplus(0.5), dtype=torch.float64))

    optimizer = torch.optim.Adam(
        [raw_c, raw_gap, raw_g0, raw_g1, raw_amp, raw_elastic, raw_bg],
        lr=cfg.torch_lr,
        weight_decay=cfg.torch_weight_decay,
    )

    for _ in range(cfg.torch_epochs):
        optimizer.zero_grad()

        c = torch.nn.functional.softplus(raw_c) + 1e-6
        gap = torch.nn.functional.softplus(raw_gap) + 1e-6
        g0 = torch.nn.functional.softplus(raw_g0) + 1e-6
        g1 = torch.nn.functional.softplus(raw_g1) + 1e-6
        amp = torch.nn.functional.softplus(raw_amp) + 1e-6
        elastic = torch.nn.functional.softplus(raw_elastic) + 1e-6
        bg = torch.nn.functional.softplus(raw_bg) + 1e-6

        pred = torch_model_map(q_t, w_t, cfg, c, gap, g0, g1, amp, elastic, bg)
        loss = torch.mean((pred - obs_t) ** 2)
        loss.backward()
        optimizer.step()

    c_fit = float((torch.nn.functional.softplus(raw_c) + 1e-6).detach().cpu().item())
    gap_fit = float((torch.nn.functional.softplus(raw_gap) + 1e-6).detach().cpu().item())
    g0_fit = float((torch.nn.functional.softplus(raw_g0) + 1e-6).detach().cpu().item())
    g1_fit = float((torch.nn.functional.softplus(raw_g1) + 1e-6).detach().cpu().item())
    amp_fit = float((torch.nn.functional.softplus(raw_amp) + 1e-6).detach().cpu().item())
    elastic_fit = float((torch.nn.functional.softplus(raw_elastic) + 1e-6).detach().cpu().item())
    bg_fit = float((torch.nn.functional.softplus(raw_bg) + 1e-6).detach().cpu().item())

    pred_final = torch_model_map(
        torch.tensor(q_grid, dtype=torch.float64),
        torch.tensor(omega_grid, dtype=torch.float64),
        cfg,
        torch.tensor(c_fit, dtype=torch.float64),
        torch.tensor(gap_fit, dtype=torch.float64),
        torch.tensor(g0_fit, dtype=torch.float64),
        torch.tensor(g1_fit, dtype=torch.float64),
        torch.tensor(amp_fit, dtype=torch.float64),
        torch.tensor(elastic_fit, dtype=torch.float64),
        torch.tensor(bg_fit, dtype=torch.float64),
    ).detach().cpu().numpy()

    mse = float(mean_squared_error(observed_map.ravel(), pred_final.ravel()))
    nmse = mse / float(np.mean(observed_map**2) + 1e-12)

    return {
        "torch_c_fit_meV": c_fit,
        "torch_gap_fit_meV": gap_fit,
        "torch_gamma0_fit_meV": g0_fit,
        "torch_gamma1_fit_meV": g1_fit,
        "torch_amp_fit": amp_fit,
        "torch_elastic_fit": elastic_fit,
        "torch_background_fit": bg_fit,
        "torch_mse": mse,
        "torch_nmse": float(nmse),
    }


def main() -> None:
    cfg = NeutronScatteringConfig()
    check_config(cfg)

    q_grid, omega_grid, clean_map, noisy_map = simulate_dataset(cfg)

    fit_rows: list[dict[str, float]] = []
    for iq, q_val in enumerate(q_grid):
        row = fit_single_q_cut(omega_grid, noisy_map[iq], float(q_val), cfg)
        if row.get("success", 0.0) > 0.5:
            row["q_index"] = float(iq)
        fit_rows.append(row)

    fit_df = pd.DataFrame(fit_rows)
    if fit_df.empty:
        raise RuntimeError("No fit rows generated")

    # Fill missing optional columns to keep a stable schema for CSV export.
    expected_cols = [
        "q_index",
        "q_Ainv",
        "amp_fit",
        "omega0_fit_meV",
        "gamma_fit_meV",
        "elastic_fit",
        "background_fit",
        "slice_r2",
        "slice_mae",
        "success",
    ]
    for col in expected_cols:
        if col not in fit_df.columns:
            fit_df[col] = np.nan

    disp = regress_dispersion(fit_df, cfg)
    torch_stats = torch_global_refinement(q_grid, omega_grid, noisy_map, cfg)

    success_mask = fit_df["success"].fillna(0.0) > 0.5
    success_ratio = float(np.mean(success_mask))
    slice_r2_mean = float(fit_df.loc[success_mask, "slice_r2"].mean())
    omega_mae = float(
        mean_absolute_error(
            dispersion_meV(cfg.c_true_meV, cfg.gap_true_meV, q_to_xi(fit_df.loc[success_mask, "q_Ainv"].to_numpy(), cfg.lattice_a_A)),
            fit_df.loc[success_mask, "omega0_fit_meV"].to_numpy(),
        )
    )

    summary = {
        "n_q_total": float(cfg.n_q),
        "n_q_success": float(np.count_nonzero(success_mask)),
        "success_ratio": success_ratio,
        "slice_r2_mean": slice_r2_mean,
        "omega_mae_meV": omega_mae,
        **disp,
        **torch_stats,
    }

    # Quality gates for a trustworthy MVP.
    if success_ratio < 0.85:
        raise RuntimeError(f"Too many failed Q-cut fits: success_ratio={success_ratio:.3f}")
    if slice_r2_mean < 0.80:
        raise RuntimeError(f"Slice fit quality too low: mean R2={slice_r2_mean:.3f}")
    if summary["dispersion_r2"] < 0.92:
        raise RuntimeError(f"Dispersion regression quality too low: R2={summary['dispersion_r2']:.3f}")
    if abs(summary["c_fit_meV"] - cfg.c_true_meV) > 1.1:
        raise RuntimeError(
            f"Recovered c deviates too much: fitted={summary['c_fit_meV']:.3f}, true={cfg.c_true_meV:.3f}"
        )
    if abs(summary["gap_fit_meV"] - cfg.gap_true_meV) > 1.2:
        raise RuntimeError(
            f"Recovered gap deviates too much: fitted={summary['gap_fit_meV']:.3f}, true={cfg.gap_true_meV:.3f}"
        )
    if summary["torch_nmse"] > 0.15:
        raise RuntimeError(f"Torch global fit NMSE too high: {summary['torch_nmse']:.3f}")

    out_dir = Path(__file__).resolve().parent
    fit_path = out_dir / "q_cut_fit_results.csv"
    fit_df.sort_values(by=["q_Ainv"], na_position="last").to_csv(fit_path, index=False)

    map_summary_path = out_dir / "map_summary.csv"
    pd.DataFrame(
        {
            "q_Ainv": np.repeat(q_grid, len(omega_grid)),
            "omega_meV": np.tile(omega_grid, len(q_grid)),
            "intensity_clean": clean_map.reshape(-1),
            "intensity_noisy": noisy_map.reshape(-1),
        }
    ).to_csv(map_summary_path, index=False)

    print("Config:")
    print(asdict(cfg))

    print("\nSummary metrics:")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")

    print("\nSuccessful Q-cut fits (head):")
    print(
        fit_df.loc[success_mask, ["q_Ainv", "omega0_fit_meV", "gamma_fit_meV", "slice_r2"]]
        .head(8)
        .to_string(index=False)
    )

    print(f"\nSaved: {fit_path.name}, {map_summary_path.name}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
