"""Minimal runnable MVP for multiferroics using a coupled Landau model.

Pipeline:
1) Simulate equilibrium (P, M) responses on synthetic E/H sweeps via SciPy solvers.
2) Estimate magnetoelectric slopes with scikit-learn linear regression.
3) Fit coupling gamma back from noisy observations using a differentiable PyTorch solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize, root
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class LandauParams:
    """Landau free-energy parameters in normalized units."""

    a_p: float = 0.018
    b_p: float = 1.0
    tc_p: float = 360.0
    a_m: float = 0.030
    b_m: float = 1.1
    tc_m: float = 280.0
    gamma: float = 0.22


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical and synthetic-data configuration."""

    seed: int = 7
    noise_std: float = 0.0005
    bias_e: float = 0.09
    bias_h: float = 0.09
    temp_min: float = 250.0
    temp_max: float = 380.0
    n_temp: int = 12
    field_delta_min: float = -0.08
    field_delta_max: float = 0.08
    n_field: int = 9
    torch_epochs: int = 250
    torch_lr: float = 0.05
    torch_newton_steps: int = 35
    torch_newton_damping: float = 0.9


def free_energy(P: float, M: float, T: float, E: float, H: float, p: LandauParams) -> float:
    """Coupled Landau free energy F(P, M; T, E, H)."""
    return (
        0.5 * p.a_p * (T - p.tc_p) * P * P
        + 0.25 * p.b_p * P**4
        + 0.5 * p.a_m * (T - p.tc_m) * M * M
        + 0.25 * p.b_m * M**4
        - p.gamma * P * M
        - E * P
        - H * M
    )


def stationarity_residual(x: np.ndarray, T: float, E: float, H: float, p: LandauParams) -> np.ndarray:
    """Residual of Euler-Lagrange stationarity equations dF/dP=0, dF/dM=0."""
    P, M = float(x[0]), float(x[1])
    return np.array(
        [
            p.a_p * (T - p.tc_p) * P + p.b_p * P**3 - p.gamma * M - E,
            p.a_m * (T - p.tc_m) * M + p.b_m * M**3 - p.gamma * P - H,
        ],
        dtype=float,
    )


def solve_equilibrium_scipy(
    T: float,
    E: float,
    H: float,
    p: LandauParams,
    guess: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Solve equilibrium by trying root-finding first, then energy minimization fallback."""
    guess_vec = np.array(guess, dtype=float)
    candidates: list[np.ndarray] = []

    initial_guesses = [
        guess_vec,
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5]),
        np.array([0.0, 0.0]),
        np.array([-1.0, -1.0]),
    ]

    for g in initial_guesses:
        sol = root(stationarity_residual, g, args=(T, E, H, p), method="hybr")
        if sol.success:
            candidates.append(sol.x.astype(float))

    if not candidates:
        for g in initial_guesses:
            sol = minimize(
                lambda z: free_energy(float(z[0]), float(z[1]), T, E, H, p),
                x0=g,
                method="L-BFGS-B",
            )
            if sol.success:
                candidates.append(sol.x.astype(float))

    if not candidates:
        raise RuntimeError(f"Equilibrium solve failed at T={T:.3f}, E={E:.3f}, H={H:.3f}")

    # Choose the energetically best candidate.
    energies = [free_energy(float(x[0]), float(x[1]), T, E, H, p) for x in candidates]
    best_idx = int(np.argmin(np.array(energies)))
    return candidates[best_idx]


def generate_synthetic_dataset(p: LandauParams, cfg: SimulationConfig) -> pd.DataFrame:
    """Generate noisy P/M measurements from two sweeps around a positive bias point.

    sweep_type == "H" : sweep H around bias_h at fixed E=bias_e, track P(H)
    sweep_type == "E" : sweep E around bias_e at fixed H=bias_h, track M(E)
    """
    rng = np.random.default_rng(cfg.seed)

    temperatures = np.linspace(cfg.temp_min, cfg.temp_max, cfg.n_temp)
    deltas = np.linspace(cfg.field_delta_min, cfg.field_delta_max, cfg.n_field)

    rows: list[dict[str, float | str]] = []

    for T in temperatures:
        x_h = np.array([1.0, 1.0], dtype=float)
        for dH in deltas:
            E = cfg.bias_e
            H = cfg.bias_h + float(dH)
            x_h = solve_equilibrium_scipy(float(T), E, H, p, guess=(float(x_h[0]), float(x_h[1])))
            P_true, M_true = float(x_h[0]), float(x_h[1])
            rows.append(
                {
                    "sweep_type": "H",
                    "T": float(T),
                    "E": E,
                    "H": H,
                    "P_true": P_true,
                    "M_true": M_true,
                    "P_obs": P_true + cfg.noise_std * float(rng.normal()),
                    "M_obs": M_true + cfg.noise_std * float(rng.normal()),
                }
            )

        x_e = np.array([1.0, 1.0], dtype=float)
        for dE in deltas:
            E = cfg.bias_e + float(dE)
            H = cfg.bias_h
            x_e = solve_equilibrium_scipy(float(T), E, H, p, guess=(float(x_e[0]), float(x_e[1])))
            P_true, M_true = float(x_e[0]), float(x_e[1])
            rows.append(
                {
                    "sweep_type": "E",
                    "T": float(T),
                    "E": E,
                    "H": H,
                    "P_true": P_true,
                    "M_true": M_true,
                    "P_obs": P_true + cfg.noise_std * float(rng.normal()),
                    "M_obs": M_true + cfg.noise_std * float(rng.normal()),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["T", "sweep_type", "E", "H"]).reset_index(drop=True)


def estimate_me_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate magnetoelectric slopes by temperature using linear regressions.

    alpha_me(T) = dP/dH from H-sweep at fixed E
    beta_me(T)  = dM/dE from E-sweep at fixed H
    """
    rows: list[dict[str, float]] = []
    for T in sorted(df["T"].unique()):
        dH = df[(df["T"] == T) & (df["sweep_type"] == "H")]
        dE = df[(df["T"] == T) & (df["sweep_type"] == "E")]

        reg_alpha = LinearRegression().fit(dH[["H"]].to_numpy(), dH["P_obs"].to_numpy())
        reg_beta = LinearRegression().fit(dE[["E"]].to_numpy(), dE["M_obs"].to_numpy())

        rows.append(
            {
                "T": float(T),
                "alpha_me_dP_dH": float(reg_alpha.coef_[0]),
                "alpha_r2": float(reg_alpha.score(dH[["H"]].to_numpy(), dH["P_obs"].to_numpy())),
                "beta_me_dM_dE": float(reg_beta.coef_[0]),
                "beta_r2": float(reg_beta.score(dE[["E"]].to_numpy(), dE["M_obs"].to_numpy())),
            }
        )

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def compute_phase_profile(p: LandauParams) -> pd.DataFrame:
    """Compute equilibrium P/M vs temperature at near-zero bias for phase labeling."""
    temperatures = np.linspace(240.0, 390.0, 31)
    x = np.array([1.0, 1.0], dtype=float)
    rows: list[dict[str, float | bool]] = []
    for T in temperatures:
        x = solve_equilibrium_scipy(float(T), 1.0e-4, 1.0e-4, p, guess=(float(x[0]), float(x[1])))
        P0, M0 = float(x[0]), float(x[1])
        has_ferroelectric = abs(P0) > 0.08
        has_ferromagnetic = abs(M0) > 0.08
        rows.append(
            {
                "T": float(T),
                "P0": P0,
                "M0": M0,
                "is_FE": has_ferroelectric,
                "is_FM": has_ferromagnetic,
                "is_multiferroic": bool(has_ferroelectric and has_ferromagnetic),
            }
        )
    return pd.DataFrame(rows)


def torch_newton_equilibrium(
    T: torch.Tensor,
    E: torch.Tensor,
    H: torch.Tensor,
    gamma: torch.Tensor,
    p_fixed: LandauParams,
    n_steps: int,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Newton solver for stationarity equations."""
    # Positive branch initialization under positive bias fields.
    P = torch.sqrt(torch.clamp(p_fixed.a_p * (p_fixed.tc_p - T) / p_fixed.b_p, min=0.0) + 1.0e-6) + 0.1
    M = torch.sqrt(torch.clamp(p_fixed.a_m * (p_fixed.tc_m - T) / p_fixed.b_m, min=0.0) + 1.0e-6) + 0.1

    for _ in range(n_steps):
        gP = p_fixed.a_p * (T - p_fixed.tc_p) * P + p_fixed.b_p * P**3 - gamma * M - E
        gM = p_fixed.a_m * (T - p_fixed.tc_m) * M + p_fixed.b_m * M**3 - gamma * P - H

        A = p_fixed.a_p * (T - p_fixed.tc_p) + 3.0 * p_fixed.b_p * P**2
        B = p_fixed.a_m * (T - p_fixed.tc_m) + 3.0 * p_fixed.b_m * M**2
        det = A * B - gamma**2
        det = torch.where(det.abs() < 1.0e-4, det + 1.0e-4, det)

        dP = (B * gP + gamma * gM) / det
        dM = (gamma * gP + A * gM) / det

        P = P - damping * dP
        M = M - damping * dM

    return P, M


def fit_gamma_torch(
    df: pd.DataFrame,
    p_fixed_except_gamma: LandauParams,
    cfg: SimulationConfig,
) -> dict[str, float]:
    """Fit coupling gamma from noisy observations with PyTorch."""
    T = torch.tensor(df["T"].to_numpy(), dtype=torch.float32)
    E = torch.tensor(df["E"].to_numpy(), dtype=torch.float32)
    H = torch.tensor(df["H"].to_numpy(), dtype=torch.float32)
    P_target = torch.tensor(df["P_obs"].to_numpy(), dtype=torch.float32)
    M_target = torch.tensor(df["M_obs"].to_numpy(), dtype=torch.float32)

    gamma_raw = torch.nn.Parameter(torch.tensor(-1.7, dtype=torch.float32))
    optimizer = torch.optim.Adam([gamma_raw], lr=cfg.torch_lr)

    final_loss = float("nan")
    mse_p = float("nan")
    mse_m = float("nan")

    for _ in range(cfg.torch_epochs):
        gamma = torch.nn.functional.softplus(gamma_raw) + 1.0e-5
        P_pred, M_pred = torch_newton_equilibrium(
            T=T,
            E=E,
            H=H,
            gamma=gamma,
            p_fixed=p_fixed_except_gamma,
            n_steps=cfg.torch_newton_steps,
            damping=cfg.torch_newton_damping,
        )

        mse_p_tensor = torch.mean((P_pred - P_target) ** 2)
        mse_m_tensor = torch.mean((M_pred - M_target) ** 2)
        loss = mse_p_tensor + mse_m_tensor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach())
        mse_p = float(mse_p_tensor.detach())
        mse_m = float(mse_m_tensor.detach())

    gamma_fit = float((torch.nn.functional.softplus(gamma_raw) + 1.0e-5).detach())
    return {
        "gamma_fit": gamma_fit,
        "final_loss": final_loss,
        "mse_P": mse_p,
        "mse_M": mse_m,
    }


def compute_stationarity_quality(df: pd.DataFrame, p: LandauParams) -> float:
    """Mean residual norm on noiseless truth points."""
    residual_norms = []
    for row in df.itertuples(index=False):
        r = stationarity_residual(
            np.array([float(row.P_true), float(row.M_true)]),
            T=float(row.T),
            E=float(row.E),
            H=float(row.H),
            p=p,
        )
        residual_norms.append(float(np.linalg.norm(r)))
    return float(np.mean(residual_norms))


def main() -> None:
    torch.manual_seed(7)

    params_true = LandauParams()
    cfg = SimulationConfig()

    df = generate_synthetic_dataset(params_true, cfg)
    slope_df = estimate_me_slopes(df)
    phase_df = compute_phase_profile(params_true)
    fit_out = fit_gamma_torch(df, params_true, cfg)

    mean_residual = compute_stationarity_quality(df, params_true)
    mean_alpha_r2 = float(slope_df["alpha_r2"].mean())
    mean_beta_r2 = float(slope_df["beta_r2"].mean())
    multiferroic_count = int(phase_df["is_multiferroic"].sum())

    summary = {
        "n_samples": int(len(df)),
        "n_temperatures": int(df["T"].nunique()),
        "mean_stationarity_residual": mean_residual,
        "mean_alpha_r2": mean_alpha_r2,
        "mean_beta_r2": mean_beta_r2,
        "gamma_true": params_true.gamma,
        "gamma_fit": fit_out["gamma_fit"],
        "gamma_abs_error": abs(fit_out["gamma_fit"] - params_true.gamma),
        "torch_final_loss": fit_out["final_loss"],
        "torch_mse_P": fit_out["mse_P"],
        "torch_mse_M": fit_out["mse_M"],
        "multiferroic_temperature_points": multiferroic_count,
    }

    print("=== Multiferroics MVP Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.8f}")
        else:
            print(f"{k}: {v}")

    print("\n=== Magnetoelectric slope table (head) ===")
    print(slope_df.head(8).to_string(index=False))

    print("\n=== Phase profile (tail) ===")
    print(phase_df.tail(8).to_string(index=False))

    # Deterministic quality gates for this MVP.
    assert summary["n_samples"] == cfg.n_temp * cfg.n_field * 2, "Unexpected sample count"
    assert summary["mean_stationarity_residual"] < 2.0e-6, "Equilibrium residual too large"
    assert summary["mean_alpha_r2"] > 0.95, "alpha regression quality too low"
    assert summary["mean_beta_r2"] > 0.95, "beta regression quality too low"
    assert summary["gamma_abs_error"] < 0.02, "Torch inversion did not recover gamma"
    assert summary["torch_final_loss"] < 2.0e-4, "Torch fit loss too high"
    assert summary["multiferroic_temperature_points"] >= 15, "Phase window unexpectedly narrow"


if __name__ == "__main__":
    main()
