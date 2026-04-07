"""Minimal runnable MVP for Lienard-Wiechert potentials (PHYS-0184).

This script demonstrates source-level, non-black-box computation for a point charge
moving at constant velocity:
1) Solve retarded time numerically by fixed-point iteration.
2) Cross-check with the analytic quadratic retarded-time solution.
3) Compute Lienard-Wiechert scalar/vector potentials (phi, A).
4) Compute velocity-field part of E,B and verify against Heaviside closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import c, e, epsilon_0, mu_0


@dataclass(frozen=True)
class LWConfig:
    """Configuration for the MVP experiment."""

    charge_coulomb: float = e
    beta: float = 0.72
    t_obs_second: float = 4.0e-9
    retarded_tol_second: float = 1.0e-16
    retarded_max_iter: int = 600
    beta_for_low_speed_check: float = 1.0e-4


def source_position_uniform(t: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Return source position r_q(t)=v*t for each time sample."""
    return t[:, None] * v_vec[None, :]


def make_observation_points(beta: float, t_obs: float) -> np.ndarray:
    """Build deterministic observation points away from singular configurations."""
    x_charge = beta * c * t_obs
    offsets = np.array(
        [
            [0.22, 0.08, 0.00],
            [0.28, -0.12, 0.05],
            [0.34, 0.00, 0.11],
            [0.42, 0.18, -0.10],
            [0.50, -0.23, 0.07],
            [0.62, 0.30, 0.12],
            [0.75, -0.28, -0.15],
            [0.90, 0.22, -0.20],
        ],
        dtype=np.float64,
    )
    points = offsets.copy()
    points[:, 0] += x_charge
    return points


def solve_retarded_time_fixed_point(
    obs_xyz: np.ndarray,
    t_obs: float,
    beta_vec: np.ndarray,
    tol_second: float,
    max_iter: int,
) -> tuple[np.ndarray, int]:
    """Solve t_r from c*(t_obs-t_r)=|r-r_q(t_r)| by fixed-point iteration."""
    if tol_second <= 0.0:
        raise ValueError("tol_second must be positive")
    if max_iter < 2:
        raise ValueError("max_iter must be >=2")

    v_vec = beta_vec * c
    n_point = obs_xyz.shape[0]

    t_now = np.full(n_point, t_obs, dtype=np.float64)
    src_now = source_position_uniform(t_now, v_vec)
    r_now = np.linalg.norm(obs_xyz - src_now, axis=1)
    t_r = t_obs - r_now / c

    for iter_id in range(1, max_iter + 1):
        src = source_position_uniform(t_r, v_vec)
        r = np.linalg.norm(obs_xyz - src, axis=1)
        t_next = t_obs - r / c
        err = float(np.max(np.abs(t_next - t_r)))
        t_r = t_next
        if err < tol_second:
            return t_r, iter_id

    raise RuntimeError("Retarded-time fixed-point solver did not converge")


def retarded_time_analytic_uniform_x(obs_xyz: np.ndarray, t_obs: float, beta: float) -> np.ndarray:
    """Analytic retarded time for uniform motion along +x by quadratic formula."""
    v = beta * c
    x = obs_xyz[:, 0]
    y = obs_xyz[:, 1]
    z = obs_xyz[:, 2]

    a = c * c - v * v
    b = -2.0 * (c * c * t_obs - x * v)
    c0 = c * c * t_obs * t_obs - (x * x + y * y + z * z)

    disc = b * b - 4.0 * a * c0
    if np.any(disc < -1.0e-18):
        raise RuntimeError("Negative discriminant in analytic retarded-time solution")
    disc = np.maximum(disc, 0.0)

    return (-b - np.sqrt(disc)) / (2.0 * a)


def lw_potentials_from_retarded(
    obs_xyz: np.ndarray,
    t_r: np.ndarray,
    beta_vec: np.ndarray,
    charge_coulomb: float,
) -> dict[str, np.ndarray]:
    """Compute Lienard-Wiechert potentials from retarded geometric quantities."""
    v_vec = beta_vec * c
    src = source_position_uniform(t_r, v_vec)
    r_vec = obs_xyz - src
    r = np.linalg.norm(r_vec, axis=1)

    if np.any(r <= 0.0):
        raise RuntimeError("Observation point hit source position, singular potential")

    n_vec = r_vec / r[:, None]
    kappa = 1.0 - np.einsum("ij,j->i", n_vec, beta_vec)

    if np.any(kappa <= 0.0):
        raise RuntimeError("Found non-causal kappa<=0, invalid configuration")

    phi = charge_coulomb / (4.0 * np.pi * epsilon_0 * kappa * r)
    a_vec = (mu_0 * charge_coulomb / (4.0 * np.pi)) * (v_vec[None, :] / (kappa * r)[:, None])

    return {
        "source_xyz": src,
        "R_vec": r_vec,
        "R": r,
        "n_vec": n_vec,
        "kappa": kappa,
        "phi": phi,
        "A_vec": a_vec,
    }


def lw_velocity_fields(
    n_vec: np.ndarray,
    r: np.ndarray,
    kappa: np.ndarray,
    beta_vec: np.ndarray,
    charge_coulomb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity-field part of LW fields (beta-dot=0 for uniform motion)."""
    beta2 = float(np.dot(beta_vec, beta_vec))
    pref = charge_coulomb / (4.0 * np.pi * epsilon_0)

    e_vec = pref * (1.0 - beta2) * (n_vec - beta_vec[None, :]) / (kappa[:, None] ** 3 * r[:, None] ** 2)
    b_vec = np.cross(n_vec, e_vec) / c
    return e_vec, b_vec


def heaviside_fields_uniform(
    obs_xyz: np.ndarray,
    t_obs: float,
    beta_vec: np.ndarray,
    charge_coulomb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form fields for uniformly moving charge (Heaviside form)."""
    v_vec = beta_vec * c
    src_now = source_position_uniform(np.full(obs_xyz.shape[0], t_obs), v_vec)
    r_now = obs_xyz - src_now

    beta2 = float(np.dot(beta_vec, beta_vec))
    pref = charge_coulomb / (4.0 * np.pi * epsilon_0)

    r2 = np.einsum("ij,ij->i", r_now, r_now)
    beta_cross_r = np.cross(beta_vec[None, :], r_now)
    cross2 = np.einsum("ij,ij->i", beta_cross_r, beta_cross_r)

    denom = (r2 - cross2) ** 1.5
    if np.any(denom <= 0.0):
        raise RuntimeError("Invalid denominator in Heaviside field")

    e_vec = pref * (1.0 - beta2) * r_now / denom[:, None]
    b_vec = np.cross(beta_vec[None, :], e_vec) / c
    return e_vec, b_vec


def potentials_for_uniform_case(
    obs_xyz: np.ndarray,
    t_obs: float,
    beta: float,
    charge_coulomb: float,
    tol_second: float,
    max_iter: int,
) -> dict[str, np.ndarray | int]:
    """Helper that solves retarded time and computes LW potentials/geometry."""
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must satisfy 0<beta<1")

    beta_vec = np.array([beta, 0.0, 0.0], dtype=np.float64)

    t_r_num, used_iter = solve_retarded_time_fixed_point(obs_xyz, t_obs, beta_vec, tol_second, max_iter)
    t_r_ref = retarded_time_analytic_uniform_x(obs_xyz, t_obs, beta)

    out = lw_potentials_from_retarded(obs_xyz, t_r_num, beta_vec, charge_coulomb)
    causal_residual = t_obs - t_r_num - out["R"] / c

    out.update(
        {
            "beta_vec": beta_vec,
            "t_r_num": t_r_num,
            "t_r_ref": t_r_ref,
            "t_r_abs_err": np.abs(t_r_num - t_r_ref),
            "causal_residual": causal_residual,
            "used_iter": np.array([used_iter], dtype=np.int64),
        }
    )
    return out


def main() -> None:
    cfg = LWConfig()

    obs_xyz = make_observation_points(cfg.beta, cfg.t_obs_second)
    case = potentials_for_uniform_case(
        obs_xyz=obs_xyz,
        t_obs=cfg.t_obs_second,
        beta=cfg.beta,
        charge_coulomb=cfg.charge_coulomb,
        tol_second=cfg.retarded_tol_second,
        max_iter=cfg.retarded_max_iter,
    )

    beta_vec = case["beta_vec"]
    phi = case["phi"]
    a_vec = case["A_vec"]

    # Consistency check for A = (phi/c) * beta.
    a_from_phi = (phi[:, None] / c) * beta_vec[None, :]
    a_relation_rel_err = np.linalg.norm(a_vec - a_from_phi, axis=1) / np.maximum(
        np.linalg.norm(a_vec, axis=1), 1.0e-30
    )

    # Field cross-check: LW retarded formula vs Heaviside closed form.
    e_lw, b_lw = lw_velocity_fields(
        n_vec=case["n_vec"],
        r=case["R"],
        kappa=case["kappa"],
        beta_vec=beta_vec,
        charge_coulomb=cfg.charge_coulomb,
    )
    e_ref, b_ref = heaviside_fields_uniform(obs_xyz, cfg.t_obs_second, beta_vec, cfg.charge_coulomb)

    e_rel_err = np.linalg.norm(e_lw - e_ref, axis=1) / np.maximum(np.linalg.norm(e_ref, axis=1), 1.0e-30)
    b_rel_err = np.linalg.norm(b_lw - b_ref, axis=1) / np.maximum(np.linalg.norm(b_ref, axis=1), 1.0e-30)

    # Low-speed limit: LW potential approaches Coulomb potential.
    low_case = potentials_for_uniform_case(
        obs_xyz=obs_xyz,
        t_obs=cfg.t_obs_second,
        beta=cfg.beta_for_low_speed_check,
        charge_coulomb=cfg.charge_coulomb,
        tol_second=cfg.retarded_tol_second,
        max_iter=cfg.retarded_max_iter,
    )
    low_beta_vec = low_case["beta_vec"]
    src_now_low = source_position_uniform(np.full(obs_xyz.shape[0], cfg.t_obs_second), low_beta_vec * c)
    r_inst_low = np.linalg.norm(obs_xyz - src_now_low, axis=1)
    phi_coulomb = cfg.charge_coulomb / (4.0 * np.pi * epsilon_0 * r_inst_low)
    phi_low_rel_err = np.abs(low_case["phi"] - phi_coulomb) / phi_coulomb

    checks = {
        "retarded time numeric vs analytic": float(np.max(case["t_r_abs_err"])) < 1.0e-15,
        "retarded equation residual": float(np.max(np.abs(case["causal_residual"]))) < 2.0e-16,
        "A-phi relation": float(np.max(a_relation_rel_err)) < 2.0e-12,
        "E field LW vs Heaviside": float(np.max(e_rel_err)) < 6.0e-8,
        "B field LW vs Heaviside": float(np.max(b_rel_err)) < 8.0e-8,
        "low-beta potential near Coulomb": float(np.max(phi_low_rel_err)) < 5.0e-4,
    }

    rows = []
    for i in range(obs_xyz.shape[0]):
        rows.append(
            {
                "point_id": i,
                "x_m": obs_xyz[i, 0],
                "y_m": obs_xyz[i, 1],
                "z_m": obs_xyz[i, 2],
                "t_r_num_s": case["t_r_num"][i],
                "t_r_ref_s": case["t_r_ref"][i],
                "t_r_abs_err_s": case["t_r_abs_err"][i],
                "kappa": case["kappa"][i],
                "R_m": case["R"][i],
                "phi_V": phi[i],
                "A_mag": np.linalg.norm(a_vec[i]),
                "E_LW_mag_V_m": np.linalg.norm(e_lw[i]),
                "E_ref_mag_V_m": np.linalg.norm(e_ref[i]),
                "E_rel_err": e_rel_err[i],
            }
        )
    df = pd.DataFrame(rows)

    print("=== Lienard-Wiechert Potentials MVP (PHYS-0184) ===")
    print(
        "Config: q={q:.6e} C, beta={beta:.4f}, t_obs={t:.3e} s, N_points={n}, retarded_tol={tol:.1e}".format(
            q=cfg.charge_coulomb,
            beta=cfg.beta,
            t=cfg.t_obs_second,
            n=obs_xyz.shape[0],
            tol=cfg.retarded_tol_second,
        )
    )
    print(f"Retarded solver iterations used: {int(case['used_iter'][0])}")

    print("\n[Global diagnostics]")
    print(f"max |t_r_num - t_r_ref|      = {np.max(case['t_r_abs_err']):.3e} s")
    print(f"max causal residual           = {np.max(np.abs(case['causal_residual'])):.3e} s")
    print(f"max rel error A(phi relation) = {np.max(a_relation_rel_err):.3e}")
    print(f"max rel error E(LW vs ref)    = {np.max(e_rel_err):.3e}")
    print(f"max rel error B(LW vs ref)    = {np.max(b_rel_err):.3e}")
    print(f"max rel error low-beta phi    = {np.max(phi_low_rel_err):.3e}")

    print("\n[Sample table]")
    print(df.to_string(index=False))

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
