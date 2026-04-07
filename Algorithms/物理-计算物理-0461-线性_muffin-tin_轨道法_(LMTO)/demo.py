"""Minimal runnable MVP for LMTO (Linear Muffin-Tin Orbital), PHYS-0440.

This script builds a pedagogical 1D LMTO-like workflow:
1) Solve single-site radial equation around a muffin-tin well.
2) Build linearized local orbitals (phi, phi_dot) on each lattice site.
3) Assemble finite-basis generalized eigenproblem H c = E S c.
4) Compare with finite-difference reference spectrum.
5) Use PyTorch to tune the phi_dot scaling (alpha) to reduce spectrum error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class LMTOConfig:
    lattice_constant: float = 2.2
    n_atoms: int = 5
    muffin_tin_radius: float = 0.8
    site_potential_depth: float = 6.0
    site_potential_sigma: float = 0.35
    linearization_energy: float = -2.3
    linearization_delta_e: float = 2e-3
    tail_decay_min: float = 0.8
    x_points: int = 2201
    radial_points: int = 1200
    reference_points: int = 1300
    n_reference_levels: int = 8
    torch_steps: int = 220
    torch_lr: float = 0.06
    seed: int = 7


def validate_config(cfg: LMTOConfig) -> None:
    if cfg.n_atoms < 2:
        raise ValueError("n_atoms must be >= 2")
    if cfg.lattice_constant <= 0.0:
        raise ValueError("lattice_constant must be positive")
    if cfg.muffin_tin_radius <= 0.0:
        raise ValueError("muffin_tin_radius must be positive")
    if cfg.muffin_tin_radius >= 0.5 * cfg.lattice_constant:
        raise ValueError("muffin_tin_radius must be smaller than half lattice_constant")
    if cfg.linearization_delta_e <= 0.0:
        raise ValueError("linearization_delta_e must be positive")
    if cfg.x_points < 401 or cfg.radial_points < 200:
        raise ValueError("grid is too coarse for a stable demo")


def site_potential(r: np.ndarray, depth: float, sigma: float) -> np.ndarray:
    return -depth * np.exp(-(r / sigma) ** 2)


def solve_radial_state(r: np.ndarray, energy: float, cfg: LMTOConfig) -> tuple[np.ndarray, np.ndarray]:
    """Solve 1D radial-like equation in [0, R_mt] with even parity init.

    Equation: -1/2 u''(r) + V(r) u(r) = E u(r), u(0)=1, u'(0)=0
    """

    def ode(rv: float, y: np.ndarray) -> np.ndarray:
        v = float(site_potential(np.array([rv]), cfg.site_potential_depth, cfg.site_potential_sigma)[0])
        # y[0] = u, y[1] = u'
        return np.array([y[1], 2.0 * (v - energy) * y[0]], dtype=float)

    y0 = np.array([1.0, 0.0], dtype=float)
    sol = solve_ivp(ode, (r[0], r[-1]), y0, t_eval=r, rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f"radial ODE solver failed: {sol.message}")
    return sol.y[0], sol.y[1]


def solve_linearized_radial_set(cfg: LMTOConfig) -> dict[str, np.ndarray]:
    r = np.linspace(0.0, cfg.muffin_tin_radius, cfg.radial_points)
    e0 = cfg.linearization_energy
    de = cfg.linearization_delta_e

    u0, up0 = solve_radial_state(r, e0, cfg)
    u_plus, up_plus = solve_radial_state(r, e0 + de, cfg)
    u_minus, up_minus = solve_radial_state(r, e0 - de, cfg)

    u_dot = (u_plus - u_minus) / (2.0 * de)
    up_dot = (up_plus - up_minus) / (2.0 * de)

    return {
        "r": r,
        "u": u0,
        "u_p": up0,
        "u_dot": u_dot,
        "u_dot_p": up_dot,
    }


def normalize_function(f: np.ndarray, x: np.ndarray) -> np.ndarray:
    norm_sq = float(np.trapezoid(f * f, x))
    if not np.isfinite(norm_sq) or norm_sq <= 0.0:
        raise ValueError("invalid norm during basis normalization")
    return f / np.sqrt(norm_sq)


def make_local_orbital(
    r_abs: np.ndarray,
    radial_r: np.ndarray,
    core_vals: np.ndarray,
    core_derivs: np.ndarray,
    r_mt: float,
    tail_decay_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise orbital: radial core + exponential tail with C1 matching."""
    core_interp = np.interp(np.clip(r_abs, 0.0, r_mt), radial_r, core_vals)
    deriv_interp = np.interp(np.clip(r_abs, 0.0, r_mt), radial_r, core_derivs)

    v_r = float(np.interp(r_mt, radial_r, core_vals))
    dv_r = float(np.interp(r_mt, radial_r, core_derivs))

    if abs(v_r) < 1e-10:
        kappa = tail_decay_min
    else:
        kappa = max(tail_decay_min, -dv_r / v_r)

    outside = np.exp(-kappa * (r_abs - r_mt))
    tail_vals = v_r * outside
    tail_dr = -kappa * tail_vals

    vals = core_interp.copy()
    dr = deriv_interp.copy()

    outer_mask = r_abs > r_mt
    vals[outer_mask] = tail_vals[outer_mask]
    dr[outer_mask] = tail_dr[outer_mask]
    return vals, dr


def build_real_space(cfg: LMTOConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    span = cfg.lattice_constant * (cfg.n_atoms - 1)
    margin = 1.8 * cfg.muffin_tin_radius
    x_min = -margin
    x_max = span + margin

    x = np.linspace(x_min, x_max, cfg.x_points)
    atom_positions = np.arange(cfg.n_atoms, dtype=float) * cfg.lattice_constant

    v = np.zeros_like(x)
    for pos in atom_positions:
        r = np.abs(x - pos)
        v += site_potential(r, cfg.site_potential_depth, cfg.site_potential_sigma)

    return x, atom_positions, v


def build_lmto_basis(
    cfg: LMTOConfig,
    x: np.ndarray,
    atom_positions: np.ndarray,
    radial: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    phi_list: list[np.ndarray] = []
    phidot_list: list[np.ndarray] = []

    for pos in atom_positions:
        r_abs = np.abs(x - pos)

        phi, _ = make_local_orbital(
            r_abs=r_abs,
            radial_r=radial["r"],
            core_vals=radial["u"],
            core_derivs=radial["u_p"],
            r_mt=cfg.muffin_tin_radius,
            tail_decay_min=cfg.tail_decay_min,
        )
        phidot, _ = make_local_orbital(
            r_abs=r_abs,
            radial_r=radial["r"],
            core_vals=radial["u_dot"],
            core_derivs=radial["u_dot_p"],
            r_mt=cfg.muffin_tin_radius,
            tail_decay_min=cfg.tail_decay_min,
        )

        phi_list.append(normalize_function(phi, x))
        phidot_list.append(normalize_function(phidot, x))

    phi_mat = np.vstack(phi_list)
    phidot_mat = np.vstack(phidot_list)
    return {"phi": phi_mat, "phidot": phidot_mat}


def spatial_derivative(arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(arr, x, axis=1, edge_order=2)


def assemble_hs_matrices(basis: np.ndarray, dbasis: np.ndarray, v: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_basis = basis.shape[0]
    h = np.zeros((n_basis, n_basis), dtype=float)
    s = np.zeros((n_basis, n_basis), dtype=float)

    for i in range(n_basis):
        for j in range(i, n_basis):
            sij = float(np.trapezoid(basis[i] * basis[j], x))
            hij = float(np.trapezoid(0.5 * dbasis[i] * dbasis[j] + v * basis[i] * basis[j], x))
            s[i, j] = s[j, i] = sij
            h[i, j] = h[j, i] = hij

    return h, s


def solve_generalized(h: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_eval = np.linalg.eigvalsh(s)
    min_eval = float(np.min(s_eval))
    if min_eval <= 1e-9:
        raise RuntimeError(f"overlap matrix is near-singular, min eig = {min_eval:.3e}")

    evals, evecs = eigh(h, s)
    return evals, evecs


def finite_difference_reference(cfg: LMTOConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    span = cfg.lattice_constant * (cfg.n_atoms - 1)
    margin = 1.8 * cfg.muffin_tin_radius
    x_min = -margin
    x_max = span + margin

    x = np.linspace(x_min, x_max, cfg.reference_points)
    dx = x[1] - x[0]

    atom_positions = np.arange(cfg.n_atoms, dtype=float) * cfg.lattice_constant
    v = np.zeros_like(x)
    for pos in atom_positions:
        r = np.abs(x - pos)
        v += site_potential(r, cfg.site_potential_depth, cfg.site_potential_sigma)

    # Dirichlet boundaries: use interior points only.
    v_in = v[1:-1]
    n = v_in.size
    main = np.full(n, 1.0 / dx**2, dtype=float) + v_in
    off = np.full(n - 1, -0.5 / dx**2, dtype=float)

    evals, evecs = eigh(
        np.diag(main) + np.diag(off, 1) + np.diag(off, -1),
        eigvals_only=False,
        subset_by_index=(0, min(cfg.n_reference_levels, n) - 1),
        check_finite=False,
    )

    full_vecs = np.zeros((x.size, evecs.shape[1]), dtype=float)
    full_vecs[1:-1, :] = evecs
    for i in range(full_vecs.shape[1]):
        norm = np.sqrt(np.trapezoid(full_vecs[:, i] ** 2, x))
        full_vecs[:, i] /= max(norm, 1e-12)

    return evals, full_vecs, x


def interpolate_reference_state(psi_ref: np.ndarray, x_ref: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    psi_interp = np.interp(x_target, x_ref, psi_ref)
    norm = np.sqrt(np.trapezoid(psi_interp**2, x_target))
    return psi_interp / max(norm, 1e-12)


def project_state_to_basis(psi: np.ndarray, basis: np.ndarray, s: np.ndarray, x: np.ndarray) -> np.ndarray:
    rhs = np.array([np.trapezoid(b * psi, x) for b in basis], dtype=float)
    coeff = np.linalg.solve(s, rhs)
    return coeff @ basis


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def build_lmto_blocks(
    phi: np.ndarray,
    phidot: np.ndarray,
    dphi: np.ndarray,
    dphidot: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    h_pp, s_pp = assemble_hs_matrices(phi, dphi, v, x)
    h_dd, s_dd = assemble_hs_matrices(phidot, dphidot, v, x)

    n = phi.shape[0]
    h_pd = np.zeros((n, n), dtype=float)
    s_pd = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            s_pd[i, j] = float(np.trapezoid(phi[i] * phidot[j], x))
            h_pd[i, j] = float(np.trapezoid(0.5 * dphi[i] * dphidot[j] + v * phi[i] * phidot[j], x))

    return {
        "h_pp": h_pp,
        "h_pd": h_pd,
        "h_dd": h_dd,
        "s_pp": s_pp,
        "s_pd": s_pd,
        "s_dd": s_dd,
    }


def assemble_lmto_from_alpha(blocks: dict[str, np.ndarray], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    h_pp = blocks["h_pp"]
    h_pd = blocks["h_pd"]
    h_dd = blocks["h_dd"]
    s_pp = blocks["s_pp"]
    s_pd = blocks["s_pd"]
    s_dd = blocks["s_dd"]

    top_h = np.hstack([h_pp, alpha * h_pd])
    bottom_h = np.hstack([alpha * h_pd.T, (alpha**2) * h_dd])
    h = np.vstack([top_h, bottom_h])

    top_s = np.hstack([s_pp, alpha * s_pd])
    bottom_s = np.hstack([alpha * s_pd.T, (alpha**2) * s_dd])
    s = np.vstack([top_s, bottom_s])
    return h, s


def torch_optimize_alpha(
    blocks: dict[str, np.ndarray],
    reference_eigs: np.ndarray,
    n_fit: int,
    steps: int,
    lr: float,
) -> tuple[float, float]:
    """Fit alpha scaling for phi_dot contribution using differentiable Rayleigh-Ritz."""

    h_pp = torch.tensor(blocks["h_pp"], dtype=torch.float64)
    h_pd = torch.tensor(blocks["h_pd"], dtype=torch.float64)
    h_dd = torch.tensor(blocks["h_dd"], dtype=torch.float64)
    s_pp = torch.tensor(blocks["s_pp"], dtype=torch.float64)
    s_pd = torch.tensor(blocks["s_pd"], dtype=torch.float64)
    s_dd = torch.tensor(blocks["s_dd"], dtype=torch.float64)

    target = torch.tensor(reference_eigs[:n_fit], dtype=torch.float64)

    raw = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
    opt = torch.optim.Adam([raw], lr=lr)

    best_loss = float("inf")
    best_alpha = 1.0

    for _ in range(steps):
        opt.zero_grad()
        alpha = torch.nn.functional.softplus(raw) + 1e-3

        top_h = torch.cat([h_pp, alpha * h_pd], dim=1)
        bottom_h = torch.cat([alpha * h_pd.T, (alpha**2) * h_dd], dim=1)
        h = torch.cat([top_h, bottom_h], dim=0)

        top_s = torch.cat([s_pp, alpha * s_pd], dim=1)
        bottom_s = torch.cat([alpha * s_pd.T, (alpha**2) * s_dd], dim=1)
        s = torch.cat([top_s, bottom_s], dim=0)

        jitter = 1e-8
        s = s + jitter * torch.eye(s.shape[0], dtype=torch.float64)

        chol = torch.linalg.cholesky(s)
        chol_inv = torch.linalg.inv(chol)
        std_h = chol_inv @ h @ chol_inv.T
        evals = torch.linalg.eigvalsh(std_h)
        pred = evals[:n_fit]

        loss = torch.mean((pred - target) ** 2) + 1e-4 * (alpha - 1.0) ** 2
        loss.backward()
        opt.step()

        cur_loss = float(loss.detach().cpu().item())
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_alpha = float(alpha.detach().cpu().item())

    return best_alpha, best_loss


def run_lmto_mvp(cfg: LMTOConfig) -> dict[str, object]:
    validate_config(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    x, atom_positions, v = build_real_space(cfg)
    radial = solve_linearized_radial_set(cfg)
    basis_dict = build_lmto_basis(cfg, x, atom_positions, radial)

    phi = basis_dict["phi"]
    phidot = basis_dict["phidot"]
    dphi = spatial_derivative(phi, x)
    dphidot = spatial_derivative(phidot, x)

    h_tb, s_tb = assemble_hs_matrices(phi, dphi, v, x)
    e_tb, vec_tb = solve_generalized(h_tb, s_tb)

    blocks = build_lmto_blocks(phi, phidot, dphi, dphidot, v, x)

    h_lmto_base, s_lmto_base = assemble_lmto_from_alpha(blocks, alpha=1.0)
    e_lmto_base, vec_lmto_base = solve_generalized(h_lmto_base, s_lmto_base)

    e_ref, psi_ref_all, x_ref = finite_difference_reference(cfg)
    # Compare only the physically matched low-energy window.
    # MT_only has n_atoms basis states, so we evaluate the shared low-energy sector.
    n_fit = min(cfg.n_atoms, cfg.n_reference_levels, e_ref.size, e_lmto_base.size, e_tb.size)

    alpha_opt, torch_loss = torch_optimize_alpha(
        blocks=blocks,
        reference_eigs=e_ref,
        n_fit=n_fit,
        steps=cfg.torch_steps,
        lr=cfg.torch_lr,
    )

    h_lmto_opt, s_lmto_opt = assemble_lmto_from_alpha(blocks, alpha=alpha_opt)
    e_lmto_opt, vec_lmto_opt = solve_generalized(h_lmto_opt, s_lmto_opt)

    n_compare_tb = n_fit
    n_compare_lmto = n_fit

    rmse_tb = np.sqrt(mean_squared_error(e_ref[:n_compare_tb], e_tb[:n_compare_tb]))
    rmse_lmto = np.sqrt(mean_squared_error(e_ref[:n_compare_lmto], e_lmto_base[:n_compare_lmto]))
    rmse_lmto_opt = np.sqrt(mean_squared_error(e_ref[:n_compare_lmto], e_lmto_opt[:n_compare_lmto]))

    psi_ref0_on_x = interpolate_reference_state(psi_ref_all[:, 0], x_ref, x)
    core_mask = np.zeros_like(x, dtype=bool)
    for pos in atom_positions:
        core_mask |= np.abs(x - pos) <= cfg.muffin_tin_radius

    psi_tb0 = project_state_to_basis(psi_ref0_on_x, phi, s_tb, x)
    lmto_basis_base = np.vstack([phi, phidot])
    lmto_basis_opt = np.vstack([phi, alpha_opt * phidot])

    psi_lmto0 = project_state_to_basis(psi_ref0_on_x, lmto_basis_base, s_lmto_base, x)
    psi_lmto0_opt = project_state_to_basis(psi_ref0_on_x, lmto_basis_opt, s_lmto_opt, x)

    # Sign alignment for fair wavefunction error comparison.
    def align_sign(psi: np.ndarray, ref: np.ndarray) -> np.ndarray:
        overlap = float(np.trapezoid(psi * ref, x))
        return psi if overlap >= 0.0 else -psi

    psi_tb0 = align_sign(psi_tb0, psi_ref0_on_x)
    psi_lmto0 = align_sign(psi_lmto0, psi_ref0_on_x)
    psi_lmto0_opt = align_sign(psi_lmto0_opt, psi_ref0_on_x)

    table = pd.DataFrame(
        [
            {
                "model": "MT_only",
                "basis_size": int(phi.shape[0]),
                "alpha": 0.0,
                "s_min_eig": float(np.min(np.linalg.eigvalsh(s_tb))),
                "s_condition": float(np.linalg.cond(s_tb)),
                "eig_rmse_vs_ref": float(rmse_tb),
                "wf_rmse_core": rmse(psi_tb0[core_mask], psi_ref0_on_x[core_mask]),
                "wf_rmse_global": rmse(psi_tb0, psi_ref0_on_x),
            },
            {
                "model": "LMTO_alpha1",
                "basis_size": int(lmto_basis_base.shape[0]),
                "alpha": 1.0,
                "s_min_eig": float(np.min(np.linalg.eigvalsh(s_lmto_base))),
                "s_condition": float(np.linalg.cond(s_lmto_base)),
                "eig_rmse_vs_ref": float(rmse_lmto),
                "wf_rmse_core": rmse(psi_lmto0[core_mask], psi_ref0_on_x[core_mask]),
                "wf_rmse_global": rmse(psi_lmto0, psi_ref0_on_x),
            },
            {
                "model": "LMTO_alpha_opt",
                "basis_size": int(lmto_basis_opt.shape[0]),
                "alpha": float(alpha_opt),
                "s_min_eig": float(np.min(np.linalg.eigvalsh(s_lmto_opt))),
                "s_condition": float(np.linalg.cond(s_lmto_opt)),
                "eig_rmse_vs_ref": float(rmse_lmto_opt),
                "wf_rmse_core": rmse(psi_lmto0_opt[core_mask], psi_ref0_on_x[core_mask]),
                "wf_rmse_global": rmse(psi_lmto0_opt, psi_ref0_on_x),
            },
        ]
    )

    return {
        "config": cfg,
        "x": x,
        "atom_positions": atom_positions,
        "potential": v,
        "reference_eigs": e_ref,
        "tb_eigs": e_tb,
        "lmto_eigs": e_lmto_base,
        "lmto_opt_eigs": e_lmto_opt,
        "alpha_opt": alpha_opt,
        "torch_loss": torch_loss,
        "table": table,
        "core_mask": core_mask,
    }


def run_checks(result: dict[str, object]) -> None:
    table = result["table"]
    assert isinstance(table, pd.DataFrame)

    mt_row = table.loc[table["model"] == "MT_only"].iloc[0]
    lmto_row = table.loc[table["model"] == "LMTO_alpha1"].iloc[0]
    lmto_opt_row = table.loc[table["model"] == "LMTO_alpha_opt"].iloc[0]

    if not np.isfinite(table[["eig_rmse_vs_ref", "wf_rmse_core", "wf_rmse_global"]].to_numpy()).all():
        raise AssertionError("non-finite metrics detected")

    if float(lmto_row["eig_rmse_vs_ref"]) >= float(mt_row["eig_rmse_vs_ref"]):
        raise AssertionError("LMTO(alpha=1) should improve eig RMSE over MT-only basis")

    if float(lmto_row["wf_rmse_core"]) >= float(mt_row["wf_rmse_core"]):
        raise AssertionError("LMTO(alpha=1) should improve core wavefunction RMSE over MT-only basis")

    if float(lmto_opt_row["eig_rmse_vs_ref"]) > float(lmto_row["eig_rmse_vs_ref"]) + 1e-6:
        raise AssertionError("optimized alpha should not worsen eig RMSE compared with alpha=1")

    for name in ("MT_only", "LMTO_alpha1", "LMTO_alpha_opt"):
        row = table.loc[table["model"] == name].iloc[0]
        if float(row["s_min_eig"]) <= 1e-9:
            raise AssertionError(f"{name} overlap matrix is near singular")


def main() -> None:
    cfg = LMTOConfig()
    result = run_lmto_mvp(cfg)

    table = result["table"]
    ref = result["reference_eigs"]
    tb = result["tb_eigs"]
    lmto = result["lmto_eigs"]
    lmto_opt = result["lmto_opt_eigs"]

    n_show = min(6, ref.size, tb.size, lmto.size, lmto_opt.size)
    levels = pd.DataFrame(
        {
            "level": np.arange(n_show),
            "E_ref": ref[:n_show],
            "E_MT_only": tb[:n_show],
            "E_LMTO_alpha1": lmto[:n_show],
            "E_LMTO_alpha_opt": lmto_opt[:n_show],
        }
    )

    print("=== LMTO toy spectrum comparison ===")
    print(levels.to_string(index=False, float_format=lambda z: f"{z: .6f}"))

    print("\n=== Quality metrics ===")
    print(table.to_string(index=False, float_format=lambda z: f"{z: .6e}"))

    print(f"\nOptimized alpha (torch): {result['alpha_opt']:.6f}")
    print(f"Torch objective value: {result['torch_loss']:.6e}")

    run_checks(result)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
