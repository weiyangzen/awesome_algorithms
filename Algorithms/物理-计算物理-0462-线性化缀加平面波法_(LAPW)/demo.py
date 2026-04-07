"""Minimal runnable MVP for LAPW (Linearized Augmented Plane Wave), PHYS-0441.

This demo implements a pedagogical 1D analogue of LAPW:
- Interstitial region: plane-wave-like cosine basis.
- Muffin-tin region: linearized radial basis u(r, E_l) + u_dot(r, E_l).
- Boundary matching: determine (A_G, B_G) per basis by value/derivative continuity.
- Solve generalized eigenproblem H c = E S c.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import eigh


@dataclass(frozen=True)
class LAPWConfig:
    """Configuration for a toy 1D LAPW-like band calculation."""

    cell_length: float = 10.0
    muffin_tin_radius: float = 2.0
    linearization_energy: float = -0.45
    energy_step: float = 2e-3
    potential_depth: float = 1.2
    potential_sigma: float = 0.8
    integration_points: int = 6001
    radial_points: int = 1200
    n_max_list: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    overlap_tol: float = 1e-8


def validate_config(cfg: LAPWConfig) -> None:
    if cfg.cell_length <= 0.0:
        raise ValueError("cell_length must be positive")
    if cfg.muffin_tin_radius <= 0.0:
        raise ValueError("muffin_tin_radius must be positive")
    if 2.0 * cfg.muffin_tin_radius >= cfg.cell_length:
        raise ValueError("Need 2*R_mt < cell_length to have interstitial region")
    if cfg.energy_step <= 0.0:
        raise ValueError("energy_step must be positive")
    if cfg.integration_points < 101 or cfg.radial_points < 101:
        raise ValueError("integration grid is too coarse")


def muffin_tin_potential(r: np.ndarray, depth: float, sigma: float) -> np.ndarray:
    """Smooth attractive potential in muffin-tin: V(r) = -V0 * exp(-(r/sigma)^2)."""
    return -depth * np.exp(-(r / sigma) ** 2)


def solve_radial_profile(r: np.ndarray, energy: float, depth: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve -0.5*u'' + V(r)u = E*u with even-parity initial condition at r=0."""

    def ode(rv: float, y: np.ndarray) -> np.ndarray:
        v = float(muffin_tin_potential(np.array([rv]), depth, sigma)[0])
        # y[0] = u, y[1] = u'
        return np.array([y[1], 2.0 * (v - energy) * y[0]], dtype=float)

    y0 = np.array([1.0, 0.0], dtype=float)
    sol = solve_ivp(ode, (r[0], r[-1]), y0, t_eval=r, rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f"radial ODE failed: {sol.message}")

    u = sol.y[0]
    up = sol.y[1]
    return u, up


def solve_linearized_radial_set(cfg: LAPWConfig) -> dict[str, np.ndarray]:
    r = np.linspace(0.0, cfg.muffin_tin_radius, cfg.radial_points)

    e0 = cfg.linearization_energy
    de = cfg.energy_step

    u0, u0p = solve_radial_profile(r, e0, cfg.potential_depth, cfg.potential_sigma)
    up, upp = solve_radial_profile(r, e0 + de, cfg.potential_depth, cfg.potential_sigma)
    um, ump = solve_radial_profile(r, e0 - de, cfg.potential_depth, cfg.potential_sigma)

    u_dot = (up - um) / (2.0 * de)
    u_dot_p = (upp - ump) / (2.0 * de)

    return {
        "r": r,
        "u": u0,
        "u_p": u0p,
        "u_dot": u_dot,
        "u_dot_p": u_dot_p,
    }


def boundary_match_coefficients(q: float, cfg: LAPWConfig, radial: dict[str, np.ndarray]) -> tuple[float, float]:
    """Find A, B from continuity at r=R for f=A*u + B*u_dot matching cos(qx)."""
    r = radial["r"]
    u = radial["u"]
    u_p = radial["u_p"]
    u_dot = radial["u_dot"]
    u_dot_p = radial["u_dot_p"]

    rmt = cfg.muffin_tin_radius
    u_r = float(np.interp(rmt, r, u))
    u_p_r = float(np.interp(rmt, r, u_p))
    ud_r = float(np.interp(rmt, r, u_dot))
    ud_p_r = float(np.interp(rmt, r, u_dot_p))

    rhs_val = np.cos(q * rmt)
    rhs_der = -q * np.sin(q * rmt)

    mat = np.array([[u_r, ud_r], [u_p_r, ud_p_r]], dtype=float)
    rhs = np.array([rhs_val, rhs_der], dtype=float)

    cond = np.linalg.cond(mat)
    if cond > 1e12:
        raise RuntimeError(f"Boundary matching matrix is ill-conditioned, cond={cond:.3e}")

    a, b = np.linalg.solve(mat, rhs)
    return float(a), float(b)


def build_cell_grid(cfg: LAPWConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-0.5 * cfg.cell_length, 0.5 * cfg.cell_length, cfg.integration_points)
    dx = x[1] - x[0]
    w = np.full_like(x, dx)
    w[0] *= 0.5
    w[-1] *= 0.5
    r_abs = np.abs(x)
    mt_mask = r_abs <= cfg.muffin_tin_radius
    return x, w, r_abs, mt_mask


def build_piecewise_basis(
    n_max: int,
    cfg: LAPWConfig,
    radial: dict[str, np.ndarray],
) -> dict[str, object]:
    x, w, r_abs, mt_mask = build_cell_grid(cfg)
    r = radial["r"]

    q_values = 2.0 * np.pi * np.arange(0, n_max + 1, dtype=float) / cfg.cell_length
    n_basis = q_values.size

    phi = np.zeros((n_basis, x.size), dtype=float)
    dphi = np.zeros_like(phi)

    continuity_value = np.zeros(n_basis, dtype=float)
    continuity_derivative = np.zeros(n_basis, dtype=float)

    for i, q in enumerate(q_values):
        a_g, b_g = boundary_match_coefficients(float(q), cfg, radial)

        interstitial_val = np.cos(q * x)
        interstitial_der = -q * np.sin(q * x)

        # r-dependent linearized radial part in muffin-tin.
        u_r = np.interp(r_abs, r, radial["u"])
        ud_r = np.interp(r_abs, r, radial["u_dot"])
        up_r = np.interp(r_abs, r, radial["u_p"])
        udp_r = np.interp(r_abs, r, radial["u_dot_p"])

        mt_val = a_g * u_r + b_g * ud_r
        mt_dr = a_g * up_r + b_g * udp_r

        mt_dx = np.sign(x) * mt_dr
        mt_dx[np.abs(x) < 1e-15] = 0.0

        phi_i = interstitial_val.copy()
        dphi_i = interstitial_der.copy()
        phi_i[mt_mask] = mt_val[mt_mask]
        dphi_i[mt_mask] = mt_dx[mt_mask]

        phi[i] = phi_i
        dphi[i] = dphi_i

        rmt = cfg.muffin_tin_radius
        val_mt = float(a_g * np.interp(rmt, r, radial["u"]) + b_g * np.interp(rmt, r, radial["u_dot"]))
        der_mt = float(a_g * np.interp(rmt, r, radial["u_p"]) + b_g * np.interp(rmt, r, radial["u_dot_p"]))
        continuity_value[i] = val_mt - np.cos(q * rmt)
        continuity_derivative[i] = der_mt + q * np.sin(q * rmt)

    potential = np.zeros_like(x)
    potential[mt_mask] = muffin_tin_potential(r_abs[mt_mask], cfg.potential_depth, cfg.potential_sigma)

    return {
        "x": x,
        "w": w,
        "phi": phi,
        "dphi": dphi,
        "q_values": q_values,
        "potential": potential,
        "continuity_value": continuity_value,
        "continuity_derivative": continuity_derivative,
    }


def assemble_hs_matrices(phi: np.ndarray, dphi: np.ndarray, potential: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_basis = phi.shape[0]
    s = np.zeros((n_basis, n_basis), dtype=float)
    h = np.zeros((n_basis, n_basis), dtype=float)

    for i in range(n_basis):
        for j in range(i, n_basis):
            sij = float(np.sum(w * phi[i] * phi[j]))
            hij = float(np.sum(w * (0.5 * dphi[i] * dphi[j] + potential * phi[i] * phi[j])))
            s[i, j] = s[j, i] = sij
            h[i, j] = h[j, i] = hij

    return h, s


def solve_generalized_eigenproblem(h: np.ndarray, s: np.ndarray, overlap_tol: float) -> tuple[np.ndarray, np.ndarray, float]:
    s_eval = np.linalg.eigvalsh(s)
    min_overlap_eig = float(np.min(s_eval))
    if min_overlap_eig <= overlap_tol:
        raise RuntimeError(
            "Overlap matrix is not safely positive definite: "
            f"min eigenvalue={min_overlap_eig:.3e}, tol={overlap_tol:.1e}"
        )

    evals, evecs = eigh(h, s)
    return evals, evecs, min_overlap_eig


def generalized_residual_norm(h: np.ndarray, s: np.ndarray, evals: np.ndarray, evecs: np.ndarray, n_check: int = 3) -> float:
    n = min(n_check, evals.size)
    worst = 0.0
    for i in range(n):
        vec = evecs[:, i]
        res = h @ vec - evals[i] * (s @ vec)
        worst = max(worst, float(np.linalg.norm(res)))
    return worst


def finite_difference_reference_energy(cfg: LAPWConfig, n_points: int = 700) -> float:
    """Periodic finite-difference reference for the same 1D cell Hamiltonian."""
    x = np.linspace(-0.5 * cfg.cell_length, 0.5 * cfg.cell_length, n_points, endpoint=False)
    dx = cfg.cell_length / n_points

    r_abs = np.abs(x)
    v = np.zeros_like(x)
    mt_mask = r_abs <= cfg.muffin_tin_radius
    v[mt_mask] = muffin_tin_potential(r_abs[mt_mask], cfg.potential_depth, cfg.potential_sigma)

    main = np.full(n_points, 1.0 / dx**2, dtype=float) + v
    off = np.full(n_points - 1, -0.5 / dx**2, dtype=float)

    h = np.diag(main)
    h += np.diag(off, 1)
    h += np.diag(off, -1)
    h[0, -1] = -0.5 / dx**2
    h[-1, 0] = -0.5 / dx**2

    evals = eigh(h, eigvals_only=True, subset_by_index=(0, 0), check_finite=False)
    return float(evals[0])


def run_lapw_study(cfg: LAPWConfig) -> dict[str, object]:
    validate_config(cfg)
    radial = solve_linearized_radial_set(cfg)
    ref_e0 = finite_difference_reference_energy(cfg)

    records: list[dict[str, float]] = []
    best_data: dict[str, object] | None = None

    for n_max in cfg.n_max_list:
        basis = build_piecewise_basis(n_max, cfg, radial)
        h, s = assemble_hs_matrices(basis["phi"], basis["dphi"], basis["potential"], basis["w"])
        evals, evecs, smin = solve_generalized_eigenproblem(h, s, cfg.overlap_tol)

        e0 = float(evals[0])
        residual = generalized_residual_norm(h, s, evals, evecs, n_check=3)
        cont_val = float(np.max(np.abs(basis["continuity_value"])))
        cont_der = float(np.max(np.abs(basis["continuity_derivative"])))

        rec = {
            "n_max": float(n_max),
            "n_basis": float(n_max + 1),
            "E0_lapw": e0,
            "E1_lapw": float(evals[1]) if evals.size > 1 else np.nan,
            "E0_ref": ref_e0,
            "abs_err_ref": abs(e0 - ref_e0),
            "S_min_eig": smin,
            "bc_val_max": cont_val,
            "bc_der_max": cont_der,
            "residual_max": residual,
        }
        records.append(rec)

        if best_data is None or rec["abs_err_ref"] < best_data["record"]["abs_err_ref"]:
            best_data = {
                "record": rec,
                "h": h,
                "s": s,
                "evals": evals,
                "evecs": evecs,
                "basis": basis,
                "n_max": n_max,
            }

    if best_data is None:
        raise RuntimeError("No LAPW run produced data")

    return {
        "cfg": cfg,
        "radial": radial,
        "records": pd.DataFrame(records),
        "best": best_data,
        "reference_energy": ref_e0,
    }


def main() -> None:
    cfg = LAPWConfig()
    result = run_lapw_study(cfg)

    records: pd.DataFrame = result["records"]
    best = result["best"]
    best_rec = best["record"]

    pd.set_option("display.float_format", lambda v: f"{v:.8f}")

    checks = {
        "overlap matrix positive": bool(np.all(records["S_min_eig"] > cfg.overlap_tol)),
        "boundary value continuity < 1e-6": bool(np.max(records["bc_val_max"]) < 1e-6),
        "boundary derivative continuity < 1e-6": bool(np.max(records["bc_der_max"]) < 1e-6),
        "generalized residual < 1e-6": bool(np.max(records["residual_max"]) < 1e-6),
        "best ground-energy error < 0.15": bool(float(best_rec["abs_err_ref"]) < 0.15),
    }

    print("=== LAPW MVP (1D pedagogical analogue) | PHYS-0441 ===")
    print(
        f"cell_length={cfg.cell_length}, R_mt={cfg.muffin_tin_radius}, "
        f"E_l={cfg.linearization_energy}, dE={cfg.energy_step}, "
        f"V0={cfg.potential_depth}, sigma={cfg.potential_sigma}"
    )
    print(f"n_max sweep: {list(cfg.n_max_list)}")

    print("\nConvergence table:")
    print(records.to_string(index=False))

    summary = pd.DataFrame(
        {
            "quantity": [
                "best_n_max",
                "best_n_basis",
                "best_E0_lapw",
                "reference_E0_fd",
                "best_abs_err_ref",
                "best_S_min_eig",
                "best_bc_val_max",
                "best_bc_der_max",
                "best_residual_max",
            ],
            "value": [
                float(best_rec["n_max"]),
                float(best_rec["n_basis"]),
                float(best_rec["E0_lapw"]),
                float(best_rec["E0_ref"]),
                float(best_rec["abs_err_ref"]),
                float(best_rec["S_min_eig"]),
                float(best_rec["bc_val_max"]),
                float(best_rec["bc_der_max"]),
                float(best_rec["residual_max"]),
            ],
        }
    )

    print("\nBest configuration summary:")
    print(summary.to_string(index=False))

    print("\nThreshold checks:")
    for k, ok in checks.items():
        print(f"- {k}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
