"""Minimal runnable MVP for Moller-Plesset perturbation theory (MP2).

This script implements a fully explicit, educational RHF -> MP2 pipeline
on a small synthetic closed-shell Hamiltonian:
1) Build one-electron and two-electron integrals.
2) Run restricted Hartree-Fock self-consistent field (SCF).
3) Compute canonical-orbital MP2 correlation energy.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MP2Config:
    n_orbitals: int = 6
    n_electrons: int = 4
    n_aux: int = 8
    random_seed: int = 42
    max_iter: int = 120
    damping: float = 0.45
    tol_energy: float = 1e-8
    tol_density: float = 1e-6


def build_toy_hamiltonian(cfg: MP2Config) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct a small synthetic molecular Hamiltonian in an orthonormal AO basis.

    Returns:
    - h_core[p, q]
    - eri[p, q, r, s] in chemists' notation (pq|rs)
    - nuclear repulsion constant E_nuc
    """

    rng = np.random.default_rng(cfg.random_seed)
    n = cfg.n_orbitals

    # One-electron core Hamiltonian: diagonal energy ladder + weak couplings.
    h_core = rng.normal(0.0, 0.03, size=(n, n))
    h_core = 0.5 * (h_core + h_core.T)
    h_core += np.diag(-2.20 - 0.55 * np.arange(n, dtype=float))

    # Cholesky-like factor for two-electron integrals to ensure positivity.
    factors = rng.normal(0.0, 0.08, size=(n, n, cfg.n_aux))
    factors = 0.5 * (factors + factors.transpose(1, 0, 2))

    eri = np.einsum("pqL,rsL->pqrs", factors, factors, optimize=True)

    # Enforce major/minor symmetries explicitly.
    eri = 0.5 * (eri + eri.transpose(1, 0, 2, 3))
    eri = 0.5 * (eri + eri.transpose(0, 1, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))

    e_nuc = 1.35
    return h_core, eri, e_nuc


def build_fock(h_core: np.ndarray, eri: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Restricted Hartree-Fock Fock matrix in an orthonormal AO basis."""

    coulomb = np.einsum("rs,pqrs->pq", density, eri, optimize=True)
    exchange = np.einsum("rs,prqs->pq", density, eri, optimize=True)
    return h_core + coulomb - 0.5 * exchange


def rhf_scf(
    h_core: np.ndarray,
    eri: np.ndarray,
    e_nuc: float,
    cfg: MP2Config,
) -> dict[str, object]:
    """Run restricted HF SCF and return canonical orbitals and energies."""

    if cfg.n_electrons % 2 != 0:
        raise ValueError("This MVP assumes a closed-shell system: n_electrons must be even.")
    if cfg.n_electrons <= 0 or cfg.n_electrons > 2 * cfg.n_orbitals:
        raise ValueError("n_electrons must satisfy 0 < n_electrons <= 2*n_orbitals.")

    n_occ = cfg.n_electrons // 2
    n = cfg.n_orbitals

    density = np.zeros((n, n), dtype=float)
    e_prev = np.inf
    history: list[dict[str, float]] = []
    converged = False

    for iteration in range(1, cfg.max_iter + 1):
        fock = build_fock(h_core, eri, density)
        orbital_energies, coeff = np.linalg.eigh(fock)

        c_occ = coeff[:, :n_occ]
        density_new = 2.0 * (c_occ @ c_occ.T)
        density_prev = density.copy()
        density = (1.0 - cfg.damping) * density + cfg.damping * density_new

        fock_mixed = build_fock(h_core, eri, density)
        e_elec = 0.5 * float(np.einsum("pq,pq->", density, h_core + fock_mixed, optimize=True))
        e_total = e_elec + e_nuc

        d_energy = abs(e_total - e_prev)
        d_density = float(np.linalg.norm(density - density_prev))

        history.append(
            {
                "iter": float(iteration),
                "E_total": e_total,
                "dE": d_energy,
                "dP": d_density,
                "HOMO": float(orbital_energies[n_occ - 1]),
                "LUMO": float(orbital_energies[n_occ]),
                "gap": float(orbital_energies[n_occ] - orbital_energies[n_occ - 1]),
                "trace_P": float(np.trace(density)),
            }
        )

        if d_energy < cfg.tol_energy and d_density < cfg.tol_density:
            converged = True
            break

        e_prev = e_total

    if not history:
        raise RuntimeError("SCF history is empty.")

    # Re-diagonalize final Fock for clean canonical orbitals from final density.
    fock_final = build_fock(h_core, eri, density)
    orbital_energies, coeff = np.linalg.eigh(fock_final)

    e_elec = 0.5 * float(np.einsum("pq,pq->", density, h_core + fock_final, optimize=True))
    e_hf = e_elec + e_nuc

    return {
        "converged": converged,
        "history": pd.DataFrame(history),
        "density": density,
        "coeff": coeff,
        "orbital_energies": orbital_energies,
        "E_HF": e_hf,
        "n_occ": n_occ,
    }


def transform_eri_to_mo(eri_ao: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """AO -> MO four-index transform: (pq|rs) -> (ij|ab)-capable MO tensor."""

    return np.einsum("pi,qj,rk,sl,pqrs->ijkl", coeff, coeff, coeff, coeff, eri_ao, optimize=True)


def mp2_correlation_energy(
    eri_mo: np.ndarray,
    orbital_energies: np.ndarray,
    n_occ: int,
) -> tuple[float, pd.DataFrame, np.ndarray]:
    """Compute closed-shell MP2 correlation energy using canonical RHF orbitals.

    Formula used:
    E_MP2 = sum_{ijab} (2*(ij|ab) - (ij|ba))*(ij|ab) / (eps_i + eps_j - eps_a - eps_b)
    """

    g_ijab = eri_mo[:n_occ, :n_occ, n_occ:, n_occ:]
    g_ijba = g_ijab.transpose(0, 1, 3, 2)

    eps_occ = orbital_energies[:n_occ]
    eps_vir = orbital_energies[n_occ:]

    denom = (
        eps_occ[:, None, None, None]
        + eps_occ[None, :, None, None]
        - eps_vir[None, None, :, None]
        - eps_vir[None, None, None, :]
    )

    contrib = g_ijab * (2.0 * g_ijab - g_ijba) / denom
    e_mp2 = float(np.sum(contrib))

    top_rows: list[dict[str, float]] = []
    flat_idx = np.argsort(np.abs(contrib).ravel())[::-1][:8]
    shape = contrib.shape
    for linear_idx in flat_idx:
        i, j, a_rel, b_rel = np.unravel_index(linear_idx, shape)
        top_rows.append(
            {
                "i": float(i),
                "j": float(j),
                "a": float(a_rel + n_occ),
                "b": float(b_rel + n_occ),
                "denom": float(denom[i, j, a_rel, b_rel]),
                "contrib": float(contrib[i, j, a_rel, b_rel]),
            }
        )

    return e_mp2, pd.DataFrame(top_rows), denom


def run_mp2_pipeline(cfg: MP2Config) -> dict[str, object]:
    h_core, eri_ao, e_nuc = build_toy_hamiltonian(cfg)
    scf = rhf_scf(h_core, eri_ao, e_nuc, cfg)

    eri_mo = transform_eri_to_mo(eri_ao, scf["coeff"])
    e_mp2, top_contrib, denom = mp2_correlation_energy(
        eri_mo=eri_mo,
        orbital_energies=scf["orbital_energies"],
        n_occ=scf["n_occ"],
    )

    e_hf = float(scf["E_HF"])
    e_total_mp2 = e_hf + e_mp2

    final_hist = scf["history"].iloc[-1]
    result = {
        "config": cfg,
        "scf_converged": bool(scf["converged"]),
        "scf_iterations": int(len(scf["history"])),
        "scf_history": scf["history"],
        "E_HF": e_hf,
        "E_MP2_corr": float(e_mp2),
        "E_MP2_total": float(e_total_mp2),
        "orbital_energies": scf["orbital_energies"],
        "n_occ": int(scf["n_occ"]),
        "gap": float(final_hist["gap"]),
        "trace_P": float(final_hist["trace_P"]),
        "top_contrib": top_contrib,
        "denom_min": float(np.min(denom)),
        "denom_max": float(np.max(denom)),
        "all_denom_negative": bool(np.all(denom < 0.0)),
    }
    return result


def validate_results(res: dict[str, object], cfg: MP2Config) -> list[tuple[str, bool]]:
    checks: list[tuple[str, bool]] = []

    checks.append(("SCF converged", bool(res["scf_converged"])))
    checks.append(("Density trace matches n_electrons", abs(float(res["trace_P"]) - cfg.n_electrons) < 1e-4))
    checks.append(("HOMO-LUMO gap positive", float(res["gap"]) > 1e-6))
    checks.append(("All MP2 denominators negative", bool(res["all_denom_negative"])))
    checks.append(("MP2 correction finite", np.isfinite(float(res["E_MP2_corr"])) ))
    checks.append(("MP2 correction lowers HF energy", float(res["E_MP2_corr"]) < 0.0))

    return checks


def main() -> None:
    cfg = MP2Config(
        n_orbitals=6,
        n_electrons=4,
        n_aux=8,
        random_seed=42,
        max_iter=120,
        damping=0.45,
        tol_energy=1e-8,
        tol_density=1e-6,
    )

    results = run_mp2_pipeline(cfg)
    checks = validate_results(results, cfg)

    print("=== SCF Iteration Tail (last 10) ===")
    print(results["scf_history"].tail(10).to_string(index=False))

    summary = pd.DataFrame(
        [
            {
                "E_HF": results["E_HF"],
                "E_MP2_corr": results["E_MP2_corr"],
                "E_MP2_total": results["E_MP2_total"],
                "gap": results["gap"],
                "trace_P": results["trace_P"],
                "denom_min": results["denom_min"],
                "denom_max": results["denom_max"],
            }
        ]
    )
    print("\n=== Energy Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Largest |MP2 term contributions| ===")
    print(results["top_contrib"].to_string(index=False))

    print("\n=== Validation Checks ===")
    all_pass = True
    for name, ok in checks:
        state = "PASS" if ok else "FAIL"
        print(f"[{state}] {name}")
        all_pass = all_pass and ok

    print(f"\nValidation: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
