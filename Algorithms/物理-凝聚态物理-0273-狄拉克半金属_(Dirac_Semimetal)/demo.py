"""Minimal runnable MVP for a lattice Dirac semimetal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

PI = float(np.pi)

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

TAU_X = SIGMA_X

GAMMA1 = np.kron(TAU_X, SIGMA_X)
GAMMA2 = np.kron(TAU_X, SIGMA_Y)
GAMMA3 = np.kron(TAU_X, SIGMA_Z)


@dataclass(frozen=True)
class DiracConfig:
    m0: float = 2.2
    velocity: float = 1.0
    grid_n: int = 41
    fit_radius: float = 0.18
    fit_samples: int = 1500
    dos_samples: int = 100000
    dos_bins: int = 36
    seed: int = 20260407
    node_gap_tolerance: float = 2e-5


def wrap_to_bz(k: np.ndarray | float) -> np.ndarray | float:
    return ((k + PI) % (2.0 * PI)) - PI


def mass_term(kx: np.ndarray | float, ky: np.ndarray | float, kz: np.ndarray | float, cfg: DiracConfig) -> np.ndarray | float:
    return cfg.m0 - np.cos(kx) - np.cos(ky) - np.cos(kz)


def gap_formula(kx: np.ndarray | float, ky: np.ndarray | float, kz: np.ndarray | float, cfg: DiracConfig) -> np.ndarray | float:
    term = np.sin(kx) ** 2 + np.sin(ky) ** 2 + mass_term(kx, ky, kz, cfg) ** 2
    return cfg.velocity * np.sqrt(term)


def hamiltonian_np(kx: float, ky: float, kz: float, cfg: DiracConfig) -> np.ndarray:
    d1 = np.sin(kx)
    d2 = np.sin(ky)
    d3 = mass_term(kx, ky, kz, cfg)
    return cfg.velocity * (d1 * GAMMA1 + d2 * GAMMA2 + d3 * GAMMA3)


def refine_node(seed: np.ndarray, cfg: DiracConfig) -> tuple[np.ndarray, float]:
    bounds = [(-PI, PI), (-PI, PI), (-PI, PI)]

    def objective(kvec: np.ndarray) -> float:
        kx, ky, kz = wrap_to_bz(kvec)
        return float(gap_formula(kx, ky, kz, cfg))

    result = minimize(objective, x0=seed, method="L-BFGS-B", bounds=bounds)
    refined_k = np.asarray(wrap_to_bz(result.x), dtype=float)
    refined_gap = float(objective(refined_k))
    return refined_k, refined_gap


def locate_nodes(cfg: DiracConfig) -> tuple[tuple[np.ndarray, np.ndarray], tuple[float, float], float]:
    k_grid = np.linspace(-PI, PI, cfg.grid_n)
    kx, ky, kz = np.meshgrid(k_grid, k_grid, k_grid, indexing="ij")
    gaps = gap_formula(kx, ky, kz, cfg)

    positive_mask = kz > 0.0
    negative_mask = kz < 0.0

    pos_index = int(np.argmin(np.where(positive_mask, gaps, np.inf)))
    neg_index = int(np.argmin(np.where(negative_mask, gaps, np.inf)))

    pos_seed = np.array([kx.ravel()[pos_index], ky.ravel()[pos_index], kz.ravel()[pos_index]], dtype=float)
    neg_seed = np.array([kx.ravel()[neg_index], ky.ravel()[neg_index], kz.ravel()[neg_index]], dtype=float)

    pos_node, pos_gap = refine_node(pos_seed, cfg)
    neg_node, neg_gap = refine_node(neg_seed, cfg)

    grid_min_gap = float(np.min(gaps))
    return (pos_node, neg_node), (pos_gap, neg_gap), grid_min_gap


def build_node_dataframe(nodes: Iterable[np.ndarray], node_gaps: Iterable[float], cfg: DiracConfig) -> pd.DataFrame:
    expected_k0 = float(np.arccos(np.clip(cfg.m0 - 2.0, -1.0, 1.0)))
    rows = []
    for idx, (kvec, gap) in enumerate(zip(nodes, node_gaps), start=1):
        rows.append(
            {
                "node_id": idx,
                "kx": float(kvec[0]),
                "ky": float(kvec[1]),
                "kz": float(kvec[2]),
                "gap": float(gap),
                "abs_kz_error_vs_k0": float(abs(abs(kvec[2]) - expected_k0)),
            }
        )
    return pd.DataFrame(rows)


def fit_linear_dispersion(node: np.ndarray, cfg: DiracConfig, rng: np.random.Generator) -> dict[str, float]:
    directions = rng.normal(size=(cfg.fit_samples, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.0, cfg.fit_radius, size=(cfg.fit_samples, 1))
    delta_k = directions * radii

    k_points = wrap_to_bz(node[None, :] + delta_k)
    energies = gap_formula(k_points[:, 0], k_points[:, 1], k_points[:, 2], cfg)

    x = np.linalg.norm(delta_k, axis=1).reshape(-1, 1)
    y = np.asarray(energies, dtype=float)

    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    pred = model.predict(x)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(model.score(x, y)),
        "rmse": rmse,
    }


def estimate_dos_curve(cfg: DiracConfig, rng: np.random.Generator) -> pd.DataFrame:
    k_points = rng.uniform(-PI, PI, size=(cfg.dos_samples, 3))
    energies = gap_formula(k_points[:, 0], k_points[:, 1], k_points[:, 2], cfg)

    e_max = float(np.quantile(energies, 0.95))
    counts, edges = np.histogram(energies, bins=cfg.dos_bins, range=(0.0, e_max), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    dos = counts / (counts.sum() * width)

    return pd.DataFrame({"energy": centers, "dos": dos, "counts": counts})


def torch_eigenvalue_consistency(nodes: Iterable[np.ndarray], cfg: DiracConfig, rng: np.random.Generator) -> float:
    max_diff = 0.0
    for node in nodes:
        for _ in range(3):
            perturb = rng.normal(scale=0.03, size=3)
            k_test = np.asarray(wrap_to_bz(node + perturb), dtype=float)
            h_np = hamiltonian_np(float(k_test[0]), float(k_test[1]), float(k_test[2]), cfg)
            eig_np = np.linalg.eigvalsh(h_np)

            h_torch = torch.tensor(h_np, dtype=torch.complex128)
            eig_torch = torch.linalg.eigvalsh(h_torch).cpu().numpy()
            max_diff = max(max_diff, float(np.max(np.abs(eig_np - eig_torch))))

    return max_diff


def run_sanity_checks(
    node_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    torch_diff: float,
    cfg: DiracConfig,
) -> None:
    gaps_ok = bool((node_df["gap"] < cfg.node_gap_tolerance).all())
    mirror_ok = bool(abs(float(node_df.loc[0, "kz"] + node_df.loc[1, "kz"])) < 3e-3)
    inplane_ok = bool((node_df["kx"].abs() < 3e-3).all() and (node_df["ky"].abs() < 3e-3).all())
    fit_ok = bool((fit_df["r2"] > 0.85).all())
    torch_ok = bool(torch_diff < 1e-10)

    assert gaps_ok, "Refined node gap is too large; did not locate near-zero-energy nodes."
    assert mirror_ok, "Two nodes are not approximately mirror partners in kz."
    assert inplane_ok, "Node kx/ky coordinates should stay close to 0 for this model."
    assert fit_ok, "Linear dispersion fit quality (R^2) is too low near the nodes."
    assert torch_ok, "NumPy/Torch eigenvalue mismatch is too large."


def main() -> None:
    cfg = DiracConfig()
    if not (1.0 < cfg.m0 < 3.0):
        raise ValueError("This MVP expects 1 < m0 < 3 to realize a Dirac semimetal phase.")

    rng = np.random.default_rng(cfg.seed)

    nodes, node_gaps, coarse_min_gap = locate_nodes(cfg)
    node_df = build_node_dataframe(nodes=nodes, node_gaps=node_gaps, cfg=cfg)

    fit_rows = []
    for idx, node in enumerate(nodes, start=1):
        metrics = fit_linear_dispersion(node=node, cfg=cfg, rng=rng)
        metrics["node_id"] = idx
        fit_rows.append(metrics)
    fit_df = pd.DataFrame(fit_rows)[["node_id", "slope", "intercept", "r2", "rmse"]]

    dos_df = estimate_dos_curve(cfg=cfg, rng=rng)
    torch_diff = torch_eigenvalue_consistency(nodes=nodes, cfg=cfg, rng=rng)

    run_sanity_checks(node_df=node_df, fit_df=fit_df, torch_diff=torch_diff, cfg=cfg)

    out_dir = Path(__file__).resolve().parent
    node_df.to_csv(out_dir / "nodes_summary.csv", index=False)
    fit_df.to_csv(out_dir / "dispersion_fit.csv", index=False)
    dos_df.to_csv(out_dir / "dos_curve.csv", index=False)

    expected_k0 = float(np.arccos(cfg.m0 - 2.0))

    print("=== Dirac Semimetal MVP ===")
    print(f"m0 = {cfg.m0:.3f}, velocity = {cfg.velocity:.3f}, expected |k0| = {expected_k0:.6f}")
    print(f"coarse-grid minimum gap = {coarse_min_gap:.6e}")

    print("\n[Refined Dirac nodes]")
    print(node_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    print("\n[Linear dispersion fit near nodes]")
    print(fit_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    print("\n[Low-energy DOS sample: first 10 bins]")
    print(dos_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    print("\n[Consistency]")
    print(f"max |eig_numpy - eig_torch| = {torch_diff:.3e}")
    print("sanity checks: PASS")

    print("\nCSV artifacts written:")
    print(f"- {out_dir / 'nodes_summary.csv'}")
    print(f"- {out_dir / 'dispersion_fit.csv'}")
    print(f"- {out_dir / 'dos_curve.csv'}")


if __name__ == "__main__":
    main()
