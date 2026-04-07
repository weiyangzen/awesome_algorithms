"""Minimal runnable MVP for Fermi surface extraction on a 2D tight-binding band."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit
from sklearn.linear_model import LinearRegression

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class TBConfig:
    t: float = 1.0
    tp: float = 0.25
    target_filling: float = 0.62
    beta: float = 120.0
    grid_n_mu: int = 241
    grid_n_fs: int = 281
    q_max: float = 0.08
    q_points: int = 31
    torch_steps: int = 1200
    torch_lr: float = 0.05
    filling_tol: float = 2.0e-3
    residual_tol: float = 1.0e-3
    slope_rel_tol: float = 8.0e-2
    min_fermi_points: int = 120
    random_seed: int = 7


def dispersion(kx: np.ndarray, ky: np.ndarray, cfg: TBConfig) -> np.ndarray:
    """2D square-lattice tight-binding dispersion with next-nearest hopping."""
    return -2.0 * cfg.t * (np.cos(kx) + np.cos(ky)) - 4.0 * cfg.tp * np.cos(kx) * np.cos(ky)


def grad_dispersion(kx: np.ndarray, ky: np.ndarray, cfg: TBConfig) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of dispersion with respect to (kx, ky)."""
    vx = 2.0 * cfg.t * np.sin(kx) + 4.0 * cfg.tp * np.sin(kx) * np.cos(ky)
    vy = 2.0 * cfg.t * np.sin(ky) + 4.0 * cfg.tp * np.cos(kx) * np.sin(ky)
    return vx, vy


def fermi_dirac(energy_minus_mu: np.ndarray, beta: float) -> np.ndarray:
    x = np.clip(beta * energy_minus_mu, -60.0, 60.0)
    return 1.0 / (np.exp(x) + 1.0)


def solve_mu_for_target_filling(cfg: TBConfig) -> tuple[float, float, float]:
    """Solve mu from smooth filling equation and report both smooth/T=0 fillings."""
    k = np.linspace(-pi, pi, cfg.grid_n_mu, endpoint=False)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    e = dispersion(kx, ky, cfg)

    def smooth_filling(mu: float) -> float:
        return float(np.mean(fermi_dirac(e - mu, cfg.beta)))

    lo = float(np.min(e) - 1.0)
    hi = float(np.max(e) + 1.0)

    mu = float(brentq(lambda m: smooth_filling(m) - cfg.target_filling, lo, hi, xtol=1e-12, rtol=1e-12))
    n_smooth = smooth_filling(mu)
    n_t0 = float(np.mean(e <= mu))
    return mu, n_smooth, n_t0


def _edge_intersections(
    p1: tuple[float, float],
    v1: float,
    p2: tuple[float, float],
    v2: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    tol = 1e-14
    a_zero = abs(v1) < tol
    b_zero = abs(v2) < tol
    if a_zero and b_zero:
        out.extend([p1, p2])
        return out
    if a_zero:
        out.append(p1)
        return out
    if b_zero:
        out.append(p2)
        return out
    if v1 * v2 < 0.0:
        t = v1 / (v1 - v2)
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        out.append((float(x), float(y)))
    return out


def extract_fermi_points(mu: float, cfg: TBConfig) -> np.ndarray:
    """Extract Fermi-surface points by sign changes on cell edges."""
    k = np.linspace(-pi, pi, cfg.grid_n_fs)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    phi = dispersion(kx, ky, cfg) - mu

    points: list[tuple[float, float]] = []
    for i in range(cfg.grid_n_fs - 1):
        for j in range(cfg.grid_n_fs - 1):
            x0, x1 = k[i], k[i + 1]
            y0, y1 = k[j], k[j + 1]
            v00 = float(phi[i, j])
            v10 = float(phi[i + 1, j])
            v11 = float(phi[i + 1, j + 1])
            v01 = float(phi[i, j + 1])

            local = []
            local.extend(_edge_intersections((x0, y0), v00, (x1, y0), v10))
            local.extend(_edge_intersections((x1, y0), v10, (x1, y1), v11))
            local.extend(_edge_intersections((x1, y1), v11, (x0, y1), v01))
            local.extend(_edge_intersections((x0, y1), v01, (x0, y0), v00))

            if local:
                points.extend(local)

    if not points:
        return np.zeros((0, 2), dtype=float)

    pts = np.asarray(points, dtype=float)
    # Quantize before unique to remove repeated vertices from neighboring cells.
    pts_q = np.round(pts, decimals=9)
    unique = np.unique(pts_q, axis=0)
    return unique


def build_fermi_dataframe(fs_points: np.ndarray, mu: float, cfg: TBConfig) -> pd.DataFrame:
    if fs_points.size == 0:
        return pd.DataFrame(columns=["kx", "ky", "eps_minus_mu", "vF"])
    kx = fs_points[:, 0]
    ky = fs_points[:, 1]
    eps_minus_mu = dispersion(kx, ky, cfg) - mu
    vx, vy = grad_dispersion(kx, ky, cfg)
    vmag = np.sqrt(vx * vx + vy * vy)
    return pd.DataFrame(
        {
            "kx": kx,
            "ky": ky,
            "eps_minus_mu": eps_minus_mu,
            "vF": vmag,
        }
    )


def find_nodal_kf(mu: float, cfg: TBConfig) -> float:
    """Find one Fermi crossing on kx=ky line within [0, pi]."""

    def f(k: float) -> float:
        return float(dispersion(np.array(k), np.array(k), cfg) - mu)

    ks = np.linspace(0.0, pi, 1201)
    vals = np.array([f(kk) for kk in ks])
    signs = vals[:-1] * vals[1:]
    idx = np.where(signs <= 0.0)[0]
    if idx.size == 0:
        raise RuntimeError("No nodal Fermi crossing found on kx=ky line; adjust target_filling or hopping.")

    i = int(idx[0])
    a, b = float(ks[i]), float(ks[i + 1])
    return float(brentq(f, a, b, xtol=1e-13, rtol=1e-13))


def sample_local_dispersion(mu: float, kf: float, cfg: TBConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample epsilon(kF + q n_hat) - mu around one Fermi point."""
    vx, vy = grad_dispersion(np.array(kf), np.array(kf), cfg)
    vx0 = float(vx)
    vy0 = float(vy)
    v0 = float(np.hypot(vx0, vy0))
    nx = vx0 / v0
    ny = vy0 / v0

    q = np.linspace(-cfg.q_max, cfg.q_max, cfg.q_points)
    kx = kf + q * nx
    ky = kf + q * ny
    de = dispersion(kx, ky, cfg) - mu
    return q, de, v0


def fit_local_slope_sklearn(q: np.ndarray, de: np.ndarray) -> float:
    model = LinearRegression(fit_intercept=True)
    model.fit(q.reshape(-1, 1), de)
    return float(model.coef_[0])


def _quad_model(q: np.ndarray, a1: float, a2: float, c: float) -> np.ndarray:
    return a1 * q + a2 * q * q + c


def fit_local_slope_curve_fit(q: np.ndarray, de: np.ndarray) -> float:
    p0 = np.array([1.0, 0.0, 0.0], dtype=float)
    popt, _ = curve_fit(_quad_model, q, de, p0=p0, maxfev=10000)
    return float(popt[0])


def fit_local_slope_torch(q: np.ndarray, de: np.ndarray, cfg: TBConfig) -> float | None:
    if torch is None:
        return None

    torch.manual_seed(cfg.random_seed)
    qt = torch.tensor(q, dtype=torch.float64)
    dt = torch.tensor(de, dtype=torch.float64)

    a1 = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
    a2 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    c = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.Adam([a1, a2, c], lr=cfg.torch_lr)

    for _ in range(cfg.torch_steps):
        optimizer.zero_grad()
        pred = a1 * qt + a2 * qt * qt + c
        loss = torch.mean((pred - dt) ** 2)
        loss.backward()
        optimizer.step()

    return float(a1.detach().cpu().item())


def main() -> None:
    cfg = TBConfig()
    np.random.seed(cfg.random_seed)

    mu, n_smooth, n_t0 = solve_mu_for_target_filling(cfg)
    fs_points = extract_fermi_points(mu, cfg)
    df_fs = build_fermi_dataframe(fs_points, mu, cfg)

    if df_fs.empty:
        raise RuntimeError("No Fermi-surface points extracted.")

    kf = find_nodal_kf(mu, cfg)
    q, de, v_exact = sample_local_dispersion(mu, kf, cfg)

    slope_sk = fit_local_slope_sklearn(q, de)
    slope_cf = fit_local_slope_curve_fit(q, de)
    slope_torch = fit_local_slope_torch(q, de, cfg)

    mean_residual = float(df_fs["eps_minus_mu"].abs().mean())
    vf_min = float(df_fs["vF"].min())
    vf_mean = float(df_fs["vF"].mean())
    vf_max = float(df_fs["vF"].max())

    print("=== Fermi Surface MVP (2D Tight-Binding) ===")
    print(f"t={cfg.t:.3f}, t'={cfg.tp:.3f}, target filling={cfg.target_filling:.4f}")
    print(f"Solved chemical potential mu = {mu:.8f}")
    print(f"Smooth filling n(mu,beta) = {n_smooth:.6f}")
    print(f"Estimated T=0 filling      = {n_t0:.6f}")
    print(f"Fermi points extracted     = {len(df_fs)}")
    print(f"Mean |epsilon-mu| on FS    = {mean_residual:.3e}")
    print(f"vF stats (min/mean/max)    = {vf_min:.6f} / {vf_mean:.6f} / {vf_max:.6f}")
    print(f"Nodal kF (kx=ky)           = {kf:.8f}")
    print(f"Local slope exact |grad e| = {v_exact:.6f}")
    print(f"Local slope sklearn        = {slope_sk:.6f}")
    print(f"Local slope curve_fit      = {slope_cf:.6f}")
    if slope_torch is None:
        print("Local slope torch          = <torch unavailable>")
    else:
        print(f"Local slope torch          = {slope_torch:.6f}")

    preview = df_fs.head(8).copy()
    print("\nFS sample points:")
    print(preview.to_string(index=False))

    # Assertions for automated validation.
    assert abs(n_smooth - cfg.target_filling) <= cfg.filling_tol, "filling mismatch too large"
    assert len(df_fs) >= cfg.min_fermi_points, "too few Fermi points extracted"
    assert mean_residual <= cfg.residual_tol, "Fermi-surface residual too large"

    rel_sk = abs(abs(slope_sk) - v_exact) / max(v_exact, 1e-12)
    rel_cf = abs(abs(slope_cf) - v_exact) / max(v_exact, 1e-12)
    assert rel_sk <= cfg.slope_rel_tol, "sklearn slope inconsistent with exact vF"
    assert rel_cf <= cfg.slope_rel_tol, "curve_fit slope inconsistent with exact vF"

    if slope_torch is not None:
        rel_torch = abs(abs(slope_torch) - v_exact) / max(v_exact, 1e-12)
        assert rel_torch <= cfg.slope_rel_tol, "torch slope inconsistent with exact vF"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
