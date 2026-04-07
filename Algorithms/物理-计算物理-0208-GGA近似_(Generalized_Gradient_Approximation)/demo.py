"""Minimal runnable MVP for GGA approximation (PHYS-0207).

This demo implements a transparent exchange-only GGA workflow:
1) define electron density n(x)
2) compute reduced gradient s(x)
3) evaluate LDA exchange and PBE-style GGA enhancement factor
4) integrate exchange energy and run sanity checks
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


PI = float(np.pi)
C_X = 0.75 * (3.0 / PI) ** (1.0 / 3.0)


@dataclass(frozen=True)
class GGAConfig:
    """Configuration for 1D model-density exchange functional evaluation."""

    grid_points: int = 4001
    x_max: float = 8.0
    pbe_mu: float = 0.2195149727645171
    pbe_kappa: float = 0.804
    density_floor: float = 1.0e-10


@dataclass(frozen=True)
class FunctionalResult:
    name: str
    e_x_lda: float
    e_x_gga: float
    correction: float
    correction_ratio: float
    s_mean: float
    s_max: float
    fx_min: float
    fx_max: float


def make_grid(cfg: GGAConfig) -> np.ndarray:
    if cfg.grid_points < 11:
        raise ValueError("grid_points must be >= 11")
    if cfg.x_max <= 0.0:
        raise ValueError("x_max must be positive")
    return np.linspace(-cfg.x_max, cfg.x_max, cfg.grid_points, dtype=float)


def uniform_density(x: np.ndarray, value: float) -> np.ndarray:
    if value <= 0.0:
        raise ValueError("uniform density value must be positive")
    return np.full_like(x, fill_value=value, dtype=float)


def gaussian_modulated_density(
    x: np.ndarray,
    n0: float,
    width: float,
    amplitude: float,
    wave_number: float,
    background: float,
) -> np.ndarray:
    """A smooth positive model density for demonstrating gradient effects."""
    if n0 <= 0.0 or width <= 0.0 or background < 0.0:
        raise ValueError("invalid density parameters")
    if abs(amplitude) >= 1.0:
        raise ValueError("amplitude must satisfy |amplitude| < 1 for positivity")

    envelope = np.exp(-(x / width) ** 2)
    oscillation = 1.0 + amplitude * np.cos(wave_number * x)
    return background + n0 * envelope * oscillation


def finite_difference_gradient(n: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(n, x, edge_order=2)


def lda_exchange_per_particle(n: np.ndarray) -> np.ndarray:
    n_safe = np.asarray(n, dtype=float)
    return -C_X * np.cbrt(n_safe)


def reduced_gradient_s(n: np.ndarray, grad_n: np.ndarray, floor: float) -> np.ndarray:
    n_safe = np.maximum(np.asarray(n, dtype=float), floor)
    grad_abs = np.abs(np.asarray(grad_n, dtype=float))

    k_f = np.cbrt(3.0 * PI * PI * n_safe)
    denom = 2.0 * k_f * n_safe
    return grad_abs / np.maximum(denom, floor)


def pbe_exchange_enhancement(s: np.ndarray, mu: float, kappa: float) -> np.ndarray:
    s2 = np.asarray(s, dtype=float) ** 2
    return 1.0 + kappa - kappa / (1.0 + (mu / kappa) * s2)


def exchange_energies(
    n: np.ndarray,
    x: np.ndarray,
    cfg: GGAConfig,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (E_x^LDA, E_x^GGA, s, F_x)."""
    n_safe = np.maximum(np.asarray(n, dtype=float), cfg.density_floor)
    grad_n = finite_difference_gradient(n_safe, x)
    s = reduced_gradient_s(n_safe, grad_n, floor=cfg.density_floor)
    f_x = pbe_exchange_enhancement(s, mu=cfg.pbe_mu, kappa=cfg.pbe_kappa)

    eps_x_lda = lda_exchange_per_particle(n_safe)
    integrand_lda = n_safe * eps_x_lda
    integrand_gga = integrand_lda * f_x

    e_x_lda = float(np.trapezoid(integrand_lda, x))
    e_x_gga = float(np.trapezoid(integrand_gga, x))
    return e_x_lda, e_x_gga, s, f_x


def evaluate_profile(name: str, n: np.ndarray, x: np.ndarray, cfg: GGAConfig) -> FunctionalResult:
    e_x_lda, e_x_gga, s, f_x = exchange_energies(n=n, x=x, cfg=cfg)
    correction = e_x_gga - e_x_lda
    correction_ratio = abs(correction) / max(abs(e_x_lda), 1.0e-20)

    return FunctionalResult(
        name=name,
        e_x_lda=e_x_lda,
        e_x_gga=e_x_gga,
        correction=correction,
        correction_ratio=correction_ratio,
        s_mean=float(np.mean(s)),
        s_max=float(np.max(s)),
        fx_min=float(np.min(f_x)),
        fx_max=float(np.max(f_x)),
    )


def grid_convergence_check(base_cfg: GGAConfig) -> tuple[float, float]:
    """Compare LDA/GGA exchange energies between coarse and fine grids."""
    cfg_coarse = replace(base_cfg, grid_points=base_cfg.grid_points)
    cfg_fine = replace(base_cfg, grid_points=2 * base_cfg.grid_points - 1)

    x_coarse = make_grid(cfg_coarse)
    x_fine = make_grid(cfg_fine)

    n_coarse = gaussian_modulated_density(
        x_coarse,
        n0=0.65,
        width=2.3,
        amplitude=0.45,
        wave_number=2.4,
        background=0.03,
    )
    n_fine = gaussian_modulated_density(
        x_fine,
        n0=0.65,
        width=2.3,
        amplitude=0.45,
        wave_number=2.4,
        background=0.03,
    )

    e_lda_c, e_gga_c, _, _ = exchange_energies(n=n_coarse, x=x_coarse, cfg=cfg_coarse)
    e_lda_f, e_gga_f, _, _ = exchange_energies(n=n_fine, x=x_fine, cfg=cfg_fine)

    rel_lda = abs(e_lda_f - e_lda_c) / max(abs(e_lda_f), 1.0e-20)
    rel_gga = abs(e_gga_f - e_gga_c) / max(abs(e_gga_f), 1.0e-20)
    return float(rel_lda), float(rel_gga)


def main() -> None:
    cfg = GGAConfig()
    x = make_grid(cfg)

    profile_uniform = uniform_density(x, value=0.35)
    profile_smooth = gaussian_modulated_density(
        x,
        n0=0.60,
        width=3.0,
        amplitude=0.20,
        wave_number=1.0,
        background=0.04,
    )
    profile_sharp = gaussian_modulated_density(
        x,
        n0=0.65,
        width=2.1,
        amplitude=0.50,
        wave_number=2.6,
        background=0.03,
    )

    results = [
        evaluate_profile("uniform", profile_uniform, x, cfg),
        evaluate_profile("smooth-gradient", profile_smooth, x, cfg),
        evaluate_profile("sharp-gradient", profile_sharp, x, cfg),
    ]

    table = pd.DataFrame(
        {
            "profile": [r.name for r in results],
            "E_x_LDA": [r.e_x_lda for r in results],
            "E_x_GGA": [r.e_x_gga for r in results],
            "GGA-LDA": [r.correction for r in results],
            "|corr|/|LDA|": [r.correction_ratio for r in results],
            "<s>": [r.s_mean for r in results],
            "max(s)": [r.s_max for r in results],
            "min(Fx)": [r.fx_min for r in results],
            "max(Fx)": [r.fx_max for r in results],
        }
    )

    uniform = results[0]
    smooth = results[1]
    sharp = results[2]

    rel_grid_lda, rel_grid_gga = grid_convergence_check(cfg)

    checks = {
        "uniform limit: |E_GGA - E_LDA| < 1e-10": abs(uniform.e_x_gga - uniform.e_x_lda) < 1.0e-10,
        "enhancement lower bound: min(Fx) >= 1": min(r.fx_min for r in results) >= 1.0 - 1.0e-12,
        "enhancement upper bound: max(Fx) <= 1+kappa": max(r.fx_max for r in results)
        <= (1.0 + cfg.pbe_kappa + 1.0e-12),
        "gradient lowers exchange energy: smooth GGA < LDA": smooth.e_x_gga < smooth.e_x_lda,
        "stronger gradient gives stronger correction": sharp.correction_ratio > smooth.correction_ratio,
        "grid convergence LDA rel diff < 5e-4": rel_grid_lda < 5.0e-4,
        "grid convergence GGA rel diff < 5e-4": rel_grid_gga < 5.0e-4,
    }

    pd.set_option("display.float_format", lambda v: f"{v:.8f}")

    print("=== GGA Approximation MVP (PHYS-0207) ===")
    print(
        f"grid_points={cfg.grid_points}, x in [{-cfg.x_max:.1f}, {cfg.x_max:.1f}], "
        f"mu={cfg.pbe_mu:.15f}, kappa={cfg.pbe_kappa:.3f}"
    )

    print("\nExchange summary:")
    print(table.to_string(index=False))

    print("\nGrid convergence check:")
    print(f"- relative LDA difference (coarse vs fine): {rel_grid_lda:.3e}")
    print(f"- relative GGA difference (coarse vs fine): {rel_grid_gga:.3e}")

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
