"""Minimal runnable MVP for Turbulence Theory.

This script implements a compact 2D incompressible turbulence toy model using the
vorticity-streamfunction form of Navier-Stokes on a periodic box with
pseudo-spectral discretization.

Model equation (vorticity form):
    dω/dt + u·∇ω = ν∇²ω - αω + f
where:
    - ω is vorticity,
    - ν is viscosity,
    - α is large-scale linear drag,
    - f is low-wavenumber deterministic forcing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TurbulenceConfig:
    grid_size: int = 64
    domain_length: float = 2.0 * np.pi
    dt: float = 0.01
    n_steps: int = 1000
    viscosity: float = 2.0e-3
    linear_drag: float = 0.2
    forcing_wavenumber: int = 4
    forcing_amplitude: float = 0.35
    seed: int = 7
    report_every: int = 100

    def validate(self) -> None:
        if self.grid_size < 16:
            raise ValueError("grid_size must be >= 16")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.viscosity <= 0.0:
            raise ValueError("viscosity must be positive")
        if self.linear_drag < 0.0:
            raise ValueError("linear_drag must be non-negative")
        if self.forcing_wavenumber <= 0:
            raise ValueError("forcing_wavenumber must be positive")
        if self.forcing_amplitude <= 0.0:
            raise ValueError("forcing_amplitude must be positive")
        if self.report_every <= 0:
            raise ValueError("report_every must be positive")


@dataclass(frozen=True)
class SpectralOperators:
    kx: np.ndarray
    ky: np.ndarray
    k2: np.ndarray
    inv_k2: np.ndarray
    dealias_mask: np.ndarray
    shell_index: np.ndarray
    forcing: np.ndarray
    dx: float
    cutoff_shell: int


def build_spectral_operators(cfg: TurbulenceConfig) -> SpectralOperators:
    n = cfg.grid_size
    dx = cfg.domain_length / n

    k_1d = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k_1d, k_1d, indexing="ij")
    k2 = kx * kx + ky * ky

    inv_k2 = np.zeros_like(k2)
    nonzero = k2 > 0.0
    inv_k2[nonzero] = 1.0 / k2[nonzero]

    k_max = float(np.max(np.abs(k_1d)))
    cutoff = (2.0 / 3.0) * k_max
    dealias_mask = (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff)

    x = np.linspace(0.0, cfg.domain_length, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    kf = float(cfg.forcing_wavenumber)
    forcing = cfg.forcing_amplitude * (
        np.sin(kf * xx) + np.sin(kf * yy) + 0.5 * np.sin(kf * (xx + yy))
    )

    shell_index = np.rint(np.sqrt(k2)).astype(int)
    cutoff_shell = int(np.floor(cutoff))

    return SpectralOperators(
        kx=kx,
        ky=ky,
        k2=k2,
        inv_k2=inv_k2,
        dealias_mask=dealias_mask,
        shell_index=shell_index,
        forcing=forcing,
        dx=dx,
        cutoff_shell=cutoff_shell,
    )


def initialize_vorticity_hat(cfg: TurbulenceConfig, ops: SpectralOperators) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    omega0 = rng.standard_normal((cfg.grid_size, cfg.grid_size))
    omega_hat = np.fft.fft2(omega0)

    spectral_filter = np.exp(-(np.sqrt(ops.k2) / 8.0) ** 4)
    omega_hat = omega_hat * spectral_filter
    omega_hat[0, 0] = 0.0
    return omega_hat


def velocity_from_vorticity_hat(omega_hat: np.ndarray, ops: SpectralOperators) -> tuple[np.ndarray, np.ndarray]:
    # In 2D incompressible flow: ∇²ψ = -ω, u = ∂ψ/∂y, v = -∂ψ/∂x.
    psi_hat = omega_hat * ops.inv_k2
    u = np.fft.ifft2(1j * ops.ky * psi_hat).real
    v = np.fft.ifft2(-1j * ops.kx * psi_hat).real
    return u, v


def vorticity_rhs(
    omega_hat: np.ndarray,
    cfg: TurbulenceConfig,
    ops: SpectralOperators,
) -> np.ndarray:
    omega_hat = omega_hat * ops.dealias_mask
    omega = np.fft.ifft2(omega_hat).real

    u, v = velocity_from_vorticity_hat(omega_hat, ops)
    domega_dx = np.fft.ifft2(1j * ops.kx * omega_hat).real
    domega_dy = np.fft.ifft2(1j * ops.ky * omega_hat).real

    advection = u * domega_dx + v * domega_dy
    nonlinear_and_forcing_hat = np.fft.fft2(-advection + ops.forcing - cfg.linear_drag * omega)

    rhs_hat = nonlinear_and_forcing_hat - cfg.viscosity * ops.k2 * omega_hat
    rhs_hat = rhs_hat * ops.dealias_mask
    rhs_hat[0, 0] = 0.0
    return rhs_hat


def diagnostics(
    omega_hat: np.ndarray,
    cfg: TurbulenceConfig,
    ops: SpectralOperators,
) -> dict[str, float | np.ndarray]:
    u, v = velocity_from_vorticity_hat(omega_hat, ops)
    omega = np.fft.ifft2(omega_hat).real

    kinetic_energy = 0.5 * float(np.mean(u * u + v * v))
    enstrophy = 0.5 * float(np.mean(omega * omega))
    u_rms = float(np.sqrt(np.mean(u * u + v * v)))
    u_max = float(np.max(np.sqrt(u * u + v * v)))
    cfl = cfg.dt * u_max / ops.dx

    # For 2D incompressible flow: epsilon = nu * <omega^2> = 2*nu*enstrophy
    dissipation_rate = 2.0 * cfg.viscosity * enstrophy

    return {
        "kinetic_energy": kinetic_energy,
        "enstrophy": enstrophy,
        "u_rms": u_rms,
        "u_max": u_max,
        "cfl": cfl,
        "dissipation_rate": dissipation_rate,
        "u": u,
        "v": v,
    }


def isotropic_energy_spectrum(
    u: np.ndarray,
    v: np.ndarray,
    ops: SpectralOperators,
    grid_size: int,
) -> np.ndarray:
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)

    # Parseval-consistent modal kinetic energy density.
    mode_energy = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / (grid_size**4)
    max_shell = int(np.max(ops.shell_index))
    spectrum = np.bincount(
        ops.shell_index.ravel(),
        weights=mode_energy.ravel(),
        minlength=max_shell + 1,
    )
    return spectrum


def integral_length_scale(spectrum: np.ndarray) -> float:
    k = np.arange(spectrum.size, dtype=float)
    valid = (k > 0.0) & (spectrum > 0.0)
    if not np.any(valid):
        return float("nan")
    numerator = float(np.sum(spectrum[valid] / k[valid]))
    denominator = float(np.sum(spectrum[valid]))
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def simulate(
    cfg: TurbulenceConfig,
    ops: SpectralOperators,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    omega_hat = initialize_vorticity_hat(cfg, ops)
    initial_energy = float(diagnostics(omega_hat, cfg, ops)["kinetic_energy"])

    rows: list[dict[str, float]] = []

    for step in range(cfg.n_steps + 1):
        if step % cfg.report_every == 0 or step == cfg.n_steps:
            d = diagnostics(omega_hat, cfg, ops)
            rows.append(
                {
                    "step": float(step),
                    "time": float(step * cfg.dt),
                    "kinetic_energy": float(d["kinetic_energy"]),
                    "enstrophy": float(d["enstrophy"]),
                    "u_rms": float(d["u_rms"]),
                    "u_max": float(d["u_max"]),
                    "cfl": float(d["cfl"]),
                    "dissipation_rate": float(d["dissipation_rate"]),
                }
            )

        if step == cfg.n_steps:
            break

        k1 = vorticity_rhs(omega_hat, cfg, ops)
        k2 = vorticity_rhs(omega_hat + cfg.dt * k1, cfg, ops)
        omega_hat = omega_hat + 0.5 * cfg.dt * (k1 + k2)
        omega_hat = omega_hat * ops.dealias_mask
        omega_hat[0, 0] = 0.0

    history_df = pd.DataFrame(rows)
    return omega_hat, history_df, initial_energy


def main() -> None:
    cfg = TurbulenceConfig()
    cfg.validate()

    ops = build_spectral_operators(cfg)
    omega_hat, history_df, initial_energy = simulate(cfg, ops)

    final_diag = diagnostics(omega_hat, cfg, ops)
    u = final_diag["u"]
    v = final_diag["v"]
    assert isinstance(u, np.ndarray) and isinstance(v, np.ndarray)

    spectrum = isotropic_energy_spectrum(u, v, ops, cfg.grid_size)
    active_max = ops.cutoff_shell

    low_band_energy = float(np.sum(spectrum[1 : cfg.forcing_wavenumber + 3]))
    high_band_start = max(cfg.forcing_wavenumber + 8, 12)
    high_band_energy = float(np.sum(spectrum[high_band_start : active_max + 1]))

    spectral_energy_sum = float(np.sum(spectrum))
    final_energy = float(final_diag["kinetic_energy"])
    spectral_closure_relerr = abs(spectral_energy_sum - final_energy) / max(final_energy, 1e-12)

    integral_scale = integral_length_scale(spectrum)
    forcing_length = cfg.domain_length / cfg.forcing_wavenumber
    reynolds_number = float(final_diag["u_rms"]) * forcing_length / cfg.viscosity

    k = np.arange(spectrum.size, dtype=float)
    fit_mask = (k >= cfg.forcing_wavenumber + 1) & (k <= active_max - 2) & (spectrum > 0.0)
    spectral_slope = float("nan")
    if int(np.sum(fit_mask)) >= 2:
        spectral_slope = float(np.polyfit(np.log(k[fit_mask]), np.log(spectrum[fit_mask]), 1)[0])

    print("=== Turbulence Theory MVP (2D Incompressible Pseudo-Spectral) ===")
    print(
        "config:",
        f"N={cfg.grid_size}, L={cfg.domain_length:.3f}, dt={cfg.dt:.4f}, steps={cfg.n_steps}, "
        f"nu={cfg.viscosity:.4e}, drag={cfg.linear_drag:.3f}, "
        f"forcing_k={cfg.forcing_wavenumber}, forcing_amp={cfg.forcing_amplitude:.3f}",
    )
    print()

    with pd.option_context("display.width", 220, "display.precision", 6):
        print(history_df.to_string(index=False))
    print()

    print(
        "final summary:",
        f"E0={initial_energy:.6e}, E_final={final_energy:.6e}, growth={final_energy / initial_energy:.3f}x, "
        f"Re_forcing={reynolds_number:.2f}, integral_scale={integral_scale:.4f}, "
        f"spectrum_slope={spectral_slope:.3f}",
    )
    print(
        "spectrum checks:",
        f"low_band={low_band_energy:.6e}, high_band={high_band_energy:.6e}, "
        f"low/high={low_band_energy / max(high_band_energy, 1e-15):.3e}, "
        f"closure_relerr={spectral_closure_relerr:.3e}",
    )

    assert np.isfinite(omega_hat).all()
    assert final_energy > 0.0
    assert float(final_diag["enstrophy"]) > 0.0
    assert final_energy > 5.0 * initial_energy

    max_cfl = float(history_df["cfl"].max())
    assert max_cfl < 0.25

    assert spectral_closure_relerr < 1e-12
    assert high_band_energy > 0.0
    assert low_band_energy > 100.0 * high_band_energy

    assert reynolds_number > 30.0
    assert float(final_diag["dissipation_rate"]) > 0.0

    print("All checks passed.")


if __name__ == "__main__":
    main()
