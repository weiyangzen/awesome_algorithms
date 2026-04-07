"""Minimal runnable MVP for TDDFT (time-dependent density functional theory).

This script demonstrates a transparent 1D real-time TDDFT toy model:
1) SCF ground-state Kohn-Sham solve (adiabatic local xc + Hartree),
2) Crank-Nicolson propagation of one occupied spatial orbital (spin-paired),
3) dipole and energy response under a Gaussian laser pulse,
4) absorbed-energy scaling fit with sklearn/scipy/(optional torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal, solve_banded
from scipy.optimize import curve_fit
from scipy.signal.windows import hann
from sklearn.linear_model import LinearRegression

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None


@dataclass(frozen=True)
class Grid1D:
    x_min: float = -12.0
    x_max: float = 12.0
    n_points: int = 220

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.n_points, dtype=np.float64)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.n_points - 1)


@dataclass(frozen=True)
class TDDFTConfig:
    n_electrons: int = 2
    nuclear_charge: float = 1.5
    soft_nuclear: float = 1.0
    soft_hartree: float = 0.8
    xc_prefactor: float = (3.0 / np.pi) ** (1.0 / 3.0)
    scf_mix: float = 0.32
    scf_max_iter: int = 120
    scf_tol: float = 2.0e-6
    dt: float = 0.06
    n_steps: int = 300
    pulse_omega: float = 1.2
    pulse_t0: float = 6.0
    pulse_sigma: float = 1.6
    pulse_amplitudes: tuple[float, ...] = (0.004, 0.006, 0.008, 0.010)


def build_kinetic_tridiagonal(n_points: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """T = -1/2 d^2/dx^2 with second-order finite differences."""
    diag = np.full(n_points, 1.0 / dx**2, dtype=np.float64)
    off = np.full(n_points - 1, -0.5 / dx**2, dtype=np.float64)
    return diag, off


def apply_tridiagonal(diag: np.ndarray, off: np.ndarray, vec: np.ndarray) -> np.ndarray:
    out = diag * vec
    out[:-1] += off * vec[1:]
    out[1:] += off * vec[:-1]
    return out


def normalize_orbital(psi: np.ndarray, dx: float) -> np.ndarray:
    norm = np.sqrt(float(np.sum(np.abs(psi) ** 2) * dx))
    if norm <= 0.0:
        raise ValueError("orbital norm must stay positive")
    return psi / norm


def density_from_orbital(psi: np.ndarray, n_electrons: int) -> np.ndarray:
    if n_electrons <= 0 or n_electrons % 2 != 0:
        raise ValueError("this MVP expects positive even electron count")
    return float(n_electrons) * np.abs(psi) ** 2


def external_static_potential(
    x: np.ndarray,
    nuclear_charge: float,
    soft_nuclear: float,
) -> np.ndarray:
    return -nuclear_charge / np.sqrt(x**2 + soft_nuclear**2)


def build_hartree_kernel(x: np.ndarray, soft_hartree: float) -> np.ndarray:
    dx = x[:, None] - x[None, :]
    return 1.0 / np.sqrt(dx**2 + soft_hartree**2)


def hartree_potential(kernel: np.ndarray, density: np.ndarray, dx: float) -> np.ndarray:
    return dx * (kernel @ density)


def xc_potential_lda_like(density: np.ndarray, prefactor: float) -> np.ndarray:
    n_safe = np.clip(density, 1.0e-12, None)
    return -prefactor * np.cbrt(n_safe)


def xc_energy_density_lda_like(density: np.ndarray, prefactor: float) -> np.ndarray:
    n_safe = np.clip(density, 1.0e-12, None)
    return -0.75 * prefactor * n_safe ** (4.0 / 3.0)


def laser_field(t: float, amplitude: float, omega: float, t0: float, sigma: float) -> float:
    env = np.exp(-((t - t0) ** 2) / (2.0 * sigma**2))
    return float(amplitude * env * np.sin(omega * t))


def ground_state_scf(
    grid: Grid1D,
    cfg: TDDFTConfig,
    t_diag: np.ndarray,
    t_off: np.ndarray,
    kernel: np.ndarray,
    v_static: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    x = grid.x
    dx = grid.dx

    density = np.exp(-(x / 2.0) ** 2)
    density *= cfg.n_electrons / float(np.sum(density) * dx)

    converged = False
    psi = np.zeros_like(x, dtype=np.complex128)

    for it in range(1, cfg.scf_max_iter + 1):
        v_h = hartree_potential(kernel, density, dx)
        v_xc = xc_potential_lda_like(density, cfg.xc_prefactor)
        h_diag = t_diag + v_static + v_h + v_xc
        eigvals, eigvecs = eigh_tridiagonal(h_diag, t_off, select="i", select_range=(0, 0), check_finite=False)
        _ = eigvals  # keep for readability; not used further in this MVP
        psi_new = normalize_orbital(eigvecs[:, 0].astype(np.complex128), dx)
        density_out = density_from_orbital(psi_new, cfg.n_electrons)
        density_new = (1.0 - cfg.scf_mix) * density + cfg.scf_mix * density_out
        density_new *= cfg.n_electrons / float(np.sum(density_new) * dx)

        drho = float(np.sqrt(np.sum((density_new - density) ** 2) * dx))
        density = density_new
        psi = psi_new
        if drho < cfg.scf_tol:
            converged = True
            return psi, density, it, converged

    return psi, density, cfg.scf_max_iter, converged


def crank_nicolson_step(
    psi: np.ndarray,
    h_diag: np.ndarray,
    h_off: np.ndarray,
    dt: float,
) -> np.ndarray:
    fac = 0.5j * dt

    a_diag = 1.0 + fac * h_diag
    a_off = fac * h_off
    b_diag = 1.0 - fac * h_diag
    b_off = -fac * h_off

    rhs = b_diag * psi
    rhs[:-1] += b_off * psi[1:]
    rhs[1:] += b_off * psi[:-1]

    ab = np.zeros((3, psi.size), dtype=np.complex128)
    ab[0, 1:] = a_off
    ab[1, :] = a_diag
    ab[2, :-1] = a_off

    return solve_banded((1, 1), ab, rhs, check_finite=False)


def internal_energy(
    psi: np.ndarray,
    density: np.ndarray,
    v_static: np.ndarray,
    v_h: np.ndarray,
    t_diag: np.ndarray,
    t_off: np.ndarray,
    dx: float,
    n_electrons: int,
    xc_prefactor: float,
) -> float:
    tpsi = apply_tridiagonal(t_diag, t_off, psi)
    e_kin = float(n_electrons * np.real(np.vdot(psi, tpsi)) * dx)
    e_ext = float(np.sum(density * v_static) * dx)
    e_h = 0.5 * float(np.sum(density * v_h) * dx)
    e_xc = float(np.sum(xc_energy_density_lda_like(density, xc_prefactor)) * dx)
    return e_kin + e_ext + e_h + e_xc


def dipole_moment(x: np.ndarray, density: np.ndarray, dx: float) -> float:
    return float(np.sum(x * density) * dx)


def dominant_omega_from_dipole(dipole: np.ndarray, dt: float) -> float:
    centered = dipole - float(np.mean(dipole))
    windowed = centered * hann(centered.size)
    spec = np.abs(np.fft.rfft(windowed))
    freq = np.fft.rfftfreq(centered.size, d=dt)
    if spec.size <= 1:
        return 0.0
    idx = int(np.argmax(spec[1:]) + 1)
    return float(2.0 * np.pi * freq[idx])


def run_tddft_response(
    amplitude: float,
    grid: Grid1D,
    cfg: TDDFTConfig,
    psi0: np.ndarray,
    t_diag: np.ndarray,
    t_off: np.ndarray,
    kernel: np.ndarray,
    v_static: np.ndarray,
    collect_trace: bool,
) -> dict[str, object]:
    x = grid.x
    dx = grid.dx

    psi = np.array(psi0, dtype=np.complex128, copy=True)
    density = density_from_orbital(psi, cfg.n_electrons)
    v_h0 = hartree_potential(kernel, density, dx)
    e0 = internal_energy(psi, density, v_static, v_h0, t_diag, t_off, dx, cfg.n_electrons, cfg.xc_prefactor)

    dipoles: list[float] = []
    norms: list[float] = []
    energies: list[float] = []
    fields: list[float] = []
    times: list[float] = []
    trace_rows: list[dict[str, float]] = []

    for step in range(cfg.n_steps):
        t = step * cfg.dt
        field_t = laser_field(t, amplitude, cfg.pulse_omega, cfg.pulse_t0, cfg.pulse_sigma)

        density = density_from_orbital(psi, cfg.n_electrons)
        v_h = hartree_potential(kernel, density, dx)
        v_xc = xc_potential_lda_like(density, cfg.xc_prefactor)
        h_diag = t_diag + v_static + x * field_t + v_h + v_xc

        psi = crank_nicolson_step(psi, h_diag=h_diag, h_off=t_off, dt=cfg.dt)
        psi = normalize_orbital(psi, dx)

        density_next = density_from_orbital(psi, cfg.n_electrons)
        v_h_next = hartree_potential(kernel, density_next, dx)
        e_int = internal_energy(
            psi,
            density_next,
            v_static,
            v_h_next,
            t_diag,
            t_off,
            dx,
            cfg.n_electrons,
            cfg.xc_prefactor,
        )
        d = dipole_moment(x, density_next, dx)
        norm = float(np.sum(np.abs(psi) ** 2) * dx)

        dipoles.append(d)
        norms.append(norm)
        energies.append(e_int)
        fields.append(field_t)
        times.append(t)

        if collect_trace and (step % 20 == 0 or step == cfg.n_steps - 1):
            trace_rows.append(
                {
                    "t_au": t,
                    "field_au": field_t,
                    "dipole_au": d,
                    "energy_internal_au": e_int,
                    "norm": norm,
                }
            )

    absorbed = float(energies[-1] - e0)
    dipoles_arr = np.array(dipoles, dtype=np.float64)
    omega_dom = dominant_omega_from_dipole(dipoles_arr, cfg.dt)

    result: dict[str, object] = {
        "amplitude": amplitude,
        "absorbed_energy_au": absorbed,
        "max_abs_dipole_au": float(np.max(np.abs(dipoles_arr))),
        "norm_drift_max": float(np.max(np.abs(np.array(norms) - 1.0))),
        "dominant_omega_au": omega_dom,
        "field_rms_au": float(np.sqrt(np.mean(np.square(fields)))),
        "trace_table": pd.DataFrame(trace_rows) if collect_trace else pd.DataFrame(),
    }
    return result


def fit_absorbed_energy_sklearn(amplitudes: np.ndarray, absorbed: np.ndarray) -> float:
    x = np.square(amplitudes).reshape(-1, 1)
    model = LinearRegression(fit_intercept=False)
    model.fit(x, absorbed)
    return float(model.coef_[0])


def fit_absorbed_energy_curve_fit(amplitudes: np.ndarray, absorbed: np.ndarray) -> float:
    def model(a: np.ndarray, beta: float) -> np.ndarray:
        return beta * np.square(a)

    beta, _ = curve_fit(model, amplitudes, absorbed, p0=np.array([1.0], dtype=np.float64))
    return float(beta[0])


def fit_absorbed_energy_torch_optional(
    amplitudes: np.ndarray,
    absorbed: np.ndarray,
    steps: int = 3000,
    lr: float = 0.08,
) -> Optional[float]:
    if torch is None:
        return None

    x = torch.tensor(np.square(amplitudes) * 1.0e4, dtype=torch.float64)
    y = torch.tensor(absorbed * 1.0e4, dtype=torch.float64)
    beta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        pred = beta * x
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()

    return float(beta.detach().cpu().item())


def main() -> None:
    grid = Grid1D()
    cfg = TDDFTConfig()

    x = grid.x
    dx = grid.dx
    t_diag, t_off = build_kinetic_tridiagonal(grid.n_points, dx)
    kernel = build_hartree_kernel(x, cfg.soft_hartree)
    v_static = external_static_potential(x, cfg.nuclear_charge, cfg.soft_nuclear)

    psi0, density0, scf_iters, scf_ok = ground_state_scf(grid, cfg, t_diag, t_off, kernel, v_static)
    if not scf_ok:
        raise RuntimeError("ground-state SCF did not converge")

    scan_rows: list[dict[str, float]] = []
    trace_table = pd.DataFrame()

    for idx, amp in enumerate(cfg.pulse_amplitudes):
        sim = run_tddft_response(
            amplitude=float(amp),
            grid=grid,
            cfg=cfg,
            psi0=psi0,
            t_diag=t_diag,
            t_off=t_off,
            kernel=kernel,
            v_static=v_static,
            collect_trace=(idx == len(cfg.pulse_amplitudes) - 1),
        )
        if idx == len(cfg.pulse_amplitudes) - 1:
            trace_table = sim["trace_table"]  # type: ignore[assignment]

        scan_rows.append(
            {
                "amplitude_au": float(sim["amplitude"]),
                "amplitude2_au2": float(sim["amplitude"]) ** 2,
                "absorbed_energy_au": float(sim["absorbed_energy_au"]),
                "max_abs_dipole_au": float(sim["max_abs_dipole_au"]),
                "dominant_omega_au": float(sim["dominant_omega_au"]),
                "norm_drift_max": float(sim["norm_drift_max"]),
                "field_rms_au": float(sim["field_rms_au"]),
            }
        )

    scan = pd.DataFrame(scan_rows)
    amplitudes = scan["amplitude_au"].to_numpy(dtype=np.float64)
    absorbed = scan["absorbed_energy_au"].to_numpy(dtype=np.float64)

    slope_sk = fit_absorbed_energy_sklearn(amplitudes, absorbed)
    slope_cf = fit_absorbed_energy_curve_fit(amplitudes, absorbed)
    slope_torch = fit_absorbed_energy_torch_optional(amplitudes, absorbed)

    monotonic = bool(np.all(np.diff(absorbed) > 0.0))
    max_norm_drift = float(scan["norm_drift_max"].max())
    omega_ref = float(scan["dominant_omega_au"].iloc[-1])

    print("=== TDDFT MVP (PHYS-0209) ===")
    print(
        f"grid=(n={grid.n_points}, dx={dx:.5f}), dt={cfg.dt:.4f}, n_steps={cfg.n_steps}, "
        f"scf_iters={scf_iters}"
    )
    print(
        f"pulse: omega={cfg.pulse_omega:.3f}, t0={cfg.pulse_t0:.2f}, sigma={cfg.pulse_sigma:.2f}, "
        f"amplitudes={cfg.pulse_amplitudes}"
    )
    print(f"density_norm_check_ground={float(np.sum(density0) * dx):.8f}")
    print(f"energy_slope_sklearn={slope_sk:.8f}")
    print(f"energy_slope_curve_fit={slope_cf:.8f}")
    if slope_torch is None:
        print("energy_slope_torch=unavailable")
    else:
        print(f"energy_slope_torch={slope_torch:.8f}")

    print("\nAmplitude scan summary:")
    print(scan.to_string(index=False))
    print("\nTime trace preview (largest amplitude):")
    print(trace_table.head(10).to_string(index=False))

    assert scf_ok, "SCF must converge"
    assert abs(float(np.sum(density0) * dx) - cfg.n_electrons) < 1.0e-8, "ground density normalization mismatch"
    assert monotonic, "absorbed energy should increase with pulse amplitude"
    assert max_norm_drift < 2.0e-9, f"wavefunction norm drift too large: {max_norm_drift}"
    assert slope_sk > 0.0 and slope_cf > 0.0, "response slope should be positive"
    assert abs(slope_sk - slope_cf) < 5.0e-7, "sklearn vs curve_fit mismatch"
    if slope_torch is not None:
        assert abs(slope_sk - slope_torch) < 2.0e-3, "torch slope mismatch"
    assert omega_ref > 0.1, "dominant frequency should be positive and finite"

    print("All checks passed.")


if __name__ == "__main__":
    main()
