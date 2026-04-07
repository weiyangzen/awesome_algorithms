"""Minimal runnable MVP for ideal 1D magnetohydrodynamics (MHD).

The script solves the Brio-Wu shock-tube setup with a first-order finite-volume
scheme and Rusanov (local Lax-Friedrichs) numerical flux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MHDConfig:
    gamma: float = 2.0
    bx: float = 0.75
    x_min: float = -0.5
    x_max: float = 0.5
    n_cells: int = 400
    cfl: float = 0.35
    t_end: float = 0.08
    density_floor: float = 1e-8
    pressure_floor: float = 1e-8
    max_steps: int = 200_000


def primitive_to_conserved(
    rho: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    p: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    cfg: MHDConfig,
) -> np.ndarray:
    kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    magnetic = 0.5 * (cfg.bx * cfg.bx + by * by + bz * bz)
    total_energy = p / (cfg.gamma - 1.0) + kinetic + magnetic

    return np.column_stack(
        [
            rho,
            rho * vx,
            rho * vy,
            rho * vz,
            by,
            bz,
            total_energy,
        ]
    )


def conserved_to_primitive(U: np.ndarray, cfg: MHDConfig) -> tuple[np.ndarray, ...]:
    rho = np.maximum(U[:, 0], cfg.density_floor)
    vx = U[:, 1] / rho
    vy = U[:, 2] / rho
    vz = U[:, 3] / rho
    by = U[:, 4]
    bz = U[:, 5]

    kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    magnetic = 0.5 * (cfg.bx * cfg.bx + by * by + bz * bz)
    p = (cfg.gamma - 1.0) * (U[:, 6] - kinetic - magnetic)
    p = np.maximum(p, cfg.pressure_floor)

    return rho, vx, vy, vz, p, by, bz


def mhd_flux(U: np.ndarray, cfg: MHDConfig) -> np.ndarray:
    rho, vx, vy, vz, p, by, bz = conserved_to_primitive(U, cfg)
    magnetic_pressure = 0.5 * (cfg.bx * cfg.bx + by * by + bz * bz)
    total_pressure = p + magnetic_pressure
    v_dot_b = vx * cfg.bx + vy * by + vz * bz

    F = np.empty_like(U)
    F[:, 0] = rho * vx
    F[:, 1] = rho * vx * vx + total_pressure - cfg.bx * cfg.bx
    F[:, 2] = rho * vx * vy - cfg.bx * by
    F[:, 3] = rho * vx * vz - cfg.bx * bz
    F[:, 4] = by * vx - cfg.bx * vy
    F[:, 5] = bz * vx - cfg.bx * vz
    F[:, 6] = (U[:, 6] + total_pressure) * vx - cfg.bx * v_dot_b
    return F


def fast_magnetosonic_speed(U: np.ndarray, cfg: MHDConfig) -> np.ndarray:
    rho, vx, _, _, p, by, bz = conserved_to_primitive(U, cfg)
    a2 = cfg.gamma * p / rho
    b2 = (cfg.bx * cfg.bx + by * by + bz * bz) / rho
    bx2 = (cfg.bx * cfg.bx) / rho

    discriminant = np.maximum((a2 + b2) ** 2 - 4.0 * a2 * bx2, 0.0)
    cf2 = 0.5 * (a2 + b2 + np.sqrt(discriminant))
    cf = np.sqrt(np.maximum(cf2, 0.0))
    return np.abs(vx) + cf


def rusanov_flux(UL: np.ndarray, UR: np.ndarray, cfg: MHDConfig) -> np.ndarray:
    FL = mhd_flux(UL, cfg)
    FR = mhd_flux(UR, cfg)

    speed = np.maximum(fast_magnetosonic_speed(UL, cfg), fast_magnetosonic_speed(UR, cfg))
    return 0.5 * (FL + FR) - 0.5 * speed[:, None] * (UR - UL)


def apply_outflow_boundary(U: np.ndarray) -> np.ndarray:
    U_pad = np.empty((U.shape[0] + 2, U.shape[1]), dtype=U.dtype)
    U_pad[1:-1] = U
    U_pad[0] = U[0]
    U_pad[-1] = U[-1]
    return U_pad


def build_brio_wu_initial_state(cfg: MHDConfig) -> tuple[np.ndarray, np.ndarray, float]:
    dx = (cfg.x_max - cfg.x_min) / cfg.n_cells
    x = np.linspace(cfg.x_min + 0.5 * dx, cfg.x_max - 0.5 * dx, cfg.n_cells)

    left = x < 0.0
    rho = np.where(left, 1.0, 0.125)
    p = np.where(left, 1.0, 0.1)
    vx = np.zeros_like(x)
    vy = np.zeros_like(x)
    vz = np.zeros_like(x)
    by = np.where(left, 1.0, -1.0)
    bz = np.zeros_like(x)

    U0 = primitive_to_conserved(rho, vx, vy, vz, p, by, bz, cfg)
    return x, U0, dx


def solve_1d_ideal_mhd(cfg: MHDConfig) -> dict[str, object]:
    x, U, dx = build_brio_wu_initial_state(cfg)

    initial_mass = float(np.sum(U[:, 0]) * dx)
    initial_energy = float(np.sum(U[:, 6]) * dx)

    t = 0.0
    steps = 0
    while t < cfg.t_end:
        max_speed = float(np.max(fast_magnetosonic_speed(U, cfg)))
        if not np.isfinite(max_speed) or max_speed <= 0.0:
            raise RuntimeError(f"invalid wave speed: {max_speed}")

        dt = cfg.cfl * dx / max_speed
        if t + dt > cfg.t_end:
            dt = cfg.t_end - t

        U_pad = apply_outflow_boundary(U)
        interface_flux = rusanov_flux(U_pad[:-1], U_pad[1:], cfg)
        U = U - (dt / dx) * (interface_flux[1:] - interface_flux[:-1])

        # Reconstruct with floors to keep physically admissible states.
        rho, vx, vy, vz, p, by, bz = conserved_to_primitive(U, cfg)
        U = primitive_to_conserved(rho, vx, vy, vz, p, by, bz, cfg)

        t += dt
        steps += 1
        if steps > cfg.max_steps:
            raise RuntimeError("maximum number of time steps exceeded")

    rho, vx, vy, vz, p, by, bz = conserved_to_primitive(U, cfg)
    magnetic_pressure = 0.5 * (cfg.bx * cfg.bx + by * by + bz * bz)
    total_pressure = p + magnetic_pressure

    final_mass = float(np.sum(U[:, 0]) * dx)
    final_energy = float(np.sum(U[:, 6]) * dx)

    frame = pd.DataFrame(
        {
            "x": x,
            "rho": rho,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "By": by,
            "Bz": bz,
            "p": p,
            "p_total": total_pressure,
            "energy": U[:, 6],
        }
    )

    return {
        "frame": frame,
        "steps": steps,
        "final_time": t,
        "initial_mass": initial_mass,
        "final_mass": final_mass,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
    }


def main() -> None:
    cfg = MHDConfig()
    result = solve_1d_ideal_mhd(cfg)
    frame: pd.DataFrame = result["frame"]

    mass_rel_drift = abs(result["final_mass"] - result["initial_mass"]) / max(
        abs(result["initial_mass"]), 1e-14
    )
    energy_rel_drift = abs(result["final_energy"] - result["initial_energy"]) / max(
        abs(result["initial_energy"]), 1e-14
    )

    rho_min = float(frame["rho"].min())
    p_min = float(frame["p"].min())
    finite_ok = bool(np.isfinite(frame.to_numpy()).all())

    checks = {
        "finite_values": finite_ok,
        "positive_density": rho_min > 0.0,
        "positive_pressure": p_min > 0.0,
        "mass_rel_drift<2e-2": mass_rel_drift < 2e-2,
        "energy_rel_drift<3e-2": energy_rel_drift < 3e-2,
    }
    passed = all(checks.values())

    print("=== Ideal 1D MHD: Brio-Wu Shock Tube (Rusanov FV) ===")
    print(f"cells={cfg.n_cells}, gamma={cfg.gamma:.3f}, Bx={cfg.bx:.3f}, CFL={cfg.cfl:.3f}")
    print(f"final_time={result['final_time']:.5f}, steps={result['steps']}")
    print(f"rho_min={rho_min:.6e}, p_min={p_min:.6e}")
    print(f"mass_rel_drift={mass_rel_drift:.6e}, energy_rel_drift={energy_rel_drift:.6e}")

    print("\nChecks:")
    for name, flag in checks.items():
        print(f"- {name}: {'PASS' if flag else 'FAIL'}")

    sample = frame.iloc[::40].reset_index(drop=True)
    print("\nSampled profile rows (every 40th cell):")
    print(sample.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
