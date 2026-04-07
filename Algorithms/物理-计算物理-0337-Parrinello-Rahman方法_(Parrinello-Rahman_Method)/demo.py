"""Parrinello-Rahman Method MVP.

This script implements a compact, source-visible anisotropic NPT molecular dynamics demo:
- periodic LJ particles in a deformable simulation cell,
- internal stress tensor estimation,
- Parrinello-Rahman-style cell-matrix evolution toward a target stress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PRConfig:
    seed: int = 7
    dim: int = 3

    cells_per_axis: tuple[int, int, int] = (2, 2, 2)
    mass: float = 1.0

    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: float = 2.5

    temperature: float = 0.7
    particle_friction: float = 0.15

    dt: float = 0.002
    n_steps: int = 5000
    sample_interval: int = 100

    barostat_mass: float = 45.0
    barostat_friction: float = 0.8

    # Target Cauchy stress tensor (reduced units).
    target_stress_diag: tuple[float, float, float] = (0.90, 2.40, 0.90)

    min_volume: float = 0.5
    max_volume: float = 250.0


def make_initial_fractional_positions(cells_per_axis: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = cells_per_axis
    frac = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                frac.append(((i + 0.5) / nx, (j + 0.5) / ny, (k + 0.5) / nz))
    return np.asarray(frac, dtype=float)


def frac_to_cart(frac: np.ndarray, h: np.ndarray) -> np.ndarray:
    # Row-wise mapping: r = s * h^T
    return frac @ h.T


def cart_to_frac(cart: np.ndarray, h: np.ndarray) -> np.ndarray:
    h_inv = np.linalg.inv(h)
    return cart @ h_inv.T


def lj_forces_energy_virial(
    frac: np.ndarray,
    h: np.ndarray,
    epsilon: float,
    sigma: float,
    cutoff: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    n, dim = frac.shape
    forces = np.zeros((n, dim), dtype=float)
    virial = np.zeros((dim, dim), dtype=float)
    potential = 0.0

    rc2 = cutoff * cutoff
    sigma2 = sigma * sigma

    inv_rc2 = 1.0 / rc2
    sr2c = sigma2 * inv_rc2
    sr6c = sr2c**3
    sr12c = sr6c**2
    u_shift = 4.0 * epsilon * (sr12c - sr6c)

    for i in range(n - 1):
        for j in range(i + 1, n):
            ds = frac[j] - frac[i]
            ds -= np.round(ds)
            dr = ds @ h.T  # from i to j in Cartesian coordinates

            r2 = float(dr @ dr)
            if r2 >= rc2 or r2 < 1e-12:
                continue

            inv_r2 = 1.0 / r2
            sr2 = sigma2 * inv_r2
            sr6 = sr2**3
            sr12 = sr6**2

            pair_u = 4.0 * epsilon * (sr12 - sr6) - u_shift
            potential += pair_u

            # magnitude helper in vector form; f_on_i points from j->i when repulsive.
            coeff = 24.0 * epsilon * inv_r2 * (2.0 * sr12 - sr6)
            f_on_i = -coeff * dr
            forces[i] += f_on_i
            forces[j] -= f_on_i

            rij = -dr  # r_i - r_j
            virial += np.outer(rij, f_on_i)

    return forces, float(potential), virial


def internal_stress_tensor(vel: np.ndarray, virial: np.ndarray, mass: float, volume: float) -> np.ndarray:
    kinetic = mass * vel.T @ vel
    return (kinetic + virial) / volume


def observables(
    frac: np.ndarray,
    h: np.ndarray,
    vel: np.ndarray,
    potential: float,
    stress: np.ndarray,
    target_stress: np.ndarray,
    mass: float,
) -> dict[str, float]:
    n, dim = frac.shape
    cart = frac_to_cart(frac, h)

    kinetic = 0.5 * mass * float(np.sum(vel * vel))
    dof = dim * max(n - 1, 1)
    temperature_inst = 2.0 * kinetic / dof

    volume = float(np.linalg.det(h))

    # Cell-vector lengths based on basis columns.
    cell_lengths = np.linalg.norm(h, axis=0)

    stress_error = stress - target_stress
    stress_error_norm = float(np.linalg.norm(stress_error))

    return {
        "potential": potential,
        "kinetic": kinetic,
        "temperature": temperature_inst,
        "volume": volume,
        "a_len": float(cell_lengths[0]),
        "b_len": float(cell_lengths[1]),
        "c_len": float(cell_lengths[2]),
        "h_xy": float(h[0, 1]),
        "h_xz": float(h[0, 2]),
        "h_yz": float(h[1, 2]),
        "stress_xx": float(stress[0, 0]),
        "stress_yy": float(stress[1, 1]),
        "stress_zz": float(stress[2, 2]),
        "stress_error_norm": stress_error_norm,
    }


def run_simulation(cfg: PRConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    frac = make_initial_fractional_positions(cfg.cells_per_axis)
    n_particles = frac.shape[0]

    # Intentionally anisotropic initial cell to demonstrate shape relaxation.
    h = np.array(
        [
            [3.2, 0.20, 0.05],
            [0.00, 2.9, 0.15],
            [0.00, 0.00, 2.6],
        ],
        dtype=float,
    )
    h0 = h.copy()

    vel = rng.normal(scale=np.sqrt(cfg.temperature / cfg.mass), size=(n_particles, cfg.dim))
    vel -= vel.mean(axis=0, keepdims=True)

    target_stress = np.diag(np.asarray(cfg.target_stress_diag, dtype=float))

    baro_vel = np.zeros((cfg.dim, cfg.dim), dtype=float)

    dt = cfg.dt
    gamma_p = cfg.particle_friction
    gamma_b = cfg.barostat_friction

    rows: list[dict[str, float]] = []

    for step in range(cfg.n_steps + 1):
        volume = float(np.linalg.det(h))
        if not np.isfinite(volume) or volume <= 1e-12:
            raise RuntimeError("Cell volume became non-positive; unstable integration.")

        forces, potential, virial = lj_forces_energy_virial(frac, h, cfg.epsilon, cfg.sigma, cfg.cutoff)
        stress = internal_stress_tensor(vel, virial, cfg.mass, volume)

        if step % cfg.sample_interval == 0 or step == cfg.n_steps:
            obs = observables(frac, h, vel, potential, stress, target_stress, cfg.mass)
            row = {
                "step": float(step),
                "time": step * dt,
                **obs,
            }
            rows.append(row)

        if step == cfg.n_steps:
            break

        # Particle update (Euler-Maruyama Langevin in Cartesian space).
        vel += (forces / cfg.mass) * dt
        damp = np.exp(-gamma_p * dt)
        noise_scale = np.sqrt(cfg.temperature * (1.0 - damp * damp) / cfg.mass)
        vel = damp * vel + noise_scale * rng.normal(size=vel.shape)

        cart = frac_to_cart(frac, h)
        cart = cart + vel * dt
        frac = cart_to_frac(cart, h)
        frac -= np.floor(frac)

        # Parrinello-Rahman style anisotropic cell update.
        stress_error = stress - target_stress
        accel = stress_error / cfg.barostat_mass

        baro_vel += accel * dt
        baro_vel = 0.5 * (baro_vel + baro_vel.T)  # remove rotational component
        baro_vel *= np.exp(-gamma_b * dt)

        h = h + (baro_vel @ h) * dt

        # Affine velocity correction due to changing cell metric.
        vel = vel + (vel @ baro_vel.T) * dt

    table = pd.DataFrame(rows)
    return table, h0, h


def validate(table: pd.DataFrame, h0: np.ndarray, h_final: np.ndarray, cfg: PRConfig) -> tuple[bool, dict[str, float]]:
    numeric = table.to_numpy(dtype=float)
    finite = bool(np.isfinite(numeric).all())

    min_volume = float(table["volume"].min())
    max_volume = float(table["volume"].max())

    initial_err = float(table["stress_error_norm"].iloc[0])
    final_err = float(table["stress_error_norm"].iloc[-1])

    cell_change_norm = float(np.linalg.norm(h_final - h0))

    passed = (
        finite
        and min_volume > cfg.min_volume
        and max_volume < cfg.max_volume
        and cell_change_norm > 1.0e-2
        and final_err < initial_err
    )

    metrics = {
        "finite": float(finite),
        "min_volume": min_volume,
        "max_volume": max_volume,
        "initial_stress_error_norm": initial_err,
        "final_stress_error_norm": final_err,
        "cell_change_norm": cell_change_norm,
    }
    return passed, metrics


def main() -> None:
    cfg = PRConfig()
    table, h0, h_final = run_simulation(cfg)
    passed, metrics = validate(table, h0, h_final, cfg)

    sample_idx = np.linspace(0, len(table) - 1, num=min(12, len(table)), dtype=int)
    sample = table.iloc[sample_idx]

    print("=== Parrinello-Rahman Method MVP ===")
    print(
        "particles={0}, steps={1}, dt={2}, T={3}, cutoff={4}".format(
            int(np.prod(cfg.cells_per_axis)),
            cfg.n_steps,
            cfg.dt,
            cfg.temperature,
            cfg.cutoff,
        )
    )
    print(
        "target_stress_diag=[{0:.3f}, {1:.3f}, {2:.3f}], barostat_mass={3}, barostat_friction={4}".format(
            cfg.target_stress_diag[0],
            cfg.target_stress_diag[1],
            cfg.target_stress_diag[2],
            cfg.barostat_mass,
            cfg.barostat_friction,
        )
    )
    print()

    print("Sampled trajectory statistics:")
    print(sample.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()

    print("Validation metrics:")
    print(f"finite                      = {bool(metrics['finite'])}")
    print(f"min_volume                  = {metrics['min_volume']:.6e} (min>{cfg.min_volume:.3e})")
    print(f"max_volume                  = {metrics['max_volume']:.6e} (max<{cfg.max_volume:.3e})")
    print(f"initial_stress_error_norm   = {metrics['initial_stress_error_norm']:.6e}")
    print(f"final_stress_error_norm     = {metrics['final_stress_error_norm']:.6e}")
    print(f"cell_change_norm            = {metrics['cell_change_norm']:.6e} (min>1e-2)")
    print(f"Validation: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
