"""Minimal runnable MVP for 1D Euler equations with HLL/HLLC Riemann solvers.

The demo solves Sod shock tube using a first-order finite-volume method and
prints conservative diagnostics plus a contact-sharpness comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


GAMMA = 1.4
EPS = 1e-12


@dataclass
class RunResult:
    scheme: str
    nx: int
    n_steps: int
    final_time: float
    cfl: float
    mass_error: float
    momentum_error: float
    energy_error: float
    min_rho: float
    min_p: float
    contact_cells: int
    max_contact_grad: float


def primitive_to_conservative(rho: np.ndarray, u: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    """Convert primitive variables to conservative state U=[rho, rho*u, E]."""
    if np.any(rho <= 0.0):
        raise ValueError("rho must be positive")
    if np.any(p <= 0.0):
        raise ValueError("p must be positive")

    mom = rho * u
    e_int = p / (gamma - 1.0)
    e_kin = 0.5 * rho * u**2
    energy = e_int + e_kin
    return np.vstack((rho, mom, energy))


def conservative_to_primitive(U: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert U=[rho, rho*u, E] to primitive variables (rho, u, p)."""
    rho = U[0]
    mom = U[1]
    energy = U[2]

    if np.any(rho <= 0.0):
        raise ValueError("non-positive density encountered")

    u = mom / rho
    kinetic = 0.5 * rho * u**2
    p = (gamma - 1.0) * (energy - kinetic)

    if np.any(p <= 0.0):
        raise ValueError("non-positive pressure encountered")

    return rho, u, p


def flux(U: np.ndarray, gamma: float) -> np.ndarray:
    """Physical Euler flux F(U)."""
    rho, u, p = conservative_to_primitive(U, gamma)
    mom = U[1]
    energy = U[2]
    return np.array(
        [
            mom,
            mom * u + p,
            u * (energy + p),
        ]
    )


def sound_speed(rho: float, p: float, gamma: float) -> float:
    """Return thermodynamic sound speed."""
    return float(np.sqrt(gamma * p / rho))


def hll_flux(UL: np.ndarray, UR: np.ndarray, gamma: float) -> np.ndarray:
    """HLL interface flux using Davis wave-speed estimates."""
    rhoL, uL, pL = conservative_to_primitive(UL.reshape(3, 1), gamma)
    rhoR, uR, pR = conservative_to_primitive(UR.reshape(3, 1), gamma)

    rhoL_s = float(rhoL[0])
    uL_s = float(uL[0])
    pL_s = float(pL[0])
    rhoR_s = float(rhoR[0])
    uR_s = float(uR[0])
    pR_s = float(pR[0])

    aL = sound_speed(rhoL_s, pL_s, gamma)
    aR = sound_speed(rhoR_s, pR_s, gamma)

    SL = min(uL_s - aL, uR_s - aR)
    SR = max(uL_s + aL, uR_s + aR)

    FL = flux(UL.reshape(3, 1), gamma).reshape(3)
    FR = flux(UR.reshape(3, 1), gamma).reshape(3)

    if SL >= 0.0:
        return FL
    if SR <= 0.0:
        return FR

    return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)


def hllc_flux(UL: np.ndarray, UR: np.ndarray, gamma: float) -> np.ndarray:
    """HLLC interface flux with contact-wave restoration."""
    rhoL, uL, pL = conservative_to_primitive(UL.reshape(3, 1), gamma)
    rhoR, uR, pR = conservative_to_primitive(UR.reshape(3, 1), gamma)

    rhoL_s = float(rhoL[0])
    uL_s = float(uL[0])
    pL_s = float(pL[0])
    rhoR_s = float(rhoR[0])
    uR_s = float(uR[0])
    pR_s = float(pR[0])

    EL = float(UL[2])
    ER = float(UR[2])

    aL = sound_speed(rhoL_s, pL_s, gamma)
    aR = sound_speed(rhoR_s, pR_s, gamma)

    SL = min(uL_s - aL, uR_s - aR)
    SR = max(uL_s + aL, uR_s + aR)

    FL = flux(UL.reshape(3, 1), gamma).reshape(3)
    FR = flux(UR.reshape(3, 1), gamma).reshape(3)

    if SL >= 0.0:
        return FL
    if SR <= 0.0:
        return FR

    denom = rhoL_s * (SL - uL_s) - rhoR_s * (SR - uR_s)
    if abs(denom) < EPS:
        return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)

    S_star = (
        pR_s
        - pL_s
        + rhoL_s * uL_s * (SL - uL_s)
        - rhoR_s * uR_s * (SR - uR_s)
    ) / denom

    p_star_L = pL_s + rhoL_s * (SL - uL_s) * (S_star - uL_s)
    p_star_R = pR_s + rhoR_s * (SR - uR_s) * (S_star - uR_s)
    p_star = 0.5 * (p_star_L + p_star_R)

    rho_star_L = rhoL_s * (SL - uL_s) / (SL - S_star)
    rho_star_R = rhoR_s * (SR - uR_s) / (SR - S_star)

    U_star_L = np.array(
        [
            rho_star_L,
            rho_star_L * S_star,
            ((SL - uL_s) * EL - pL_s * uL_s + p_star * S_star) / (SL - S_star),
        ]
    )
    U_star_R = np.array(
        [
            rho_star_R,
            rho_star_R * S_star,
            ((SR - uR_s) * ER - pR_s * uR_s + p_star * S_star) / (SR - S_star),
        ]
    )

    if S_star >= 0.0:
        return FL + SL * (U_star_L - UL)
    return FR + SR * (U_star_R - UR)


def apply_outflow_ghosts(U: np.ndarray, ng: int) -> np.ndarray:
    """Pad conservative array with transmissive (zero-gradient) ghost cells."""
    U_ext = np.zeros((3, U.shape[1] + 2 * ng), dtype=float)
    U_ext[:, ng:-ng] = U
    U_ext[:, :ng] = U[:, [0]]
    U_ext[:, -ng:] = U[:, [-1]]
    return U_ext


def max_signal_speed(U: np.ndarray, gamma: float) -> float:
    """Maximum |u|+a in all cells."""
    rho, u, p = conservative_to_primitive(U, gamma)
    a = np.sqrt(gamma * p / rho)
    return float(np.max(np.abs(u) + a))


def finite_volume_step(
    U: np.ndarray,
    dx: float,
    dt: float,
    gamma: float,
    riemann_flux: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
) -> np.ndarray:
    """One first-order Godunov update with selected Riemann solver."""
    ng = 1
    U_ext = apply_outflow_ghosts(U, ng=ng)
    n_faces = U.shape[1] + 1

    F = np.zeros((3, n_faces), dtype=float)
    for i in range(n_faces):
        UL = U_ext[:, i]
        UR = U_ext[:, i + 1]
        F[:, i] = riemann_flux(UL, UR, gamma)

    U_next = U - (dt / dx) * (F[:, 1:] - F[:, :-1])

    rho, _u, p = conservative_to_primitive(U_next, gamma)
    if np.min(rho) <= 0.0 or np.min(p) <= 0.0:
        raise ValueError("positivity lost after update; reduce CFL")
    if not np.all(np.isfinite(U_next)):
        raise ValueError("non-finite state detected")

    return U_next


def sod_initial_condition(x: np.ndarray) -> np.ndarray:
    """Sod shock tube primitive state at t=0."""
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.where(x < 0.5, 1.0, 0.1)
    return primitive_to_conservative(rho, u, p, gamma=GAMMA)


def contact_region_metrics(rho: np.ndarray, x: np.ndarray) -> tuple[int, float]:
    """Return (cells in contact-density band, max density gradient in right half)."""
    in_band = (x > 0.55) & (x < 0.9) & (rho > 0.28) & (rho < 0.42)
    contact_cells = int(np.count_nonzero(in_band))

    right_half = (x > 0.55) & (x < 0.9)
    grad = np.abs(np.diff(rho))
    grad_mask = right_half[:-1] | right_half[1:]
    max_grad = float(np.max(grad[grad_mask])) if np.any(grad_mask) else 0.0
    return contact_cells, max_grad


def run_solver(nx: int, cfl: float, t_end: float, scheme: str) -> tuple[RunResult, np.ndarray, np.ndarray]:
    """Run Sod tube with either HLL or HLLC."""
    if nx < 50:
        raise ValueError("nx must be >= 50 for meaningful shock-tube output")
    if not (0.0 < cfl < 1.0):
        raise ValueError("cfl must be in (0,1)")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")

    riemann = hll_flux if scheme.upper() == "HLL" else hllc_flux

    x = np.linspace(0.0, 1.0, nx, endpoint=False) + 0.5 / nx
    dx = 1.0 / nx
    U = sod_initial_condition(x)

    mass0 = float(np.sum(U[0]) * dx)
    mom0 = float(np.sum(U[1]) * dx)
    energy0 = float(np.sum(U[2]) * dx)

    t = 0.0
    n_steps = 0
    while t < t_end - 1e-14:
        amax = max_signal_speed(U, gamma=GAMMA)
        dt = cfl * dx / max(amax, EPS)
        if t + dt > t_end:
            dt = t_end - t

        U = finite_volume_step(U, dx=dx, dt=dt, gamma=GAMMA, riemann_flux=riemann)
        t += dt
        n_steps += 1

    rho, _u, p = conservative_to_primitive(U, gamma=GAMMA)

    mass1 = float(np.sum(U[0]) * dx)
    mom1 = float(np.sum(U[1]) * dx)
    energy1 = float(np.sum(U[2]) * dx)

    contact_cells, max_contact_grad = contact_region_metrics(rho, x)

    result = RunResult(
        scheme=scheme.upper(),
        nx=nx,
        n_steps=n_steps,
        final_time=t,
        cfl=cfl,
        mass_error=mass1 - mass0,
        momentum_error=mom1 - mom0,
        energy_error=energy1 - energy0,
        min_rho=float(np.min(rho)),
        min_p=float(np.min(p)),
        contact_cells=contact_cells,
        max_contact_grad=max_contact_grad,
    )

    return result, x, U


def print_result_table(results: list[RunResult]) -> None:
    """Pretty-print summary metrics for multiple schemes."""
    print(
        "scheme  nx   steps  t_final  cfl   mass_err      mom_err       energy_err    "
        "min_rho   min_p    contact_cells  max|drho|"
    )
    print(
        "------  ---  -----  -------  ----  ------------  ------------  ------------  "
        "-------  -------  -------------  ---------"
    )
    for r in results:
        print(
            f"{r.scheme:<6}  {r.nx:3d}  {r.n_steps:5d}  {r.final_time:7.4f}  {r.cfl:4.2f}  "
            f"{r.mass_error:12.3e}  {r.momentum_error:12.3e}  {r.energy_error:12.3e}  "
            f"{r.min_rho:7.4f}  {r.min_p:7.4f}  {r.contact_cells:13d}  {r.max_contact_grad:9.4f}"
        )


def main() -> None:
    nx = 400
    cfl = 0.45
    t_end = 0.20

    print("=== HLL vs HLLC MVP on Sod shock tube (1D Euler) ===")
    print(f"gamma={GAMMA}, nx={nx}, cfl={cfl}, t_end={t_end}")

    hll_result, _x_hll, U_hll = run_solver(nx=nx, cfl=cfl, t_end=t_end, scheme="HLL")
    hllc_result, _x_hllc, U_hllc = run_solver(nx=nx, cfl=cfl, t_end=t_end, scheme="HLLC")

    print_result_table([hll_result, hllc_result])

    rho_hll = U_hll[0]
    rho_hllc = U_hllc[0]
    density_l1_diff = float(np.mean(np.abs(rho_hllc - rho_hll)))
    print(f"\nMean |rho_HLLC - rho_HLL| = {density_l1_diff:.6e}")

    if hll_result.min_rho <= 0.0 or hll_result.min_p <= 0.0:
        raise AssertionError("HLL result violates positivity")
    if hllc_result.min_rho <= 0.0 or hllc_result.min_p <= 0.0:
        raise AssertionError("HLLC result violates positivity")

    if hllc_result.contact_cells < hll_result.contact_cells:
        print("Contact-capturing check: HLLC uses fewer mixed cells than HLL (expected).")
    else:
        print("Contact-capturing check: no clear win in mixed-cell count at this resolution.")

    print("All MVP checks completed.")


if __name__ == "__main__":
    main()
