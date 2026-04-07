"""Minimal runnable MVP for Rydberg-atoms blockade modeling.

This script builds a compact, transparent numerical pipeline for:
1) Rydberg-state energy (quantum-defect corrected),
2) transition frequency estimation,
3) C6 ~ n^11 van der Waals scaling,
4) blockade-radius computation,
5) geometric blockade graph simulation,
6) approximate maximum simultaneous excitations via randomized greedy MIS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Fundamental constants (SI)
HBAR_J_S = 1.054_571_817e-34
PLANCK_J_S = 6.626_070_15e-34
RYDBERG_ENERGY_J = 2.179_872_361_103_5e-18


@dataclass
class RydbergSimulationResult:
    """Container for key outputs of the MVP simulation."""

    n: int
    delta_l: float
    n_eff: float
    energy_j: float
    transition_frequency_ghz: float
    c6_j_m6: float
    blockade_radius_um: float
    n_atoms: int
    cloud_side_um: float
    best_excitation_count: int
    excitation_fraction: float
    mean_degree: float
    selected_indices: np.ndarray
    positions_um: np.ndarray
    adjacency: np.ndarray


def effective_principal_quantum_number(n: int, delta_l: float) -> float:
    """Return effective principal quantum number n* = n - delta_l."""
    if n <= 0:
        raise ValueError("n must be positive")
    n_eff = float(n) - float(delta_l)
    if n_eff <= 0.0:
        raise ValueError("effective principal quantum number must be > 0")
    return n_eff


def rydberg_binding_energy_joule(n: int, delta_l: float) -> float:
    """Hydrogenic-like binding energy with quantum defect correction.

    E_n = -Ry / (n - delta_l)^2
    """
    n_eff = effective_principal_quantum_number(n=n, delta_l=delta_l)
    return -RYDBERG_ENERGY_J / (n_eff * n_eff)


def transition_frequency_ghz(n1: int, n2: int, delta_l: float) -> float:
    """Estimate transition frequency |E2 - E1| / h in GHz."""
    e1 = rydberg_binding_energy_joule(n=n1, delta_l=delta_l)
    e2 = rydberg_binding_energy_joule(n=n2, delta_l=delta_l)
    freq_hz = abs(e2 - e1) / PLANCK_J_S
    return freq_hz / 1e9


def c6_from_scaling_j_m6(
    n: int,
    n_ref: int,
    c6_ref_j_m6: float,
    exponent: float = 11.0,
) -> float:
    """Estimate C6 by power-law scaling C6(n) = C6_ref * (n / n_ref)^exponent."""
    if n_ref <= 0:
        raise ValueError("n_ref must be positive")
    if c6_ref_j_m6 <= 0.0:
        raise ValueError("c6_ref_j_m6 must be positive")
    scale = (float(n) / float(n_ref)) ** float(exponent)
    return c6_ref_j_m6 * scale


def blockade_radius_um(c6_j_m6: float, rabi_omega_rad_s: float) -> float:
    """Compute blockade radius Rb = (|C6| / (hbar * Omega))^(1/6), returned in um."""
    if c6_j_m6 <= 0.0:
        raise ValueError("c6_j_m6 must be positive")
    if rabi_omega_rad_s <= 0.0:
        raise ValueError("rabi_omega_rad_s must be positive")
    rb_m = (abs(c6_j_m6) / (HBAR_J_S * rabi_omega_rad_s)) ** (1.0 / 6.0)
    return rb_m * 1e6


def sample_positions_square_um(
    n_atoms: int,
    side_um: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample atom positions in a 2D square cloud [0, side]^2 (um)."""
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive")
    if side_um <= 0.0:
        raise ValueError("side_um must be positive")
    return rng.uniform(0.0, side_um, size=(n_atoms, 2))


def blockade_adjacency(positions_um: np.ndarray, rb_um: float) -> np.ndarray:
    """Build adjacency matrix: edge(i,j)=True if distance(i,j) < rb_um."""
    if positions_um.ndim != 2 or positions_um.shape[1] != 2:
        raise ValueError("positions_um must have shape (N, 2)")
    if rb_um <= 0.0:
        raise ValueError("rb_um must be positive")

    diff = positions_um[:, None, :] - positions_um[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    rb2 = rb_um * rb_um
    adjacency = (dist2 < rb2) & (dist2 > 0.0)
    return adjacency


def randomized_greedy_independent_set(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    restarts: int = 300,
) -> np.ndarray:
    """Approximate maximum independent set with randomized greedy restarts.

    In blockade language, an independent set corresponds to atoms that can be
    simultaneously excited without violating pairwise blockade constraints.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if restarts <= 0:
        raise ValueError("restarts must be positive")

    n = adjacency.shape[0]
    best = np.empty(0, dtype=int)

    for _ in range(restarts):
        order = rng.permutation(n)
        blocked = np.zeros(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)

        for idx in order:
            if blocked[idx]:
                continue
            selected[idx] = True
            blocked[idx] = True
            blocked |= adjacency[idx]

        chosen = np.flatnonzero(selected)
        if chosen.size > best.size:
            best = chosen

    return best


def run_rydberg_blockade_mvp(
    *,
    n: int = 70,
    delta_l: float = 3.131_180_4,
    n_transition_target: int = 71,
    n_ref: int = 60,
    c6_ref_hz_um6: float = 1.0e9,
    rabi_omega_rad_s: float = 2.0 * np.pi * 1.5e6,
    n_atoms: int = 120,
    cloud_side_um: float = 80.0,
    restarts: int = 350,
    seed: int = 7,
) -> RydbergSimulationResult:
    """Run the end-to-end minimal simulation pipeline."""
    rng = np.random.default_rng(seed)

    n_eff = effective_principal_quantum_number(n=n, delta_l=delta_l)
    energy_j = rydberg_binding_energy_joule(n=n, delta_l=delta_l)
    trans_ghz = transition_frequency_ghz(n1=n, n2=n_transition_target, delta_l=delta_l)

    # Convert C6 reference from (Hz * um^6) to (J * m^6): multiply by h and (1e-6)^6.
    c6_ref_j_m6 = PLANCK_J_S * c6_ref_hz_um6 * 1e-36
    c6_j_m6 = c6_from_scaling_j_m6(n=n, n_ref=n_ref, c6_ref_j_m6=c6_ref_j_m6)
    rb_um = blockade_radius_um(c6_j_m6=c6_j_m6, rabi_omega_rad_s=rabi_omega_rad_s)

    positions_um = sample_positions_square_um(n_atoms=n_atoms, side_um=cloud_side_um, rng=rng)
    adjacency = blockade_adjacency(positions_um=positions_um, rb_um=rb_um)

    selected = randomized_greedy_independent_set(adjacency=adjacency, rng=rng, restarts=restarts)
    best_count = int(selected.size)
    excitation_fraction = float(best_count / n_atoms)
    mean_degree = float(np.mean(np.sum(adjacency, axis=1)))

    return RydbergSimulationResult(
        n=n,
        delta_l=delta_l,
        n_eff=n_eff,
        energy_j=energy_j,
        transition_frequency_ghz=trans_ghz,
        c6_j_m6=c6_j_m6,
        blockade_radius_um=rb_um,
        n_atoms=n_atoms,
        cloud_side_um=cloud_side_um,
        best_excitation_count=best_count,
        excitation_fraction=excitation_fraction,
        mean_degree=mean_degree,
        selected_indices=selected,
        positions_um=positions_um,
        adjacency=adjacency,
    )


def run_checks(result: RydbergSimulationResult) -> None:
    """Basic sanity and consistency checks for the MVP output."""
    if result.n_eff <= 0.0:
        raise AssertionError("n_eff must be positive")
    if not (result.energy_j < 0.0):
        raise AssertionError("Rydberg bound-state energy should be negative")
    if result.transition_frequency_ghz <= 0.0:
        raise AssertionError("transition frequency should be positive")
    if result.blockade_radius_um <= 0.0:
        raise AssertionError("blockade radius must be positive")
    if not (0 <= result.best_excitation_count <= result.n_atoms):
        raise AssertionError("best excitation count out of bounds")
    if not (0.0 <= result.excitation_fraction <= 1.0):
        raise AssertionError("excitation fraction out of bounds")

    # Validate independent-set condition on selected nodes.
    s = result.selected_indices
    if s.size > 1:
        sub = result.adjacency[np.ix_(s, s)]
        if np.any(sub):
            raise AssertionError("selected set violates blockade independence condition")


def preview_table(
    positions_um: np.ndarray,
    selected_indices: np.ndarray,
    max_rows: int = 12,
) -> str:
    """Build a small ASCII preview table for terminal output."""
    selected_mask = np.zeros(positions_um.shape[0], dtype=bool)
    selected_mask[selected_indices] = True

    rows = ["idx   x_um      y_um      selected"]
    upper = min(max_rows, positions_um.shape[0])
    for idx in range(upper):
        x_um, y_um = positions_um[idx]
        rows.append(f"{idx:3d}  {x_um:8.3f}  {y_um:8.3f}  {str(bool(selected_mask[idx])):>8s}")
    return "\n".join(rows)


def main() -> None:
    result = run_rydberg_blockade_mvp()
    run_checks(result)

    print("=== Rydberg Atoms MVP (Blockade Graph) ===")
    print(f"n = {result.n}, delta_l = {result.delta_l:.6f}, n* = {result.n_eff:.6f}")
    print(f"Binding energy E_n = {result.energy_j:.6e} J")
    print(f"Transition frequency (n -> n+1) = {result.transition_frequency_ghz:.6f} GHz")
    print(f"Estimated C6 = {result.c6_j_m6:.6e} J*m^6")
    print(f"Blockade radius Rb = {result.blockade_radius_um:.3f} um")
    print(f"Atoms = {result.n_atoms}, cloud side = {result.cloud_side_um:.1f} um")
    print(f"Mean blockade-graph degree = {result.mean_degree:.3f}")
    print(f"Approx. max simultaneous excitations = {result.best_excitation_count}")
    print(f"Excitation fraction = {result.excitation_fraction:.3f}")
    print()
    print(preview_table(result.positions_um, result.selected_indices, max_rows=12))
    print("All checks passed.")


if __name__ == "__main__":
    main()
