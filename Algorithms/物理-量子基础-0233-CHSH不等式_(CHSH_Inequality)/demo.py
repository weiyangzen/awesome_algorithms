"""Minimal runnable MVP for CHSH inequality."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class ChshSummary:
    """Container for one CHSH result set."""

    label: str
    e_ab: float
    e_abp: float
    e_apb: float
    e_apbp: float
    s_value: float
    bound: float


def _chsh_from_correlators(
    e_ab: float,
    e_abp: float,
    e_apb: float,
    e_apbp: float,
) -> float:
    """Compute CHSH combination S = E(a,b)+E(a,b')+E(a',b)-E(a',b')."""
    return e_ab + e_abp + e_apb - e_apbp


def classical_lhv_max_bound() -> ChshSummary:
    """Exhaustively evaluate deterministic local hidden-variable strategies.

    There are 2 choices for each predefined output:
    A(a), A(a'), B(b), B(b') in {-1,+1}, i.e., 16 strategies in total.
    """
    max_abs_s = -1.0
    best = None

    for aa, aap, bb, bbp in product((-1, 1), repeat=4):
        e_ab = float(aa * bb)
        e_abp = float(aa * bbp)
        e_apb = float(aap * bb)
        e_apbp = float(aap * bbp)
        s = _chsh_from_correlators(e_ab, e_abp, e_apb, e_apbp)
        if abs(s) > max_abs_s:
            max_abs_s = abs(s)
            best = ChshSummary(
                label="classical deterministic LHV",
                e_ab=e_ab,
                e_abp=e_abp,
                e_apb=e_apb,
                e_apbp=e_apbp,
                s_value=s,
                bound=2.0,
            )

    if best is None:
        raise RuntimeError("failed to evaluate classical strategies")
    return best


def _unit_vector_xy(angle_rad: float) -> np.ndarray:
    """Unit vector in x-y plane."""
    return np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)


def _pauli() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return sx, sy, sz


def _spin_observable(direction: np.ndarray) -> np.ndarray:
    """Build sigma·n for one qubit measurement direction n."""
    sx, sy, sz = _pauli()
    nx, ny, nz = direction
    return nx * sx + ny * sy + nz * sz


def _singlet_state() -> np.ndarray:
    """|psi-> = (|01> - |10>)/sqrt(2) in computational basis."""
    state = np.zeros(4, dtype=complex)
    state[1] = 1.0 / np.sqrt(2.0)
    state[2] = -1.0 / np.sqrt(2.0)
    return state


def quantum_correlator(state: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """E(a,b) = <psi| (sigma·a ⊗ sigma·b) |psi>."""
    op = np.kron(_spin_observable(a), _spin_observable(b))
    val = np.vdot(state, op @ state)
    return float(np.real_if_close(val))


def quantum_ideal_chsh() -> ChshSummary:
    """Compute ideal quantum CHSH value at Tsirelson-optimal angles."""
    # Standard angle set in x-y plane:
    # a=0, a'=pi/2, b=pi/4, b'=-pi/4 -> |S| = 2*sqrt(2) for singlet correlations.
    a = _unit_vector_xy(0.0)
    ap = _unit_vector_xy(np.pi / 2.0)
    b = _unit_vector_xy(np.pi / 4.0)
    bp = _unit_vector_xy(-np.pi / 4.0)

    state = _singlet_state()
    e_ab = quantum_correlator(state, a, b)
    e_abp = quantum_correlator(state, a, bp)
    e_apb = quantum_correlator(state, ap, b)
    e_apbp = quantum_correlator(state, ap, bp)
    s = _chsh_from_correlators(e_ab, e_abp, e_apb, e_apbp)

    return ChshSummary(
        label="quantum singlet (ideal)",
        e_ab=e_ab,
        e_abp=e_abp,
        e_apb=e_apb,
        e_apbp=e_apbp,
        s_value=s,
        bound=2.0,
    )


def sample_binary_products(correlator: float, shots: int, rng: np.random.Generator) -> float:
    """Sample ±1 products with expectation equal to correlator.

    For outcomes x in {-1,+1}, E[x] = correlator.
    So P(x=+1)=(1+E)/2 and P(x=-1)=(1-E)/2.
    """
    p_plus = (1.0 + correlator) / 2.0
    if p_plus < 0.0 or p_plus > 1.0:
        raise ValueError("correlator must be in [-1, 1]")
    draws = rng.random(shots)
    values = np.where(draws < p_plus, 1.0, -1.0)
    return float(np.mean(values))


def quantum_sampled_chsh(shots_per_setting: int = 30_000, seed: int = 7) -> ChshSummary:
    """Estimate CHSH by finite-shot sampling around ideal quantum correlators."""
    ideal = quantum_ideal_chsh()
    rng = np.random.default_rng(seed)

    e_ab = sample_binary_products(ideal.e_ab, shots_per_setting, rng)
    e_abp = sample_binary_products(ideal.e_abp, shots_per_setting, rng)
    e_apb = sample_binary_products(ideal.e_apb, shots_per_setting, rng)
    e_apbp = sample_binary_products(ideal.e_apbp, shots_per_setting, rng)
    s = _chsh_from_correlators(e_ab, e_abp, e_apb, e_apbp)
    return ChshSummary(
        label=f"quantum singlet (sampled, shots={shots_per_setting})",
        e_ab=e_ab,
        e_abp=e_abp,
        e_apb=e_apb,
        e_apbp=e_apbp,
        s_value=s,
        bound=2.0,
    )


def print_report(results: list[ChshSummary]) -> None:
    """Print compact CHSH diagnostics."""
    print("CHSH inequality demo")
    print("Definition: S = E(a,b) + E(a,b') + E(a',b) - E(a',b')")
    print("Local hidden-variable bound: |S| <= 2")
    print()
    print("label                                 Eab      Eab'     Ea'b     Ea'b'       S      |S|")
    for item in results:
        print(
            f"{item.label:36s} "
            f"{item.e_ab:7.4f}  {item.e_abp:7.4f}  {item.e_apb:7.4f}  {item.e_apbp:7.4f}  "
            f"{item.s_value:7.4f}  {abs(item.s_value):7.4f}"
        )


def run_checks(
    classical: ChshSummary,
    quantum_ideal: ChshSummary,
    quantum_sampled: ChshSummary,
) -> None:
    """Minimal sanity checks for classical bound and quantum violation."""
    if abs(classical.s_value) > classical.bound + 1e-12:
        raise AssertionError("classical deterministic strategy exceeded CHSH bound")

    tsirelson = 2.0 * np.sqrt(2.0)
    if abs(abs(quantum_ideal.s_value) - tsirelson) > 1e-9:
        raise AssertionError("ideal quantum CHSH does not match Tsirelson value")

    if abs(quantum_sampled.s_value) <= quantum_sampled.bound:
        raise AssertionError("sampled quantum experiment failed to violate CHSH")

    if abs(quantum_sampled.s_value) < 2.5:
        raise AssertionError("sampled CHSH is too low; increase shots or check implementation")


def main() -> None:
    classical = classical_lhv_max_bound()
    quantum_ideal = quantum_ideal_chsh()
    quantum_sampled = quantum_sampled_chsh(shots_per_setting=30_000, seed=7)

    results = [classical, quantum_ideal, quantum_sampled]
    print_report(results)
    print()
    print(f"Tsirelson bound 2*sqrt(2) = {2.0 * np.sqrt(2.0):.6f}")
    run_checks(classical, quantum_ideal, quantum_sampled)
    print("All checks passed.")


if __name__ == "__main__":
    main()
