"""Quantum Measurement Theory MVP.

This script demonstrates three core ideas in quantum measurement:
1) Born rule probabilities for projective and generalized (POVM) measurements.
2) State-update (collapse/backaction) after a measurement outcome.
3) Non-commuting measurement order effects (direct Z vs X->Z sequence).

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


Array = np.ndarray


@dataclass(frozen=True)
class Measurement:
    """A finite-outcome quantum measurement represented by Kraus operators."""

    name: str
    kraus_ops: dict[str, Array]

    def effects(self) -> dict[str, Array]:
        """Return POVM effects E_k = M_k^dagger M_k."""
        return {label: op.conj().T @ op for label, op in self.kraus_ops.items()}


def density_from_ket(psi: Array) -> Array:
    """Build a density matrix rho = |psi><psi| from a state vector."""
    vec = np.asarray(psi, dtype=complex).reshape(-1, 1)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("State vector must have non-zero norm.")
    vec = vec / norm
    return vec @ vec.conj().T


def born_probabilities(rho: Array, measurement: Measurement) -> dict[str, float]:
    """Compute p(k) = Tr(E_k rho), normalized for numerical robustness."""
    probs: dict[str, float] = {}
    for label, effect in measurement.effects().items():
        prob = float(np.real(np.trace(effect @ rho)))
        probs[label] = max(prob, 0.0)

    total = float(sum(probs.values()))
    if total <= 0.0:
        raise ValueError("Invalid probability mass (<= 0).")

    for label in probs:
        probs[label] /= total
    return probs


def validate_density_matrix(rho: Array, atol: float = 1e-9) -> None:
    """Basic physicality checks: Hermitian, unit trace, PSD within tolerance."""
    if not np.allclose(rho, rho.conj().T, atol=atol):
        raise ValueError("Density matrix is not Hermitian.")

    tr = np.trace(rho)
    if not np.isclose(float(np.real(tr)), 1.0, atol=atol):
        raise ValueError("Density matrix trace is not 1.")

    eigvals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    if float(np.min(eigvals)) < -1e-8:
        raise ValueError("Density matrix is not positive semidefinite.")


def assert_measurement_complete(measurement: Measurement, atol: float = 1e-9) -> None:
    """Check completeness: sum_k M_k^dagger M_k = I."""
    dim = next(iter(measurement.kraus_ops.values())).shape[0]
    accum = np.zeros((dim, dim), dtype=complex)
    for op in measurement.kraus_ops.values():
        accum += op.conj().T @ op
    if not np.allclose(accum, np.eye(dim, dtype=complex), atol=atol):
        raise ValueError(f"Measurement '{measurement.name}' is not complete.")


def sample_prepared_state(
    rho: Array,
    measurement: Measurement,
    n_shots: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Sample outcomes with fresh state preparation each shot."""
    probs = born_probabilities(rho, measurement)
    labels = list(probs.keys())
    pvals = np.array([probs[label] for label in labels], dtype=float)

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_shots, pvals)

    empirical = {
        label: float(count) / float(n_shots) for label, count in zip(labels, counts)
    }
    return probs, empirical


def measure_once(rho: Array, measurement: Measurement, rng: np.random.Generator) -> tuple[str, Array]:
    """Perform one measurement shot and return (outcome, post_state)."""
    probs = born_probabilities(rho, measurement)
    labels = list(probs.keys())
    pvals = np.array([probs[label] for label in labels], dtype=float)
    outcome_idx = int(rng.choice(len(labels), p=pvals))
    outcome = labels[outcome_idx]

    op = measurement.kraus_ops[outcome]
    unnormalized = op @ rho @ op.conj().T
    prob = float(np.real(np.trace(unnormalized)))
    if prob <= 0.0:
        raise RuntimeError("Sampled an outcome with zero probability.")

    post_rho = unnormalized / prob
    post_rho = 0.5 * (post_rho + post_rho.conj().T)
    post_rho = post_rho / np.trace(post_rho)
    validate_density_matrix(post_rho)
    return outcome, post_rho


def collapse_repeatability(
    rho: Array,
    measurement: Measurement,
    n_trials: int,
    seed: int,
) -> float:
    """Measure twice with same projective measurement and count mismatches."""
    rng = np.random.default_rng(seed)
    mismatches = 0
    for _ in range(n_trials):
        first, post = measure_once(rho, measurement, rng)
        second, _ = measure_once(post, measurement, rng)
        if first != second:
            mismatches += 1
    return float(mismatches) / float(n_trials)


def sequence_final_distribution(
    rho: Array,
    sequence: list[Measurement],
    n_shots: int,
    seed: int,
) -> dict[str, float]:
    """Run sequential measurements and report final-step outcome frequencies."""
    rng = np.random.default_rng(seed)
    final_labels = list(sequence[-1].kraus_ops.keys())
    counts = {label: 0 for label in final_labels}

    for _ in range(n_shots):
        state = rho.copy()
        last_outcome = final_labels[0]
        for measurement in sequence:
            last_outcome, state = measure_once(state, measurement, rng)
        counts[last_outcome] += 1

    return {label: float(counts[label]) / float(n_shots) for label in final_labels}


def z_measurement() -> Measurement:
    p0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    p1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    return Measurement(name="Projective-Z", kraus_ops={"0": p0, "1": p1})


def x_measurement() -> Measurement:
    p_plus = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
    p_minus = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=complex)
    return Measurement(name="Projective-X", kraus_ops={"+": p_plus, "-": p_minus})


def unsharp_z_measurement(eta: float) -> Measurement:
    """Two-outcome diagonal POVM with strength eta in [0.5, 1]."""
    if not (0.5 <= eta <= 1.0):
        raise ValueError("eta must be in [0.5, 1.0].")
    m0 = np.array([[np.sqrt(eta), 0.0], [0.0, np.sqrt(1.0 - eta)]], dtype=complex)
    m1 = np.array([[np.sqrt(1.0 - eta), 0.0], [0.0, np.sqrt(eta)]], dtype=complex)
    return Measurement(name=f"POVM-Unsharp-Z(eta={eta:.2f})", kraus_ops={"0": m0, "1": m1})


def run_born_rule_suite(n_shots: int, base_seed: int) -> pd.DataFrame:
    ket0 = np.array([1.0, 0.0], dtype=complex)
    ket_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)

    rho0 = density_from_ket(ket0)
    rho_plus = density_from_ket(ket_plus)

    m_z = z_measurement()
    m_x = x_measurement()
    m_povm = unsharp_z_measurement(eta=0.80)

    for measurement in (m_z, m_x, m_povm):
        assert_measurement_complete(measurement)

    cases = [
        ("Z on |0>", rho0, m_z),
        ("Z on |+>", rho_plus, m_z),
        ("X on |0>", rho0, m_x),
        ("UnsharpZ(eta=0.80) on |0>", rho0, m_povm),
    ]

    rows: list[dict[str, float | str]] = []
    for idx, (case_name, rho, measurement) in enumerate(cases):
        theory, empirical = sample_prepared_state(
            rho=rho,
            measurement=measurement,
            n_shots=n_shots,
            seed=base_seed + idx,
        )
        for outcome in theory:
            th = theory[outcome]
            em = empirical[outcome]
            rows.append(
                {
                    "case": case_name,
                    "measurement": measurement.name,
                    "outcome": outcome,
                    "theory_prob": th,
                    "empirical_prob": em,
                    "abs_error": abs(th - em),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    base_seed = 20260407

    born_df = run_born_rule_suite(n_shots=30_000, base_seed=base_seed)

    ket0 = np.array([1.0, 0.0], dtype=complex)
    ket_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    rho0 = density_from_ket(ket0)
    rho_plus = density_from_ket(ket_plus)

    m_z = z_measurement()
    m_x = x_measurement()

    repeat_mismatch_rate = collapse_repeatability(
        rho=rho_plus,
        measurement=m_z,
        n_trials=5_000,
        seed=base_seed + 100,
    )

    _, direct_z = sample_prepared_state(
        rho=rho0,
        measurement=m_z,
        n_shots=20_000,
        seed=base_seed + 200,
    )
    x_then_z = sequence_final_distribution(
        rho=rho0,
        sequence=[m_x, m_z],
        n_shots=20_000,
        seed=base_seed + 300,
    )

    sequence_df = pd.DataFrame(
        [
            {
                "experiment": "Direct Z on |0>",
                "outcome": "0",
                "empirical_prob": direct_z["0"],
                "reference": 1.0,
            },
            {
                "experiment": "Direct Z on |0>",
                "outcome": "1",
                "empirical_prob": direct_z["1"],
                "reference": 0.0,
            },
            {
                "experiment": "X then Z on |0>",
                "outcome": "0",
                "empirical_prob": x_then_z["0"],
                "reference": 0.5,
            },
            {
                "experiment": "X then Z on |0>",
                "outcome": "1",
                "empirical_prob": x_then_z["1"],
                "reference": 0.5,
            },
        ]
    )
    sequence_df["abs_error"] = (sequence_df["empirical_prob"] - sequence_df["reference"]).abs()

    print("Quantum Measurement Theory MVP")
    print("-" * 72)
    print("[Born Rule Frequency Check]")
    print(born_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("[Projective Collapse Repeatability]")
    print(f"Mismatch rate after two consecutive Z measurements on |+>: {repeat_mismatch_rate:.6f}")
    print()
    print("[Non-Commuting Order Effect]")
    print(sequence_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    max_born_error = float(born_df["abs_error"].max())
    assert max_born_error < 0.02, "Born-rule frequency error is too large."

    assert repeat_mismatch_rate == 0.0, "Projective repeatability should be exact here."

    direct_p0 = float(direct_z["0"])
    seq_p0 = float(x_then_z["0"])
    assert direct_p0 > 0.999, "Direct Z on |0> should be almost deterministic outcome 0."
    assert abs(seq_p0 - 0.5) < 0.03, "X->Z sequence should produce near-uniform Z outcomes."
    assert abs(direct_p0 - seq_p0) > 0.4, "Order effect should be clearly visible."

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
