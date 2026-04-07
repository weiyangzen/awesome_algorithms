"""Minimal runnable MVP for Bell's inequality (CHSH form)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CHSHAngles:
    """Measurement angles (radians) for CHSH settings."""

    a: float = 0.0
    a_prime: float = math.pi / 2
    b: float = math.pi / 4
    b_prime: float = -math.pi / 4


def local_response(theta: float, lam: np.ndarray) -> np.ndarray:
    """Deterministic local hidden-variable response in {-1, +1}."""
    return np.where(np.cos(theta - lam) >= 0.0, 1, -1).astype(np.int8)


def quantum_theory_correlator(theta_a: float, theta_b: float) -> float:
    """Singlet-state correlator E = -cos(theta_a - theta_b)."""
    return -math.cos(theta_a - theta_b)


def sample_quantum_correlator(
    theta_a: float,
    theta_b: float,
    n_shots: int,
    rng: np.random.Generator,
) -> float:
    """Sample a pair of ±1 outcomes with target correlator E."""
    if n_shots <= 0:
        raise ValueError("n_shots must be positive")

    target_e = quantum_theory_correlator(theta_a, theta_b)
    p_same = np.clip((1.0 + target_e) / 2.0, 0.0, 1.0)

    # A is unbiased ±1, and B is equal/opposite to match E[A*B]=target_e.
    a = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_shots)
    same_mask = rng.random(n_shots) < p_same
    b = np.where(same_mask, a, -a)
    return float(np.mean(a * b))


def compute_chsh(correlators: dict[str, float]) -> float:
    """Compute CHSH value from four correlators."""
    return (
        correlators["ab"]
        + correlators["abp"]
        + correlators["apb"]
        - correlators["apbp"]
    )


def run_local_hidden_variable_experiment(
    angles: CHSHAngles,
    n_shots: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run one local hidden-variable simulation for all CHSH settings."""
    if n_shots <= 0:
        raise ValueError("n_shots must be positive")

    lam = rng.uniform(0.0, 2.0 * math.pi, size=n_shots)

    a = local_response(angles.a, lam)
    ap = local_response(angles.a_prime, lam)
    b = local_response(angles.b, lam)
    bp = local_response(angles.b_prime, lam)

    corr_ab = float(np.mean(a * b))
    corr_abp = float(np.mean(a * bp))
    corr_apb = float(np.mean(ap * b))
    corr_apbp = float(np.mean(ap * bp))

    x = a * b + a * bp + ap * b - ap * bp
    max_abs_x = int(np.max(np.abs(x)))
    if max_abs_x > 2:
        raise RuntimeError(f"Local hidden-variable bound violated at sample level: {max_abs_x}")

    return {
        "ab": corr_ab,
        "abp": corr_abp,
        "apb": corr_apb,
        "apbp": corr_apbp,
        "S": float(np.mean(x)),
    }


def build_report(n_shots: int = 20_000, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build detailed correlator table and CHSH summary table."""
    if n_shots <= 0:
        raise ValueError("n_shots must be positive")

    angles = CHSHAngles()
    rng_local = np.random.default_rng(seed)
    rng_quantum = np.random.default_rng(seed + 1)

    local = run_local_hidden_variable_experiment(angles, n_shots, rng_local)

    settings = [
        ("ab", "a,b", angles.a, angles.b),
        ("abp", "a,b'", angles.a, angles.b_prime),
        ("apb", "a',b", angles.a_prime, angles.b),
        ("apbp", "a',b'", angles.a_prime, angles.b_prime),
    ]

    quantum_theory: dict[str, float] = {}
    quantum_empirical: dict[str, float] = {}
    rows: list[dict[str, float | str]] = []

    for key, label, theta_a, theta_b in settings:
        e_theory = quantum_theory_correlator(theta_a, theta_b)
        e_emp = sample_quantum_correlator(theta_a, theta_b, n_shots, rng_quantum)
        quantum_theory[key] = e_theory
        quantum_empirical[key] = e_emp
        rows.append(
            {
                "pair": label,
                "theta_a_deg": math.degrees(theta_a),
                "theta_b_deg": math.degrees(theta_b),
                "local_E": local[key],
                "quantum_E_theory": e_theory,
                "quantum_E_empirical": e_emp,
                "abs_empirical_error": abs(e_emp - e_theory),
            }
        )

    detail_df = pd.DataFrame(rows)

    s_local = local["S"]
    s_quantum_theory = compute_chsh(quantum_theory)
    s_quantum_emp = compute_chsh(quantum_empirical)

    # Approximate SE for S by independent setting blocks: Var(E)= (1-E^2)/n.
    se_s = math.sqrt(
        sum(max(0.0, 1.0 - quantum_empirical[key] ** 2) / n_shots for key, *_ in settings)
    )
    if se_s > 0:
        z_score = (abs(s_quantum_emp) - 2.0) / se_s
        p_value = float(stats.norm.sf(z_score)) if z_score > 0 else 1.0
    else:
        z_score = float("inf") if abs(s_quantum_emp) > 2.0 else 0.0
        p_value = 0.0 if abs(s_quantum_emp) > 2.0 else 1.0

    summary_df = pd.DataFrame(
        [
            {
                "metric": "local_hidden_variable_S",
                "value": s_local,
            },
            {
                "metric": "quantum_theory_S",
                "value": s_quantum_theory,
            },
            {
                "metric": "quantum_empirical_S",
                "value": s_quantum_emp,
            },
            {
                "metric": "classical_bound",
                "value": 2.0,
            },
            {
                "metric": "tsirelson_bound",
                "value": 2.0 * math.sqrt(2.0),
            },
            {
                "metric": "z_score_vs_classical_bound",
                "value": z_score,
            },
            {
                "metric": "one_sided_p_value",
                "value": p_value,
            },
        ]
    )

    # Deterministic sanity checks.
    if abs(s_local) > 2.0 + 1e-12:
        raise RuntimeError("Local hidden-variable model exceeded CHSH classical bound")

    if abs(abs(s_quantum_theory) - 2.0 * math.sqrt(2.0)) > 1e-12:
        raise RuntimeError("Unexpected CHSH quantum-theory value for chosen angles")

    if abs(s_quantum_emp) <= 2.0:
        raise RuntimeError("Quantum empirical CHSH did not violate classical bound")

    return detail_df, summary_df


def main() -> None:
    detail_df, summary_df = build_report(n_shots=20_000, seed=2026)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)

    print("=== Bell-CHSH Correlators ===")
    print(detail_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print()
    print("=== CHSH Summary ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
