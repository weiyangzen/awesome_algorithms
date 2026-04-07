"""Relativistic Transformation of EM Fields MVP.

This script implements Lorentz transformation of electromagnetic fields
in SI units, then validates core relativistic invariants.
"""

from __future__ import annotations

import numpy as np


C = 299_792_458.0  # m/s


def _as_batch3(x: np.ndarray | list[float], name: str) -> tuple[np.ndarray, bool]:
    """Return shape (n, 3) array and whether input was a single 3-vector."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == 3:
        return arr[np.newaxis, :], True
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr, False
    raise ValueError(f"{name} must be shape (3,) or (n,3), got {arr.shape}")


def _broadcast_velocity(v: np.ndarray | list[float], n: int) -> np.ndarray:
    """Broadcast velocity to shape (n, 3)."""

    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == 3:
        return np.broadcast_to(arr, (n, 3)).copy()
    if arr.ndim == 2 and arr.shape == (n, 3):
        return arr
    raise ValueError(f"v must be shape (3,) or ({n},3), got {arr.shape}")


def lorentz_transform_fields(
    e: np.ndarray | list[float],
    b: np.ndarray | list[float],
    v: np.ndarray | list[float],
    c: float = C,
) -> tuple[np.ndarray, np.ndarray]:
    """Lorentz-transform (E, B) from frame S to frame S' moving with velocity v.

    Vector form (SI units):
      E' = gamma * (E + v x B) - ((gamma - 1) / |v|^2) * v * (v·E)
      B' = gamma * (B - (v x E)/c^2) - ((gamma - 1) / |v|^2) * v * (v·B)
    """

    e_arr, e_single = _as_batch3(e, "E")
    b_arr, b_single = _as_batch3(b, "B")
    if e_arr.shape[0] != b_arr.shape[0]:
        raise ValueError("E and B must have same batch size")

    v_arr = _broadcast_velocity(v, e_arr.shape[0])
    beta2 = np.sum(v_arr * v_arr, axis=1) / (c * c)
    if np.any(beta2 >= 1.0):
        raise ValueError("speed must satisfy |v| < c")

    gamma = 1.0 / np.sqrt(1.0 - beta2)
    v2 = np.sum(v_arr * v_arr, axis=1)
    coeff = np.zeros_like(v2)
    nz = v2 > 0.0
    coeff[nz] = (gamma[nz] - 1.0) / v2[nz]

    v_dot_e = np.sum(v_arr * e_arr, axis=1)
    v_dot_b = np.sum(v_arr * b_arr, axis=1)

    e_prime = (
        gamma[:, np.newaxis] * (e_arr + np.cross(v_arr, b_arr))
        - coeff[:, np.newaxis] * v_arr * v_dot_e[:, np.newaxis]
    )
    b_prime = (
        gamma[:, np.newaxis] * (b_arr - np.cross(v_arr, e_arr) / (c * c))
        - coeff[:, np.newaxis] * v_arr * v_dot_b[:, np.newaxis]
    )

    if e_single and b_single:
        return e_prime[0], b_prime[0]
    return e_prime, b_prime


def field_invariants(
    e: np.ndarray | list[float],
    b: np.ndarray | list[float],
    c: float = C,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Lorentz invariants I1=E^2-c^2B^2 and I2=E·B."""

    e_arr, e_single = _as_batch3(e, "E")
    b_arr, b_single = _as_batch3(b, "B")
    if e_arr.shape[0] != b_arr.shape[0]:
        raise ValueError("E and B must have same batch size")

    i1 = np.einsum("ij,ij->i", e_arr, e_arr) - (c * c) * np.einsum("ij,ij->i", b_arr, b_arr)
    i2 = np.einsum("ij,ij->i", e_arr, b_arr)
    if e_single and b_single:
        return np.array(i1[0]), np.array(i2[0])
    return i1, i2


def run_random_invariance_demo() -> None:
    print("=== Scenario A: random fields + random boosts (invariance check) ===")

    rng = np.random.default_rng(7)
    n = 256
    e = rng.normal(0.0, 2.0e4, size=(n, 3))
    b = rng.normal(0.0, 8.0e-5, size=(n, 3))

    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    speed = rng.uniform(0.0, 0.92 * C, size=n)
    v = direction * speed[:, np.newaxis]

    e_prime, b_prime = lorentz_transform_fields(e, b, v)
    i1, i2 = field_invariants(e, b)
    i1_prime, i2_prime = field_invariants(e_prime, b_prime)

    d1 = i1_prime - i1
    d2 = i2_prime - i2
    rel1 = np.max(np.abs(d1) / np.maximum(1.0, np.abs(i1)))
    rel2 = np.max(np.abs(d2) / np.maximum(1.0, np.abs(i2)))

    print(f"max abs(I1' - I1) = {np.max(np.abs(d1)):.6e}")
    print(f"max abs(I2' - I2) = {np.max(np.abs(d2)):.6e}")
    print(f"max rel(I1' - I1) = {rel1:.6e}")
    print(f"max rel(I2' - I2) = {rel2:.6e}")

    assert rel1 < 1e-11, "I1 should be Lorentz invariant within numerical precision."
    assert rel2 < 1e-11, "I2 should be Lorentz invariant within numerical precision."


def run_plane_wave_demo() -> None:
    print("\n=== Scenario B: EM radiation (plane wave) under longitudinal boost ===")

    e0 = 300.0  # V/m
    beta = 0.6

    e = np.array([0.0, e0, 0.0])
    b = np.array([0.0, 0.0, e0 / C])
    v = np.array([beta * C, 0.0, 0.0])  # along propagation direction x

    e_prime, b_prime = lorentz_transform_fields(e, b, v)
    i1_prime, i2_prime = field_invariants(e_prime, b_prime)

    scale_numeric = np.linalg.norm(e_prime) / np.linalg.norm(e)
    scale_theory = np.sqrt((1.0 - beta) / (1.0 + beta))
    wave_ratio = np.linalg.norm(e_prime) / (C * np.linalg.norm(b_prime))
    orthogonality = float(np.dot(e_prime, b_prime))

    print(f"E' = {e_prime}")
    print(f"B' = {b_prime}")
    print(f"amplitude scale numeric = {scale_numeric:.8f}")
    print(f"amplitude scale theory  = {scale_theory:.8f}")
    print(f"|E'|/(c|B'|) = {wave_ratio:.8f}")
    print(f"E'·B' = {orthogonality:.6e}")
    print(f"I1' = {float(i1_prime):.6e}, I2' = {float(i2_prime):.6e}")

    assert abs(scale_numeric - scale_theory) < 1e-12, "Plane-wave amplitude scaling mismatch."
    assert abs(wave_ratio - 1.0) < 1e-12, "Vacuum radiation should keep |E|=c|B|."
    assert abs(orthogonality) < 1e-12, "Plane wave should keep E·B=0."
    assert abs(float(i1_prime)) < 1e-8 and abs(float(i2_prime)) < 1e-12, "Radiation invariants should be ~0."


def run_pure_electric_demo() -> None:
    print("\n=== Scenario C: pure electric field generates magnetic field in moving frame ===")

    e = np.array([1.0e5, 0.0, 0.0])  # V/m
    b = np.array([0.0, 0.0, 0.0])  # T
    beta = 0.4
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    v = np.array([0.0, beta * C, 0.0])  # boost along y

    e_prime, b_prime = lorentz_transform_fields(e, b, v)
    expected_b = -gamma * np.cross(v, e) / (C * C)

    print(f"E' = {e_prime}")
    print(f"B' = {b_prime}")
    print(f"expected B' (analytic) = {expected_b}")

    rel_b = np.linalg.norm(b_prime - expected_b) / np.maximum(1.0, np.linalg.norm(expected_b))
    assert rel_b < 1e-12, "Pure-electric transformed magnetic field mismatch."


def run_inverse_transform_demo() -> None:
    print("\n=== Scenario D: inverse transform consistency ===")

    e = np.array([1200.0, -350.0, 800.0])
    b = np.array([2.0e-6, -8.0e-7, 1.0e-6])
    v = np.array([0.2 * C, -0.35 * C, 0.15 * C])

    e_prime, b_prime = lorentz_transform_fields(e, b, v)
    e_back, b_back = lorentz_transform_fields(e_prime, b_prime, -v)

    rel_e = np.linalg.norm(e_back - e) / np.maximum(1.0, np.linalg.norm(e))
    rel_b = np.linalg.norm(b_back - b) / np.maximum(1.0, np.linalg.norm(b))

    print(f"relative recovery error (E) = {rel_e:.6e}")
    print(f"relative recovery error (B) = {rel_b:.6e}")

    assert rel_e < 1e-12, "Inverse transform should recover E."
    assert rel_b < 1e-12, "Inverse transform should recover B."


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    run_random_invariance_demo()
    run_plane_wave_demo()
    run_pure_electric_demo()
    run_inverse_transform_demo()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
