"""Minimal runnable MVP for radiosity (diffuse global illumination).

This script builds a small patch scene, computes an approximate form-factor
matrix, solves the radiosity equation both iteratively and directly, and
prints deterministic validation tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigvals, solve

EPS = 1e-12


@dataclass(frozen=True)
class Patch:
    idx: int
    name: str
    center: np.ndarray
    normal: np.ndarray
    area: float
    rho_rgb: np.ndarray
    emit_rgb: np.ndarray


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise ValueError("zero-length vector cannot be normalized")
    return v / n


def build_scene() -> list[Patch]:
    """Construct a tiny Cornell-box-like patch set with one area light."""
    return [
        Patch(
            idx=0,
            name="ceiling_light",
            center=np.array([0.0, 1.9, 0.0], dtype=np.float64),
            normal=_normalize(np.array([0.0, -1.0, 0.0], dtype=np.float64)),
            area=1.0,
            rho_rgb=np.array([0.10, 0.10, 0.10], dtype=np.float64),
            emit_rgb=np.array([24.0, 22.0, 18.0], dtype=np.float64),
        ),
        Patch(
            idx=1,
            name="floor",
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            normal=_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float64)),
            area=16.0,
            rho_rgb=np.array([0.72, 0.70, 0.66], dtype=np.float64),
            emit_rgb=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        Patch(
            idx=2,
            name="left_wall",
            center=np.array([-2.0, 1.0, 0.0], dtype=np.float64),
            normal=_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float64)),
            area=8.0,
            rho_rgb=np.array([0.62, 0.24, 0.24], dtype=np.float64),
            emit_rgb=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        Patch(
            idx=3,
            name="right_wall",
            center=np.array([2.0, 1.0, 0.0], dtype=np.float64),
            normal=_normalize(np.array([-1.0, 0.0, 0.0], dtype=np.float64)),
            area=8.0,
            rho_rgb=np.array([0.24, 0.24, 0.64], dtype=np.float64),
            emit_rgb=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        Patch(
            idx=4,
            name="back_wall",
            center=np.array([0.0, 1.0, -2.0], dtype=np.float64),
            normal=_normalize(np.array([0.0, 0.0, 1.0], dtype=np.float64)),
            area=8.0,
            rho_rgb=np.array([0.58, 0.58, 0.58], dtype=np.float64),
            emit_rgb=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        Patch(
            idx=5,
            name="front_wall",
            center=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            normal=_normalize(np.array([0.0, 0.0, -1.0], dtype=np.float64)),
            area=8.0,
            rho_rgb=np.array([0.54, 0.54, 0.54], dtype=np.float64),
            emit_rgb=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
    ]


def build_form_factor_matrix(
    patches: list[Patch],
    row_target_max: float = 0.92,
) -> np.ndarray:
    """Build a dense approximate form-factor matrix via pairwise geometry."""
    n = len(patches)
    f = np.zeros((n, n), dtype=np.float64)

    for i, pi in enumerate(patches):
        for j, pj in enumerate(patches):
            if i == j:
                continue
            rij = pj.center - pi.center
            dist2 = float(np.dot(rij, rij))
            if dist2 <= EPS:
                continue

            rhat = rij / np.sqrt(dist2)
            cos_i = float(np.dot(pi.normal, rhat))
            cos_j = float(np.dot(pj.normal, -rhat))
            if cos_i <= 0.0 or cos_j <= 0.0:
                continue

            # Differential diffuse transfer approximation.
            fij = (pj.area * cos_i * cos_j) / (np.pi * dist2)
            if fij > 0.0:
                f[i, j] = fij

    np.fill_diagonal(f, 0.0)

    row_sums = f.sum(axis=1)
    for i in range(n):
        if row_sums[i] > row_target_max:
            scale = row_target_max / row_sums[i]
            f[i, :] *= scale

    return f


def iterate_radiosity(
    f: np.ndarray,
    rho_rgb: np.ndarray,
    emit_rgb: np.ndarray,
    max_iters: int = 4000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobi fixed-point iteration: B_{k+1} = E + rho * (F @ B_k)."""
    b = emit_rgb.copy()
    history: list[float] = []

    for _ in range(max_iters):
        next_b = emit_rgb + rho_rgb * (f @ b)
        delta = float(np.max(np.abs(next_b - b)))
        history.append(delta)
        b = next_b
        if delta < tol:
            return b, np.array(history, dtype=np.float64)

    raise AssertionError("radiosity iteration did not converge in max_iters")


def solve_radiosity_direct(
    f: np.ndarray,
    rho_rgb: np.ndarray,
    emit_rgb: np.ndarray,
) -> np.ndarray:
    """Direct linear solve per RGB channel.

    For each channel c: (I - diag(rho[:,c])F) * B[:,c] = E[:,c].
    """
    n = f.shape[0]
    eye = np.eye(n, dtype=np.float64)
    out = np.zeros_like(emit_rgb)

    for c in range(3):
        a = eye - np.diag(rho_rgb[:, c]) @ f
        out[:, c] = solve(a, emit_rgb[:, c], assume_a="gen")

    return out


def spectral_radius_max(f: np.ndarray, rho_rgb: np.ndarray) -> float:
    """Return max spectral radius over RGB iteration operators diag(rho_c)F."""
    max_r = 0.0
    for c in range(3):
        m = np.diag(rho_rgb[:, c]) @ f
        vals = eigvals(m)
        max_r = max(max_r, float(np.max(np.abs(vals))))
    return max_r


def run_experiment(
    patches: list[Patch],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = [p.name for p in patches]
    f = build_form_factor_matrix(patches, row_target_max=0.92)
    row_sums = f.sum(axis=1)

    rho_rgb = np.vstack([p.rho_rgb for p in patches])
    emit_rgb = np.vstack([p.emit_rgb for p in patches])

    b_iter, history = iterate_radiosity(f, rho_rgb, emit_rgb, max_iters=4000, tol=1e-10)
    b_direct = solve_radiosity_direct(f, rho_rgb, emit_rgb)

    abs_err = np.abs(b_iter - b_direct)
    max_abs_err = float(np.max(abs_err))
    residual = b_iter - (emit_rgb + rho_rgb * (f @ b_iter))
    residual_inf = float(np.max(np.abs(residual)))
    rho_spec = spectral_radius_max(f, rho_rgb)

    if float(np.max(row_sums)) >= 0.999:
        raise AssertionError("form-factor rows are not contractive enough")
    if rho_spec >= 1.0:
        raise AssertionError("iteration operator spectral radius must be < 1")
    if max_abs_err > 5e-7:
        raise AssertionError(f"iterative/direct mismatch too large: {max_abs_err}")
    if residual_inf > 1e-8:
        raise AssertionError(f"fixed-point residual too large: {residual_inf}")
    if np.any(b_iter < -1e-10):
        raise AssertionError("radiosity must stay non-negative")

    non_emissive = np.where(np.linalg.norm(emit_rgb, axis=1) <= EPS)[0]
    if non_emissive.size == 0:
        raise AssertionError("scene must include non-emissive patches")
    bounce_gain = b_iter[non_emissive] - emit_rgb[non_emissive]
    if float(np.max(bounce_gain)) <= 0.0:
        raise AssertionError("global illumination effect not observed")

    detail_rows: list[dict[str, float | int | str]] = []
    for i, p in enumerate(patches):
        detail_rows.append(
            {
                "idx": p.idx,
                "name": p.name,
                "area": p.area,
                "rho_r": float(p.rho_rgb[0]),
                "rho_g": float(p.rho_rgb[1]),
                "rho_b": float(p.rho_rgb[2]),
                "emit_r": float(p.emit_rgb[0]),
                "emit_g": float(p.emit_rgb[1]),
                "emit_b": float(p.emit_rgb[2]),
                "B_iter_r": float(b_iter[i, 0]),
                "B_iter_g": float(b_iter[i, 1]),
                "B_iter_b": float(b_iter[i, 2]),
                "B_dir_r": float(b_direct[i, 0]),
                "B_dir_g": float(b_direct[i, 1]),
                "B_dir_b": float(b_direct[i, 2]),
                "max_abs_err": float(np.max(abs_err[i])),
                "row_sum_F": float(row_sums[i]),
            }
        )

    detail_df = pd.DataFrame(detail_rows).sort_values("idx").reset_index(drop=True)

    ff_df = pd.DataFrame(f, columns=names, index=names)

    summary_df = pd.DataFrame(
        [
            {
                "patch_count": int(len(patches)),
                "iterations": int(len(history)),
                "final_delta": float(history[-1]),
                "max_row_sum_F": float(np.max(row_sums)),
                "min_row_sum_F": float(np.min(row_sums)),
                "spectral_radius_max": rho_spec,
                "max_abs_err_iter_vs_direct": max_abs_err,
                "fixed_point_residual_inf": residual_inf,
                "non_emissive_patch_max_bounce": float(np.max(bounce_gain)),
            }
        ]
    )

    return ff_df, detail_df, summary_df


def main() -> None:
    patches = build_scene()
    ff_df, detail_df, summary_df = run_experiment(patches)

    print("Patch names:")
    print(", ".join(p.name for p in patches))

    print("\nForm-factor matrix F (rounded to 4 decimals):")
    print(ff_df.round(4).to_string())

    preview_cols = [
        "idx",
        "name",
        "emit_r",
        "emit_g",
        "emit_b",
        "B_iter_r",
        "B_iter_g",
        "B_iter_b",
        "max_abs_err",
        "row_sum_F",
    ]
    print("\nPer-patch radiosity:")
    print(detail_df[preview_cols].to_string(index=False, justify="center"))

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
