"""Runnable MVP for Umbrella Sampling with a WHAM reconstruction.

This script builds a 1D toy free-energy problem, runs umbrella sampling in
multiple windows using Metropolis-Hastings, and combines biased histograms with
WHAM to estimate the unbiased PMF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp


@dataclass(frozen=True)
class WindowSample:
    center: float
    kappa: float
    samples: np.ndarray
    acceptance_rate: float


@dataclass(frozen=True)
class WhamResult:
    bin_centers: np.ndarray
    probability: np.ndarray
    pmf: np.ndarray
    free_energy_offsets: np.ndarray
    converged: bool
    iterations: int
    max_delta: float


@dataclass(frozen=True)
class ValidationReport:
    pass_all: bool
    mean_acceptance: float
    min_adjacent_overlap: float
    coverage_fraction: float
    pmf_rmse_vs_reference: float
    comparable_bins: int
    reasons: list[str]


def base_potential(x: np.ndarray | float) -> np.ndarray | float:
    """Dimensionless double-well potential U(x)."""
    x_arr = np.asarray(x)
    return 0.25 * x_arr**4 - 1.5 * x_arr**2 + 0.15 * x_arr


def harmonic_bias(x: np.ndarray | float, center: float, kappa: float) -> np.ndarray | float:
    x_arr = np.asarray(x)
    return 0.5 * kappa * (x_arr - center) ** 2


def metropolis_samples(
    *,
    beta: float,
    n_steps: int,
    burn_in: int,
    thin: int,
    proposal_sigma: float,
    init_x: float,
    rng: np.random.Generator,
    center: float,
    kappa: float,
) -> tuple[np.ndarray, float]:
    """Draw samples from the biased distribution exp[-beta*(U(x)+w_i(x))]."""
    if n_steps <= burn_in:
        raise ValueError("n_steps must be larger than burn_in")
    if thin <= 0:
        raise ValueError("thin must be positive")

    x = float(init_x)
    accepted = 0
    kept: list[float] = []

    def total_energy(z: float) -> float:
        return float(base_potential(z) + harmonic_bias(z, center, kappa))

    current_energy = total_energy(x)

    for step in range(n_steps):
        proposal = x + float(rng.normal(loc=0.0, scale=proposal_sigma))
        proposal_energy = total_energy(proposal)
        delta = beta * (proposal_energy - current_energy)

        if delta <= 0.0 or rng.random() < np.exp(-delta):
            x = proposal
            current_energy = proposal_energy
            accepted += 1

        if step >= burn_in and ((step - burn_in) % thin == 0):
            kept.append(x)

    samples = np.asarray(kept, dtype=float)
    acceptance_rate = accepted / n_steps
    return samples, acceptance_rate


def run_umbrella_windows(
    *,
    centers: np.ndarray,
    kappa: float,
    beta: float,
    n_steps: int,
    burn_in: int,
    thin: int,
    proposal_sigma: float,
    base_seed: int,
) -> list[WindowSample]:
    windows: list[WindowSample] = []
    for i, center in enumerate(centers):
        rng = np.random.default_rng(base_seed + 7919 * i)
        samples, acceptance = metropolis_samples(
            beta=beta,
            n_steps=n_steps,
            burn_in=burn_in,
            thin=thin,
            proposal_sigma=proposal_sigma,
            init_x=float(center),
            rng=rng,
            center=float(center),
            kappa=kappa,
        )
        windows.append(
            WindowSample(
                center=float(center),
                kappa=float(kappa),
                samples=samples,
                acceptance_rate=float(acceptance),
            )
        )
    return windows


def build_window_histograms(windows: list[WindowSample], bin_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = []
    totals = []
    for window in windows:
        c, _ = np.histogram(window.samples, bins=bin_edges)
        counts.append(c.astype(float))
        totals.append(float(c.sum()))
    return np.vstack(counts), np.asarray(totals, dtype=float)


def run_wham(
    *,
    counts: np.ndarray,
    totals: np.ndarray,
    beta: float,
    bias_matrix: np.ndarray,
    bin_centers: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> WhamResult:
    """Solve WHAM fixed-point equations for discrete histogram bins."""
    n_windows, n_bins = counts.shape
    if bias_matrix.shape != (n_windows, n_bins):
        raise ValueError("bias_matrix shape mismatch")

    f = np.zeros(n_windows, dtype=float)
    numerator = counts.sum(axis=0)
    converged = False
    max_delta = np.inf

    for it in range(1, max_iter + 1):
        log_terms = np.log(totals)[:, None] + beta * (f[:, None] - bias_matrix)
        log_denom = logsumexp(log_terms, axis=0)
        denom = np.exp(log_denom)

        p = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0)
        p_sum = p.sum()
        if p_sum <= 0:
            raise RuntimeError("WHAM produced non-positive normalization")
        p /= p_sum

        p_safe = np.clip(p, 1e-300, None)
        log_norm = logsumexp(np.log(p_safe)[None, :] - beta * bias_matrix, axis=1)
        f_new = -(1.0 / beta) * log_norm
        f_new -= f_new.mean()

        max_delta = float(np.max(np.abs(f_new - f)))
        f = f_new
        if max_delta < tol:
            converged = True
            break

    pmf = np.full(n_bins, np.inf, dtype=float)
    positive_mask = p > 0
    pmf[positive_mask] = -(1.0 / beta) * np.log(p[positive_mask])
    if np.any(np.isfinite(pmf)):
        pmf -= np.min(pmf[np.isfinite(pmf)])

    return WhamResult(
        bin_centers=bin_centers,
        probability=p,
        pmf=pmf,
        free_energy_offsets=f,
        converged=converged,
        iterations=it,
        max_delta=max_delta,
    )


def adjacent_overlap_metric(counts: np.ndarray, totals: np.ndarray) -> float:
    """Bhattacharyya overlap between neighboring windows (minimum value)."""
    overlaps: list[float] = []
    for i in range(len(totals) - 1):
        p_i = counts[i] / totals[i]
        p_j = counts[i + 1] / totals[i + 1]
        bc = float(np.sum(np.sqrt(p_i * p_j)))
        overlaps.append(bc)
    return float(min(overlaps)) if overlaps else 0.0


def unbiased_reference_pmf(
    *,
    beta: float,
    bin_edges: np.ndarray,
    n_steps: int,
    burn_in: int,
    thin: int,
    proposal_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples, _ = metropolis_samples(
        beta=beta,
        n_steps=n_steps,
        burn_in=burn_in,
        thin=thin,
        proposal_sigma=proposal_sigma,
        init_x=0.0,
        rng=rng,
        center=0.0,
        kappa=0.0,
    )
    ref_counts, _ = np.histogram(samples, bins=bin_edges)
    ref_counts = ref_counts.astype(float)
    p_ref = ref_counts / ref_counts.sum()

    pmf_ref = np.full_like(p_ref, np.inf)
    positive = p_ref > 0
    pmf_ref[positive] = -np.log(p_ref[positive])
    pmf_ref -= np.min(pmf_ref[np.isfinite(pmf_ref)])
    return pmf_ref, ref_counts


def validate_result(
    *,
    windows: list[WindowSample],
    wham: WhamResult,
    counts: np.ndarray,
    totals: np.ndarray,
    pmf_ref: np.ndarray,
    ref_counts: np.ndarray,
) -> ValidationReport:
    mean_accept = float(np.mean([w.acceptance_rate for w in windows]))
    min_overlap = adjacent_overlap_metric(counts, totals)
    coverage = float(np.mean(np.isfinite(wham.pmf)))

    compare_mask = np.isfinite(wham.pmf) & np.isfinite(pmf_ref) & (ref_counts >= 20)
    comparable_bins = int(compare_mask.sum())
    if comparable_bins > 0:
        rmse = float(np.sqrt(np.mean((wham.pmf[compare_mask] - pmf_ref[compare_mask]) ** 2)))
    else:
        rmse = float("inf")

    reasons: list[str] = []
    if not wham.converged:
        reasons.append("WHAM did not converge within max_iter")
    if not (0.15 <= mean_accept <= 0.85):
        reasons.append("mean window acceptance out of [0.15, 0.85]")
    if min_overlap < 0.03:
        reasons.append("adjacent-window overlap too small (<0.03)")
    if coverage < 0.75:
        reasons.append("finite PMF coverage too low (<0.75)")
    if comparable_bins < 30:
        reasons.append("too few bins for WHAM-vs-reference comparison (<30)")
    if rmse > 1.0:
        reasons.append("PMF RMSE vs unbiased reference too large (>1.0 kBT)")

    return ValidationReport(
        pass_all=(len(reasons) == 0),
        mean_acceptance=mean_accept,
        min_adjacent_overlap=min_overlap,
        coverage_fraction=coverage,
        pmf_rmse_vs_reference=rmse,
        comparable_bins=comparable_bins,
        reasons=reasons,
    )


def summarize_windows(windows: list[WindowSample]) -> pd.DataFrame:
    rows = []
    for i, window in enumerate(windows):
        rows.append(
            {
                "window": i,
                "center": window.center,
                "kappa": window.kappa,
                "samples": int(window.samples.size),
                "acceptance": window.acceptance_rate,
                "sample_mean": float(np.mean(window.samples)),
                "sample_std": float(np.std(window.samples, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def summarize_pmf(
    *,
    x: np.ndarray,
    pmf_wham: np.ndarray,
    pmf_ref: np.ndarray,
    counts_total: np.ndarray,
) -> pd.DataFrame:
    step = max(1, len(x) // 12)
    idx = np.arange(0, len(x), step)
    return pd.DataFrame(
        {
            "x": x[idx],
            "pmf_wham": pmf_wham[idx],
            "pmf_reference": pmf_ref[idx],
            "total_counts": counts_total[idx].astype(int),
        }
    )


# Global histogram grid used by WHAM and reference estimation.
BIN_EDGES = np.linspace(-2.5, 2.5, 121)


def main() -> None:
    beta = 1.0
    centers = np.linspace(-2.2, 2.2, 13)
    kappa = 35.0

    windows = run_umbrella_windows(
        centers=centers,
        kappa=kappa,
        beta=beta,
        n_steps=24000,
        burn_in=4000,
        thin=10,
        proposal_sigma=0.35,
        base_seed=20260407,
    )

    counts, totals = build_window_histograms(windows, BIN_EDGES)
    bin_centers = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
    bias_matrix = harmonic_bias(bin_centers[None, :], centers[:, None], kappa)

    wham = run_wham(
        counts=counts,
        totals=totals,
        beta=beta,
        bias_matrix=bias_matrix,
        bin_centers=bin_centers,
        tol=1e-8,
        max_iter=10000,
    )

    pmf_ref, ref_counts = unbiased_reference_pmf(
        beta=beta,
        bin_edges=BIN_EDGES,
        n_steps=320000,
        burn_in=10000,
        thin=20,
        proposal_sigma=0.8,
        seed=20261234,
    )

    validation = validate_result(
        windows=windows,
        wham=wham,
        counts=counts,
        totals=totals,
        pmf_ref=pmf_ref,
        ref_counts=ref_counts,
    )

    window_df = summarize_windows(windows)
    pmf_df = summarize_pmf(
        x=wham.bin_centers,
        pmf_wham=wham.pmf,
        pmf_ref=pmf_ref,
        counts_total=counts.sum(axis=0),
    )

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("=== Umbrella Sampling + WHAM MVP ===")
    print(f"windows={len(windows)}, beta={beta:.2f}, kappa={kappa:.1f}, bins={len(wham.bin_centers)}")
    print()

    print("[Window diagnostics]")
    print(window_df.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print()

    print("[WHAM diagnostics]")
    print(f"converged={wham.converged}, iterations={wham.iterations}, max_delta={wham.max_delta:.3e}")
    print(f"mean acceptance={validation.mean_acceptance:.4f}")
    print(f"min adjacent overlap={validation.min_adjacent_overlap:.4f}")
    print(f"finite PMF coverage={validation.coverage_fraction:.4f}")
    print(f"PMF RMSE vs reference={validation.pmf_rmse_vs_reference:.4f} kBT over {validation.comparable_bins} bins")
    print()

    print("[PMF sample points]")
    print(pmf_df.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print()

    if validation.pass_all:
        print("Validation: PASS")
    else:
        print("Validation: FAIL")
        for reason in validation.reasons:
            print(f"- {reason}")
        raise RuntimeError("Umbrella sampling MVP validation failed")


if __name__ == "__main__":
    main()
