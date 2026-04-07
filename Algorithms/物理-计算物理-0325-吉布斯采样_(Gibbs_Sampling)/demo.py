"""Gibbs Sampling MVP for a coupled Gaussian system in computational physics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gibbs_bivariate_gaussian(
    rho: float,
    n_samples: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a correlated 2D Gaussian using coordinate-wise Gibbs updates.

    Target distribution:
        (x, y) ~ N(0, [[1, rho], [rho, 1]])
    Conditional updates:
        x | y ~ N(rho * y, 1 - rho^2)
        y | x ~ N(rho * x, 1 - rho^2)
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be in (-1, 1) for a valid covariance matrix.")
    if n_samples <= 0 or burn_in < 0 or thin <= 0:
        raise ValueError("n_samples > 0, burn_in >= 0, thin > 0 are required.")

    sigma_cond = np.sqrt(1.0 - rho * rho)
    x, y = 0.0, 0.0

    out = np.empty((n_samples, 2), dtype=np.float64)
    kept = 0
    total_iters = burn_in + n_samples * thin

    for step in range(total_iters):
        x = rng.normal(loc=rho * y, scale=sigma_cond)
        y = rng.normal(loc=rho * x, scale=sigma_cond)
        if step >= burn_in and (step - burn_in) % thin == 0:
            out[kept, 0] = x
            out[kept, 1] = y
            kept += 1

    return out


def lag1_autocorr(series: np.ndarray) -> float:
    """Estimate lag-1 autocorrelation with a numerically safe formula."""
    centered = series - series.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 1e-15:
        return 0.0
    numer = float(np.dot(centered[1:], centered[:-1]))
    return numer / denom


def approx_ess(n: int, rho1: float) -> float:
    """Crude ESS approximation for AR(1)-like chains."""
    rho1 = float(np.clip(rho1, -0.999, 0.999))
    return n * (1.0 - rho1) / (1.0 + rho1)


def main() -> None:
    seed = 20260407
    rho = 0.92
    n_samples = 12_000
    burn_in = 2_500
    thin = 2

    rng = np.random.default_rng(seed)
    samples = gibbs_bivariate_gaussian(
        rho=rho,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        rng=rng,
    )

    df = pd.DataFrame(samples, columns=["x", "y"])
    sample_mean = df.mean().to_numpy()
    sample_cov = df.cov(ddof=1).to_numpy()
    target_cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)

    mean_l2_error = float(np.linalg.norm(sample_mean, ord=2))
    cov_fro_error = float(np.linalg.norm(sample_cov - target_cov, ord="fro"))

    acf1_x = lag1_autocorr(df["x"].to_numpy())
    acf1_y = lag1_autocorr(df["y"].to_numpy())
    ess_x = approx_ess(n_samples, acf1_x)
    ess_y = approx_ess(n_samples, acf1_y)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    # Quadratic energy for the correlated Gaussian (analogous to a coupled mode).
    energy = (x * x - 2.0 * rho * x * y + y * y) / (2.0 * (1.0 - rho * rho))
    avg_energy = float(energy.mean())

    summary = pd.DataFrame(
        [
            ("samples_kept", float(n_samples)),
            ("burn_in", float(burn_in)),
            ("thin", float(thin)),
            ("rho_target", rho),
            ("mean_l2_error", mean_l2_error),
            ("cov_fro_error", cov_fro_error),
            ("lag1_acf_x", acf1_x),
            ("lag1_acf_y", acf1_y),
            ("approx_ess_x", ess_x),
            ("approx_ess_y", ess_y),
            ("avg_energy", avg_energy),
        ],
        columns=["metric", "value"],
    )

    print("=== Gibbs Sampling MVP: Correlated 2D Gaussian ===")
    print(f"seed={seed}, rho={rho}, n_samples={n_samples}, burn_in={burn_in}, thin={thin}")
    print("\nSample mean vector [x, y]:")
    print(np.array2string(sample_mean, precision=6, floatmode="fixed"))
    print("\nSample covariance:")
    print(np.array2string(sample_cov, precision=6, floatmode="fixed"))
    print("\nTarget covariance:")
    print(np.array2string(target_cov, precision=6, floatmode="fixed"))
    print("\nDiagnostics:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
