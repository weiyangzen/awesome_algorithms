"""Minimal runnable MVP for 线性判别分析 (LDA).

This script implements LDA in a transparent way (no black-box estimator):
1) build a reproducible multiclass Gaussian dataset,
2) compute within-class / between-class scatter matrices,
3) solve eigenvectors of pinv(Sw) @ Sb for projection,
4) fit a shared-covariance linear discriminant classifier,
5) evaluate raw-space and projected-space accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LDAConfig:
    """Configuration for deterministic demo execution."""

    n_classes: int = 3
    n_features: int = 6
    n_per_class: int = 90
    test_ratio: float = 0.30
    n_components: int = 2
    reg: float = 1e-4
    seed: int = 103


@dataclass
class LDAProjectionModel:
    """Model artifacts for LDA projection."""

    classes: np.ndarray
    class_means: np.ndarray
    global_mean: np.ndarray
    sw: np.ndarray
    sb: np.ndarray
    eigenvalues: np.ndarray
    w: np.ndarray


@dataclass
class LDAClassifierModel:
    """Shared-covariance Gaussian classifier parameters."""

    classes: np.ndarray
    priors: np.ndarray
    means: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray


def generate_synthetic_dataset(config: LDAConfig) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible multiclass Gaussian dataset."""
    if config.n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if config.n_features < 1:
        raise ValueError("n_features must be >= 1")
    if config.n_per_class < 2:
        raise ValueError("n_per_class must be >= 2")

    rng = np.random.default_rng(config.seed)
    d = config.n_features
    k = config.n_classes

    # Build a shared positive-definite covariance matrix.
    a = rng.normal(size=(d, d))
    shared_cov = (a @ a.T) / float(d) + 0.55 * np.eye(d)

    # Class means are shifted along a structured direction to ensure separability.
    means = rng.normal(loc=0.0, scale=1.0, size=(k, d))
    direction = np.linspace(0.7, 1.3, num=d)
    offsets = np.linspace(-3.0, 3.0, num=k)
    means += offsets[:, None] * direction[None, :]

    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    for class_id in range(k):
        x_class = rng.multivariate_normal(
            mean=means[class_id],
            cov=shared_cov,
            size=config.n_per_class,
        )
        y_class = np.full(config.n_per_class, class_id, dtype=int)
        x_chunks.append(x_class)
        y_chunks.append(y_class)

    x = np.vstack(x_chunks)
    y = np.concatenate(y_chunks)
    return x.astype(float), y.astype(int)


def stratified_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple stratified split implemented with NumPy only."""
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        shuffled = cls_idx.copy()
        rng.shuffle(shuffled)

        n_test = int(round(shuffled.size * test_ratio))
        n_test = max(1, min(n_test, shuffled.size - 1))

        test_indices.extend(shuffled[:n_test].tolist())
        train_indices.extend(shuffled[n_test:].tolist())

    train_idx = np.array(train_indices, dtype=int)
    test_idx = np.array(test_indices, dtype=int)

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def compute_scatter_matrices(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute class means, global mean, Sw and Sb."""
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y must be 1D and aligned with x")

    classes = np.unique(y)
    d = x.shape[1]

    global_mean = x.mean(axis=0)
    class_means = np.zeros((classes.size, d), dtype=float)
    sw = np.zeros((d, d), dtype=float)
    sb = np.zeros((d, d), dtype=float)

    for i, cls in enumerate(classes):
        x_cls = x[y == cls]
        mu_cls = x_cls.mean(axis=0)
        class_means[i] = mu_cls

        centered = x_cls - mu_cls
        sw += centered.T @ centered

        diff = (mu_cls - global_mean).reshape(-1, 1)
        sb += x_cls.shape[0] * (diff @ diff.T)

    return classes, class_means, global_mean, sw, sb


def fit_lda_projection(
    x: np.ndarray,
    y: np.ndarray,
    n_components: int,
    reg: float,
) -> LDAProjectionModel:
    """Fit LDA projection matrix by eigendecomposition of pinv(Sw) @ Sb."""
    classes, class_means, global_mean, sw, sb = compute_scatter_matrices(x, y)

    max_components = min(classes.size - 1, x.shape[1])
    if not (1 <= n_components <= max_components):
        raise ValueError(
            f"n_components must be in [1, {max_components}], got {n_components}"
        )
    if reg <= 0.0:
        raise ValueError("reg must be positive")

    sw_reg = sw + reg * np.eye(sw.shape[0])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        m = np.linalg.pinv(sw_reg) @ sb

    eigvals, eigvecs = np.linalg.eig(m)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    w = eigvecs[:, :n_components]
    norms = np.linalg.norm(w, axis=0)
    norms = np.where(norms > 0.0, norms, 1.0)
    w = w / norms

    return LDAProjectionModel(
        classes=classes,
        class_means=class_means,
        global_mean=global_mean,
        sw=sw,
        sb=sb,
        eigenvalues=eigvals,
        w=w,
    )


def transform_with_projection(x: np.ndarray, proj_model: LDAProjectionModel) -> np.ndarray:
    """Project data to LDA subspace."""
    return (x - proj_model.global_mean) @ proj_model.w


def fit_lda_classifier(
    x: np.ndarray,
    y: np.ndarray,
    reg: float,
) -> LDAClassifierModel:
    """Fit shared-covariance Gaussian classifier (LDA classifier form)."""
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y must be 1D and aligned with x")
    if reg <= 0.0:
        raise ValueError("reg must be positive")

    classes = np.unique(y)
    n, d = x.shape
    k = classes.size

    priors = np.zeros(k, dtype=float)
    means = np.zeros((k, d), dtype=float)
    pooled_scatter = np.zeros((d, d), dtype=float)

    for i, cls in enumerate(classes):
        x_cls = x[y == cls]
        priors[i] = x_cls.shape[0] / float(n)
        means[i] = x_cls.mean(axis=0)

        centered = x_cls - means[i]
        pooled_scatter += centered.T @ centered

    denom = max(n - k, 1)
    cov = pooled_scatter / float(denom)
    cov += reg * np.eye(d)
    inv_cov = np.linalg.pinv(cov)

    return LDAClassifierModel(
        classes=classes,
        priors=priors,
        means=means,
        cov=cov,
        inv_cov=inv_cov,
    )


def predict_lda_classifier(model: LDAClassifierModel, x: np.ndarray) -> np.ndarray:
    """Predict class labels using linear discriminant scores."""
    if x.ndim != 2:
        raise ValueError("x must be 2D")

    score_columns: list[np.ndarray] = []
    for i in range(model.classes.size):
        mu = model.means[i]
        # Some BLAS backends may emit spurious floating warnings on matmul.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            linear_term = x @ (model.inv_cov @ mu)
        const_term = -0.5 * float(mu @ model.inv_cov @ mu) + np.log(model.priors[i] + 1e-12)
        score_columns.append(linear_term + const_term)

    scores = np.column_stack(score_columns)
    if np.any(~np.isfinite(scores)):
        raise FloatingPointError("Non-finite discriminant score encountered.")
    best = np.argmax(scores, axis=1)
    return model.classes[best]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return classification accuracy."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    return float(np.mean(y_true == y_pred))


def fisher_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Trace(Sb) / Trace(Sw) as a simple class-separation indicator."""
    _, _, _, sw, sb = compute_scatter_matrices(x, y)
    return float(np.trace(sb) / (np.trace(sw) + 1e-12))


def run_checks(
    config: LDAConfig,
    proj_model: LDAProjectionModel,
    acc_train_raw: float,
    acc_test_raw: float,
    acc_train_proj: float,
    acc_test_proj: float,
    fisher_raw: float,
    fisher_proj: float,
) -> None:
    """Fail fast when metrics violate MVP expectations."""
    if proj_model.w.shape != (config.n_features, config.n_components):
        raise AssertionError("Projection matrix has unexpected shape.")

    if acc_train_raw < 0.92:
        raise AssertionError(f"Raw-space train accuracy too low: {acc_train_raw:.4f}")
    if acc_test_raw < 0.88:
        raise AssertionError(f"Raw-space test accuracy too low: {acc_test_raw:.4f}")
    if acc_train_proj < 0.90:
        raise AssertionError(f"Projected-space train accuracy too low: {acc_train_proj:.4f}")
    if acc_test_proj < 0.86:
        raise AssertionError(f"Projected-space test accuracy too low: {acc_test_proj:.4f}")

    # Projection should generally improve class-separation density on this dataset.
    if fisher_proj <= fisher_raw:
        raise AssertionError(
            f"Expected projected Fisher ratio > raw Fisher ratio, got {fisher_proj:.4f} <= {fisher_raw:.4f}"
        )

    # At most K-1 non-trivial discriminant directions.
    max_components = config.n_classes - 1
    top_eigs = proj_model.eigenvalues[: max_components + 1]
    if np.any(~np.isfinite(top_eigs)):
        raise AssertionError("Top eigenvalues contain non-finite values.")


def main() -> None:
    config = LDAConfig()

    x, y = generate_synthetic_dataset(config)
    x_train, x_test, y_train, y_test = stratified_train_test_split(
        x=x,
        y=y,
        test_ratio=config.test_ratio,
        seed=config.seed + 17,
    )

    proj_model = fit_lda_projection(
        x=x_train,
        y=y_train,
        n_components=config.n_components,
        reg=config.reg,
    )
    x_train_proj = transform_with_projection(x_train, proj_model)
    x_test_proj = transform_with_projection(x_test, proj_model)

    clf_raw = fit_lda_classifier(x_train, y_train, reg=config.reg)
    clf_proj = fit_lda_classifier(x_train_proj, y_train, reg=config.reg)

    y_train_pred_raw = predict_lda_classifier(clf_raw, x_train)
    y_test_pred_raw = predict_lda_classifier(clf_raw, x_test)
    y_train_pred_proj = predict_lda_classifier(clf_proj, x_train_proj)
    y_test_pred_proj = predict_lda_classifier(clf_proj, x_test_proj)

    acc_train_raw = accuracy(y_train, y_train_pred_raw)
    acc_test_raw = accuracy(y_test, y_test_pred_raw)
    acc_train_proj = accuracy(y_train, y_train_pred_proj)
    acc_test_proj = accuracy(y_test, y_test_pred_proj)

    fisher_raw = fisher_ratio(x_train, y_train)
    fisher_proj = fisher_ratio(x_train_proj, y_train)

    run_checks(
        config=config,
        proj_model=proj_model,
        acc_train_raw=acc_train_raw,
        acc_test_raw=acc_test_raw,
        acc_train_proj=acc_train_proj,
        acc_test_proj=acc_test_proj,
        fisher_raw=fisher_raw,
        fisher_proj=fisher_proj,
    )

    total_samples = x.shape[0]
    top_eigs = proj_model.eigenvalues[: config.n_components]

    print("LDA MVP report (NumPy, from-scratch)")
    print(
        f"samples={total_samples}, features={config.n_features}, classes={config.n_classes}, "
        f"train={x_train.shape[0]}, test={x_test.shape[0]}"
    )
    print(
        f"components={config.n_components} (max={config.n_classes - 1}), reg={config.reg:.1e}, seed={config.seed}"
    )
    print("top eigenvalues:", np.array2string(top_eigs, precision=6, separator=", "))
    print()
    print("Accuracy:")
    print(f"  raw-space train  : {acc_train_raw:.4f}")
    print(f"  raw-space test   : {acc_test_raw:.4f}")
    print(f"  proj-space train : {acc_train_proj:.4f}")
    print(f"  proj-space test  : {acc_test_proj:.4f}")
    print()
    print("Fisher ratio (trace(Sb)/trace(Sw)):")
    print(f"  raw-space        : {fisher_raw:.4f}")
    print(f"  proj-space       : {fisher_proj:.4f}")
    print()
    print("All LDA checks passed.")


if __name__ == "__main__":
    main()
