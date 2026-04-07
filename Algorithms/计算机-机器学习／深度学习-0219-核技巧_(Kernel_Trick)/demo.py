"""Kernel Trick minimal runnable MVP.

This script demonstrates why kernels help on nonlinearly separable data by:
1) comparing linear SVM vs RBF-kernel SVM,
2) implementing dual-form kernel ridge classification from scratch,
3) verifying an explicit polynomial feature map equals a polynomial kernel matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


KernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class KernelRidgeModel:
    x_train: np.ndarray
    alpha: np.ndarray
    kernel_name: str
    kernel_param: float
    reg: float


def set_seed(seed: int = 219) -> None:
    np.random.seed(seed)


def polynomial_kernel(x: np.ndarray, z: np.ndarray, degree: int = 2, coef0: float = 1.0) -> np.ndarray:
    return (x @ z.T + coef0) ** degree


def rbf_kernel(x: np.ndarray, z: np.ndarray, gamma: float) -> np.ndarray:
    x_sq = np.sum(x**2, axis=1, keepdims=True)
    z_sq = np.sum(z**2, axis=1, keepdims=True).T
    sq_dist = x_sq + z_sq - 2.0 * (x @ z.T)
    sq_dist = np.maximum(sq_dist, 0.0)
    return np.exp(-gamma * sq_dist)


def poly2_explicit_feature_map(x: np.ndarray) -> np.ndarray:
    """Feature map phi(x) for k(x,z)=(x·z+1)^2 when x in R^2."""
    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.column_stack(
        [
            np.ones(x.shape[0]),
            np.sqrt(2.0) * x1,
            np.sqrt(2.0) * x2,
            x1**2,
            np.sqrt(2.0) * x1 * x2,
            x2**2,
        ]
    )


def kernel_ridge_fit(x_train: np.ndarray, y_pm1: np.ndarray, kernel_fn: KernelFn, reg: float) -> KernelRidgeModel:
    k_train = kernel_fn(x_train, x_train)
    n = k_train.shape[0]
    alpha = np.linalg.solve(k_train + reg * np.eye(n), y_pm1)
    return KernelRidgeModel(
        x_train=x_train,
        alpha=alpha,
        kernel_name="rbf",
        kernel_param=float(getattr(kernel_fn, "gamma", np.nan)) if hasattr(kernel_fn, "gamma") else np.nan,
        reg=reg,
    )


def kernel_ridge_decision(model: KernelRidgeModel, x_test: np.ndarray, kernel_fn: KernelFn) -> np.ndarray:
    k_test_train = kernel_fn(x_test, model.x_train)
    return k_test_train @ model.alpha


def kernel_ridge_predict(model: KernelRidgeModel, x_test: np.ndarray, kernel_fn: KernelFn) -> np.ndarray:
    scores = kernel_ridge_decision(model, x_test, kernel_fn)
    return (scores >= 0.0).astype(int)


def main() -> None:
    set_seed(219)

    x, y = make_circles(n_samples=1200, factor=0.45, noise=0.08, random_state=219)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=219,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Check the kernel trick identity for degree-2 polynomial kernel on R^2.
    x_probe = x_train[:120]
    phi = poly2_explicit_feature_map(x_probe)
    k_explicit = phi @ phi.T
    k_poly = polynomial_kernel(x_probe, x_probe, degree=2, coef0=1.0)
    poly_kernel_match_max_abs_err = float(np.max(np.abs(k_explicit - k_poly)))

    linear_svm = SVC(kernel="linear", C=1.0)
    linear_svm.fit(x_train, y_train)
    pred_linear = linear_svm.predict(x_test)
    acc_linear = float(accuracy_score(y_test, pred_linear))

    rbf_gamma = 2.0
    rbf_svm = SVC(kernel="rbf", C=2.0, gamma=rbf_gamma)
    rbf_svm.fit(x_train, y_train)
    pred_rbf_svm = rbf_svm.predict(x_test)
    acc_rbf_svm = float(accuracy_score(y_test, pred_rbf_svm))

    y_train_pm1 = np.where(y_train == 1, 1.0, -1.0)
    rbf_kernel_fn: KernelFn = lambda a, b: rbf_kernel(a, b, gamma=rbf_gamma)
    krr_model = kernel_ridge_fit(x_train, y_train_pm1, kernel_fn=rbf_kernel_fn, reg=1e-2)
    pred_krr = kernel_ridge_predict(krr_model, x_test, kernel_fn=rbf_kernel_fn)
    acc_krr = float(accuracy_score(y_test, pred_krr))

    summary = pd.DataFrame(
        [
            {
                "model": "Linear SVM (no kernel trick)",
                "test_accuracy": acc_linear,
                "detail": "kernel=linear, C=1.0",
            },
            {
                "model": "RBF SVM (kernel trick)",
                "test_accuracy": acc_rbf_svm,
                "detail": f"kernel=rbf, C=2.0, gamma={rbf_gamma}",
            },
            {
                "model": "Custom Dual Kernel Ridge",
                "test_accuracy": acc_krr,
                "detail": f"rbf gamma={rbf_gamma}, reg=1e-2",
            },
        ]
    ).sort_values("test_accuracy", ascending=False, ignore_index=True)

    print("=== Kernel Trick MVP: Nonlinear Circles Classification ===")
    print(f"train size={x_train.shape[0]}, test size={x_test.shape[0]}")
    print(f"poly2 explicit-map vs kernel max |diff| = {poly_kernel_match_max_abs_err:.3e}")
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Quality gates: reproducible and educationally meaningful.
    assert poly_kernel_match_max_abs_err < 1e-10, "Explicit map and polynomial kernel mismatch."
    assert acc_rbf_svm > acc_linear + 0.10, "Kernelized model should clearly beat linear baseline."
    assert acc_rbf_svm > 0.90, "RBF SVM accuracy unexpectedly low."
    assert acc_krr > 0.90, "Custom dual kernel ridge accuracy unexpectedly low."
    assert abs(acc_krr - acc_rbf_svm) < 0.08, "Custom kernel model too far from SVM performance."

    print("\nQuality gates passed.")


if __name__ == "__main__":
    main()
