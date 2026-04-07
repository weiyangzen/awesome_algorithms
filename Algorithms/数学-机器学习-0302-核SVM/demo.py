"""Minimal runnable MVP for kernel SVM (binary classification).

This demo implements a small, explicit kernel SVM solver using a simplified
SMO-style optimization over the dual problem. It avoids treating external
libraries as a black box and keeps the training loop transparent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


class StandardScalerMVP:
    """Small feature standardization helper (mean/std)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScalerMVP":
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values")

        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        self.scale_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerMVP must be fitted before transform")
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if x.shape[1] != self.mean_.shape[0]:
            raise ValueError("feature dimension mismatch")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class KernelSVMConfig:
    c: float = 1.0
    kernel: str = "rbf"  # linear | poly | rbf
    gamma: float = 1.0
    degree: int = 3
    coef0: float = 1.0
    tol: float = 1e-3
    eps: float = 1e-5
    alpha_tol: float = 1e-6
    max_passes: int = 8
    max_iter: int = 50_000
    seed: int = 7


class KernelSVMMVP:
    """Binary kernel SVM with simplified SMO optimization."""

    def __init__(self, config: KernelSVMConfig) -> None:
        if config.c <= 0:
            raise ValueError("c must be > 0")
        if config.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if config.degree < 1:
            raise ValueError("degree must be >= 1")
        if config.tol <= 0 or config.eps <= 0 or config.alpha_tol <= 0:
            raise ValueError("tol, eps, alpha_tol must be > 0")
        if config.max_passes < 1 or config.max_iter < 1:
            raise ValueError("max_passes and max_iter must be >= 1")
        if config.kernel not in {"linear", "poly", "rbf"}:
            raise ValueError("kernel must be one of: linear, poly, rbf")

        self.cfg = config
        self.scaler = StandardScalerMVP()

        self.neg_label_: int | float | str | None = None
        self.pos_label_: int | float | str | None = None

        self.x_sv_: np.ndarray | None = None
        self.y_sv_: np.ndarray | None = None
        self.alpha_sv_: np.ndarray | None = None
        self.b_: float = 0.0

        self.alpha_full_: np.ndarray | None = None
        self.y_train_signed_: np.ndarray | None = None
        self.x_train_scaled_: np.ndarray | None = None
        self.train_kernel_matrix_: np.ndarray | None = None

        self.iterations_: int = 0
        self.support_count_: int = 0
        self.dual_objective_: float = float("nan")

    @staticmethod
    def _safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out = a @ b
        return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)

    @staticmethod
    def _validate_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y)

        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x_arr.shape[0] < 2:
            raise ValueError("need at least 2 samples")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("x contains non-finite values")

        unique = np.unique(y_arr)
        if unique.shape[0] != 2:
            raise ValueError("kernel SVM MVP supports exactly 2 classes")

        return x_arr, y_arr

    def _to_signed_labels(self, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        self.neg_label_, self.pos_label_ = classes[0], classes[1]
        return np.where(y == self.pos_label_, 1.0, -1.0)

    def _kernel_matrix(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.cfg.kernel == "linear":
            return self._safe_matmul(x1, x2.T)

        if self.cfg.kernel == "poly":
            cross = self._safe_matmul(x1, x2.T)
            return (self.cfg.gamma * cross + self.cfg.coef0) ** self.cfg.degree

        # RBF kernel
        x1_sq = np.sum(x1 * x1, axis=1, keepdims=True)
        x2_sq = np.sum(x2 * x2, axis=1, keepdims=True).T
        cross = self._safe_matmul(x1, x2.T)
        sq_dist = np.maximum(x1_sq + x2_sq - 2.0 * cross, 0.0)
        return np.exp(-self.cfg.gamma * sq_dist)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelSVMMVP":
        x_arr, y_arr = self._validate_xy(x, y)
        y_signed = self._to_signed_labels(y_arr)

        x_scaled = self.scaler.fit_transform(x_arr)
        n_samples = x_scaled.shape[0]

        k = self._kernel_matrix(x_scaled, x_scaled)
        if not np.all(np.isfinite(k)):
            raise RuntimeError("kernel matrix contains non-finite values")

        alpha = np.zeros(n_samples, dtype=float)
        b = 0.0
        rng = np.random.default_rng(self.cfg.seed)

        passes_without_change = 0
        iterations = 0

        while passes_without_change < self.cfg.max_passes and iterations < self.cfg.max_iter:
            num_changed = 0

            for i in range(n_samples):
                if iterations >= self.cfg.max_iter:
                    break

                weighted = alpha * y_signed
                f_i = float(np.dot(weighted, k[i, :]) + b)
                e_i = f_i - y_signed[i]

                violates_kkt = (
                    (y_signed[i] * e_i < -self.cfg.tol and alpha[i] < self.cfg.c)
                    or (y_signed[i] * e_i > self.cfg.tol and alpha[i] > 0)
                )
                if not violates_kkt:
                    iterations += 1
                    continue

                j = i
                while j == i:
                    j = int(rng.integers(0, n_samples))

                f_j = float(np.dot(weighted, k[j, :]) + b)
                e_j = f_j - y_signed[j]

                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]

                if y_signed[i] != y_signed[j]:
                    low = max(0.0, alpha_j_old - alpha_i_old)
                    high = min(self.cfg.c, self.cfg.c + alpha_j_old - alpha_i_old)
                else:
                    low = max(0.0, alpha_i_old + alpha_j_old - self.cfg.c)
                    high = min(self.cfg.c, alpha_i_old + alpha_j_old)

                if abs(low - high) < 1e-15:
                    iterations += 1
                    continue

                eta = 2.0 * k[i, j] - k[i, i] - k[j, j]
                if eta >= -1e-12:
                    iterations += 1
                    continue

                alpha_j_new = alpha_j_old - y_signed[j] * (e_i - e_j) / eta
                alpha_j_new = float(np.clip(alpha_j_new, low, high))
                if not np.isfinite(alpha_j_new):
                    iterations += 1
                    continue

                if abs(alpha_j_new - alpha_j_old) < self.cfg.eps:
                    iterations += 1
                    continue

                alpha_i_new = alpha_i_old + y_signed[i] * y_signed[j] * (alpha_j_old - alpha_j_new)
                alpha_i_new = float(np.clip(alpha_i_new, 0.0, self.cfg.c))
                if not np.isfinite(alpha_i_new):
                    iterations += 1
                    continue

                b1 = (
                    b
                    - e_i
                    - y_signed[i] * (alpha_i_new - alpha_i_old) * k[i, i]
                    - y_signed[j] * (alpha_j_new - alpha_j_old) * k[i, j]
                )
                b2 = (
                    b
                    - e_j
                    - y_signed[i] * (alpha_i_new - alpha_i_old) * k[i, j]
                    - y_signed[j] * (alpha_j_new - alpha_j_old) * k[j, j]
                )

                if 0 < alpha_i_new < self.cfg.c:
                    b = float(b1)
                elif 0 < alpha_j_new < self.cfg.c:
                    b = float(b2)
                else:
                    b = float(0.5 * (b1 + b2))

                alpha[i] = alpha_i_new
                alpha[j] = alpha_j_new

                num_changed += 1
                iterations += 1

            if num_changed == 0:
                passes_without_change += 1
            else:
                passes_without_change = 0

        support_mask = alpha > self.cfg.alpha_tol
        if not np.any(support_mask):
            support_mask[np.argmax(alpha)] = True

        self.x_sv_ = x_scaled[support_mask]
        self.y_sv_ = y_signed[support_mask]
        self.alpha_sv_ = alpha[support_mask]
        self.b_ = b

        self.alpha_full_ = alpha
        self.y_train_signed_ = y_signed
        self.x_train_scaled_ = x_scaled
        self.train_kernel_matrix_ = k

        self.iterations_ = iterations
        self.support_count_ = int(np.sum(support_mask))

        weighted = alpha * y_signed
        kw = self._safe_matmul(k, weighted)
        dual = alpha.sum() - 0.5 * float(np.dot(weighted, kw))
        self.dual_objective_ = float(dual)

        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.x_sv_ is None or self.y_sv_ is None or self.alpha_sv_ is None:
            raise RuntimeError("model is not fitted yet")

        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D array")

        x_scaled = self.scaler.transform(x_arr)
        k_eval = self._kernel_matrix(x_scaled, self.x_sv_)
        weights = self.alpha_sv_ * self.y_sv_
        scores = self._safe_matmul(k_eval, weights) + self.b_
        return np.asarray(scores, dtype=float)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.neg_label_ is None or self.pos_label_ is None:
            raise RuntimeError("model is not fitted yet")

        scores = self.decision_function(x)
        signed_pred = np.where(scores >= 0.0, 1.0, -1.0)
        return np.where(signed_pred > 0.0, self.pos_label_, self.neg_label_)


def make_xor_dataset(
    n_per_cluster: int = 80,
    noise_std: float = 0.35,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a non-linearly separable XOR-like 2D dataset."""

    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    labels = np.array([1, 0, 0, 1], dtype=int)

    blocks_x = []
    blocks_y = []
    for center, label in zip(centers, labels):
        x_block = rng.normal(loc=center, scale=noise_std, size=(n_per_cluster, 2))
        y_block = np.full(n_per_cluster, label, dtype=int)
        blocks_x.append(x_block)
        blocks_y.append(y_block)

    x_all = np.vstack(blocks_x)
    y_all = np.concatenate(blocks_y)

    # Add one mild nonlinear feature to emphasize kernel benefit.
    radial = np.sum(x_all * x_all, axis=1, keepdims=True)
    x_all = np.hstack([x_all, radial])

    return x_all, y_all


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be in (0, 1)")

    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    cut = int(round(n * (1.0 - test_ratio)))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> Dict[str, int]:
    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    tn = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> Dict[str, float]:
    c = confusion_counts(y_true, y_pred, positive_label=positive_label)
    precision = c["tp"] / (c["tp"] + c["fp"] + 1e-12)
    recall = c["tp"] / (c["tp"] + c["fn"] + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def summarize_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = accuracy(y_true, y_pred)
    prf = precision_recall_f1(y_true, y_pred)
    cm = confusion_counts(y_true, y_pred)
    print(f"[{name}] acc={acc:.4f} precision={prf['precision']:.4f} recall={prf['recall']:.4f} f1={prf['f1']:.4f}")
    print(f"[{name}] confusion: TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")


def main() -> None:
    x, y = make_xor_dataset(n_per_cluster=80, noise_std=0.35, seed=2026)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.3, seed=2027)

    rbf_cfg = KernelSVMConfig(
        c=1.5,
        kernel="rbf",
        gamma=0.9,
        tol=1e-3,
        eps=1e-5,
        max_passes=8,
        max_iter=50_000,
        seed=11,
    )
    rbf_model = KernelSVMMVP(rbf_cfg).fit(x_train, y_train)

    linear_cfg = KernelSVMConfig(
        c=1.5,
        kernel="linear",
        gamma=1.0,
        tol=1e-3,
        eps=1e-5,
        max_passes=8,
        max_iter=50_000,
        seed=11,
    )
    linear_model = KernelSVMMVP(linear_cfg).fit(x_train, y_train)

    y_train_pred_rbf = rbf_model.predict(x_train)
    y_test_pred_rbf = rbf_model.predict(x_test)

    y_train_pred_linear = linear_model.predict(x_train)
    y_test_pred_linear = linear_model.predict(x_test)

    majority_label = int(np.bincount(y_train).argmax())
    y_test_pred_majority = np.full_like(y_test, fill_value=majority_label)

    print("=== Kernel SVM MVP (Simplified SMO) ===")
    print(f"train size: {x_train.shape[0]} | test size: {x_test.shape[0]} | feature dim: {x_train.shape[1]}")
    print(
        "rbf hyperparams: "
        f"C={rbf_cfg.c}, gamma={rbf_cfg.gamma}, tol={rbf_cfg.tol}, "
        f"eps={rbf_cfg.eps}, max_passes={rbf_cfg.max_passes}, max_iter={rbf_cfg.max_iter}"
    )
    print(
        f"rbf training summary: iterations={rbf_model.iterations_}, "
        f"support_vectors={rbf_model.support_count_}, dual_objective={rbf_model.dual_objective_:.6f}"
    )

    summarize_metrics("RBF train", y_train, y_train_pred_rbf)
    summarize_metrics("RBF test ", y_test, y_test_pred_rbf)
    summarize_metrics("Linear train", y_train, y_train_pred_linear)
    summarize_metrics("Linear test ", y_test, y_test_pred_linear)
    summarize_metrics("Majority test", y_test, y_test_pred_majority)

    print("sample test predictions (first 8):")
    scores = rbf_model.decision_function(x_test)
    for i in range(min(8, x_test.shape[0])):
        print(
            f"idx={i:02d} y_true={int(y_test[i])} y_pred={int(y_test_pred_rbf[i])} "
            f"score={scores[i]: .4f}"
        )


if __name__ == "__main__":
    main()
