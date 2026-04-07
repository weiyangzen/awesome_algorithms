"""SMO algorithm MVP for binary SVM training.

This demo intentionally implements SMO updates in source code
instead of using sklearn.svm.SVC as a black box.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class SMOConfig:
    """Hyper-parameters for simplified SMO."""

    C: float = 2.0
    tol: float = 1e-3
    max_passes: int = 8
    max_iter: int = 300
    kernel: str = "rbf"
    gamma: float = 1.5
    seed: int = 42
    alpha_eps: float = 1e-6


class BinarySMO:
    """Binary SVM trained by Platt-style SMO pair updates."""

    def __init__(self, config: SMOConfig):
        self.config = config
        self.alphas_: np.ndarray | None = None
        self.b_: float = 0.0
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.K_train_: np.ndarray | None = None
        self.support_mask_: np.ndarray | None = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.config.kernel == "linear":
            return X1 @ X2.T
        if self.config.kernel == "rbf":
            sq_dist = cdist(X1, X2, metric="sqeuclidean")
            return np.exp(-self.config.gamma * sq_dist)
        raise ValueError(f"Unsupported kernel: {self.config.kernel}")

    def _f_idx(self, idx: int) -> float:
        """Decision value f(x_i) for a training sample index."""
        assert self.alphas_ is not None
        assert self.y_train_ is not None
        assert self.K_train_ is not None
        return float(np.dot(self.alphas_ * self.y_train_, self.K_train_[:, idx]) + self.b_)

    def _error_idx(self, idx: int) -> float:
        assert self.y_train_ is not None
        return self._f_idx(idx) - float(self.y_train_[idx])

    def fit(self, X: np.ndarray, y: np.ndarray, verbose_every: int = 40) -> "BinarySMO":
        """Train SVM with simplified SMO.

        Parameters
        ----------
        X:
            Feature matrix (n_samples, n_features).
        y:
            Labels in {-1, +1}.
        """
        y = y.astype(np.float64)
        if set(np.unique(y)) != {-1.0, 1.0}:
            raise ValueError("SMO expects labels encoded as -1/+1.")

        n_samples = X.shape[0]
        self.X_train_ = X.astype(np.float64, copy=True)
        self.y_train_ = y
        self.alphas_ = np.zeros(n_samples, dtype=np.float64)
        self.b_ = 0.0
        self.K_train_ = self._kernel(self.X_train_, self.X_train_)

        rng = np.random.default_rng(self.config.seed)
        passes = 0
        iteration = 0

        while passes < self.config.max_passes and iteration < self.config.max_iter:
            changed_alphas = 0

            for i in range(n_samples):
                E_i = self._error_idx(i)
                y_i = self.y_train_[i]
                a_i_old = self.alphas_[i]

                violate_low = y_i * E_i < -self.config.tol and a_i_old < self.config.C - self.config.alpha_eps
                violate_high = y_i * E_i > self.config.tol and a_i_old > self.config.alpha_eps
                if not (violate_low or violate_high):
                    continue

                j = i
                while j == i:
                    j = int(rng.integers(0, n_samples))

                E_j = self._error_idx(j)
                y_j = self.y_train_[j]
                a_j_old = self.alphas_[j]

                if y_i != y_j:
                    L = max(0.0, a_j_old - a_i_old)
                    H = min(self.config.C, self.config.C + a_j_old - a_i_old)
                else:
                    L = max(0.0, a_i_old + a_j_old - self.config.C)
                    H = min(self.config.C, a_i_old + a_j_old)

                if H - L < self.config.alpha_eps:
                    continue

                K_ii = self.K_train_[i, i]
                K_jj = self.K_train_[j, j]
                K_ij = self.K_train_[i, j]
                eta = K_ii + K_jj - 2.0 * K_ij
                if eta <= 1e-12:
                    continue

                a_j_new = a_j_old + y_j * (E_i - E_j) / eta
                a_j_new = float(np.clip(a_j_new, L, H))

                if abs(a_j_new - a_j_old) < self.config.alpha_eps:
                    continue

                a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)

                b1 = (
                    self.b_
                    - E_i
                    - y_i * (a_i_new - a_i_old) * K_ii
                    - y_j * (a_j_new - a_j_old) * K_ij
                )
                b2 = (
                    self.b_
                    - E_j
                    - y_i * (a_i_new - a_i_old) * K_ij
                    - y_j * (a_j_new - a_j_old) * K_jj
                )

                self.alphas_[i] = a_i_new
                self.alphas_[j] = a_j_new

                if self.config.alpha_eps < a_i_new < self.config.C - self.config.alpha_eps:
                    self.b_ = b1
                elif self.config.alpha_eps < a_j_new < self.config.C - self.config.alpha_eps:
                    self.b_ = b2
                else:
                    self.b_ = 0.5 * (b1 + b2)

                changed_alphas += 1

            iteration += 1
            passes = passes + 1 if changed_alphas == 0 else 0

            should_log = (
                iteration % verbose_every == 0
                or passes == self.config.max_passes
                or iteration == self.config.max_iter
            )
            if should_log:
                objective = self.dual_objective()
                support_count = int(np.sum(self.alphas_ > self.config.alpha_eps))
                print(
                    f"[{self.config.kernel}] iter={iteration:03d} "
                    f"changed={changed_alphas:03d} passes={passes} "
                    f"support={support_count:03d} dual_obj={objective:.4f}"
                )

        self.support_mask_ = self.alphas_ > self.config.alpha_eps
        return self

    def dual_objective(self) -> float:
        """Current dual objective value for diagnostics."""
        assert self.alphas_ is not None
        assert self.y_train_ is not None
        assert self.K_train_ is not None
        alpha = self.alphas_
        y = self.y_train_
        term1 = np.sum(alpha)
        yy = np.outer(y, y)
        aa = np.outer(alpha, alpha)
        term2 = 0.5 * np.sum(aa * yy * self.K_train_)
        return float(term1 - term2)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.alphas_ is not None
        assert self.y_train_ is not None
        assert self.X_train_ is not None
        sv_mask = self.alphas_ > self.config.alpha_eps
        if not np.any(sv_mask):
            raise RuntimeError("No support vectors found. Training likely failed to converge.")

        alpha_sv = self.alphas_[sv_mask]
        y_sv = self.y_train_[sv_mask]
        X_sv = self.X_train_[sv_mask]
        K = self._kernel(X.astype(np.float64), X_sv)
        return K @ (alpha_sv * y_sv) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        score = self.decision_function(X)
        return np.where(score >= 0.0, 1, -1)


def build_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a non-linear binary classification dataset."""
    X, y01 = make_moons(n_samples=260, noise=0.22, random_state=seed)
    y = np.where(y01 == 1, 1, -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64)


def run_experiment(config: SMOConfig, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float | str | int]:
    model = BinarySMO(config)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        "kernel": config.kernel,
        "C": config.C,
        "gamma": config.gamma if config.kernel == "rbf" else np.nan,
        "train_acc": accuracy_score(y_train, y_pred_train),
        "test_acc": accuracy_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test, pos_label=1),
        "support_vectors": int(np.sum(model.alphas_ > config.alpha_eps)),
        "dual_objective": model.dual_objective(),
        "train_seconds": elapsed,
    }


def main() -> None:
    seed = 42
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = build_dataset(seed=seed)

    configs = [
        SMOConfig(kernel="linear", C=1.5, gamma=1.0, seed=seed),
        SMOConfig(kernel="rbf", C=2.0, gamma=1.5, seed=seed),
    ]

    rows: list[dict[str, float | str | int]] = []
    for cfg in configs:
        print(f"\n=== Training SMO SVM: kernel={cfg.kernel} ===")
        rows.append(run_experiment(cfg, X_train, X_test, y_train, y_test))

    df = pd.DataFrame(rows)
    metric_cols = ["kernel", "C", "gamma", "train_acc", "test_acc", "test_f1", "support_vectors", "dual_objective", "train_seconds"]

    print("\n=== Summary Metrics ===")
    print(df[metric_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_idx = int(df["test_acc"].idxmax())
    best_kernel = str(df.loc[best_idx, "kernel"])
    print(f"\nBest config by test_acc: {best_kernel}")


if __name__ == "__main__":
    main()
