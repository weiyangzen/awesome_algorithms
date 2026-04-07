"""Minimal runnable MVP for Hyperband hyper-parameter optimization.

This script implements Hyperband + Successive Halving from scratch for a
binary classification task using a hand-written logistic regression trainer.
- No interactive input
- Deterministic via fixed seeds
- Source-level traceable algorithm flow (no Hyperband black box)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log
from typing import Dict, List, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class HyperConfig:
    config_id: int
    alpha: float
    eta0: float
    batch_size: int
    momentum: float
    seed: int


@dataclass
class EvalResult:
    val_logloss: float
    val_accuracy: float


@dataclass
class EvalRecord:
    bracket_s: int
    stage_i: int
    config_id: int
    resource: int
    val_logloss: float
    val_accuracy: float


@dataclass
class StageRecord:
    bracket_s: int
    stage_i: int
    target_n: int
    target_r: int
    evaluated: int
    kept: int
    best_logloss: float
    median_logloss: float


@dataclass
class DatasetPack:
    x_train: Array
    y_train: Array
    x_val: Array
    y_val: Array
    x_test: Array
    y_test: Array


@dataclass
class HyperbandResult:
    best_config: HyperConfig
    best_val_logloss: float
    best_val_accuracy: float
    test_logloss: float
    test_accuracy: float
    all_evals: List[EvalRecord]
    stage_records: List[StageRecord]


def sigmoid(z: Array) -> Array:
    """Numerically stable sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    pos = z >= 0.0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def stratified_split_indices(y: Array, seed: int) -> Tuple[Array, Array, Array]:
    """Create deterministic stratified train/val/test indices with ratio 60/20/20."""
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)

    train_parts: List[Array] = []
    val_parts: List[Array] = []
    test_parts: List[Array] = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)

        n = idx.size
        n_train = int(round(0.60 * n))
        n_val = int(round(0.20 * n))
        n_test = n - n_train - n_val

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            raise ValueError("Stratified split failed: class count too small.")

        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train : n_train + n_val])
        test_parts.append(idx[n_train + n_val :])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)

    return (
        rng.permutation(train_idx),
        rng.permutation(val_idx),
        rng.permutation(test_idx),
    )


def standardize_by_train(x_train: Array, x_val: Array, x_test: Array) -> Tuple[Array, Array, Array]:
    """Standardize features with train-set statistics only."""
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    x_train_s = (x_train - mean) / std
    x_val_s = (x_val - mean) / std
    x_test_s = (x_test - mean) / std

    return x_train_s, x_val_s, x_test_s


def build_dataset(seed: int = 2026) -> DatasetPack:
    """Create a deterministic non-linear binary dataset."""
    rng = np.random.default_rng(seed)
    n_samples = 1800
    n_features = 20

    x = rng.normal(0.0, 1.0, size=(n_samples, n_features))

    w = rng.normal(0.0, 1.0, size=n_features)
    nonlinear = (
        0.8 * x[:, 0] * x[:, 1]
        - 0.5 * (x[:, 2] ** 2)
        + 0.35 * np.sin(2.2 * x[:, 3])
        - 0.2 * x[:, 4] * x[:, 5]
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logits = x @ w + nonlinear + 0.30 * rng.normal(size=n_samples)
    prob = sigmoid(np.clip(logits, -30.0, 30.0))
    y = (rng.random(n_samples) < prob).astype(int)

    train_idx, val_idx, test_idx = stratified_split_indices(y, seed + 1)

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    x_train, x_val, x_test = standardize_by_train(x_train, x_val, x_test)

    return DatasetPack(
        x_train=x_train.astype(float),
        y_train=y_train.astype(int),
        x_val=x_val.astype(float),
        y_val=y_val.astype(int),
        x_test=x_test.astype(float),
        y_test=y_test.astype(int),
    )


def sample_config(rng: np.random.Generator, config_id: int) -> HyperConfig:
    """Sample one hyper-parameter configuration from a compact search space."""
    batch_choices = [16, 32, 64, 128]
    momentum_choices = [0.0, 0.6, 0.9]

    return HyperConfig(
        config_id=config_id,
        alpha=10.0 ** float(rng.uniform(-6.0, -2.0)),
        eta0=10.0 ** float(rng.uniform(-3.0, -0.7)),
        batch_size=int(batch_choices[int(rng.integers(0, len(batch_choices)))]),
        momentum=float(momentum_choices[int(rng.integers(0, len(momentum_choices)))]),
        seed=10000 + config_id,
    )


def binary_logloss(y_true: Array, y_prob: Array) -> float:
    y_true = y_true.astype(float)
    p = np.clip(y_prob.astype(float), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def accuracy_from_prob(y_true: Array, y_prob: Array) -> float:
    pred = (y_prob >= 0.5).astype(int)
    return float(np.mean(pred == y_true))


def train_for_resource(
    config: HyperConfig,
    x_train: Array,
    y_train: Array,
    x_eval: Array,
    y_eval: Array,
    resource_epochs: int,
) -> EvalResult:
    """Train hand-written logistic regression with mini-batch SGD."""
    if resource_epochs <= 0:
        raise ValueError("resource_epochs must be positive")

    n_samples, n_features = x_train.shape
    if y_train.shape[0] != n_samples:
        raise ValueError("x_train and y_train row mismatch")

    rng = np.random.default_rng(config.seed)

    w = rng.normal(0.0, 0.05, size=n_features)
    b = 0.0

    vel_w = np.zeros_like(w)
    vel_b = 0.0

    bs = max(1, min(config.batch_size, n_samples))

    for _ in range(resource_epochs):
        perm = rng.permutation(n_samples)

        for start in range(0, n_samples, bs):
            idx = perm[start : start + bs]
            xb = x_train[idx]
            yb = y_train[idx].astype(float)

            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                logits = np.clip(xb @ w + b, -40.0, 40.0)
            probs = sigmoid(logits)
            err = probs - yb

            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                grad_w = (xb.T @ err) / xb.shape[0] + config.alpha * w
            grad_b = float(np.mean(err))

            if not np.all(np.isfinite(grad_w)) or not np.isfinite(grad_b):
                return EvalResult(val_logloss=float("inf"), val_accuracy=0.0)

            grad_norm = float(np.linalg.norm(grad_w))
            if grad_norm > 10.0:
                grad_w = grad_w * (10.0 / grad_norm)
                grad_b = float(np.clip(grad_b, -5.0, 5.0))

            if config.momentum > 0.0:
                vel_w = config.momentum * vel_w + grad_w
                vel_b = config.momentum * vel_b + grad_b
                w -= config.eta0 * vel_w
                b -= config.eta0 * vel_b
            else:
                w -= config.eta0 * grad_w
                b -= config.eta0 * grad_b

            w = np.clip(w, -50.0, 50.0)
            b = float(np.clip(b, -50.0, 50.0))

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        eval_prob = sigmoid(np.clip(x_eval @ w + b, -40.0, 40.0))
    return EvalResult(
        val_logloss=binary_logloss(y_eval, eval_prob),
        val_accuracy=accuracy_from_prob(y_eval, eval_prob),
    )


def run_hyperband(
    data: DatasetPack,
    max_resource: int = 27,
    eta: int = 3,
    seed: int = 2026,
) -> HyperbandResult:
    """Run Hyperband outer loop with Successive Halving inner loops."""
    if max_resource < 1:
        raise ValueError("max_resource must be >= 1")
    if eta < 2:
        raise ValueError("eta must be >= 2")

    rng = np.random.default_rng(seed)

    s_max = int(floor(log(max_resource, eta)))
    budget_total = (s_max + 1) * max_resource

    config_counter = 0
    stage_records: List[StageRecord] = []
    all_evals: List[EvalRecord] = []

    global_best_cfg: HyperConfig | None = None
    global_best_loss = float("inf")
    global_best_acc = 0.0

    for s in range(s_max, -1, -1):
        n = int(ceil((budget_total / max_resource) * (eta**s) / (s + 1)))
        r = max_resource * (eta ** (-s))

        configs: List[HyperConfig] = []
        for _ in range(n):
            config_counter += 1
            configs.append(sample_config(rng, config_counter))

        current = list(configs)

        for i in range(s + 1):
            n_i = int(floor(n * (eta ** (-i))))
            r_i = int(round(r * (eta**i)))
            r_i = max(1, min(max_resource, r_i))

            scored: List[Tuple[float, float, HyperConfig]] = []
            for cfg in current:
                ev = train_for_resource(
                    config=cfg,
                    x_train=data.x_train,
                    y_train=data.y_train,
                    x_eval=data.x_val,
                    y_eval=data.y_val,
                    resource_epochs=r_i,
                )
                all_evals.append(
                    EvalRecord(
                        bracket_s=s,
                        stage_i=i,
                        config_id=cfg.config_id,
                        resource=r_i,
                        val_logloss=ev.val_logloss,
                        val_accuracy=ev.val_accuracy,
                    )
                )
                scored.append((ev.val_logloss, ev.val_accuracy, cfg))

                if ev.val_logloss < global_best_loss:
                    global_best_loss = ev.val_logloss
                    global_best_acc = ev.val_accuracy
                    global_best_cfg = cfg

            scored.sort(key=lambda x: (x[0], -x[1]))

            if i < s:
                keep = max(1, int(floor(n_i / eta)))
                keep = min(keep, len(scored))
                current = [cfg for _, _, cfg in scored[:keep]]
            else:
                keep = len(scored)

            losses = np.array([x[0] for x in scored], dtype=float)
            stage_records.append(
                StageRecord(
                    bracket_s=s,
                    stage_i=i,
                    target_n=max(1, n_i),
                    target_r=r_i,
                    evaluated=len(scored),
                    kept=keep,
                    best_logloss=float(np.min(losses)),
                    median_logloss=float(np.median(losses)),
                )
            )

    if global_best_cfg is None:
        raise RuntimeError("Hyperband produced no evaluated configuration.")

    x_trainval = np.vstack([data.x_train, data.x_val])
    y_trainval = np.concatenate([data.y_train, data.y_val])
    test_eval = train_for_resource(
        config=global_best_cfg,
        x_train=x_trainval,
        y_train=y_trainval,
        x_eval=data.x_test,
        y_eval=data.y_test,
        resource_epochs=max_resource,
    )

    return HyperbandResult(
        best_config=global_best_cfg,
        best_val_logloss=global_best_loss,
        best_val_accuracy=global_best_acc,
        test_logloss=test_eval.val_logloss,
        test_accuracy=test_eval.val_accuracy,
        all_evals=all_evals,
        stage_records=stage_records,
    )


def print_stage_table(stage_records: Sequence[StageRecord]) -> None:
    print(
        "s | i | target_n | target_r | evaluated | kept | "
        "best_logloss | median_logloss"
    )
    print(
        "--+---+----------+----------+-----------+------+--------------+--------------"
    )
    for rec in stage_records:
        print(
            f"{rec.bracket_s:>1d} | {rec.stage_i:>1d} | {rec.target_n:>8d} | "
            f"{rec.target_r:>8d} | {rec.evaluated:>9d} | {rec.kept:>4d} | "
            f"{rec.best_logloss:>12.6f} | {rec.median_logloss:>12.6f}"
        )


def print_top_evaluations(
    all_evals: Sequence[EvalRecord],
    top_k: int = 12,
) -> None:
    top = sorted(all_evals, key=lambda x: (x.val_logloss, -x.val_accuracy))[:top_k]

    print("\nTop evaluations by validation logloss")
    print("rank | s | i | cfg_id | resource | val_logloss | val_accuracy")
    print("-----+---+---+--------+----------+-------------+-------------")
    for rank, rec in enumerate(top, start=1):
        print(
            f"{rank:>4d} | {rec.bracket_s:>1d} | {rec.stage_i:>1d} | {rec.config_id:>6d} | "
            f"{rec.resource:>8d} | {rec.val_logloss:>11.6f} | {rec.val_accuracy:>11.4f}"
        )


def format_config(config: HyperConfig) -> Dict[str, object]:
    return {
        "config_id": config.config_id,
        "alpha": round(config.alpha, 8),
        "eta0": round(config.eta0, 8),
        "batch_size": config.batch_size,
        "momentum": config.momentum,
        "seed": config.seed,
    }


def main() -> None:
    dataset_seed = 2026
    hyperband_seed = 2042
    max_resource = 27
    eta = 3

    data = build_dataset(seed=dataset_seed)
    result = run_hyperband(
        data=data,
        max_resource=max_resource,
        eta=eta,
        seed=hyperband_seed,
    )

    print("Hyperband demo (binary classification hyper-parameter search)")
    print(
        f"dataset_seed={dataset_seed}, hyperband_seed={hyperband_seed}, "
        f"max_resource={max_resource}, eta={eta}"
    )

    print("\nBracket/Stage summary")
    print_stage_table(result.stage_records)
    print_top_evaluations(result.all_evals, top_k=12)

    print("\nBest configuration")
    print(format_config(result.best_config))
    print(
        f"Best validation: logloss={result.best_val_logloss:.6f}, "
        f"accuracy={result.best_val_accuracy:.4f}"
    )
    print(
        f"Test (retrain on train+val @ max_resource): "
        f"logloss={result.test_logloss:.6f}, accuracy={result.test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
