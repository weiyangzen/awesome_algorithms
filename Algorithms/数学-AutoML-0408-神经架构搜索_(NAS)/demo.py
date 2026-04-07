"""Neural Architecture Search (NAS) MVP on a toy classification task.

This script demonstrates a minimal discrete NAS pipeline:
1) define a small architecture search space,
2) train each candidate on train split,
3) rank by validation accuracy,
4) retrain the best architecture on train+val,
5) report final test accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ArchSpec:
    depth: int
    hidden_dim: int
    activation: str


@dataclass(frozen=True)
class SearchResult:
    spec: ArchSpec
    val_acc: float
    best_epoch: int
    param_count: int
    train_loss_at_best: float


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU-only deterministic behavior for this MVP.
    torch.use_deterministic_algorithms(True)


def build_dataset(seed: int = 408) -> dict[str, torch.Tensor]:
    """Create reproducible train/val/test tensors from two-moons data."""
    x, y = make_moons(n_samples=1500, noise=0.25, random_state=seed)

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x,
        y,
        test_size=0.4,
        random_state=seed,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp,
        y_tmp,
        test_size=0.5,
        random_state=seed,
        stratify=y_tmp,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return {
        "train_x": torch.tensor(x_train, dtype=torch.float32),
        "train_y": torch.tensor(y_train, dtype=torch.long),
        "val_x": torch.tensor(x_val, dtype=torch.float32),
        "val_y": torch.tensor(y_val, dtype=torch.long),
        "test_x": torch.tensor(x_test, dtype=torch.float32),
        "test_y": torch.tensor(y_test, dtype=torch.long),
    }


def build_model(spec: ArchSpec, input_dim: int = 2, num_classes: int = 2) -> nn.Module:
    """Build an MLP candidate from an architecture spec."""
    if spec.activation not in {"relu", "tanh"}:
        raise ValueError("activation must be one of {'relu', 'tanh'}")

    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(spec.depth):
        layers.append(nn.Linear(in_dim, spec.hidden_dim))
        layers.append(nn.ReLU() if spec.activation == "relu" else nn.Tanh())
        in_dim = spec.hidden_dim
    layers.append(nn.Linear(in_dim, num_classes))

    return nn.Sequential(*layers)


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)
        return float((pred == y).float().mean().item())


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_with_validation(
    model: nn.Module,
    train_loader: DataLoader,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[float, int, float]:
    """Train model and return (best_val_acc, best_epoch, train_loss_at_best)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_epoch = -1
    best_train_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = yb.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        mean_train_loss = running_loss / max(seen, 1)
        val_acc = accuracy(model, val_x, val_y)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_train_loss = mean_train_loss

    return best_val_acc, best_epoch, best_train_loss


def all_candidates() -> list[ArchSpec]:
    depths = [1, 2, 3]
    hidden_dims = [16, 32, 64]
    activations = ["relu", "tanh"]
    return [ArchSpec(d, h, a) for d, h, a in product(depths, hidden_dims, activations)]


def run_search(
    data: dict[str, torch.Tensor],
    candidates: Iterable[ArchSpec],
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> list[SearchResult]:
    """Evaluate each architecture on the validation split."""
    results: list[SearchResult] = []

    train_loader_base = make_loader(data["train_x"], data["train_y"], batch_size, shuffle=True)
    candidate_list = list(candidates)
    for i, spec in enumerate(candidate_list):
        # Fixed, architecture-specific seed for full reproducibility.
        set_global_seed(10000 + i)

        model = build_model(spec)
        best_val_acc, best_epoch, train_loss = train_with_validation(
            model=model,
            train_loader=train_loader_base,
            val_x=data["val_x"],
            val_y=data["val_y"],
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )

        results.append(
            SearchResult(
                spec=spec,
                val_acc=best_val_acc,
                best_epoch=best_epoch,
                param_count=count_trainable_parameters(model),
                train_loss_at_best=train_loss,
            )
        )

    return results


def retrain_best_and_test(
    spec: ArchSpec,
    data: dict[str, torch.Tensor],
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 7e-3,
    weight_decay: float = 5e-5,
) -> tuple[float, float]:
    """Retrain selected architecture on train+val, then evaluate on test."""
    set_global_seed(20408)

    trainval_x = torch.cat([data["train_x"], data["val_x"]], dim=0)
    trainval_y = torch.cat([data["train_y"], data["val_y"]], dim=0)
    trainval_loader = make_loader(trainval_x, trainval_y, batch_size=batch_size, shuffle=True)

    model = build_model(spec)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        model.train()
        for xb, yb in trainval_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    trainval_acc = accuracy(model, trainval_x, trainval_y)
    test_acc = accuracy(model, data["test_x"], data["test_y"])
    return trainval_acc, test_acc


def results_to_dataframe(results: list[SearchResult]) -> pd.DataFrame:
    records = [
        {
            "depth": r.spec.depth,
            "hidden_dim": r.spec.hidden_dim,
            "activation": r.spec.activation,
            "params": r.param_count,
            "best_epoch": r.best_epoch,
            "val_acc": r.val_acc,
            "train_loss_at_best": r.train_loss_at_best,
        }
        for r in results
    ]
    return pd.DataFrame.from_records(records).sort_values(
        by=["val_acc", "params"], ascending=[False, True]
    ).reset_index(drop=True)


def main() -> None:
    set_global_seed(408)
    data = build_dataset(seed=408)

    candidates = all_candidates()
    results = run_search(data, candidates)
    df = results_to_dataframe(results)

    best_row = df.iloc[0]
    best_spec = ArchSpec(
        depth=int(best_row["depth"]),
        hidden_dim=int(best_row["hidden_dim"]),
        activation=str(best_row["activation"]),
    )

    trainval_acc, test_acc = retrain_best_and_test(best_spec, data)

    print("=== NAS MVP (Discrete Search on MLPs) ===")
    print(f"Search space size: {len(candidates)}")
    print("Dataset split sizes:")
    print(
        f"  train={len(data['train_x'])}, val={len(data['val_x'])}, test={len(data['test_x'])}"
    )
    print()

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        140,
        "display.float_format",
        "{:.4f}".format,
    ):
        print("Top-10 architectures by validation accuracy:")
        print(df.head(10).to_string(index=False))

    print()
    print(
        "Selected architecture: "
        f"depth={best_spec.depth}, hidden_dim={best_spec.hidden_dim}, activation={best_spec.activation}"
    )
    print(f"Validation accuracy (search): {float(best_row['val_acc']):.4f}")
    print(f"Train+Val accuracy (retrain): {trainval_acc:.4f}")
    print(f"Test accuracy (final): {test_acc:.4f}")

    # Deterministic quality checks for this toy NAS run.
    assert len(df) == 18, "search space size mismatch"
    assert df["val_acc"].between(0.0, 1.0).all(), "invalid validation accuracy"
    assert not df[["val_acc", "train_loss_at_best"]].isna().any().any(), "NaN found"

    best_val_acc = float(df.iloc[0]["val_acc"])
    median_val_acc = float(df["val_acc"].median())
    assert best_val_acc >= median_val_acc, "best architecture ranking is broken"
    assert best_val_acc > 0.85, "search did not find a strong enough candidate"
    assert test_acc > 0.84, "final test accuracy unexpectedly low"

    print("All checks passed.")


if __name__ == "__main__":
    main()
