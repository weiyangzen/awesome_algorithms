"""Vision Transformer minimal runnable MVP.

This script builds a tiny ViT classifier with PyTorch and trains it on the
built-in scikit-learn digits dataset (8x8 grayscale images).

No interactive input is required.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class ViTConfig:
    image_size: int = 8
    patch_size: int = 2
    emb_dim: int = 64
    num_heads: int = 4
    depth: int = 2
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    num_classes: int = 10


class PatchDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, patches: np.ndarray, labels: np.ndarray) -> None:
        if patches.ndim != 3:
            raise ValueError(f"patches must be 3D, got shape={patches.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape={labels.shape}")
        if patches.shape[0] != labels.shape[0]:
            raise ValueError("patches and labels must have the same sample size")

        self.x = torch.tensor(patches, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class TinyViTClassifier(nn.Module):
    """Tiny Vision Transformer for 8x8 digits classification."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if config.emb_dim % config.num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")

        num_patches_per_side = config.image_size // config.patch_size
        self.num_patches = num_patches_per_side * num_patches_per_side
        patch_dim = config.patch_size * config.patch_size

        self.patch_embed = nn.Linear(patch_dim, config.emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.emb_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.emb_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.emb_dim)
        self.head = nn.Linear(config.emb_dim, config.num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_patches, patch_dim]
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        cls_repr = self.norm(x[:, 0, :])
        logits = self.head(cls_repr)
        return logits


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_patches(images: np.ndarray, patch_size: int) -> np.ndarray:
    """Convert images [N, H, W] into flattened non-overlapping patches.

    Returns shape [N, num_patches, patch_size*patch_size].
    """
    if images.ndim != 3:
        raise ValueError(f"images must be 3D, got shape={images.shape}")

    n_samples, h, w = images.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")

    n_h = h // patch_size
    n_w = w // patch_size

    patches = (
        images.reshape(n_samples, n_h, patch_size, n_w, patch_size)
        .transpose(0, 1, 3, 2, 4)
        .reshape(n_samples, n_h * n_w, patch_size * patch_size)
    )
    return patches.astype(np.float32)


def build_dataloaders(patch_size: int, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    digits = load_digits()
    images = (digits.images.astype(np.float32) / 16.0).clip(0.0, 1.0)
    labels = digits.target.astype(np.int64)

    patches = extract_patches(images=images, patch_size=patch_size)

    x_train, x_test, y_train, y_test = train_test_split(
        patches,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    train_ds = PatchDataset(x_train, y_train)
    test_ds = PatchDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        losses.append(float(loss.item()))
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 18,
    lr: float = 3e-3,
) -> pd.DataFrame:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    records: List[Dict[str, float]] = []

    initial_train = evaluate(model, train_loader, device)
    initial_test = evaluate(model, test_loader, device)
    print(
        "[initial] "
        f"train_loss={initial_train['loss']:.4f}, train_acc={initial_train['accuracy']:.4f}, "
        f"test_loss={initial_test['loss']:.4f}, test_acc={initial_test['accuracy']:.4f}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        record = {
            "epoch": float(epoch),
            "train_loss_batch_mean": float(np.mean(epoch_losses)),
            "train_acc": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
        }
        records.append(record)

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={record['train_loss_batch_mean']:.4f}, "
            f"train_acc={record['train_acc']:.4f}, "
            f"test_loss={record['test_loss']:.4f}, "
            f"test_acc={record['test_acc']:.4f}"
        )

    return pd.DataFrame.from_records(records)


@torch.no_grad()
def show_prediction_examples(model: nn.Module, loader: DataLoader, device: torch.device, n: int = 8) -> None:
    model.eval()

    xb, yb = next(iter(loader))
    xb = xb[:n].to(device)
    yb = yb[:n]

    logits = model(xb).cpu().numpy()
    probs = softmax(logits, axis=1)
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    df = pd.DataFrame(
        {
            "true": yb.numpy(),
            "pred": pred,
            "confidence": np.round(confidence, 4),
        }
    )
    print("\nSample predictions:")
    print(df.to_string(index=False))


def main() -> None:
    set_global_seed(42)
    torch.set_num_threads(max(torch.get_num_threads(), 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ViTConfig()

    train_loader, test_loader = build_dataloaders(
        patch_size=config.patch_size,
        batch_size=64,
    )

    model = TinyViTClassifier(config).to(device)
    print(f"Device: {device}")
    print(
        "Model config: "
        f"image_size={config.image_size}, patch_size={config.patch_size}, "
        f"emb_dim={config.emb_dim}, heads={config.num_heads}, depth={config.depth}"
    )

    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=18,
        lr=3e-3,
    )

    final_row = history.iloc[-1]
    print("\nFinal epoch summary:")
    print(final_row.to_string())

    best_test_acc = float(history["test_acc"].max())
    final_test_acc = float(history["test_acc"].iloc[-1])
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")

    if final_test_acc < 0.90:
        raise RuntimeError(
            "Final test accuracy is below 0.90; MVP did not reach the expected quality floor."
        )

    show_prediction_examples(model, test_loader, device, n=8)


if __name__ == "__main__":
    main()
