"""Attention mechanism minimal runnable MVP.

This script implements a tiny self-attention classifier from scratch with
explicit Q/K/V projections and scaled dot-product attention.

Run:
    uv run python demo.py
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class Config:
    seq_len: int = 8
    input_dim: int = 8
    d_model: int = 64
    d_k: int = 64
    depth: int = 2
    ff_mult: int = 2
    num_classes: int = 10
    dropout: float = 0.10
    batch_size: int = 64
    epochs: int = 18
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 42
    quality_floor: float = 0.90


class SequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 3:
            raise ValueError(f"x must be 3D [N, seq_len, input_dim], got {x.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D [N], got {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same sample count")

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class ScaledDotSelfAttention(nn.Module):
    """Single-head self-attention with explicit Q/K/V projections."""

    def __init__(self, d_model: int, d_k: int, dropout: float) -> None:
        super().__init__()
        self.d_k = d_k
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_k)
        self.out_proj = nn.Linear(d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.d_k))
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, v)
        out = self.out_proj(context)
        return out, probs


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, d_k: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn = ScaledDotSelfAttention(d_model=d_model, d_k=d_k, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_probs = self.attn(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_probs


class TinyAttentionClassifier(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len + 1, cfg.d_model))
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    d_model=cfg.d_model,
                    d_k=cfg.d_k,
                    ff_mult=cfg.ff_mult,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        # x: [batch, seq_len, input_dim]
        if x.ndim != 3:
            raise ValueError(f"Expected input rank 3, got {x.ndim}")

        h = self.input_proj(x)
        cls = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embed

        last_attn: torch.Tensor | None = None
        for block in self.blocks:
            h, last_attn = block(h)

        cls_repr = self.norm(h[:, 0, :])
        logits = self.head(cls_repr)

        if return_attention:
            return logits, last_attn
        return logits, None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    digits = load_digits()

    x = digits.images.astype(np.float32) / 16.0  # [N, 8, 8]
    y = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    train_ds = SequenceDataset(x_train, y_train)
    test_ds = SequenceDataset(x_test, y_test)

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

        logits, _ = model(xb)
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
    epochs: int,
    lr: float,
    weight_decay: float,
) -> pd.DataFrame:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    logs: List[Dict[str, float]] = []

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
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        row = {
            "epoch": float(epoch),
            "train_loss_batch_mean": float(np.mean(epoch_losses)),
            "train_acc": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
        }
        logs.append(row)

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={row['train_loss_batch_mean']:.4f}, "
            f"train_acc={row['train_acc']:.4f}, "
            f"test_loss={row['test_loss']:.4f}, "
            f"test_acc={row['test_acc']:.4f}"
        )

    return pd.DataFrame.from_records(logs)


@torch.no_grad()
def show_predictions_with_attention(
    model: TinyAttentionClassifier,
    loader: DataLoader,
    device: torch.device,
    n: int = 8,
) -> None:
    model.eval()

    xb, yb = next(iter(loader))
    xb = xb[:n].to(device)
    yb = yb[:n]

    logits, attn = model(xb, return_attention=True)
    if attn is None:
        raise RuntimeError("Expected attention weights but got None")

    logits_np = logits.cpu().numpy()
    probs = softmax(logits_np, axis=1)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    # attention from cls token (index 0) to image rows (indices 1..seq_len)
    cls_to_rows = attn[:, 0, 1:].cpu().numpy()
    focus_row = cls_to_rows.argmax(axis=1)

    df = pd.DataFrame(
        {
            "true": yb.numpy(),
            "pred": pred,
            "confidence": np.round(conf, 4),
            "focus_row": focus_row,
        }
    )

    print("\nSample predictions with attention focus:")
    print(df.to_string(index=False))

    mean_attention = cls_to_rows.mean(axis=0)
    print("\nMean cls->row attention (rounded):")
    print(np.round(mean_attention, 4))


def main() -> None:
    cfg = Config()
    set_global_seed(cfg.seed)
    torch.set_num_threads(max(torch.get_num_threads(), 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        "Config: "
        f"seq_len={cfg.seq_len}, input_dim={cfg.input_dim}, d_model={cfg.d_model}, "
        f"d_k={cfg.d_k}, depth={cfg.depth}, epochs={cfg.epochs}"
    )

    train_loader, test_loader = build_dataloaders(batch_size=cfg.batch_size)

    model = TinyAttentionClassifier(cfg).to(device)
    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    final_row = history.iloc[-1]
    print("\nFinal epoch summary:")
    print(final_row.to_string())

    best_test_acc = float(history["test_acc"].max())
    final_test_acc = float(history["test_acc"].iloc[-1])
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")

    if final_test_acc < cfg.quality_floor:
        raise RuntimeError(
            f"Final test accuracy {final_test_acc:.4f} is below quality floor {cfg.quality_floor:.2f}."
        )

    show_predictions_with_attention(model, test_loader, device, n=8)


if __name__ == "__main__":
    main()
