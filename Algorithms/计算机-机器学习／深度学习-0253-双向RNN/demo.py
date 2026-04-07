"""Minimal runnable MVP for Bidirectional RNN (CS-0120).

This script demonstrates a source-transparent bidirectional RNN for token
classification. To make the benefit observable, we compare:
- a manual unidirectional RNN tagger
- a manual bidirectional RNN tagger

Task (synthetic): for each token position i, predict whether x_i matches at
least one immediate neighbor (x_{i-1} or x_{i+1}). This requires future context
for many positions, so bidirectional modeling has a clear advantage.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    positive_rate: float


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_neighbor_match_dataset(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic token-label pairs.

    Tokens: integers in [1, vocab_size - 1], shape [N, L].
    Label y[i, t] = 1 iff token at position t equals one immediate neighbor.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if vocab_size < 3:
        raise ValueError("vocab_size must be >= 3")

    rng = np.random.default_rng(seed)
    x = rng.integers(low=1, high=vocab_size, size=(num_samples, seq_len), dtype=np.int64)

    y = np.zeros((num_samples, seq_len), dtype=np.int64)
    y[:, 0] = (x[:, 0] == x[:, 1]).astype(np.int64)
    y[:, -1] = (x[:, -1] == x[:, -2]).astype(np.int64)
    y[:, 1:-1] = ((x[:, 1:-1] == x[:, :-2]) | (x[:, 1:-1] == x[:, 2:])).astype(np.int64)
    return x, y


class UniDirectionalRNNTagger(nn.Module):
    """Manual unidirectional RNN token classifier using RNNCell."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.forward_cell = nn.RNNCell(embed_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        emb = self.embedding(token_ids)  # [B, L, E]

        h = torch.zeros(batch_size, self.hidden_dim, device=token_ids.device)
        forward_states: List[torch.Tensor] = []

        for t in range(seq_len):
            h = self.forward_cell(emb[:, t, :], h)
            forward_states.append(h)

        features = torch.stack(forward_states, dim=1)  # [B, L, H]
        logits = self.classifier(features)  # [B, L, 2]
        return logits


class BidirectionalRNNTagger(nn.Module):
    """Manual bidirectional RNN token classifier using two RNNCell chains."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.forward_cell = nn.RNNCell(embed_dim, hidden_dim)
        self.backward_cell = nn.RNNCell(embed_dim, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, 2)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        emb = self.embedding(token_ids)  # [B, L, E]

        h_forward = torch.zeros(batch_size, self.hidden_dim, device=token_ids.device)
        h_backward = torch.zeros(batch_size, self.hidden_dim, device=token_ids.device)

        forward_states: List[torch.Tensor] = []
        backward_states: List[torch.Tensor] = [torch.empty(0)] * seq_len

        for t in range(seq_len):
            h_forward = self.forward_cell(emb[:, t, :], h_forward)
            forward_states.append(h_forward)

        for t in range(seq_len - 1, -1, -1):
            h_backward = self.backward_cell(emb[:, t, :], h_backward)
            backward_states[t] = h_backward

        fwd = torch.stack(forward_states, dim=1)  # [B, L, H]
        bwd = torch.stack(backward_states, dim=1)  # [B, L, H]
        features = torch.cat([fwd, bwd], dim=-1)  # [B, L, 2H]
        logits = self.classifier(features)  # [B, L, 2]
        return logits


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> List[float]:
    """Train token classifier and return epoch loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    x_tensor = torch.tensor(x_train, dtype=torch.long)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    n = x_train.shape[0]
    rng = np.random.default_rng(seed)
    loss_history: List[float] = []

    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        epoch_loss_sum = 0.0
        seen = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(x_tensor[idx])
            loss = criterion(logits.reshape(-1, 2), y_tensor[idx].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_real = int(idx.shape[0])
            epoch_loss_sum += float(loss.item()) * batch_size_real
            seen += batch_size_real

        loss_history.append(epoch_loss_sum / seen)

    return loss_history


def evaluate_model(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray) -> EvalResult:
    """Evaluate token classifier with binary metrics on flattened positions."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.long))
        pred = logits.argmax(dim=-1).cpu().numpy()

    y_true = y_test.reshape(-1)
    y_pred = pred.reshape(-1)

    accuracy = float(np.mean(y_true == y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    return EvalResult(
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        positive_rate=float(np.mean(y_pred)),
    )


def summarize_samples(
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: nn.Module,
    n_rows: int,
) -> pd.DataFrame:
    """Build a compact table of example predictions."""
    with torch.no_grad():
        pred = model(torch.tensor(x_test[:n_rows], dtype=torch.long)).argmax(dim=-1).cpu().numpy()

    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "tokens": " ".join(map(str, x_test[i].tolist())),
                "true_labels": " ".join(map(str, y_test[i].tolist())),
                "pred_labels": " ".join(map(str, pred[i].tolist())),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    set_seed(253)

    vocab_size = 12
    seq_len = 18
    train_size = 1800
    test_size = 500

    x_train, y_train = build_neighbor_match_dataset(
        num_samples=train_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=253,
    )
    x_test, y_test = build_neighbor_match_dataset(
        num_samples=test_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=260,
    )

    uni_model = UniDirectionalRNNTagger(vocab_size=vocab_size, embed_dim=16, hidden_dim=20)
    bi_model = BidirectionalRNNTagger(vocab_size=vocab_size, embed_dim=16, hidden_dim=20)

    uni_losses = train_model(
        uni_model,
        x_train,
        y_train,
        epochs=12,
        batch_size=128,
        lr=1e-2,
        seed=901,
    )
    bi_losses = train_model(
        bi_model,
        x_train,
        y_train,
        epochs=12,
        batch_size=128,
        lr=1e-2,
        seed=902,
    )

    uni_eval = evaluate_model(uni_model, x_test, y_test)
    bi_eval = evaluate_model(bi_model, x_test, y_test)

    summary = pd.DataFrame(
        {
            "model": ["Unidirectional-RNN", "Bidirectional-RNN"],
            "accuracy": [uni_eval.accuracy, bi_eval.accuracy],
            "precision": [uni_eval.precision, bi_eval.precision],
            "recall": [uni_eval.recall, bi_eval.recall],
            "f1": [uni_eval.f1, bi_eval.f1],
            "pred_positive_rate": [uni_eval.positive_rate, bi_eval.positive_rate],
            "loss_start": [uni_losses[0], bi_losses[0]],
            "loss_end": [uni_losses[-1], bi_losses[-1]],
        }
    )

    sample_table = summarize_samples(x_test, y_test, bi_model, n_rows=4)

    assert bi_losses[-1] < bi_losses[0], "Bidirectional model loss did not decrease"
    assert uni_losses[-1] < uni_losses[0], "Unidirectional model loss did not decrease"
    assert bi_eval.accuracy >= 0.97, "Bidirectional accuracy too low"
    assert bi_eval.f1 >= 0.97, "Bidirectional F1 too low"
    assert bi_eval.accuracy - uni_eval.accuracy >= 0.05, "Bidirectional gain on accuracy is too small"
    assert bi_eval.f1 > uni_eval.f1, "Bidirectional model should outperform unidirectional on F1"

    print("Bidirectional RNN MVP (manual forward/backward RNNCell)")
    print(
        f"train_samples={train_size}, test_samples={test_size}, seq_len={seq_len}, vocab_size={vocab_size}"
    )
    print(f"positive_label_rate_in_test={float(y_test.mean()):.4f}")
    print()

    print("metric_summary=")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
    print()

    print("bidirectional_prediction_examples=")
    print(sample_table.to_string(index=False))


if __name__ == "__main__":
    main()
