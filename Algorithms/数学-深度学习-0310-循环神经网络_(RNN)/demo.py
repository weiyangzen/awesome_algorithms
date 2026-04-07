"""循环神经网络 (RNN) 最小可运行 MVP。

任务：给定离散 token 序列，判断 token A(0) 是否先于 token B(1) 首次出现。
该任务需要沿时间维度累计上下文信息，适合用基础 RNN 演示。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Config:
    """实验配置。"""

    seed: int = 2026
    n_samples: int = 2400
    seq_len: int = 20
    vocab_size: int = 8
    token_a: int = 0
    token_b: int = 1

    test_size: float = 0.2
    batch_size: int = 64
    epochs: int = 14
    learning_rate: float = 1e-2

    embed_dim: int = 16
    hidden_dim: int = 32


def set_global_seed(seed: int) -> None:
    """固定随机性，便于复现实验结果。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_config(cfg: Config) -> None:
    """基础参数合法性检查。"""
    if cfg.n_samples <= 0:
        raise ValueError("n_samples 必须为正整数")
    if cfg.seq_len < 2:
        raise ValueError("seq_len 至少为 2，才能同时放置 token_a/token_b")
    if cfg.vocab_size < 4:
        raise ValueError("vocab_size 至少为 4，保留 token_a/token_b 与填充 token")
    if cfg.token_a == cfg.token_b:
        raise ValueError("token_a 与 token_b 不能相同")
    if not (0.0 < cfg.test_size < 1.0):
        raise ValueError("test_size 必须在 (0, 1) 之间")
    if cfg.batch_size <= 0 or cfg.epochs <= 0:
        raise ValueError("batch_size 与 epochs 必须为正整数")
    if cfg.learning_rate <= 0.0:
        raise ValueError("learning_rate 必须为正数")


def generate_order_dataset(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成序列顺序判别数据。

    返回：
    - sequences: shape (n_samples, seq_len)
    - labels: shape (n_samples,)；1 表示 A 在 B 之前，0 表示 B 在 A 之前
    - pos_a / pos_b: token_a/token_b 在各样本中的位置，用于审计输出
    """
    filler_tokens = np.arange(2, cfg.vocab_size, dtype=np.int64)
    if filler_tokens.size == 0:
        raise ValueError("可用填充 token 为空，请增大 vocab_size")

    sequences = np.random.choice(filler_tokens, size=(cfg.n_samples, cfg.seq_len), replace=True)
    labels = np.random.randint(0, 2, size=cfg.n_samples, dtype=np.int64)
    pos_a = np.empty(cfg.n_samples, dtype=np.int64)
    pos_b = np.empty(cfg.n_samples, dtype=np.int64)

    for i in range(cfg.n_samples):
        pa, pb = np.random.choice(cfg.seq_len, size=2, replace=False)
        if labels[i] == 1 and pa > pb:
            pa, pb = pb, pa
        elif labels[i] == 0 and pa < pb:
            pa, pb = pb, pa

        sequences[i, pa] = cfg.token_a
        sequences[i, pb] = cfg.token_b
        pos_a[i] = pa
        pos_b[i] = pb

    return sequences, labels, pos_a, pos_b


class RNNOrderClassifier(nn.Module):
    """Embedding + 单层 tanh RNN + 全连接分类头。"""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, hidden = self.rnn(emb)
        last_hidden = hidden[-1]
        return self.classifier(last_hidden)


def build_dataloaders(
    sequences: np.ndarray,
    labels: np.ndarray,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    cfg: Config,
) -> tuple[DataLoader, DataLoader, dict[str, np.ndarray]]:
    """划分数据并构建 DataLoader。"""
    (x_train, x_test, y_train, y_test, pa_train, pa_test, pb_train, pb_test) = train_test_split(
        sequences,
        labels,
        pos_a,
        pos_b,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=labels,
    )

    train_ds = TensorDataset(
        torch.from_numpy(x_train).long(),
        torch.from_numpy(y_train).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test).long(),
        torch.from_numpy(y_test).long(),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    holdout = {
        "x_test": x_test,
        "y_test": y_test,
        "pa_test": pa_test,
        "pb_test": pb_test,
        "train_size": np.array([x_train.shape[0]], dtype=np.int64),
        "test_size": np.array([x_test.shape[0]], dtype=np.int64),
    }
    return train_loader, test_loader, holdout


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """执行一轮训练或评估。"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = yb.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == yb).sum().item())
        total_count += batch_size

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


@torch.no_grad()
def predict_numpy(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """对 numpy 输入做批量预测，返回类别 id。"""
    model.eval()
    preds: list[np.ndarray] = []
    for i in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).long().to(device)
        logits = model(xb)
        pb = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pb)
    return np.concatenate(preds, axis=0)


def print_samples(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pa_test: np.ndarray,
    pb_test: np.ndarray,
    max_rows: int = 6,
) -> None:
    """打印少量样本，审计模型是否学到“顺序规则”。"""
    print("\n[Sample predictions]")
    print("idx | pos_A | pos_B | true | pred | first_10_tokens")
    print("----+-------+-------+------+------|----------------")
    for i in range(min(max_rows, x_test.shape[0])):
        head_tokens = " ".join(str(int(v)) for v in x_test[i, :10])
        print(
            f"{i:>3d} | {int(pa_test[i]):>5d} | {int(pb_test[i]):>5d} |"
            f" {int(y_true[i]):>4d} | {int(y_pred[i]):>4d} | {head_tokens}"
        )


def main() -> None:
    cfg = Config()
    validate_config(cfg)
    set_global_seed(cfg.seed)

    sequences, labels, pos_a, pos_b = generate_order_dataset(cfg)

    train_loader, test_loader, holdout = build_dataloaders(sequences, labels, pos_a, pos_b, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNOrderClassifier(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print("RNN MVP: token A/B 顺序分类")
    print(
        "Config: "
        f"samples={cfg.n_samples}, seq_len={cfg.seq_len}, vocab={cfg.vocab_size}, "
        f"train={int(holdout['train_size'][0])}, test={int(holdout['test_size'][0])}, "
        f"epochs={cfg.epochs}, batch={cfg.batch_size}, device={device.type}"
    )
    print("\n[Epoch metrics]")
    print("epoch | train_loss | train_acc | test_loss | test_acc")
    print("------+------------+-----------+-----------+---------")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)
        print(f"{epoch:>5d} | {train_loss:>10.4f} | {train_acc:>9.4f} | {test_loss:>9.4f} | {test_acc:>7.4f}")

    y_test = holdout["y_test"]
    y_pred = predict_numpy(model, holdout["x_test"], device=device, batch_size=cfg.batch_size)

    final_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("\n[Final evaluation]")
    print(f"test_accuracy: {final_acc:.4f}")
    print("confusion_matrix [rows=true, cols=pred]:")
    print(cm)

    print_samples(
        holdout["x_test"],
        y_test,
        y_pred,
        holdout["pa_test"],
        holdout["pb_test"],
        max_rows=6,
    )


if __name__ == "__main__":
    main()
