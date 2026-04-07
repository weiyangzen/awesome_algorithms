"""GRU (Gated Recurrent Unit) 最小可运行 MVP。

任务：给定离散 token 序列，判断 token A(0) 是否先于 token B(1) 首次出现。
该任务需要沿时间维度保留顺序信息，适合用 GRU 演示门控循环机制。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Config:
    """实验配置。"""

    seed: int = 2026
    n_samples: int = 2800
    seq_len: int = 30
    vocab_size: int = 10
    token_a: int = 0
    token_b: int = 1

    test_size: float = 0.2
    batch_size: int = 64
    epochs: int = 12
    learning_rate: float = 8e-3

    embed_dim: int = 24
    hidden_dim: int = 40
    grad_clip: float = 1.0


def set_global_seed(seed: int) -> None:
    """固定随机行为，便于复现实验。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_config(cfg: Config) -> None:
    """参数合法性检查。"""
    if cfg.n_samples <= 0:
        raise ValueError("n_samples 必须为正整数")
    if cfg.seq_len < 2:
        raise ValueError("seq_len 至少为 2，才能同时容纳 token_a 和 token_b")
    if cfg.vocab_size < 4:
        raise ValueError("vocab_size 至少为 4，需留出 token_a/token_b 与填充 token")
    if cfg.token_a == cfg.token_b:
        raise ValueError("token_a 与 token_b 不能相同")
    if not (0.0 < cfg.test_size < 1.0):
        raise ValueError("test_size 必须在 (0, 1) 之间")
    if cfg.batch_size <= 0 or cfg.epochs <= 0:
        raise ValueError("batch_size 与 epochs 必须为正整数")
    if cfg.learning_rate <= 0.0:
        raise ValueError("learning_rate 必须为正数")
    if cfg.grad_clip <= 0.0:
        raise ValueError("grad_clip 必须为正数")


def generate_order_dataset(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成顺序判别数据。

    返回：
    - sequences: (n_samples, seq_len)
    - labels: (n_samples,), 1 表示 A 在 B 前；0 表示 B 在 A 前
    - pos_a / pos_b: 各样本中 A/B 的位置
    """
    filler_tokens = np.arange(2, cfg.vocab_size, dtype=np.int64)
    if filler_tokens.size == 0:
        raise ValueError("填充 token 为空，请增大 vocab_size")

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


class GRUOrderClassifier(nn.Module):
    """Embedding + 单层 GRU + 线性分类头。"""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        last_hidden = h_n[-1]
        return self.classifier(last_hidden)


def build_dataloaders(
    sequences: np.ndarray,
    labels: np.ndarray,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    cfg: Config,
) -> tuple[DataLoader, DataLoader, dict[str, np.ndarray]]:
    """分层划分训练/测试集并构建 DataLoader。"""
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
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    """执行一轮训练或评估，返回平均损失和准确率。"""
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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
    """对 numpy 序列批量推理，返回预测类别。"""
    model.eval()
    preds: list[np.ndarray] = []
    for i in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).long().to(device)
        logits = model(xb)
        pb = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pb)
    return np.concatenate(preds, axis=0)


def analyze_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """计算分类指标与二项检验。"""
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    correct = int((y_true == y_pred).sum())
    n_test = int(y_true.shape[0])
    btest = binomtest(k=correct, n=n_test, p=0.5, alternative="greater")
    ci = btest.proportion_ci(confidence_level=0.95, method="exact")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "binom_pvalue": float(btest.pvalue),
        "acc_ci95_low": float(ci.low),
        "acc_ci95_high": float(ci.high),
    }


def print_sample_audit(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pa_test: np.ndarray,
    pb_test: np.ndarray,
    max_rows: int = 8,
) -> None:
    """打印少量样本，人工审计是否学到顺序规则。"""
    print("\n[Sample audit]")
    print("idx | pos_A | pos_B | true | pred | first_12_tokens")
    print("----+-------+-------+------+------|----------------")
    for i in range(min(max_rows, x_test.shape[0])):
        token_head = " ".join(str(int(v)) for v in x_test[i, :12])
        print(
            f"{i:>3d} | {int(pa_test[i]):>5d} | {int(pb_test[i]):>5d} |"
            f" {int(y_true[i]):>4d} | {int(y_pred[i]):>4d} | {token_head}"
        )


def main() -> None:
    cfg = Config()
    validate_config(cfg)
    set_global_seed(cfg.seed)

    sequences, labels, pos_a, pos_b = generate_order_dataset(cfg)
    train_loader, test_loader, holdout = build_dataloaders(sequences, labels, pos_a, pos_b, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUOrderClassifier(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history: list[dict[str, float]] = []

    print("GRU MVP: token A/B 顺序分类")
    print(
        "Config: "
        f"samples={cfg.n_samples}, seq_len={cfg.seq_len}, vocab={cfg.vocab_size}, "
        f"train={int(holdout['train_size'][0])}, test={int(holdout['test_size'][0])}, "
        f"epochs={cfg.epochs}, batch={cfg.batch_size}, device={device.type}"
    )
    print("\n[Epoch metrics]")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=cfg.grad_clip,
        )
        test_loss, test_acc = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            grad_clip=cfg.grad_clip,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    history_df = pd.DataFrame(history)

    y_test = holdout["y_test"]
    y_pred = predict_numpy(model=model, x=holdout["x_test"], device=device, batch_size=256)
    stats = analyze_classification(y_true=y_test, y_pred=y_pred)

    print("\n[Epoch metrics tail]")
    print(history_df.tail(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n[Final evaluation]")
    print(f"accuracy={stats['accuracy']:.4f}")
    print(f"f1={stats['f1']:.4f}")
    print(f"binom_pvalue={stats['binom_pvalue']:.3e}")
    print(f"acc_ci95=({stats['acc_ci95_low']:.4f}, {stats['acc_ci95_high']:.4f})")
    print("confusion_matrix (rows=true [0,1], cols=pred [0,1]):")
    print(np.array2string(stats["confusion_matrix"]))

    print_sample_audit(
        x_test=holdout["x_test"],
        y_true=y_test,
        y_pred=y_pred,
        pa_test=holdout["pa_test"],
        pb_test=holdout["pb_test"],
        max_rows=8,
    )


if __name__ == "__main__":
    main()
