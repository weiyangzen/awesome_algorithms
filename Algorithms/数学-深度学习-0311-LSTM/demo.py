"""LSTM 最小可运行 MVP。

任务：对单变量时间序列做一步预测（next-step forecasting）。
实现目标：提供一个可直接运行、可复现、可评估的 LSTM 入门范例。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Config:
    """实验配置。"""

    seed: int = 2026
    n_points: int = 2800
    window_size: int = 32
    test_size: float = 0.2

    batch_size: int = 64
    epochs: int = 24
    learning_rate: float = 2e-3

    hidden_size: int = 40
    num_layers: int = 1


def set_global_seed(seed: int) -> None:
    """固定随机种子，提升复现性。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_config(cfg: Config) -> None:
    """参数合法性检查。"""
    if cfg.n_points <= cfg.window_size + 20:
        raise ValueError("n_points 太小，无法构造稳定训练/测试样本")
    if cfg.window_size < 4:
        raise ValueError("window_size 至少为 4")
    if not (0.0 < cfg.test_size < 1.0):
        raise ValueError("test_size 必须在 (0, 1) 范围内")
    if cfg.batch_size <= 0 or cfg.epochs <= 0:
        raise ValueError("batch_size 和 epochs 必须为正整数")
    if cfg.learning_rate <= 0.0:
        raise ValueError("learning_rate 必须为正数")
    if cfg.hidden_size <= 0 or cfg.num_layers <= 0:
        raise ValueError("hidden_size 和 num_layers 必须为正整数")


def generate_series(cfg: Config) -> np.ndarray:
    """生成含长滞后依赖的合成时序，并做轻量平滑。"""
    series = np.zeros(cfg.n_points, dtype=np.float32)
    warmup = 24
    series[:warmup] = np.random.normal(loc=0.0, scale=0.5, size=warmup).astype(np.float32)

    for t in range(warmup, cfg.n_points):
        seasonal = 0.9 * np.sin(2.0 * np.pi * t / 30.0)
        long_lag = 0.58 * series[t - 12] - 0.27 * series[t - 24]
        nonlinear = 0.18 * np.tanh(series[t - 7] * series[t - 13])
        noise = np.random.normal(loc=0.0, scale=0.12)
        series[t] = long_lag + seasonal + nonlinear + noise

    # 使用 scipy 进行轻量去噪，保留主周期结构。
    smooth_series = savgol_filter(series, window_length=11, polyorder=2, mode="interp")
    return smooth_series.astype(np.float32)


def build_supervised_dataset(series: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """将时序转为监督学习样本。"""
    windows = np.lib.stride_tricks.sliding_window_view(series, window_shape=window_size + 1)
    x = windows[:, :window_size]
    y = windows[:, -1]
    return x.astype(np.float32), y.astype(np.float32)


def split_and_scale(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """按时间顺序划分训练/测试，并分别标准化输入与目标。"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_scaled = x_scaler.fit_transform(x_train.reshape(-1, 1)).reshape(x_train.shape)
    x_test_scaled = x_scaler.transform(x_test.reshape(-1, 1)).reshape(x_test.shape)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return (
        x_train_scaled.astype(np.float32),
        x_test_scaled.astype(np.float32),
        y_train_scaled.astype(np.float32),
        y_test_scaled.astype(np.float32),
        x_scaler,
        y_scaler,
    )


def make_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """构建训练与测试 DataLoader。"""
    train_ds = TensorDataset(
        torch.from_numpy(x_train).unsqueeze(-1),
        torch.from_numpy(y_train).unsqueeze(-1),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test).unsqueeze(-1),
        torch.from_numpy(y_test).unsqueeze(-1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class LSTMForecaster(nn.Module):
    """单层 LSTM 回归器。"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.lstm(x)
        last_state = seq_out[:, -1, :]
        return self.output(last_state)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """执行一轮训练或评估，返回平均损失。"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(xb)
        loss = criterion(pred, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = yb.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size

    return total_loss / total_count


@torch.no_grad()
def predict_scaled(model: nn.Module, x_scaled: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """对标准化输入做批量预测，输出标准化空间下的 y。"""
    model.eval()
    preds: list[np.ndarray] = []
    x_tensor = torch.from_numpy(x_scaled).unsqueeze(-1)

    for i in range(0, x_tensor.shape[0], batch_size):
        xb = x_tensor[i : i + batch_size].to(device)
        yb = model(xb).squeeze(-1).cpu().numpy()
        preds.append(yb)

    return np.concatenate(preds, axis=0)


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    """计算并汇总核心指标。"""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline)))

    corr, p_value = pearsonr(y_true, y_pred)

    return pd.DataFrame(
        {
            "metric": ["rmse", "mae", "baseline_rmse", "pearson_r", "pearson_p"],
            "value": [rmse, mae, baseline_rmse, float(corr), float(p_value)],
        }
    )


def main() -> None:
    cfg = Config()
    validate_config(cfg)
    set_global_seed(cfg.seed)

    series = generate_series(cfg)
    x_raw, y_raw = build_supervised_dataset(series, cfg.window_size)

    x_train, x_test, y_train, y_test, _x_scaler, y_scaler = split_and_scale(
        x_raw,
        y_raw,
        test_size=cfg.test_size,
    )

    train_loader, test_loader = make_dataloaders(x_train, y_train, x_test, y_test, cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(input_size=1, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print("LSTM MVP: 单变量时间序列一步预测")
    print(
        "Config: "
        f"points={cfg.n_points}, window={cfg.window_size}, epochs={cfg.epochs}, "
        f"batch={cfg.batch_size}, hidden={cfg.hidden_size}, device={device.type}"
    )
    print("\n[Epoch metrics]")
    print("epoch | train_mse | test_mse")
    print("------+-----------+---------")

    first_train_loss: float | None = None
    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        test_loss = run_epoch(model, test_loader, criterion, device, optimizer=None)

        if first_train_loss is None:
            first_train_loss = train_loss

        if epoch == 1 or epoch % 4 == 0 or epoch == cfg.epochs:
            print(f"{epoch:>5d} | {train_loss:>9.5f} | {test_loss:>7.5f}")

    y_pred_scaled = predict_scaled(model, x_test, device=device, batch_size=cfg.batch_size)

    y_test_real = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_real = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_baseline_real = x_raw[-y_test_real.shape[0] :, -1]

    metrics_df = summarize_metrics(y_test_real, y_pred_real, y_baseline_real)

    sample_df = pd.DataFrame(
        {
            "y_true": y_test_real[:10],
            "y_pred": y_pred_real[:10],
            "naive_last_value": y_baseline_real[:10],
            "abs_err": np.abs(y_test_real[:10] - y_pred_real[:10]),
        }
    )

    print("\n[Final metrics]")
    print(metrics_df.to_string(index=False))

    print("\n[First 10 predictions]")
    print(sample_df.to_string(index=False))

    final_train_loss = run_epoch(model, train_loader, criterion, device, optimizer=None)
    model_rmse = float(metrics_df.loc[metrics_df["metric"] == "rmse", "value"].iloc[0])
    baseline_rmse = float(metrics_df.loc[metrics_df["metric"] == "baseline_rmse", "value"].iloc[0])

    if not np.isfinite(model_rmse):
        raise RuntimeError("模型 RMSE 非有限值，训练失败")
    if first_train_loss is not None and final_train_loss > first_train_loss * 1.05:
        raise RuntimeError("训练后损失未改善，可能存在实现错误")
    if model_rmse > baseline_rmse * 1.25:
        raise RuntimeError("模型显著劣于朴素基线，需检查超参数或实现")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
