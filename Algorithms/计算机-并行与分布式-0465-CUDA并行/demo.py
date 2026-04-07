"""CUDA parallel MVP: vector add + matrix multiplication benchmarks.

The script is deterministic and requires no interactive input.
If CUDA is unavailable, it automatically falls back to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class BenchmarkConfig:
    vector_size: int
    mat_shape: Tuple[int, int, int]  # (M, K, N)
    tile_size: int
    warmup: int
    repeats: int
    seed: int


@dataclass
class BenchStats:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_gops: float


def pick_config(use_cuda: bool) -> BenchmarkConfig:
    if use_cuda:
        return BenchmarkConfig(
            vector_size=4_000_000,
            mat_shape=(512, 512, 512),
            tile_size=64,
            warmup=2,
            repeats=6,
            seed=2026,
        )
    return BenchmarkConfig(
        vector_size=1_000_000,
        mat_shape=(256, 256, 256),
        tile_size=64,
        warmup=1,
        repeats=4,
        seed=2026,
    )


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def validate_tensor_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} contains non-finite values.")


def time_callable(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
    operation_count: float,
) -> BenchStats:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >=0 and repeats must be >0.")
    if operation_count <= 0:
        raise ValueError("operation_count must be >0.")

    for _ in range(warmup):
        _ = fn()
    sync_if_needed(device)

    samples_ms = []
    for _ in range(repeats):
        t0 = perf_counter()
        out = fn()
        sync_if_needed(device)
        t1 = perf_counter()
        validate_tensor_finite("benchmark output", out)
        samples_ms.append((t1 - t0) * 1e3)

    arr = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    throughput_gops = float(operation_count / (mean_ms / 1e3) / 1e9)
    return BenchStats(
        mean_ms=mean_ms,
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        throughput_gops=throughput_gops,
    )


def make_vector_data(size: int, seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if size <= 0:
        raise ValueError("size must be > 0.")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x_cpu = torch.randn(size, dtype=torch.float32, generator=g)
    y_cpu = torch.randn(size, dtype=torch.float32, generator=g)
    return x_cpu.to(device), y_cpu.to(device)


def make_matrix_data(
    shape: Tuple[int, int, int],
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    m, k, n = shape
    if min(m, k, n) <= 0:
        raise ValueError(f"Invalid matrix shape: {shape}")

    rng = np.random.default_rng(seed)
    a_np = rng.normal(size=(m, k)).astype(np.float32)
    b_np = rng.normal(size=(k, n)).astype(np.float32)

    a = torch.from_numpy(a_np).to(device)
    b = torch.from_numpy(b_np).to(device)
    return a, b, a_np, b_np


def tiled_matmul(a: torch.Tensor, b: torch.Tensor, tile_size: int) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("tiled_matmul expects 2D tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: a={tuple(a.shape)}, b={tuple(b.shape)}.")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0.")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("This MVP expects float32 tensors.")

    m, k = a.shape
    _, n = b.shape
    c = torch.zeros((m, n), dtype=a.dtype, device=a.device)

    for i in range(0, m, tile_size):
        i_end = min(i + tile_size, m)
        for j in range(0, n, tile_size):
            j_end = min(j + tile_size, n)
            acc = torch.zeros((i_end - i, j_end - j), dtype=a.dtype, device=a.device)
            for kk in range(0, k, tile_size):
                k_end = min(kk + tile_size, k)
                a_block = a[i:i_end, kk:k_end]
                b_block = b[kk:k_end, j:j_end]
                acc = acc + (a_block @ b_block)
            c[i:i_end, j:j_end] = acc

    return c


def run_vector_add_benchmark(device: torch.device, cfg: BenchmarkConfig) -> Dict[str, float]:
    x, y = make_vector_data(size=cfg.vector_size, seed=cfg.seed, device=device)

    z_device = x + y
    z_cpu_ref = x.cpu() + y.cpu()
    vector_max_abs_error = float(torch.max(torch.abs(z_device.cpu() - z_cpu_ref)).item())

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    cpu_stats = time_callable(
        fn=lambda: x_cpu + y_cpu,
        device=torch.device("cpu"),
        warmup=cfg.warmup,
        repeats=cfg.repeats,
        operation_count=float(cfg.vector_size),
    )
    device_stats = time_callable(
        fn=lambda: x + y,
        device=device,
        warmup=cfg.warmup,
        repeats=cfg.repeats,
        operation_count=float(cfg.vector_size),
    )

    speedup = cpu_stats.mean_ms / device_stats.mean_ms

    return {
        "vector_max_abs_error": vector_max_abs_error,
        "vector_cpu_mean_ms": cpu_stats.mean_ms,
        "vector_device_mean_ms": device_stats.mean_ms,
        "vector_cpu_gops": cpu_stats.throughput_gops,
        "vector_device_gops": device_stats.throughput_gops,
        "vector_speedup_vs_cpu": float(speedup),
    }


def run_matmul_benchmark(device: torch.device, cfg: BenchmarkConfig) -> Dict[str, float]:
    a, b, a_np, b_np = make_matrix_data(shape=cfg.mat_shape, seed=cfg.seed + 1, device=device)
    m, k, n = cfg.mat_shape

    c_ref = a @ b
    c_np = a_np @ b_np
    c_ref_cpu = c_ref.cpu().numpy()

    matmul_max_abs_error_vs_numpy = float(np.max(np.abs(c_ref_cpu - c_np)))

    c_tiled = tiled_matmul(a, b, tile_size=cfg.tile_size)
    tiled_max_abs_error_vs_torch = float(torch.max(torch.abs(c_tiled - c_ref)).item())

    matmul_ops = float(2 * m * k * n)
    matmul_stats = time_callable(
        fn=lambda: a @ b,
        device=device,
        warmup=cfg.warmup,
        repeats=cfg.repeats,
        operation_count=matmul_ops,
    )

    tiled_repeats = max(1, min(2, cfg.repeats))
    tiled_stats = time_callable(
        fn=lambda: tiled_matmul(a, b, tile_size=cfg.tile_size),
        device=device,
        warmup=0,
        repeats=tiled_repeats,
        operation_count=matmul_ops,
    )

    a_cpu = a.cpu()
    b_cpu = b.cpu()
    cpu_stats = time_callable(
        fn=lambda: a_cpu @ b_cpu,
        device=torch.device("cpu"),
        warmup=1,
        repeats=max(2, cfg.repeats // 2),
        operation_count=matmul_ops,
    )

    return {
        "matmul_max_abs_error_vs_numpy": matmul_max_abs_error_vs_numpy,
        "tiled_max_abs_error_vs_torch": tiled_max_abs_error_vs_torch,
        "matmul_device_mean_ms": matmul_stats.mean_ms,
        "matmul_device_gops": matmul_stats.throughput_gops,
        "tiled_device_mean_ms": tiled_stats.mean_ms,
        "tiled_device_gops": tiled_stats.throughput_gops,
        "matmul_cpu_mean_ms": cpu_stats.mean_ms,
        "matmul_cpu_gops": cpu_stats.throughput_gops,
        "matmul_speedup_vs_cpu": float(cpu_stats.mean_ms / matmul_stats.mean_ms),
    }


def main() -> None:
    torch.set_grad_enabled(False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    cfg = pick_config(use_cuda=use_cuda)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(cfg.seed)

    print("CUDA Parallel MVP")
    print(f"device={device.type}, cuda_available={use_cuda}")
    if use_cuda:
        print(f"gpu_name={torch.cuda.get_device_name(device)}")
    print(
        "config: "
        f"vector_size={cfg.vector_size}, "
        f"mat_shape={cfg.mat_shape}, "
        f"tile_size={cfg.tile_size}, "
        f"warmup={cfg.warmup}, repeats={cfg.repeats}"
    )

    vec_metrics = run_vector_add_benchmark(device=device, cfg=cfg)
    mat_metrics = run_matmul_benchmark(device=device, cfg=cfg)

    print("\n=== Vector Add Metrics ===")
    print(f"vector_max_abs_error={vec_metrics['vector_max_abs_error']:.6e}")
    print(f"vector_cpu_mean_ms={vec_metrics['vector_cpu_mean_ms']:.4f}")
    print(f"vector_device_mean_ms={vec_metrics['vector_device_mean_ms']:.4f}")
    print(f"vector_cpu_gops={vec_metrics['vector_cpu_gops']:.4f}")
    print(f"vector_device_gops={vec_metrics['vector_device_gops']:.4f}")
    print(f"vector_speedup_vs_cpu={vec_metrics['vector_speedup_vs_cpu']:.4f}")

    print("\n=== Matrix Multiply Metrics ===")
    print(f"matmul_max_abs_error_vs_numpy={mat_metrics['matmul_max_abs_error_vs_numpy']:.6e}")
    print(f"tiled_max_abs_error_vs_torch={mat_metrics['tiled_max_abs_error_vs_torch']:.6e}")
    print(f"matmul_cpu_mean_ms={mat_metrics['matmul_cpu_mean_ms']:.4f}")
    print(f"matmul_device_mean_ms={mat_metrics['matmul_device_mean_ms']:.4f}")
    print(f"tiled_device_mean_ms={mat_metrics['tiled_device_mean_ms']:.4f}")
    print(f"matmul_cpu_gops={mat_metrics['matmul_cpu_gops']:.4f}")
    print(f"matmul_device_gops={mat_metrics['matmul_device_gops']:.4f}")
    print(f"tiled_device_gops={mat_metrics['tiled_device_gops']:.4f}")
    print(f"matmul_speedup_vs_cpu={mat_metrics['matmul_speedup_vs_cpu']:.4f}")

    checks = {
        "vector_error_ok": vec_metrics["vector_max_abs_error"] < 1e-5,
        "matmul_error_ok": mat_metrics["matmul_max_abs_error_vs_numpy"] < 1e-3,
        "tiled_error_ok": mat_metrics["tiled_max_abs_error_vs_torch"] < 1e-3,
        "all_times_positive": (
            vec_metrics["vector_cpu_mean_ms"] > 0.0
            and vec_metrics["vector_device_mean_ms"] > 0.0
            and mat_metrics["matmul_cpu_mean_ms"] > 0.0
            and mat_metrics["matmul_device_mean_ms"] > 0.0
            and mat_metrics["tiled_device_mean_ms"] > 0.0
        ),
    }
    global_checks_pass = all(checks.values())

    print("\n=== Checks ===")
    for key, value in checks.items():
        print(f"{key}={value}")
    print(f"global_checks_pass={global_checks_pass}")


if __name__ == "__main__":
    main()
