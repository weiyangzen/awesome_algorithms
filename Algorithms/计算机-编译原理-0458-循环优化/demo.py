"""循环优化（Loop Optimization）最小可运行示例。

该脚本对比两种实现：
1) 基线版本：在循环体内重复计算循环不变式，并使用乘法计算步长项。
2) 优化版本：执行三种经典循环优化。
   - 循环不变式外提（Loop-Invariant Code Motion）
   - 强度削弱（Strength Reduction）
   - 循环展开（Loop Unrolling, 因子=4）

运行方式：
    uv run python Algorithms/计算机-编译原理-0458-循环优化/demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class KernelConfig:
    """测试参数。"""

    n: int = 1_000_000
    a: int = 17
    b: int = 29
    bias: int = 11
    stride: int = 6
    repeats: int = 5


def baseline_loop(cfg: KernelConfig) -> np.ndarray:
    """基线版本：故意保留可优化点。"""
    out = np.empty(cfg.n, dtype=np.int64)
    for i in range(cfg.n):
        invariant = cfg.a * cfg.b + cfg.bias
        stride_term = i * cfg.stride
        poly = (i % 7) - (i % 5)
        out[i] = invariant + stride_term + poly
    return out


def optimized_loop(cfg: KernelConfig) -> np.ndarray:
    """优化版本：外提不变式 + 强度削弱 + 展开。"""
    out = np.empty(cfg.n, dtype=np.int64)

    # 1) 循环不变式外提
    invariant = cfg.a * cfg.b + cfg.bias

    # 2) 强度削弱：i * stride -> 累加型归纳变量
    stride_term = 0

    # 3) 展开：每轮处理 4 个元素
    i = 0
    unroll = 4
    step4 = cfg.stride * unroll

    while i + (unroll - 1) < cfg.n:
        poly0 = (i % 7) - (i % 5)
        i1 = i + 1
        poly1 = (i1 % 7) - (i1 % 5)
        i2 = i + 2
        poly2 = (i2 % 7) - (i2 % 5)
        i3 = i + 3
        poly3 = (i3 % 7) - (i3 % 5)

        out[i] = invariant + stride_term + poly0
        out[i1] = invariant + (stride_term + cfg.stride) + poly1
        out[i2] = invariant + (stride_term + 2 * cfg.stride) + poly2
        out[i3] = invariant + (stride_term + 3 * cfg.stride) + poly3

        i += unroll
        stride_term += step4

    # 处理尾部
    while i < cfg.n:
        poly = (i % 7) - (i % 5)
        out[i] = invariant + stride_term + poly
        i += 1
        stride_term += cfg.stride

    return out


def benchmark_ms(fn, cfg: KernelConfig, repeats: int) -> tuple[float, float, int]:
    """返回 (最短耗时ms, 平均耗时ms, 校验和)。"""
    timings: list[float] = []
    checksum = 0
    for _ in range(repeats):
        t0 = perf_counter()
        arr = fn(cfg)
        dt = (perf_counter() - t0) * 1000.0
        timings.append(dt)
        checksum = int(arr.sum())
    return min(timings), sum(timings) / len(timings), checksum


def main() -> None:
    cfg = KernelConfig()

    base = baseline_loop(cfg)
    opt = optimized_loop(cfg)

    if not np.array_equal(base, opt):
        diff_idx = int(np.nonzero(base != opt)[0][0])
        raise RuntimeError(
            f"优化结果与基线不一致：idx={diff_idx}, base={base[diff_idx]}, opt={opt[diff_idx]}"
        )

    best_base, avg_base, checksum_base = benchmark_ms(baseline_loop, cfg, cfg.repeats)
    best_opt, avg_opt, checksum_opt = benchmark_ms(optimized_loop, cfg, cfg.repeats)

    speedup_best = best_base / best_opt if best_opt > 0 else float("inf")
    speedup_avg = avg_base / avg_opt if avg_opt > 0 else float("inf")

    print("== Loop Optimization MVP ==")
    print(f"n={cfg.n}, repeats={cfg.repeats}")
    print(f"checksum(base)={checksum_base}")
    print(f"checksum(opt) ={checksum_opt}")
    print("correctness=PASS")
    print(f"baseline: best={best_base:.2f} ms, avg={avg_base:.2f} ms")
    print(f"optimized: best={best_opt:.2f} ms, avg={avg_opt:.2f} ms")
    print(f"speedup(best)={speedup_best:.2f}x")
    print(f"speedup(avg) ={speedup_avg:.2f}x")


if __name__ == "__main__":
    main()
