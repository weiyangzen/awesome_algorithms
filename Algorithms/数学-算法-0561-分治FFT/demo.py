"""MATH-0561: 分治 FFT (Cooley-Tukey) 的最小可运行 MVP。

实现目标：
1) 用递归分治实现 FFT（非黑箱）。
2) 基于 FFT 实现多项式卷积（线性卷积）。
3) 用朴素 O(n^2) 卷积和 NumPy 参考结果做自动校验。
"""

from __future__ import annotations

import numpy as np


def next_power_of_two(n: int) -> int:
    """返回大于等于 n 的最小 2 的幂。"""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fft_divide_conquer(x: np.ndarray) -> np.ndarray:
    """递归 Cooley-Tukey FFT。

    约束：len(x) 必须为 2 的幂。
    返回：x 的离散傅里叶变换结果（complex128）。
    """
    n = x.shape[0]
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("FFT input length must be a non-zero power of two.")

    if n == 1:
        return x.astype(np.complex128, copy=True)

    even = fft_divide_conquer(x[::2])
    odd = fft_divide_conquer(x[1::2])

    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    out = np.empty(n, dtype=np.complex128)
    out[: n // 2] = even + twiddle * odd
    out[n // 2 :] = even - twiddle * odd
    return out


def ifft_divide_conquer(X: np.ndarray) -> np.ndarray:
    """用共轭技巧实现 IFFT：ifft(X) = conj(fft(conj(X))) / n。"""
    n = X.shape[0]
    return np.conjugate(fft_divide_conquer(np.conjugate(X))) / n


def convolution_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """基于 FFT 的线性卷积，时间复杂度 O(n log n)。"""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Only 1-D arrays are supported.")
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Input arrays must be non-empty.")

    out_len = len(a) + len(b) - 1
    n = next_power_of_two(out_len)

    fa = np.zeros(n, dtype=np.complex128)
    fb = np.zeros(n, dtype=np.complex128)
    fa[: len(a)] = a
    fb[: len(b)] = b

    Fa = fft_divide_conquer(fa)
    Fb = fft_divide_conquer(fb)
    Fc = Fa * Fb
    c = ifft_divide_conquer(Fc)
    return c[:out_len].real


def convolution_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """朴素线性卷积，时间复杂度 O(n^2)。"""
    out = np.zeros(len(a) + len(b) - 1, dtype=np.float64)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def main() -> None:
    # 固定样例：演示非 2 次幂长度输入的卷积。
    a = np.array([1.0, 2.0, 3.0, 4.0, 2.0], dtype=np.float64)
    b = np.array([3.0, -1.0, 0.5], dtype=np.float64)

    fast = convolution_fft(a, b)
    slow = convolution_naive(a, b)

    print("Input a:", a.tolist())
    print("Input b:", b.tolist())
    print("FFT convolution:", np.round(fast, 8).tolist())
    print("Naive convolution:", np.round(slow, 8).tolist())

    if not np.allclose(fast, slow, atol=1e-9):
        raise AssertionError("FFT convolution does not match naive convolution.")

    # 对拍 1：FFT 与 NumPy 参考实现比对（仅用于验证）。
    rng = np.random.default_rng(561)
    for bits in range(1, 8):
        n = 1 << bits
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        ours = fft_divide_conquer(x)
        ref = np.fft.fft(x)
        if not np.allclose(ours, ref, atol=1e-9):
            raise AssertionError(f"FFT mismatch at n={n}")

    # 对拍 2：卷积随机测试，覆盖不同长度。
    for n in [2, 3, 5, 8]:
        for m in [2, 4, 7]:
            x = rng.integers(-5, 6, size=n).astype(np.float64)
            y = rng.integers(-5, 6, size=m).astype(np.float64)
            lhs = convolution_fft(x, y)
            rhs = convolution_naive(x, y)
            if not np.allclose(lhs, rhs, atol=1e-9):
                raise AssertionError(f"Convolution mismatch at n={n}, m={m}")

    print("All checks passed.")


if __name__ == "__main__":
    main()
