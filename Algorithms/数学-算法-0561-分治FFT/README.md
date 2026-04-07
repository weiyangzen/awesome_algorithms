# 分治FFT

- UID: `MATH-0561`
- 学科: `数学`
- 分类: `算法`
- 源序号: `561`
- 目标目录: `Algorithms/数学-算法-0561-分治FFT`

## R01

分治 FFT（Fast Fourier Transform）通常指 Cooley-Tukey 思路：把长度为 `n` 的 DFT 递归拆成两个长度 `n/2` 的子问题（偶数位与奇数位），再通过蝶形合并。它把 DFT 从 `O(n^2)` 降到 `O(n log n)`，是信号处理与卷积加速的基础算法。

## R02

目标问题是高效计算离散傅里叶变换（DFT）：

`X[k] = Σ_{t=0}^{n-1} x[t] * exp(-2π i k t / n)`。

朴素实现对每个 `k` 都要累加 `n` 次，总成本 `n*n`。分治 FFT 通过复用子结构，避免重复计算旋转因子。

## R03

核心拆分公式（`n` 为 2 的幂）：

- 偶数项序列 `x_even[m] = x[2m]`
- 奇数项序列 `x_odd[m] = x[2m+1]`

则有

`X[k] = E[k] + W_n^k O[k]`，
`X[k + n/2] = E[k] - W_n^k O[k]`，

其中 `E, O` 分别是偶/奇子序列的 DFT，`W_n = exp(-2π i / n)`。这就是蝶形合并。

## R04

分治结构：

1. 若长度为 1，直接返回。
2. 递归计算偶数位 FFT 与奇数位 FFT。
3. 预计算当前层旋转因子 `W_n^k`。
4. 逐点执行蝶形合并得到长度 `n` 结果。

每一层总计算量 `O(n)`，层数 `log2(n)`，总复杂度 `O(n log n)`。

## R05

本目录 MVP 的实现约定：

- 仅支持一维输入向量。
- FFT 核心函数要求输入长度为 2 的幂。
- 线性卷积会自动补零到 `>= len(a)+len(b)-1` 的最小 2 次幂。
- 数值类型使用 `complex128`（频域）和 `float64`（实卷积输出）。

## R06

伪代码（递归 FFT + 卷积）：

```text
FFT(x):
    n = len(x)
    if n == 1: return x
    even = FFT(x[0], x[2], ...)
    odd  = FFT(x[1], x[3], ...)
    for k in [0..n/2-1]:
        tw = exp(-2π i k / n)
        y[k]       = even[k] + tw * odd[k]
        y[k+n/2]   = even[k] - tw * odd[k]
    return y

IFFT(X):
    return conj(FFT(conj(X))) / n

CONV(a, b):
    n = next_power_of_two(len(a)+len(b)-1)
    A = FFT(pad(a, n))
    B = FFT(pad(b, n))
    C = A * B
    c = IFFT(C)
    return real(c[:len(a)+len(b)-1])
```

## R07

正确性直觉：

- DFT 的求和指标可按奇偶拆分，得到两个更小 DFT。
- 旋转因子 `W_n^k` 恰好把奇数分支映射回原频域坐标。
- 蝶形同时产出 `k` 和 `k+n/2` 两个频点，覆盖全部输出。
- IFFT 的共轭公式来自 DFT 矩阵的共轭对称关系，因此可复用同一 FFT 例程。

## R08

复杂度：

- 单次 FFT：`T(n)=2T(n/2)+O(n)`，解得 `O(n log n)`。
- 卷积流程：两次 FFT + 一次点乘 + 一次 IFFT，仍为 `O(n log n)`。
- 额外空间：递归版本存在切片与栈开销，量级 `O(n)` 到 `O(n log n)`（实现相关）。

## R09

数值与工程注意点：

- 浮点误差不可避免，验证用 `np.allclose` 而非逐元素全等。
- IFFT 后理论实数结果可能带极小虚部，应取 `.real`。
- 输入非 2 次幂时必须补零；否则递归拆分不满足结构假设。

## R10

FFT 卷积定理：

- 时域线性卷积 `c = a * b`
- 频域逐点乘法 `C = A · B`

因此卷积可通过“补零 -> FFT -> 点乘 -> IFFT”实现。这个流程在多项式乘法、滤波和相关计算里非常常见。

## R11

边界条件：

- 单元素输入：FFT 直接返回该元素。
- 空数组：本 MVP 显式报错，不做隐式约定。
- 非 2 次幂长度：在卷积接口中自动补零到合适长度。
- 负数或小数输入：算法同样适用。

## R12

常见错误：

- 把 `exp(-2π i k / n)` 写成正号，导致方向相反。
- 忘记 IFFT 最后除以 `n`。
- 卷积补零长度不足，造成循环卷积“绕回污染”。
- 蝶形写回时覆盖了未读取值（应先保存子结果）。

## R13

`demo.py` 结构：

- `next_power_of_two`：计算补零目标长度。
- `fft_divide_conquer`：递归 Cooley-Tukey FFT 主体。
- `ifft_divide_conquer`：共轭技巧实现逆变换。
- `convolution_fft`：FFT 线性卷积实现。
- `convolution_naive`：`O(n^2)` 参考实现。
- `main`：固定样例 + 随机对拍 + 打印结果。

## R14

运行方式：

```bash
python3 demo.py
```

无命令行参数，无交互输入，脚本会直接执行示例与验证。

## R15

预期输出特征：

- 打印 `Input a` 与 `Input b`。
- 打印 `FFT convolution` 与 `Naive convolution`。
- 所有断言通过后打印 `All checks passed.`。

若实现错误，脚本会抛出 `AssertionError` 指示失败维度。

## R16

可扩展方向：

- 迭代版（bit-reversal）FFT，减少递归与切片开销。
- 实数优化（RFFT）减少一半频谱冗余。
- 模数域 NTT（如 `998244353`）用于整数精确卷积。
- 2D FFT 用于图像卷积与频域滤波。

## R17

交付文件说明：

- `README.md`：分治 FFT 的原理、复杂度、坑点与实现说明。
- `demo.py`：可直接运行的 Python MVP（依赖 `numpy`）。
- `meta.json`：任务元数据（UID/目录/蓝图路径）供校验脚本使用。

## R18

`demo.py` 的源码级算法流（8 步）：

1. 在 `main` 读取固定向量 `a, b`，调用 `convolution_fft(a, b)`。  
2. `convolution_fft` 先计算输出长度 `len(a)+len(b)-1`，再用 `next_power_of_two` 得到补零长度 `n`。  
3. 创建复数缓冲区 `fa, fb`，把输入复制进去，其余位置补零。  
4. 调用 `fft_divide_conquer(fa)` 与 `fft_divide_conquer(fb)`：函数递归拆分偶/奇子序列直到长度 1。  
5. 在递归回溯阶段，按 `twiddle = exp(-2j*pi*k/n)` 执行蝶形合并，产出完整频谱 `Fa, Fb`。  
6. 逐点相乘 `Fc = Fa * Fb`，把时域卷积转化为频域乘法。  
7. 调用 `ifft_divide_conquer(Fc)`，其内部执行 `conj(FFT(conj(Fc))) / n`，再截取前 `out_len` 项并取实部。  
8. `main` 用 `convolution_naive` 与 `np.fft.fft` 参考结果进行断言对拍；全部通过后输出 `All checks passed.`。  

这 8 步完整展开了本实现的“非黑箱”路径：递归拆分、蝶形合并、逆变换归一化与自动验证都在源码中可直接追踪。
