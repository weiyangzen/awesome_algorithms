# FWT (快速沃尔什变换)

- UID: `MATH-0562`
- 学科: `数学`
- 分类: `算法`
- 源序号: `562`
- 目标目录: `Algorithms/数学-算法-0562-FWT_(快速沃尔什变换)`

## R01

FWT（Fast Walsh Transform，常与 FWHT Fast Walsh-Hadamard Transform 混用）是把序列映射到由 `+1/-1` 组成的沃尔什-哈达玛基上的快速变换。  
它和 FFT 的共同点是都用分治+蝶形把复杂度从二次降到 `O(n log n)`；不同点是 FWT 处理的是按位运算卷积（尤其是 XOR/AND/OR）场景，而不是普通加法卷积。

## R02

本题聚焦 `XOR 卷积`：给定序列 `a, b`，定义

`c[k] = Σ a[i] * b[j]`，其中求和条件是 `i xor j = k`。

朴素算法需要双重循环 `O(n^2)`；当 `n` 为 2 的幂时，可用 FWT 在 `O(n log n)` 内完成。

## R03

核心线性代数基础是哈达玛矩阵 `H_n`（`n=2^m`）递归构造：

- `H_1 = [1]`
- `H_{2n} = [[H_n, H_n], [H_n, -H_n]]`

FWT 的每一层蝶形更新：

- `(u, v) -> (u + v, u - v)`

正变换与逆变换形式几乎相同，仅归一化不同。本文实现采用“逆变换最后整体除以 `n`”。

## R04

XOR 卷积的频域性质：

- 先对 `a`、`b` 做 FWT 得到 `A`、`B`
- 逐点相乘 `C[t] = A[t] * B[t]`
- 对 `C` 做逆 FWT，得到时域结果 `c`

这与 FFT 计算普通卷积的流程同构，只是“基函数”和“卷积定义（xor）”不同。

## R05

输入约束与实现约定（本 MVP）：

- 输入为一维数值向量（`numpy.ndarray`）
- 若长度不是 2 的幂，自动零填充到不小于 `max(len(a), len(b))` 的最小 2 次幂
- 计算使用 `float64`，便于统一演示与误差比较
- 输出为填充后长度的 XOR 卷积结果

## R06

伪代码（XOR 版本）：

```text
function FWHT(a, inverse):
    n = len(a)
    step = 1
    while step < n:
        for i in range(0, n, 2*step):
            for j in [0..step-1]:
                u = a[i+j]
                v = a[i+j+step]
                a[i+j] = u + v
                a[i+j+step] = u - v
        step *= 2
    if inverse:
        a /= n

function XOR_CONV(a, b):
    pad to same power-of-two length n
    A = FWHT(a, false)
    B = FWHT(b, false)
    C = pointwise_mul(A, B)
    c = FWHT(C, true)
    return c
```

## R07

正确性直觉：

- 每层蝶形等价于把长度为 `2*step` 的块映射到哈达玛子空间
- 多层迭代后等价于乘上完整哈达玛矩阵 `H_n`
- `H_n * H_n = nI`，因此逆变换可由同样蝶形流程再除 `n` 得到
- XOR 卷积在该基下可对角化，所以时域卷积变成频域逐点乘法

## R08

复杂度分析：

- 单次 FWT：`O(n log n)`，空间可原地 `O(1)`（不计输入输出）
- XOR 卷积总流程：两次正变换 + 一次逆变换 + 一次逐点乘  
  总时间 `O(n log n)`，远优于朴素 `O(n^2)`

## R09

数值与工程注意事项：

- 整数输入经 `+/-` 操作理论上仍为整数，但实现常用浮点，最终可按需求 `round`
- 若 `n` 很大，浮点误差会累积，验证时应使用 `np.allclose` 而非逐元素全等
- 逆变换归一化可放在每层或末尾一次完成；本实现选择末尾一次除 `n`，代码更直观

## R10

与常见变换对比：

- FFT：处理普通加法卷积（`i + j = k`）
- FWT-XOR：处理按位异或卷积（`i xor j = k`）
- FWT-AND/OR：可通过改蝶形公式处理按位与/或卷积

因此 FWT 是“位运算卷积工具箱”，常见于组合计数、子集 DP、位掩码优化题。

## R11

边界条件：

- 长度为 1：变换应保持稳定，卷积退化为单点乘法
- 空数组：本 MVP 不支持，调用前应保证非空
- 非 2 次幂长度：通过 `next_power_of_two` 自动补零
- 输入符号混合（正负值）：算法同样适用

## R12

常见错误清单：

- 忘记逆变换除以 `n`，导致结果整体放大
- 使用原地更新但未暂存 `left/right`，造成数据覆盖污染
- 把 XOR 卷积误写成普通卷积下标
- 忽略长度补齐，导致 `i xor j` 访问越界或定义域不一致

## R13

`demo.py` 的最小可运行内容：

- `next_power_of_two`：长度补齐
- `fwht_inplace`：原地 FWT/逆 FWT
- `xor_convolution_fwt`：主算法
- `xor_convolution_naive`：二次复杂度参考实现
- `main`：固定样例 + 随机对拍，脚本运行即给出验证结果

## R14

运行方式：

```bash
python3 demo.py
```

无需命令行参数，无交互输入，直接打印样例结果和校验结论。

## R15

预期输出特征（数值会按样例确定）：

- 打印输入向量 `x`、`y`
- 打印 `FWT XOR convolution` 与 `Naive XOR convolution`
- 两者一致时输出 `All checks passed.`

若实现出错，会抛出 `AssertionError`，便于自动化检测。

## R16

可扩展方向：

- 将 XOR 蝶形推广到 AND/OR 变换
- 在计数型任务中改用模数域（如 `mod 998244353`）以避免浮点误差
- 批量处理多组向量时可复用变换缓冲区，减少重复分配
- 对大规模输入可考虑并行分块或 GPU 张量实现

## R17

本目录交付物说明：

- `README.md`：给出 FWT 的定义、原理、复杂度、坑点与实现解读
- `demo.py`：独立可运行的 Python MVP（仅依赖 `numpy`）
- `meta.json`：任务元数据与目录映射，用于自动化索引校验

## R18

`demo.py` 的源码级算法流（8 步）：

1. 读取输入向量 `x, y`，调用 `next_power_of_two` 计算统一长度 `n`。  
2. 创建长度 `n` 的零数组 `fx, fy`，把原始数据拷贝进去（其余位置补零）。  
3. 在 `fwht_inplace` 中从 `step=1` 开始分层迭代，每层处理块长 `2*step`。  
4. 对每个块做蝶形：暂存 `left/right`，再写回 `left+right` 与 `left-right`。  
5. 对 `fx`、`fy` 分别完成正变换，得到谱域表示。  
6. 逐点相乘 `fz = fx * fy`，完成卷积定理对应的频域乘法。  
7. 对 `fz` 执行同样蝶形流程并在末尾除以 `n`（`inverse=True`），得到时域 XOR 卷积。  
8. 在 `main` 中用 `xor_convolution_naive` 做对拍，并进行多组随机测试，确保实现正确。  

这 8 步完整展开了“不是黑箱”的 FWT 实现路径：从长度补齐、蝶形更新到逆变换归一化与结果验证。
