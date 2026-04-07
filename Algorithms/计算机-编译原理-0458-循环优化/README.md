# 循环优化

- UID: `CS-0298`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `458`
- 目标目录: `Algorithms/计算机-编译原理-0458-循环优化`

## R01

循环优化（Loop Optimization）是编译器中最核心的性能优化方向之一，目标是在不改变程序语义的前提下，降低循环体内每次迭代的指令开销。本 MVP 聚焦三个经典子技术：
- 循环不变式外提（Loop-Invariant Code Motion）
- 强度削弱（Strength Reduction）
- 循环展开（Loop Unrolling）

## R02

问题抽象：给定一个固定结构的数值循环，循环体中包含重复计算和可替代的高成本运算。需要构造“基线版本”与“优化版本”，并验证：
- 结果完全一致（逐元素相等）
- 优化后平均耗时更低

## R03

MVP 输入输出定义：
- 输入：`KernelConfig(n, a, b, bias, stride, repeats)`
- 计算目标：对 `i=0..n-1` 计算 `invariant + i*stride + ((i%7)-(i%5))`
- 输出：`np.ndarray[int64]` 结果向量，以及性能统计（best/avg 毫秒与加速比）

## R04

基线实现保留可优化点：
- 每次迭代都计算 `invariant = a*b+bias`
- 每次迭代都通过乘法计算 `i*stride`

这些写法在语义上正确，但会产生不必要的重复工作，适合作为循环优化的教学对照。

## R05

优化实现包含 3 个 pass（手工模拟编译器效果）：
- 不变式外提：把 `a*b+bias` 移到循环外计算 1 次
- 强度削弱：把 `i*stride` 改为归纳变量 `stride_term += stride`
- 循环展开：每轮处理 4 个元素，减少分支与循环控制开销

## R06

正确性要点：
- 不变式外提不改变值域，因为 `a,b,bias` 在循环内不修改
- 强度削弱保持等价关系：第 `k` 次迭代前 `stride_term == k*stride`
- 展开后按 `i,i+1,i+2,i+3` 顺序写回，尾部循环覆盖剩余元素
- 通过 `np.array_equal(base, opt)` 做逐元素校验

## R07

复杂度分析：
- 时间复杂度：两版本均为 `O(n)`
- 空间复杂度：均为 `O(n)`（输出数组）
- 优化收益来源：降低常数项，而非改变渐进阶

## R08

实验配置：
- 语言：Python 3.10+
- 依赖：`numpy`
- 默认参数：`n=1_000_000`, `repeats=5`
- 计时方法：`time.perf_counter()`，输出 best 与 avg

## R09

工程实现文件：
- `demo.py`

关键函数：
- `baseline_loop(cfg)`：未优化循环
- `optimized_loop(cfg)`：三种优化叠加
- `benchmark_ms(fn, cfg, repeats)`：重复测量耗时
- `main()`：正确性校验 + 性能对比 + 打印结果

## R10

运行方式：

```bash
uv run python Algorithms/计算机-编译原理-0458-循环优化/demo.py
```

脚本无需交互输入，直接输出正确性和性能结果。

## R11

一次典型输出（不同机器会有波动）：

```text
== Loop Optimization MVP ==
n=1000000, repeats=5
checksum(base)=3000501999997
checksum(opt) =3000501999997
correctness=PASS
baseline: best=xxx.xx ms, avg=xxx.xx ms
optimized: best=yyy.yy ms, avg=yyy.yy ms
speedup(best)=z.zzx
speedup(avg) =w.wwx
```

## R12

参数说明与调优建议：
- `n`：问题规模，越大越能观察到稳定加速比
- `stride`：步长常量，用于展示乘法到加法的替换
- `repeats`：重复测量次数，建议至少 3 次
- 若希望更快完成 CI，可把 `n` 调低到 `200000`

## R13

与真实编译器优化 pass 的对应关系：
- LICM（Loop-Invariant Code Motion）对应不变式外提
- Induction Variable Optimization 对应强度削弱
- Loop Unroll Pass 对应 4 倍展开

该示例是“源码层模拟”，便于理解 pass 的语义，不依赖特定编译器后端。

## R14

适用边界：
- 适合数值密集、循环体可分析、无复杂别名副作用的场景
- 不适合大量 I/O 或分支极不规则的循环
- 在解释器语言中收益可能受解释器开销影响，在编译型语言中通常更明显

## R15

常见风险与规避：
- 风险：展开后尾部处理遗漏导致越界或漏算
- 风险：强度削弱初始化错误导致整体偏移
- 规避：保留基线版本并做逐元素一致性断言
- 规避：使用固定测试参数进行回归对比

## R16

可扩展方向：
- 增加循环融合（loop fusion）与循环分裂（loop fission）案例
- 增加自动选择展开因子（2/4/8）并比较收益
- 引入简化 IR，做 pass 前后代码打印，贴近编译器流水线

## R17

验收清单：
- `README.md` 与 `demo.py` 已全部替换占位符
- `uv run python .../demo.py` 可直接运行
- 输出包含 `correctness=PASS`
- 输出包含 `baseline/optimized` 耗时与加速比

## R18

`demo.py` 的源码级算法流程（8 步）：
1. 读取 `KernelConfig`，确定 `n/a/b/bias/stride/repeats`。
2. 执行 `baseline_loop`：在每次迭代内重复计算 `invariant` 与 `i*stride`，得到基线数组。
3. 执行 `optimized_loop` 前先在循环外计算 `invariant`，完成不变式外提。
4. 在优化循环中维护 `stride_term` 归纳变量，每次加 `stride` 或每轮展开后加 `4*stride`，完成强度削弱。
5. 使用 4 倍展开一次写入 `i..i+3` 四个位置，减少循环控制开销。
6. 对尾部不足 4 个元素的部分进入尾循环，保证全范围覆盖。
7. 用 `np.array_equal` 对基线与优化结果做逐元素一致性检查，不一致则抛错定位。
8. 用 `benchmark_ms` 重复计时并打印 best/avg 与 speedup，形成可复现实验结论。
