# zk-STARKs

- UID: `CS-0150`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `294`
- 目标目录: `Algorithms/计算机-密码学-0294-zk-STARKs`

## R01

zk-STARKs（Zero-Knowledge Scalable Transparent ARguments of Knowledge）是一类“透明设置、可扩展、后量子友好”的零知识证明体系。

本条目给出一个教学型 MVP：
- 用 Fibonacci 递推构造 AIR（Algebraic Intermediate Representation）约束；
- 用 Merkle 树承诺评估向量；
- 用 Fiat-Shamir 生成随机挑战；
- 用简化 FRI 折叠流程做低度性检查演示。

说明：本实现强调“可读可跑”，不是生产级密码系统，也不提供真正零知识。

## R02

本条目问题定义：

- 输入：
  - 有限域参数 `p = 2^61 - 1`；
  - 公共初值 `a0, a1`；
  - 轨迹长度 `n = 32`（2 的幂）；
  - 查询数 `queries_per_round` 与次数学界 `degree_bound`。
- 输出：
  - 一个教学型 `StarkProof`（轨迹、约束、Merkle 根、FRI 轮次开口等）；
  - 验证函数对“诚实轨迹”为 `True`，对“被篡改轨迹”为 `False`。

## R03

MVP 的核心数学对象：

1. 计算轨迹：
   - `t[0]=a0, t[1]=a1`
   - `t[i+2] = t[i+1] + t[i] (mod p)`
2. AIR 约束（长度与轨迹相同）：
   - 过渡约束：`c[i] = t[i+2] - t[i+1] - t[i] (mod p), i=0..n-3`
   - 边界约束：`c[n-2] = t[0]-a0, c[n-1] = t[1]-a1`
3. FRI 风格折叠：
   - `v_next[j] = v[2j] + beta * v[2j+1] (mod p)`

若轨迹合法，则约束向量应全 0，折叠到最后也应为 0。

## R04

时间复杂度（设 `n` 为轨迹长度，MVP 里 `n=32`）：

- 轨迹构造：`O(n)`；
- 约束计算：`O(n)`；
- Merkle 建树（每层线性，总和）：`O(n)`；
- FRI 折叠总和：`O(n)`；
- 模 `p` Vandermonde 线性方程求解（用于次数估计）：`O(n^3)`。

因此总复杂度由插值步骤主导，为 `O(n^3)`（在小规模教学场景可接受）。

## R05

空间复杂度：

- 轨迹与约束各 `O(n)`；
- 单次 Merkle 树存储 `O(n)`；
- 各轮折叠向量总存储 `O(n)`（几何级数）；
- Vandermonde 增广矩阵 `O(n^2)`。

总体空间复杂度为 `O(n^2)`。

## R06

微型例子（不含大整数细节）：

- 设 `a0=a1=1`，前几项轨迹：`1,1,2,3,5,8,...`；
- 过渡约束首项：`2-1-1=0`；第二项：`3-2-1=0`；
- 边界约束：`t[0]-1=0, t[1]-1=0`；
- 因此约束向量应为全 0，FRI 每轮折叠仍保持 0。

若篡改 `t[10]`，则至少若干过渡约束会变成非零，验证会失败。

## R07

zk-STARKs 的工程价值：

- 透明设置：不依赖可信初始化（区别于很多 SNARK 方案）；
- 可扩展：大规模约束下验证成本仍可控；
- 哈希与域算术驱动，具有较好的后量子兼容性；
- 在 Rollup、可验证计算、证明系统工程中非常关键。

## R08

理论基础（工程摘要）：

1. 将程序正确性转写成代数约束（AIR）。
2. 把“约束成立”转成“某些多项式在域上满足低次结构”。
3. 通过 Merkle 承诺获得可验证的一致性（防止事后改值）。
4. 通过 Fiat-Shamir 把交互挑战转成非交互随机性。
5. 用 FRI 递归折叠检查“函数接近低次多项式”。

本 MVP 对以上每步都给了对应代码骨架。

## R09

适用条件与局限：

- 适用：教学、流程验证、算法拆解、单元测试式理解；
- 局限：
  - 未实现零知识掩蔽；
  - 验证端可见完整轨迹，不满足隐私目标；
  - FRI/承诺策略为简化版，不可直接用于安全生产环境；
  - 参数与安全性分析未达到密码工程标准。

## R10

正确性框架（本实现）：

1. 先验证轨迹承诺根（Merkle root）一致。  
2. 再验证边界与递推过渡条件。  
3. 重算 AIR 约束并比对约束承诺根。  
4. 对约束向量做次数重算并检查 `degree_bound`。  
5. 逐轮校验 FRI 挑战、折叠关系和开口路径。  
6. 检查末轮值与全零约束推导一致（最终值应为 0）。

上述任一步失败即拒绝证明。

## R11

误差与稳定性：

- 在有限域中运算，避免浮点误差；
- 使用 Python 大整数 + 模运算，数值稳定性高；
- 挑战与查询由哈希派生，结果可复现；
- 教学实现中最昂贵也最敏感的是模线性方程求解（次数估计），`n` 较大时应改用更高效多项式工具链。

## R12

性能视角：

- 当 `n=32` 时，`demo.py` 可快速完成证明与验证；
- 主要计算热点：
  - `interpolate_degree` 的 `O(n^3)` 消元；
  - 多轮 Merkle 构建与路径验证；
- 该实现优先“算法可追踪性”，未做底层 SIMD/并行优化。

## R13

本目录可验证保证：

- `demo.py` 无交互输入，固定参数可直接运行；
- 运行会打印两组案例：
  - `Case 1` 诚实轨迹应通过；
  - `Case 2` 篡改轨迹应失败；
- 程序末尾用断言强制检查上述结论，成功后输出 `All checks passed.`。

## R14

鲁棒性与常见失效模式：

- 若 `trace_len` 非 2 的幂，Merkle/FRI 简化实现会拒绝；
- 若人为破坏开口路径，Merkle 校验会失败；
- 若修改任一折叠值但不匹配 `a + beta*b`，FRI 一致性会失败；
- 若轨迹不满足 Fibonacci 递推，AIR 过渡约束会失败；
- 若约束次数超界，低次检查会失败。

## R15

实现设计（`demo.py`）：

- `StarkConfig`：统一管理轨迹长度、边界、查询数和次数界。  
- `fibonacci_trace` / `compute_air_constraints`：构造 AIR 实例。  
- `build_merkle_tree` / `merkle_path` / `verify_merkle_path`：承诺与开口。  
- `derive_beta` / `derive_queries`：Fiat-Shamir 挑战和查询。  
- `fold_layer`：FRI 风格二元折叠。  
- `interpolate_degree`：用模线性代数估计约束多项式次数。  
- `prove` / `verify`：证明与验证主流程。  
- `main`：执行诚实与篡改两组端到端测试。

## R16

相关算法链路：

- 证明系统：zk-SNARK、PLONK、Halo、Nova；
- 低次测试：FRI 及其变体；
- 承诺：Merkle commitments、Polynomial commitments；
- 虚拟机证明：AIR、R1CS、算术电路。

本条目位于“STARK 思路入门”位置，强调构件关系而非工业参数。

## R17

运行方式：

```bash
cd Algorithms/计算机-密码学-0294-zk-STARKs
uv run python demo.py
```

预期输出特征：

- 显示域参数与配置；
- `Case 1` 的 `verify = True`；
- `Case 2` 的 `verify = False`；
- 最后一行 `All checks passed.`。

依赖：

- `numpy`
- Python 标准库：`dataclasses`、`hashlib`、`typing`

## R18

源码级算法流程拆解（`prove + verify`，9 步）：

1. 用 `fibonacci_trace` 生成执行轨迹 `t`。  
2. 用 `compute_air_constraints` 把“递推正确性 + 边界条件”转成约束向量 `c`。  
3. 分别对 `t` 与 `c` 建 `Merkle tree`，得到 `trace_root` 与 `constraint_root`。  
4. 用 `interpolate_degree`（模 Vandermonde 消元）估计 `c` 的多项式次数。  
5. 从 `c` 开始迭代执行 `fold_layer`，每轮由 `derive_beta` 生成挑战 `beta`。  
6. 每轮用 `derive_queries` 选查询点，并导出三类开口：`a,b`（当前层）与 `c`（下一层）。  
7. 验证端重建根并检查轨迹约束：边界、过渡、约束重算一致。  
8. 验证端逐轮检查 Merkle 路径与折叠关系 `c == a + beta*b (mod p)`。  
9. 验证端确认末轮值、次数界与全零约束逻辑一致后接受证明。

说明：真实 STARK 工程会引入更严格的域扩展、随机线性组合、查询策略和零知识 masking；本 MVP 仅保留可追踪的核心机制。
