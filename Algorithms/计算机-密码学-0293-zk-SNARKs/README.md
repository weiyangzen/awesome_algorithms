# zk-SNARKs

- UID: `CS-0149`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `293`
- 目标目录: `Algorithms/计算机-密码学-0293-zk-SNARKs`

## R01

`zk-SNARK`（Zero-Knowledge Succinct Non-interactive Argument of Knowledge）是一类证明系统：
- `Zero-Knowledge`：验证者应当学不到见证（witness）本身；
- `Succinct`：证明与验证都很短小，验证复杂度通常远低于重算原问题；
- `Non-interactive`：证明者一次性给出证明，无需多轮交互；
- `Argument of Knowledge`：在特定密码学假设下，伪造有效证明应当很难。

本目录给出一个可运行的教学 MVP：用 `Groth16` 风格的代数结构，证明“知道私密 `w` 使得 `x * w = y`（模素数域）”。

## R02

本条目要解决的问题：
- 公共输入：`x, y`；
- 私有见证：`w`；
- 关系：`x * w = y (mod p)`。

目标是构建 `setup -> prove -> verify` 的最小闭环，展示 zk-SNARK 的数据流，而不是实现工业级密码库。

## R03

MVP 的核心思想：
- 用单约束电路表示关系：`(x) * (w) = (y)`；
- 采用 Groth16 常见的参数符号（`alpha, beta, gamma, delta`）；
- 证明输出三元组 `(A, B, C)`；
- 验证通过一个“pairing 方程”完成。

为了保持代码短小，`demo.py` 把群元素用有限域指数表示、把 pairing 用乘法模拟，这样能直观看到代数关系如何闭合。

## R04

电路与变量约定：
- 完整赋值向量 `z = [1, x, y, w]`；
- 线性形式系数：
  - `u = [0, 1, 0, 0]`（取出 `x`）
  - `v = [0, 0, 0, 1]`（取出 `w`）
  - `w = [0, 0, 1, 0]`（取出 `y`）
- 约束：`<u, z> * <v, z> = <w, z>`。

该约束成立当且仅当 `x * w = y`。

## R05

高层伪代码：

```text
Setup:
  采样 alpha, beta, gamma, delta, tau
  构造 public IC 和 private K 查询参数

Prove(x, y, witness_w):
  组装 z=[1,x,y,w]
  计算 a_eval=<u,z>, b_eval=<v,z>
  采样随机数 r,s
  A = alpha + a_eval + r*delta
  B = beta + b_eval + s*delta
  C = private_term + A*s + B*r - r*s*delta
  输出 proof=(A,B,C)

Verify(x, y, proof):
  计算 vk_x = IC0 + x*IC1 + y*IC2
  检查 pairing(A,B) == pairing(alpha,beta)+pairing(vk_x,gamma)+pairing(C,delta)
```

## R06

时间复杂度（本 MVP）：
- `setup`：常数规模参数计算，`O(1)`；
- `prove`：固定长度向量点积与常数次域运算，`O(1)`；
- `verify`：常数次域乘加，`O(1)`。

因为电路只有 1 条约束、4 个变量，复杂度是常数量级；真实 zk-SNARK 会随约束数增长。

## R07

空间复杂度：
- 参数与证明都是常数大小；
- 无外部数据集、无中间大矩阵。

因此整体空间复杂度 `O(1)`（相对于当前固定规模电路）。

## R08

`demo.py` 关键参数：
- `prime=2147483647`：有限域模数；
- `setup seed=2026`：固定 trusted setup 随机性，保证可复现；
- `prove seed=99`：固定证明随机掩码 `r,s`；
- 示例语句：`x=13, w=21, y=x*w mod p`。

这些参数都是教学用默认值，可在代码内直接修改观察行为。

## R09

实现边界与假设：
- 使用了“指数空间 + 乘法 pairing”的教学替身，不是安全群实现；
- 没有曲线、pairing 库、子群检查、序列化安全等工程细节；
- 只覆盖一个非常小的关系约束；
- 目标是帮助理解 Groth16 风格公式如何从电路数据流到验证方程。

## R10

运行方式（无交互）：

```bash
cd Algorithms/计算机-密码学-0293-zk-SNARKs
uv run python demo.py
```

程序会自动执行：
- 正确见证证明；
- 错误公开输入验证；
- 篡改证明验证；
- 错误见证生成证明后验证。

## R11

输出解读：
- `valid proof: True`：正确语句+正确见证通过；
- `wrong public y: False`：同一证明不能用于错误公开语句；
- `tampered proof element C: False`：证明被改动后应失败；
- `wrong witness: False`：不满足关系的见证不能通过。

若这些布尔值模式被破坏，说明代数公式或实现有 bug。

## R12

正确性快速检查清单：
1. `y = x*w mod p` 时，`verify(...)` 为 `True`。  
2. 仅改 `y`，验证应变 `False`。  
3. 仅改 `proof.C`，验证应变 `False`。  
4. 用错误 `w` 重新出证明，验证应为 `False`。  
5. 多次运行（固定 seed）输出应一致。

## R13

常见失败模式：
- 模运算遗漏（整数溢出语义错误）；
- 逆元对 `0` 求逆导致异常；
- 公共输入索引顺序错位（`[1,x,y]` 与代码不一致）；
- 把 `A,B,C` 公式里的随机项符号写错；
- 把教学替身当作生产安全实现。

## R14

与相近证明系统对比（简述）：
- 相比 `Schnorr`：Schnorr 更像离散对数知识证明，表达一般算术电路能力较弱；
- 相比 `zk-STARK`：STARK 通常透明设置（无 trusted setup）但证明更大；
- 相比通用 `SNARK` 框架：本 MVP 只演示最小关系，不覆盖大规模电路与完整安全属性。

## R15

工程实践建议：
- 学习路径先从“单约束、可手算”开始，再扩展到多约束 R1CS/QAP；
- 把随机源、域参数、序列化格式集中管理，减少实现错误面；
- 对证明系统做属性测试（正确性/鲁棒性/边界条件）；
- 生产场景应直接采用审计过的密码库而非手写实现。

## R16

局限性与可扩展方向：
- 当前仅是教学代数模型，不具备生产安全性；
- 可扩展到多约束电路（例如多个乘法门与加法门）；
- 可接入真实椭圆曲线与 pairing 库，替换 toy pairing；
- 可加入 witness 生成器与约束编译器，形成更完整 pipeline。

## R17

MVP 依赖与自包含性：
- 依赖：Python 标准库（`dataclasses`, `random`）；
- 输入：无命令行参数、无外部文件、无交互输入；
- 输出：标准输出打印验证结果布尔值；
- 目录内文件：`README.md` + `demo.py` + `meta.json` 即可复现。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `build_multiplication_circuit()` 定义约束系数 `u,v,w` 与变量布局 `z=[1,x,y,w]`。  
2. `trusted_setup()` 采样 `tau,alpha,beta,gamma,delta`，并计算 public `IC` 与 private `K` 查询参数。  
3. `prove()` 用 `make_assignment()` 组装完整赋值，计算 `a_eval=<u,z>`、`b_eval=<v,z>`。  
4. `prove()` 采样随机掩码 `r,s`，构造 `A=alpha+a_eval+r*delta` 与 `B=beta+b_eval+s*delta`。  
5. `prove()` 计算 `private_term=<K_private,z>`，再构造 `C=private_term + A*s + B*r - r*s*delta`，得到证明 `(A,B,C)`。  
6. `verify()` 按公开输入计算 `vk_x = IC0 + x*IC1 + y*IC2`。  
7. `verify()` 计算左侧 `pairing(A,B)` 与右侧 `pairing(alpha,beta)+pairing(vk_x,gamma)+pairing(C,delta)`。  
8. `main()` 运行四组测试（正确证明、错误 public、篡改 proof、错误 witness），输出布尔结果验证流程完整性。

这 8 步对应了一个“极小 Groth16 风格”证明闭环，并显式展示了 setup/prove/verify 的代数连通关系。
