# DFA最小化

- UID: `CS-0283`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `443`
- 目标目录: `Algorithms/计算机-编译原理-0443-DFA最小化`

## R01

DFA 最小化（Deterministic Finite Automaton Minimization）是在保持语言不变的前提下，找到与原 DFA 等价且状态数最少的 DFA。它是词法分析器生成中的基础步骤：先由正则表达式构造自动机，再通过最小化减少状态数量，从而降低扫描器表大小和分支成本。

本目录 MVP 使用两阶段方法：
- 阶段 1：从初态出发做可达性分析，删除不可达状态；
- 阶段 2：在可达子图上使用 Hopcroft 分割细化算法，合并 Myhill-Nerode 等价状态。

## R02

问题定义（本实现）：
- 输入：一个完整 DFA `M=(Q, Σ, δ, q0, F)`，其中 `δ` 对每个 `(q, a)` 都定义。
- 输出：
  - 最小 DFA `M_min`；
  - 原状态到新状态的映射 `old -> new`；
  - 最终等价类分割（partition blocks）。

正确性目标：
- 语言保持不变：`L(M) = L(M_min)`；
- 状态数最小（在 DFA 等价类意义下唯一到同构）。

## R03

基本理论：
- 两状态 `p,q` 等价，当且仅当对任意字符串 `w`，从 `p,q` 出发读入 `w` 的接受性一致；
- 最小 DFA 的状态就是等价关系的商集（equivalence classes）；
- 接受态与非接受态必不等价，因此可作为初始粗分割；
- Hopcroft 算法通过“前驱集合切分”持续细化分割，直到稳定。

## R04

算法策略（实现级）：
1. `validate_dfa` 检查 DFA 完整性（起始态、终态子集、迁移函数闭包与完备）。
2. `reachable_states` 从 `q0` 做 BFS 得到可达集合。
3. `prune_unreachable` 构造去除不可达状态的新 DFA。
4. `hopcroft_minimize`：
   - 初始分割 `P = {F, Q\F}`（空集不加入）；
   - 工作集 `W` 初始化为较小块；
   - 反复用 `X = Pred(A, a)` 切分每个块 `Y` 为 `Y∩X` 与 `Y\X`；
   - 无可切分后收敛。
5. 以每个分割块生成一个新状态并重建转移。

## R05

核心数据结构：
- `DFA`（`@dataclass(frozen=True)`）
  - `states: FrozenSet[str]`
  - `alphabet: FrozenSet[str]`
  - `transition: Dict[(state,symbol), state]`
  - `start: str`
  - `finals: FrozenSet[str]`
- 分割结构：`List[FrozenSet[str]]`。
- 映射：`Dict[str, str]`（原状态 -> 最小化后状态名）。

## R06

正确性要点：
- 先删不可达状态，否则会引入“语言无关”状态影响最小化结果判读。
- 细化过程只分裂、不会合并跨块状态，保证不丢失已发现可区分性。
- 当分割稳定时，每个块内状态行为不可再区分，即得到等价类。
- 重建最小 DFA 时，每块取任意代表状态转移都一致（否则该块早应被继续分裂）。

## R07

复杂度分析：
- 设状态数 `n=|Q|`，字母表大小 `k=|Σ|`。
- 可达性 BFS：`O(n*k)`。
- Hopcroft 细化：经典上界 `O(k * n * log n)`。
- 构建最小 DFA：`O(n*k)`。
- 总体时间：`O(k*n*log n)`（主导项）；空间：`O(n*k)`（转移表 + 分割 + 辅助集合）。

## R08

边界与异常处理：
- 空字母表：抛 `ValueError`。
- 缺失转移或转移越界到未知状态：抛 `ValueError`。
- 终态不在状态集合中：抛 `ValueError`。
- 比较两个 DFA 语言一致性时若字母表不同：抛 `ValueError`。

## R09

MVP 取舍：
- 采用 Python 标准库实现，不依赖第三方自动机库，保证算法流程可追踪。
- 输入示例使用内置 DFA，不做命令行参数化和文件解析。
- 语言等价验证采用“有界穷举串”回归测试（`len<=8`），作为工程上快速自检；
  完全等价证明仍以理论上的最小化构造为准。

## R10

`demo.py` 主要函数职责：
- `validate_dfa`：完整性与合法性检查。
- `reachable_states` / `prune_unreachable`：可达剪枝。
- `_predecessors`：计算某块在某字符下的前驱集合。
- `hopcroft_minimize`：执行分割细化并构建最小 DFA。
- `accepts`：运行 DFA 判断给定串是否接受。
- `equivalent_on_bounded_words`：做 bounded regression check。
- `build_demo_dfa`：构造一个包含重复状态和不可达状态的非最小 DFA。

## R11

运行方式：

```bash
cd Algorithms/计算机-编译原理-0443-DFA最小化
uv run python demo.py
```

脚本无交互输入，直接打印原 DFA、分割结果、最小 DFA 与一致性检查结果。

## R12

输出说明：
- `Original DFA`：原状态机全部状态和转移表。
- `reachable states` / `pruned ... unreachable`：可达性剪枝统计。
- `final partition blocks`：最小化结束时的等价类。
- `old -> minimized state mapping`：原状态映射到新状态。
- `Minimized DFA`：最终最小状态机。
- `sample acceptance checks`：示例字符串接受结果。
- `bounded equivalence test`：长度 `<=8` 的全部二进制串回归一致性结果。

## R13

建议验证项：
- 可达性：示例中的 `U`、`UA` 应被剪枝。
- 状态数：原 8 状态经最小化应降为 3 状态。
- 语言保持：对示例串 `"01"`、`"1001"` 应接受；`"1"`、`"11111"` 应拒绝。
- 回归一致：`equivalent_on_bounded_words(..., max_len=8)` 返回 `True`。

## R14

关键参数与调节：
- `equivalent_on_bounded_words` 的 `max_len`：
  - 值越大，回归检查覆盖越强；
  - 代价约 `O(|Σ|^max_len)`，仅用于测试而非算法主体。
- 状态命名策略：当前按分割块排序生成 `Q0,Q1,...`，便于输出稳定对比。

## R15

方法对比：
- 表填充法（table-filling）：概念直观，实现简单，通常 `O(n^2*k)`。
- Hopcroft：实现复杂一些，但理论复杂度更优 `O(k*n*log n)`，适合状态较多情形。
- Brzozowski（两次反转+确定化）：实现优雅但在某些输入上可能指数膨胀。

本项目选择 Hopcroft，兼顾工程可用性与复杂度上界。

## R16

典型应用场景：
- 编译器/解释器词法分析器（正则 -> DFA -> 最小 DFA）。
- 网络入侵规则匹配中的自动机压缩。
- 形式验证与模型检查中的状态机规约。
- 嵌入式协议解析器的状态表缩减。

## R17

可扩展方向：
- 支持从正则表达式直接构建 DFA 并串联最小化流水线。
- 支持从文件（JSON/YAML）读写自动机定义。
- 增加 DOT/Graphviz 可视化导出。
- 增加随机 DFA fuzz 测试，自动验证最小化前后语言一致。
- 支持 Mealy/Moore 机的等价最小化变体。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `build_demo_dfa` 构造 8 状态 DFA：其中 `A-F` 可达且含重复等价结构，`U/UA` 不可达。  
2. `validate_dfa` 检查输入自动机完整性，确保每个 `(state, symbol)` 都有定义且目标合法。  
3. `prune_unreachable` 调用 `reachable_states` 做 BFS，只保留从初态可达的子图。  
4. `hopcroft_minimize` 初始化分割 `P={F, Q\F}` 和工作集 `W`（优先较小块）。  
5. 循环取出 `A in W`，对每个字符 `a` 计算 `X=Pred(A,a)`，并将每个块 `Y` 按 `Y∩X`/`Y\X` 进行细化。  
6. 分割稳定后，为每个等价类分配新状态名 `Q0,Q1,...`，得到原状态到新状态映射。  
7. 以每个等价类代表状态的转移重建最小 DFA（起始态与终态映射同步更新）。  
8. `main` 打印最小化结果，并通过 `equivalent_on_bounded_words` 对 `len<=8` 全部串做回归一致性验证。  
