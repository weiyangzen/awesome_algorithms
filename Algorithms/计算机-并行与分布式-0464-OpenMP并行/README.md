# OpenMP并行

- UID: `CS-0303`
- 学科: `计算机`
- 分类: `并行与分布式`
- 源序号: `464`
- 目标目录: `Algorithms/计算机-并行与分布式-0464-OpenMP并行`

## R01

OpenMP（Open Multi-Processing）是共享内存并行编程标准，典型写法是：
- `#pragma omp parallel for`：把循环迭代分发给多个线程；
- `schedule(static|dynamic|guided)`：控制任务分配策略；
- `reduction(+:sum)`：并行累加后自动做归约。

本条目用 Python 做一个“语义可运行 MVP”：不依赖 C/C++ 编译链，但完整复现 OpenMP 的核心思想（并行 for + 调度 + 归约 + 校验）。

## R02

问题定义（MVP 范围）：
- 输入：
1. 连续浮点向量 `data`（长度 `N=1_200_000`）；
2. 并行 worker 数 `workers`（由 CPU 核数推导，上限 8）；
3. 动态调度块大小 `dynamic_chunk_size=40_000`；
4. 标量核函数 `f(x)=sin^2(x)+cos^2(x)+sqrt(x)`。
- 输出：
1. 串行求和结果；
2. `schedule(static)` 并行求和结果；
3. `schedule(dynamic)` 并行求和结果；
4. 三者相对误差与耗时、相对串行速度比；
5. 最终断言结论。

## R03

OpenMP 语义在本实现中的映射：
- `parallel for` -> `ProcessPoolExecutor` 并行处理多个 chunk；
- `schedule(static)` -> 启动时用 `np.array_split` 固定分块；
- `schedule(dynamic)` -> 运行时按小块队列拉取任务；
- `reduction(+:sum)` -> 每个 worker 先做局部和，再在主进程合并。

这让读者可以直接观察调度策略差异，而不是只调用一个黑箱 API。

## R04

`demo.py` 高层流程：
1. 生成固定规模输入向量；
2. 用恒等式 `sin^2+cos^2=1` 计算参考值 `expected`；
3. 串行循环执行核函数并求和；
4. 构建静态分块并行执行，合并局部和；
5. 构建动态分块并行执行，合并局部和；
6. 计算每种模式相对误差与速度比；
7. 做误差阈值断言；
8. 打印表格结果与通过结论。

## R05

关键数据结构：
- `RunStat(dataclass)`：记录 `mode / total / elapsed_sec / rel_err_vs_expected`；
- `np.ndarray`：输入向量与 chunk 切片；
- `list[np.ndarray]`：待分发任务列表（静态或动态）；
- `list[float]`：并行 partial sums。

## R06

正确性依据：
- 核函数是纯函数（无共享可变状态），每个 chunk 独立可并行；
- 并行版本与串行版本使用同一 `kernel`，只改变执行顺序；
- `reduction` 只在最后一层做加法聚合，语义等价于串行求和；
- 通过 `expected` 做相对误差校验，可快速识别调度或归约错误。

## R07

复杂度分析（设数据长度为 `N`，worker 数为 `P`）：
- 串行：时间 `O(N)`，空间 `O(1)`（不计输入）；
- 并行（理想）：计算部分约 `O(N/P)`，总工作量仍 `O(N)`；
- 调度与通信开销：
1. `static` 任务数约为 `P`，调度开销较小；
2. `dynamic` 任务数约为 `N/chunk_size`，负载均衡更好但调度开销更高。

## R08

边界与异常处理：
- `make_dynamic_chunks` 对 `chunk_size<=0` 直接抛 `ValueError`；
- worker 数最低为 1，保证单核环境可运行；
- 误差超过阈值 `2e-12` 触发 `AssertionError`；
- 无交互输入，脚本默认参数下可直接执行。

## R09

MVP 取舍：
- 选择 Python 进程池模拟 OpenMP 语义，而不是直接编译 C/OpenMP；
- 保留最小核心：调度 + 归约 + 验证，不扩展到 NUMA/线程绑定；
- 使用显式 `kernel` 循环，避免“第三方函数一行求和看不见细节”；
- 强调“可解释且可跑通”优先于极限性能。

## R10

`demo.py` 函数职责：
- `kernel`：定义单点计算逻辑；
- `reduce_chunk`：worker 内局部归约；
- `serial_reduction`：串行基线；
- `make_static_chunks`：静态调度切块；
- `make_dynamic_chunks`：动态调度切块；
- `parallel_reduction`：并行执行 + partial sums 合并；
- `timed_call`：统一计时包装；
- `relative_error`：相对误差计算；
- `print_result_table`：结果打印；
- `main`：组织实验、断言、输出。

## R11

运行方式：

```bash
cd Algorithms/计算机-并行与分布式-0464-OpenMP并行
uv run python demo.py
```

脚本无交互输入，运行后输出三种模式对比表。

## R12

输出说明：
- `mode`：执行模式（`serial` / `parallel-static` / `parallel-dynamic`）；
- `sum`：归约后的总和；
- `seconds`：该模式总耗时；
- `speedup`：相对串行加速比（`serial_time / mode_time`）；
- `rel_err`：对参考值 `expected` 的相对误差；
- `All checks passed for CS-0303`：所有断言通过。

## R13

最小验证清单：
- 三种模式都能正常运行到结束；
- 三种模式 `rel_err <= 2e-12`；
- `parallel-static` 与 `parallel-dynamic` 结果与串行一致（浮点误差范围内）；
- 输出中包含 OpenMP 映射提示行与最终通过结论。

## R14

默认参数与意义：
- `N=1_200_000`：保证任务量足够观察并行开销；
- `workers=min(8, cpu_count)`：避免过多进程导致额外争抢；
- `dynamic_chunk_size=40_000`：在负载均衡与调度开销之间折中；
- `tolerance=2e-12`：用于浮点归约结果一致性验收。

## R15

与相关模型对比：
- 相比 MPI：OpenMP 假设共享内存，通信隐式；MPI 是分布式显式消息传递；
- 相比 CUDA：OpenMP 多用于 CPU 共享内存并行；CUDA 主要面向 GPU kernel；
- 相比 Python `ThreadPoolExecutor`：本实现用进程池可绕过 GIL，更接近 CPU 并行计算场景。

## R16

适用场景：
- 并行计算教学中理解 `parallel for + schedule + reduction`；
- 在无 C/OpenMP 编译环境下做并行语义验证；
- 快速评估静态调度与动态调度的行为差异。

不适用场景：
- 生产级 HPC 性能评测；
- 需要线程绑定、缓存亲和、NUMA 优化的精细调优；
- 需要真实 OpenMP runtime 指标（如 `OMP_PROC_BIND` 级别分析）。

## R17

可扩展方向：
1. 增加 `guided` 调度策略模拟并与 `static/dynamic` 对比；
2. 支持更多 reduction 操作（`min/max/product`）；
3. 引入不均匀计算负载，观察动态调度优势；
4. 增加 CSV 输出与多轮重复统计（均值/方差）；
5. 对接真实 C/OpenMP 实现做结果与性能交叉验证。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `main` 生成输入向量 `data`，确定 `workers` 与 `dynamic_chunk_size`。  
2. 通过恒等式 `sin^2+cos^2=1` 计算参考值 `expected`，作为精度锚点。  
3. `serial_reduction` 在单进程中逐元素执行 `kernel`，得到串行基线。  
4. `make_static_chunks` 把数据按 worker 数固定切块，模拟 `schedule(static)`。  
5. `parallel_reduction` 把各块分配到进程池并运行 `reduce_chunk`，每块先算局部和；主进程再 `sum(partials)` 完成 `reduction(+:sum)`。  
6. `make_dynamic_chunks` 生成大量小块任务，模拟 `schedule(dynamic)` 的“运行时取任务”行为。  
7. 再次调用 `parallel_reduction` 得到动态调度总和，并计算各模式耗时与加速比。  
8. 逐项检查 `rel_err <= 2e-12`；全部通过后打印结果表和 `All checks passed for CS-0303 (OpenMP并行).`。
