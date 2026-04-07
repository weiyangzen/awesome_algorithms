# CUDA并行

- UID: `CS-0304`
- 学科: `计算机`
- 分类: `并行与分布式`
- 源序号: `465`
- 目标目录: `Algorithms/计算机-并行与分布式-0465-CUDA并行`

## R01

本条目实现一个最小可运行的 CUDA 并行 MVP，核心目标是把“同一算法在 CPU 与 GPU 上的执行路径、正确性和性能差异”透明展示出来。

实现范围：
- 使用 `PyTorch` 作为 CUDA kernel 调度入口（避免 C++/CUDA 编译链依赖）；
- 选择两个典型并行任务：向量加法（带宽型）与矩阵乘法（计算密集型）；
- 提供无 CUDA 环境下的自动 CPU 回退，保证 `uv run python demo.py` 可直接运行。

## R02

问题定义（MVP 版本）：
- 输入：
  - 向量规模 `N`；
  - 矩阵维度 `(M, K, N)`；
  - 分块大小 `tile_size`；
  - 基准测试参数（`warmup`, `repeats`）。
- 输出：
  - 向量加法的数值误差、CPU/设备时间、吞吐与加速比；
  - 矩阵乘法（标准 `torch.matmul` 与显式分块版本）的误差与耗时；
  - 汇总结论 `global_checks_pass`。

约束：
- 脚本不读交互输入；
- 如果无 CUDA，脚本仍在 CPU 完成全流程并给出结果。

## R03

并行模型说明：
- CUDA 执行模型可抽象为：网格（grid）-线程块（block）-线程（thread）。
- 在本 MVP 中，`torch` 负责将高层张量操作映射为底层 CUDA kernel：
  - `x + y` 触发逐元素并行 kernel；
  - `A @ B` 触发 GEMM kernel（通常为分块/向量化实现）。
- 为了避免“黑盒即结论”，我们在 `demo.py` 中额外实现了 `tiled_matmul`，显式展示分块矩阵乘法的数据流（每个输出 tile 累加多个 K 方向子块）。

## R04

算法流程（高层）：
1. 检测设备：优先 `cuda`，否则回退 `cpu`。
2. 构造固定随机种子的向量和矩阵输入。
3. 执行向量加法：
   - 设备侧计算；
   - CPU 侧参考计算；
   - 对比误差并计时。
4. 执行标准矩阵乘法 `torch.matmul`，并与 NumPy 结果做误差校验。
5. 执行显式分块矩阵乘法 `tiled_matmul`，验证其与标准结果一致。
6. 汇总基准结果，输出加速比与验收布尔值。

## R05

核心数据结构：
- `BenchmarkConfig(dataclass)`：
  - `vector_size`, `mat_shape`, `tile_size`, `warmup`, `repeats`。
- `BenchStats(dataclass)`：
  - `mean_ms`, `std_ms`, `min_ms`, `max_ms`, `throughput_gops`。
- 张量：
  - 向量 `x, y`：`shape=(N,)`；
  - 矩阵 `A, B, C`：`A(M,K)`, `B(K,N)`, `C(M,N)`；
  - 显式分块输出 `C_tiled`。

## R06

正确性校验要点：
- 向量加法：比较设备结果与 CPU 参考结果，检查 `max_abs_error`。
- 矩阵乘法（标准）：比较 `torch.matmul` 与 `numpy.matmul`。
- 矩阵乘法（分块）：比较 `tiled_matmul` 与 `torch.matmul`。
- 全部误差须在 `float32` 合理范围内（默认阈值 `1e-4`），且输出均为有限值。

## R07

复杂度分析：
- 向量加法：
  - 时间复杂度 `O(N)`；
  - 空间复杂度 `O(N)`（输出向量）。
- 矩阵乘法：
  - 时间复杂度 `O(M*K*N)`；
  - 空间复杂度 `O(M*N)`（不计输入存储）。
- 显式分块矩阵乘法与标准 GEMM 的渐进复杂度一致，但常数因子更高（Python 层循环与更细粒度 kernel 调度开销）。

## R08

并行与内存策略：
- 向量加法属于内存带宽主导任务，适合大规模连续访存。
- 矩阵乘法属于计算密集型任务，重点在分块复用与访存局部性。
- `tiled_matmul` 通过 `(i,j,k)` 三重分块实现：
  - `(i,j)` 决定输出 tile；
  - `k` 方向循环累加部分乘积；
  - 与 CUDA block-tile 思想一致，便于教学理解。
- 计时时对 CUDA 显式 `synchronize`，避免异步执行导致时间统计失真。

## R09

MVP 取舍：
- 不写自定义 CUDA C++ 扩展，避免构建复杂度；
- 不引入多 GPU、流（stream）并发与通信优化；
- 不做自动混合精度与张量核心专项调优；
- 保留最小且可解释的数据路径，重点展示“并行原理 + 可运行验证”。

## R10

`demo.py` 函数职责：
- `pick_config`：按设备选择较稳妥的默认规模；
- `sync_if_needed`：CUDA 场景下同步设备；
- `time_callable`：统一 warmup/repeat 计时；
- `make_vector_data`：生成向量测试输入；
- `make_matrix_data`：生成矩阵测试输入；
- `tiled_matmul`：显式分块矩阵乘法（教学实现）；
- `run_vector_add_benchmark`：向量加法正确性与性能评估；
- `run_matmul_benchmark`：标准/分块矩阵乘法正确性与性能评估；
- `main`：组织执行、打印汇总与验收结论。

## R11

运行方式：

```bash
cd Algorithms/计算机-并行与分布式-0465-CUDA并行
uv run python demo.py
```

脚本无交互输入，直接输出设备信息、各项基准与最终检查结果。

## R12

输出字段解读：
- `device`：当前计算设备（`cuda` 或 `cpu`）。
- `vector_max_abs_error`：向量加法设备结果与 CPU 结果的最大绝对误差。
- `vector_device_mean_ms / vector_cpu_mean_ms`：向量加法平均耗时（毫秒）。
- `vector_speedup_vs_cpu`：向量加法加速比（CPU 均值 / 设备均值）。
- `matmul_max_abs_error_vs_numpy`：标准矩阵乘法对 NumPy 的误差。
- `tiled_max_abs_error_vs_torch`：分块矩阵乘法对标准 Torch 结果的误差。
- `matmul_device_mean_ms`：标准矩阵乘法设备耗时。
- `tiled_device_mean_ms`：分块矩阵乘法设备耗时。
- `global_checks_pass`：全局验收布尔值。

## R13

建议最小测试集：
- 基础运行：默认参数下脚本完成并输出 `global_checks_pass`。
- 设备回退：在无 CUDA 环境运行，确认流程完整且无异常。
- 误差阈值：将 `float32` 改为 `float64` 观察误差进一步降低。
- 分块敏感性：调整 `tile_size`（如 32/64/128）观察耗时变化。

## R14

关键可调参数：
- `vector_size`：向量长度，影响带宽压力；
- `mat_shape=(M,K,N)`：矩阵尺寸，影响 FLOPs；
- `tile_size`：显式分块粒度；
- `warmup`：预热次数，降低首轮波动；
- `repeats`：计时重复次数；
- `dtype`：默认 `float32`，可改 `float64` 做精度验证。

## R15

方法对比（简述）：
- 标准 `torch.matmul`：
  - 优点：通常调用高度优化的底层 GEMM，性能最好；
  - 缺点：内部细节不直接可见。
- 显式 `tiled_matmul`：
  - 优点：算法结构透明，便于理解分块累加；
  - 缺点：Python 循环开销大，性能明显低于优化库。
- 纯 NumPy CPU：
  - 优点：依赖少、可移植；
  - 缺点：无法利用 CUDA 并行能力。

## R16

应用场景：
- 深度学习训练/推理中的大规模张量运算；
- 科学计算中的线性代数核心算子加速；
- 并行计算课程中讲解 GPU 与 CPU 执行差异；
- 算法原型阶段验证“可并行化收益是否明显”。

## R17

后续扩展方向：
1. 增加自定义 CUDA kernel（`torch.utils.cpp_extension` 或 Triton）；
2. 加入 pinned memory 与异步数据传输（H2D/D2H）评估；
3. 支持混合精度（FP16/BF16）与张量核心路径；
4. 扩展到多 GPU（`DistributedDataParallel` / NCCL）；
5. 输出 CSV 基准日志，便于跨设备对比与回归跟踪。

## R18

`demo.py` 源码级算法流（8 步）：
1. `main` 调用 `pick_config` 与 `torch.cuda.is_available()` 选择运行设备和默认规模。  
2. `run_vector_add_benchmark` 生成向量 `x,y`，先在设备侧执行 `z=x+y`，再在 CPU 侧执行参考计算。  
3. 使用 `max_abs_error` 检查向量结果一致性，并通过 `time_callable` 在 warmup 后重复计时。  
4. `run_matmul_benchmark` 生成矩阵 `A,B`，先运行标准 `C_ref = torch.matmul(A,B)`，并与 `numpy.matmul` 做误差对比。  
5. 调用 `tiled_matmul(A,B,tile_size)`，按 `(i,j)` 输出块与 `k` 方向块累加构造 `C_tiled`。  
6. 将 `C_tiled` 与 `C_ref` 比较，得到分块实现误差，确认算法等价。  
7. 对标准矩阵乘法与分块矩阵乘法分别计时，统计均值/方差/吞吐指标。  
8. `main` 汇总误差阈值与有限值检查，打印 `global_checks_pass` 作为最终验收信号。  
