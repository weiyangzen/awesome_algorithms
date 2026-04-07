# 赝势方法 (Pseudopotential Method)

- UID: `PHYS-0438`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `459`
- 目标目录: `Algorithms/物理-计算物理-0459-赝势方法_(Pseudopotential_Method)`

## R01

赝势方法用于把“价电子在周期晶格中的量子运动”转换成可计算问题。核心动机是避免显式处理原子核附近剧烈振荡的全电子波函数，把核心电子与强库仑势的细节吸收到一个更平滑的有效势中，从而以较小基组得到能带、带隙与态密度等量。

## R02

本目录采用最小 MVP 版本: `1D + 平面波基 + 局域赝势`。在倒空间中，哈密顿量写成稠密矩阵并逐个 `k` 点对角化。相较直接实空间离散，平面波-赝势在周期体系上实现简单、数值行为稳定，且能直接展示“区边界开隙”这一物理现象。

## R03

数学模型:

- 倒格矢: `G_n = 2πn/a`, `n ∈ [-N_G, N_G]`
- 哈密顿量矩阵元:
  `H_{G,G'}(k) = 0.5 * (k+G)^2 * δ_{G,G'} + V_{G-G'}`
- 本征方程:
  `H(k) c_{n,k} = E_n(k) c_{n,k}`
- 局域赝势形式因子:
  `V_q = -V0 * exp[-0.5 * (qσ)^2]`

其中 `a` 为晶格常数，`V0` 决定势强度，`σ` 控制势在倒空间衰减速度。

## R04

输入:

- 物理参数: `a, V0, σ`
- 截断参数: `g_cut`（平面波个数 `2*g_cut+1`）
- 采样参数: `n_kpoints, n_bands`

输出:

- 每个 `k` 点的前 `n_bands` 条能量本征值
- `bands.csv`（列为 `k`, `k_over_pi`, `band_1...`）
- 终端摘要（Γ 点能量、区边界带隙、自由电子对照）

## R05

MVP 假设与简化:

- 单电子近似，不含自洽电荷密度循环（非 DFT-SCF）
- 一维周期体系，用于教学和算法演示
- 局域、各向同性、解析可写的赝势形式因子
- 去除 `q=0` 平均势项，只保留能带形状与带隙信息

## R06

算法流程:

1. 构造倒格矢集合 `G`
2. 在第一布里渊区生成 `k` 网格
3. 对每个 `k` 构造 `H(k)` 的动能项与势能项
4. 调用对称本征求解器得到升序本征值
5. 截取前 `n_bands` 存入能带数组
6. 计算区边界第一带隙，并和 `V0=0` 结果对比
7. 导出 `bands.csv` 供外部绘图或验证

## R07

复杂度分析（`M = 2*g_cut+1`, `K = n_kpoints`）:

- 构造单个哈密顿量: `O(M^2)`
- 单个 `k` 点对角化（稠密实对称）: `O(M^3)`
- 总时间复杂度: `O(K * M^3)`
- 内存复杂度: `O(M^2)`（单个哈密顿量）与 `O(K * n_bands)`（结果）

## R08

数值稳定性与可解释性:

- `H` 显式对称化 `0.5*(H+H.T)`，抑制浮点非对称扰动
- 使用 `scipy.linalg.eigh`（针对 Hermitian/Symmetric），比一般 `eig` 更稳定
- 平滑高斯形式因子使高频耦合快速衰减，降低截断误差
- 通过 `np.diff(bands, axis=1)` 检查各带升序，作为运行期自检

## R09

伪代码:

```text
input a, g_cut, V0, sigma, n_kpoints, n_bands
G <- reciprocal_vectors(g_cut, a)
K <- linspace(-pi/a, pi/a, n_kpoints)
for k in K:
    kinetic[i] = 0.5 * (k + G[i])^2
    potential[i,j] = -V0 * exp(-0.5 * ((G[i]-G[j]) * sigma)^2)
    set potential diagonal to 0 (remove average term)
    H = diag(kinetic) + potential
    E = eigh(H)
    bands[k,:] = first n_bands of E
report gamma energies and boundary gap
save bands.csv
```

## R10

`demo.py` 的实现要点:

- `ModelConfig`: 集中管理物理参数与离散参数
- `build_hamiltonian`: 向量化构造 `G-G'` 差分矩阵，避免双重 Python 循环
- `solve_bands`: 逐 `k` 点求解，输出二维 `bands[k_index, band_index]`
- `bands_to_dataframe`: 用 `pandas` 生成可持久化表格
- `main`: 增加 `V0=0` 基线对照，证明赝势导致开隙

## R11

运行方式（仓库根目录）:

```bash
uv run python Algorithms/物理-计算物理-0459-赝势方法_(Pseudopotential_Method)/demo.py
```

预期行为:

- 程序无交互输入
- 打印参数、Γ 点能量、两种带隙
- 在当前算法目录生成 `bands.csv`

## R12

结果解读建议:

- 当 `V0=0` 时，区边界第一/第二带近简并，带隙接近 0
- 当 `V0>0` 时，简并被打破，区边界出现有限带隙
- `sigma` 越小（倒空间衰减越慢），远程耦合更强，色散会更明显

## R13

正确性检查清单:

- 维度检查: `bands.shape == (n_kpoints, n_bands)`
- 排序检查: 每个 `k` 上 `E1 <= E2 <= ...`
- 物理检查: `gap(V0>0) > gap(V0=0)`（通常成立）
- 收敛检查: 增大 `g_cut` 后低能带变化应减小

## R14

调参建议:

- `g_cut`: 先从 `4~6` 起步，确认趋势后加大至 `8~12`
- `n_kpoints`: 画图时建议 `>= 100`，仅验证可用 `40~60`
- `V0`: 太小不易开隙，太大可能需要更大基组保证收敛
- `sigma`: 决定势在倒空间的“带宽”，影响耦合范围

## R15

异常与边界处理:

- 若 `n_bands > 2*g_cut+1`，应降低 `n_bands` 或提高 `g_cut`
- 若出现非升序本征值，代码会抛出 `RuntimeError`
- 若带隙异常震荡，优先检查 `g_cut` 是否过小

## R16

可扩展方向:

- 从 1D 扩展到 2D/3D 晶格与真实倒格矢网格
- 用非局域赝势（角动量道）替代当前局域高斯模型
- 加入 SCF 循环，升级到 Kohn-Sham DFT 风格流程
- 加入总能量、力与结构优化模块

## R17

当前 MVP 限制:

- 仅教学级模型，不对应具体材料参数拟合
- 未包含自旋轨道耦合、交换关联泛函与温度效应
- 稠密对角化在大基组下扩展性有限

## R18

`scipy.linalg.eigh` 在本任务中的“非黑盒”算法流（对应单个 `k` 点）:

1. Python 层把 `H(k)` 作为实对称矩阵传给 SciPy 线性代数封装。
2. SciPy 检查矩阵形状、数据类型与存储布局，然后路由到 LAPACK 对称本征求解例程。
3. LAPACK 先用 Householder 变换把稠密对称矩阵化为三对角矩阵 `T`（保持本征值不变）。
4. 在 `T` 上执行分治/QR 类对角化流程，求出三对角问题的本征值（及需要时的本征向量）。
5. 若请求本征向量，再把三对角空间的向量反变换回原始基底。
6. LAPACK 返回按升序排列的本征值数组给 SciPy。
7. SciPy 把结果转为 `numpy.ndarray`，`demo.py` 截取前 `n_bands`。
8. 外层 `k` 循环重复上述过程，最终拼成离散能带 `E_n(k)`。

这 8 步中第 3-5 步是经典对称本征值分解核心，解释了为什么该问题可稳定地由高性能线性代数库求解。
