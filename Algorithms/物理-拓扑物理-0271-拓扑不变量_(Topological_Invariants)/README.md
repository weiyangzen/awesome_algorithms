# 拓扑不变量 (Topological Invariants)

- UID: `PHYS-0268`
- 学科: `物理`
- 分类: `拓扑物理`
- 源序号: `271`
- 目标目录: `Algorithms/物理-拓扑物理-0271-拓扑不变量_(Topological_Invariants)`

## R01

本条目实现一个最小可运行的拓扑不变量计算任务：在二维两能带晶格模型中计算第一 Chern 数（整数拓扑不变量）。

MVP 目标：
- 给定模型参数 `m` 与布里渊区离散网格 `N x N`。
- 数值计算占据带的 Berry 曲率总通量。
- 输出接近整数的 `chern_raw` 与取整后的 `chern_integer`。

## R02

采用 Qi-Wu-Zhang (QWZ) 两能带哈密顿量：

\[
H(\mathbf{k}) = \sin k_x\,\sigma_x + \sin k_y\,\sigma_y + (m + \cos k_x + \cos k_y)\,\sigma_z.
\]

对占据带本征矢 \(|u(\mathbf{k})\rangle\)，离散 FHS（Fukui-Hatsugai-Suzuki）链变量定义为：

\[
U_x(\mathbf{k}) = \frac{\langle u(\mathbf{k})|u(\mathbf{k}+\hat{x})\rangle}{|\langle u(\mathbf{k})|u(\mathbf{k}+\hat{x})\rangle|},
\quad
U_y(\mathbf{k}) = \frac{\langle u(\mathbf{k})|u(\mathbf{k}+\hat{y})\rangle}{|\langle u(\mathbf{k})|u(\mathbf{k}+\hat{y})\rangle|}.
\]

每个小方格的离散 Berry 曲率（相位）为：

\[
F_{xy}(\mathbf{k}) = \arg\!\left[U_x(\mathbf{k})U_y(\mathbf{k}+\hat{x})U_x(\mathbf{k}+\hat{y})^{-1}U_y(\mathbf{k})^{-1}\right].
\]

Chern 数估计：

\[
C = \frac{1}{2\pi}\sum_{\mathbf{k}} F_{xy}(\mathbf{k}).
\]

## R03

输入：
- `m: float`，模型质量项。
- `grid_size: int`，布里渊区离散采样数（每个方向）。

输出：
- `chern_raw: float`，离散求和结果。
- `chern_integer: float`（实质为整数值），`chern_raw` 的四舍五入。
- 扫描模式下返回 `pandas.DataFrame`，列为 `m / chern_integer / chern_raw`。

## R04

核心思想：
- 不直接对 Berry 连接做数值微分（那样相位规选择敏感）。
- 使用 FHS 链变量构造规不变（gauge-invariant）回路相位。
- 每个 plaquette 的相位累加天然接近 `2π` 的整数倍，最终给出稳定的整数拓扑数。

## R05

算法流程（高层）：
1. 在 `k_x, k_y in [0, 2π)` 上建立周期网格。
2. 对每个网格点对角化 `H(k)`，取低能占据带本征矢。
3. 计算 `x/y` 方向单位化重叠链变量 `U_x, U_y`。
4. 对每个小方格计算闭合相位 `F_xy`。
5. 累加全布里渊区 `F_xy`，除以 `2π` 得 `chern_raw`。
6. 取整得到 `chern_integer`。

## R06

正确性直觉：
- 连续极限下，Berry 曲率积分给出第一 Chern 类；这是能带纤维丛的拓扑不变量。
- FHS 用离散 Wilson loop 相位替代连续联络，保留了规不变性与积分拓扑结构。
- 当能隙不闭合时，离散误差只会让结果在整数附近小幅波动，取整可恢复拓扑相。

## R07

复杂度：
- 网格点数 `N^2`。
- 每点对角化 `2x2` 矩阵，常数开销。
- 总时间复杂度 `O(N^2)`，空间复杂度 `O(N^2)`。

在当前 MVP 中，`N=51` 时可在普通 CPU 上快速完成。

## R08

数值稳定性策略：
- 重叠 `<u|v>` 归一化成单位复数，避免模长漂移。
- 当重叠极小（`< eps`）时返回 `1+0j` 作为保底，避免除零。
- 使用 `np.angle` 取主值相位，稳定地限制在 `(-π, π]`。

## R09

边界与失效场景：
- 在拓扑相变点附近（如 QWZ 的临界 `m`），能隙趋近 0，离散结果会更敏感。
- 网格太粗时可能出现 `chern_raw` 偏离整数较大。
- 该 MVP 为两能带单占据带版本；多带系统需推广到非阿贝尔 Berry 联络。

## R10

实现选择（MVP）：
- 语言：Python 3。
- 依赖：`numpy`, `pandas`。
- 无交互输入，`uv run python demo.py` 直接输出结果表。

文件职责：
- `demo.py`: 完整算法与批量扫描示例。
- `README.md`: 理论、步骤、复杂度、验证说明。
- `meta.json`: 任务元数据。

## R11

运行方式：

```bash
uv run python Algorithms/物理-拓扑物理-0271-拓扑不变量_(Topological_Invariants)/demo.py
```

## R12

预期输出特征：
- `m` 在不同区间会落在不同拓扑相。
- `chern_raw` 应非常接近整数（如 `-1.00000x`, `0.00000x`, `1.00000x`）。
- `chern_integer` 给出最终拓扑标签。

## R13

参数建议：
- 常规验证：`grid_size=41` 或 `51`。
- 若靠近相变点：增大到 `81` 或 `101`。
- 扫描建议 `m` 选取跨越多个区间（如 `[-3, -1, -0.5, 0.5, 1.5, 3]`）。

## R14

可扩展方向：
- 支持多带占据子空间（非阿贝尔版本）。
- 增加 `phase diagram` 扫描并可视化 `m-grid` 的拓扑相图。
- 扩展到实空间拓扑标记（如局域 Chern marker）。
- 接入 `PyTorch` 做参数反演或鲁棒性学习（当前未引入，保持 MVP 简洁）。

## R15

常见错误：
- 忘记周期边界（索引未 `% grid_size`）。
- 直接差分本征矢相位，导致规依赖数值噪声。
- 将 `np.linalg.eig` 结果不排序直接取带，可能拿错占据带。
- 在临界点附近用过小网格并强行解释为“稳定整数”。

## R16

验证清单：
- 同一 `m` 下提高 `grid_size`，`chern_raw` 是否收敛到整数附近。
- 改变本征矢整体相位后，结果是否不变（规不变性）。
- 取远离相变点参数，是否稳定得到固定整数相。
- 运行是否零交互、零异常。

## R17

参考：
- T. Fukui, Y. Hatsugai, H. Suzuki, *Chern Numbers in Discretized Brillouin Zone*, J. Phys. Soc. Jpn. 74, 1674 (2005).
- X.-L. Qi, Y.-S. Wu, S.-C. Zhang, *Topological quantization of the spin Hall effect in two-dimensional paramagnetic semiconductors*, Phys. Rev. B 74, 085308 (2006).
- M. Nakahara, *Geometry, Topology and Physics*（Berry 相与 Chern 类基础）。

## R18

`demo.py` 的源码级算法流（非黑箱）如下：

1. `qwz_hamiltonian(kx, ky, m)`：显式构造两能带 `2x2` 复哈密顿量矩阵。
2. `occupied_eigenvector(...)`：对角化该矩阵，选取最低本征值对应本征矢并归一化。
3. `chern_number_fhs(...)` 先建立 `k` 空间离散网格，并缓存每个网格点的占据带本征矢。
4. `_unit_link(v1, v2)`：计算相邻网格点本征矢重叠，并单位化成 U(1) 链变量；处理极小重叠防止数值崩溃。
5. 在 `x/y` 两方向为每个网格点构建 `Ux, Uy` 两类链变量。
6. 对每个 plaquette 计算闭合复数回路 `Ux(k) * Uy(k+ex) / (Ux(k+ey) * Uy(k))`。
7. 用 `np.angle` 取该回路相位并累加到总 Berry 通量，再除以 `2π` 得 `chern_raw`。
8. 对 `chern_raw` 四舍五入得到 `chern_integer`；`run_scan` 对多个 `m` 批量执行并汇总成表格输出。
