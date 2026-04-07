# 海森堡不确定性原理 (Heisenberg Uncertainty Principle)

- UID: `PHYS-0022`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `22`
- 目标目录: `Algorithms/物理-量子力学-0022-海森堡不确定性原理_(Heisenberg_Uncertainty_Principle)`

## R01

海森堡不确定性原理给出位置与动量测量精度的下界：

`Delta x * Delta p >= hbar / 2`

其中 `Delta x` 与 `Delta p` 分别是位置和动量的标准差。本条目实现一个最小可运行数值 MVP，用离散波函数直接计算 `Delta x`、`Delta p` 并验证该不等式。

## R02

MVP 目标与范围：

- 构造可归一化的一维复波函数 `psi(x)`；
- 通过期望值与方差计算 `Delta x` 和 `Delta p`；
- 对比高斯最小不确定态与非高斯叠加态；
- 在终端输出 `Delta x * Delta p` 与 `hbar/2` 的比值；
- 脚本无交互输入，可直接运行验证。

## R03

离散数学模型（对应 `demo.py`）：

1. 概率密度：`rho(x)=|psi(x)|^2`，并满足 `integral rho dx = 1`。
2. 位置矩：
   - `<x>=integral x rho(x) dx`
   - `<x^2>=integral x^2 rho(x) dx`
   - `Delta x = sqrt(<x^2>-<x>^2)`
3. 动量算符：`p = -i hbar d/dx`，`p^2 = -hbar^2 d^2/dx^2`。
4. 动量矩：
   - `<p>=integral psi* (p psi) dx`
   - `<p^2>=integral psi* (p^2 psi) dx`
   - `Delta p = sqrt(<p^2>-<p>^2)`
5. 数值上用 FFT 谱导数构造 `dpsi/dx` 与 `d2psi/dx2`。

## R04

直观理解：

- `Delta x` 越小，波包在空间越集中；
- 集中在空间中的波函数需要更宽的频率（波数）成分；
- 更宽的波数分布对应更大的动量不确定度 `Delta p`；
- 因而 `Delta x` 与 `Delta p` 存在无法同时任意缩小的下界。

## R05

正确性要点：

1. 波函数先归一化，保证概率解释成立。
2. 位置矩直接由 `|psi|^2` 的离散积分给出。
3. 动量矩通过算符作用计算，而非仅凭经验拟合。
4. `Delta x`、`Delta p` 都由方差定义，始终非负。
5. 代码里对最小不确定高斯态执行“接近下界”断言，并验证非高斯态乘积更大。

## R06

复杂度分析（网格点数为 `N`）：

- 归一化与位置矩计算：`O(N)`
- 一次 FFT 与逆 FFT：`O(N log N)`
- 一阶与二阶导数各需一次频域乘法和逆变换，主成本仍为 `O(N log N)`
- 总体时间复杂度：`O(N log N)`，空间复杂度：`O(N)`

## R07

标准实现流程：

1. 创建均匀网格 `x`。
2. 构造待测波函数（高斯态、非高斯叠加态）。
3. 对每个态做归一化。
4. 用 FFT 谱方法得到 `dpsi/dx` 与 `d2psi/dx2`。
5. 计算 `<x>`、`<x^2>` 与 `Delta x`。
6. 计算 `<p>`、`<p^2>` 与 `Delta p`。
7. 输出 `Delta x * Delta p` 与理论下界 `hbar/2` 的对比。
8. 运行自动检查，确认结果符合物理预期。

## R08

`demo.py` 的 MVP 设计：

- 依赖：`numpy` + `scipy.fft`
- 输入：固定在脚本内的参数（无需命令行交互）
- 对比对象：
  - `gaussian packet`（应接近最小不确定度）
  - `bimodal superposition`（应明显高于下界）
- 输出：表格化指标 + 自动断言结果 `All checks passed.`

## R09

核心函数接口：

- `normalize_wavefunction(psi, dx) -> np.ndarray`
- `gaussian_wave_packet(x, x0, sigma, k0) -> np.ndarray`
- `bimodal_superposition(x, sigma, separation, k0, relative_phase) -> np.ndarray`
- `spectral_derivatives(psi, dx) -> tuple[np.ndarray, np.ndarray]`
- `uncertainty_from_state(x, psi, hbar) -> UncertaintyResult`
- `run_case(name, x, psi, hbar) -> UncertaintyResult`
- `print_report(results) -> None`
- `run_checks(results) -> None`

## R10

测试策略：

- 归一化测试：`integral |psi|^2 dx` 必须为 1（函数内部保证）。
- 下界测试：高斯态 `Delta x * Delta p` 不能低于 `hbar/2`。
- 接近性测试：高斯态乘积应接近下界（默认容差 8%）。
- 区分性测试：双峰非高斯态乘积应大于高斯态。
- 稳定性测试：网格足够密 (`N=4096`) 以降低离散误差。

## R11

边界条件与异常处理：

- `x` 非一维或与 `psi` 长度不一致：`ValueError`
- 网格点太少（`<8`）或非均匀：`ValueError`
- `hbar <= 0`：`ValueError`
- 波函数范数非正或非有限：`ValueError`
- 若数值结果违背核心物理预期：抛 `AssertionError`

## R12

与相关量子概念关系：

- 薛定谔方程给出 `psi` 的时间演化；
- 不确定性原理约束任意时刻态的可观测涨落；
- 对高斯包络，位置与动量分布互为傅里叶意义下的“最紧”配对；
- 非高斯结构（如双峰叠加、尖峰截断）通常抬高不确定度乘积。

## R13

示例参数（`demo.py`）：

- `hbar = 1.0`
- 网格：`x in [-40, 40)`，`N = 4096`
- 高斯态：`x0=-6.0, sigma=1.1, k0=2.2`
- 双峰态：`sigma=0.9, separation=10.0, k0=1.5, relative_phase=0.6`

这组参数在普通 CPU 下运行迅速，同时能稳定展示“高斯接近下界、非高斯更大”的对比。

## R14

工程实现注意点：

- FFT 谱导数默认周期边界假设，因此要选较大空间区间，使波函数在边界处足够小。
- 计算 `<p>` 与 `<p^2>` 时使用共轭内积 `vdot`，并保留积分权重 `dx`。
- 方差受浮点误差影响可能出现极小负值，代码使用 `max(var, 0.0)` 稳定化。
- 输出中同时给出绝对值与 `ratio = (Delta x*Delta p)/(hbar/2)`，便于审查。

## R15

最小示例输出应体现：

- `gaussian packet` 的 `ratio` 接近 `1.0`
- `bimodal superposition` 的 `ratio` 显著大于 `1.0`
- 程序结尾打印 `All checks passed.`

这三点构成“数值上验证海森堡不确定性原理”的最小证据链。

## R16

可扩展方向：

- 加入时间演化 `psi(x,t)`，追踪 `Delta x(t)` 与 `Delta p(t)` 的动态关系；
- 扩展到其他算符对（如角度-角动量、不同可观测量组合）；
- 对比有限差分导数与 FFT 谱导数的误差；
- 引入噪声与采样误差，研究实验数据下的不确定性估计鲁棒性。

## R17

本条目交付内容：

- `README.md`：完成 R01-R18 的方法说明、测试策略与工程细节；
- `demo.py`：最小可运行实现，直接输出不确定性验证结果；
- `meta.json`：保持与任务元数据一致。

运行方式：

```bash
cd Algorithms/物理-量子力学-0022-海森堡不确定性原理_(Heisenberg_Uncertainty_Principle)
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，9 步）：

1. **初始化离散空间**
   在 `main` 里构建均匀网格 `x`（`N=4096`），并设置 `hbar` 与两类测试态参数。

2. **构造待测波函数**
   `gaussian_wave_packet` 生成高斯包络乘平面波相位；`bimodal_superposition` 生成双峰非高斯叠加态。

3. **归一化概率质量**
   `normalize_wavefunction` 用 `sqrt(sum(|psi|^2)*dx)` 归一化，保证离散积分意义上的总概率为 1。

4. **进入频域并建立波数轴**
   `spectral_derivatives` 调用 `fftfreq` 得到离散波数 `k`，对 `psi` 做 `fft` 得到 `psi_k`。

5. **频域微分算子作用**
   在频域中一阶导是乘 `i*k`，二阶导是乘 `-(k^2)`；分别对 `psi_k` 乘这些因子。

6. **逆变换回位置域导数**
   对乘子结果做 `ifft`，得到 `dpsi/dx` 与 `d2psi/dx2`，用于动量算符期望计算。

7. **计算位置和动量矩**
   `uncertainty_from_state` 计算 `<x>`、`<x^2>`、`<p>`、`<p^2>`，再得到 `Delta x` 与 `Delta p`。

8. **形成不确定度乘积并与理论下界比较**
   计算 `product = Delta x * Delta p` 和 `bound = hbar/2`，保存到 `UncertaintyResult`。

9. **输出与自动校验**
   `print_report` 打印对比表；`run_checks` 断言高斯态接近最小下界且非高斯态更大，最后输出 `All checks passed.`。
