# 布拉格定律 (Bragg's Law)

- UID: `PHYS-0100`
- 学科: `物理`
- 分类: `X射线物理`
- 源序号: `100`
- 目标目录: `Algorithms/物理-X射线物理-0100-布拉格定律_(Bragg's_Law)`

## R01

布拉格定律描述了晶体平行晶面上的相长干涉条件：

`n*lambda = 2*d*sin(theta)`

其中 `n` 是衍射级次、`lambda` 是 X 射线波长、`d` 是晶面间距、`theta` 是布拉格角（入射光与晶面夹角）。

## R02

本条目 MVP 要解决的最小问题：

1. 构造一条可控的合成 XRD 一维谱线（2theta-强度）；
2. 自动检测主峰并按级次 `n` 反演 `d`；
3. 用线性关系 `n ~ sin(theta)` 做二次交叉验证；
4. 给出可重复运行的 PASS/FAIL 判断。

## R03

建模边界与假设：

- 采用单色 X 射线（默认 Cu K-alpha, `lambda=1.5406 A`）；
- 用高斯峰模拟各级衍射峰，并加入小幅高斯噪声；
- 仅演示 1D 粉末衍射主峰，不建模真实结构因子与吸收修正；
- 级次从 `n=1` 开始，最大级次受 `n*lambda/(2d) <= 1` 约束。

## R04

脚本使用的核心方程：

1. 布拉格角位置：
   `2theta_n = 2*arcsin(n*lambda/(2*d))`
2. 反演晶面间距：
   `d_n = n*lambda/(2*sin(theta_n))`
3. 线性化关系（过原点）：
   `n = (2*d/lambda)*sin(theta)`
4. 验证残差：
   `r_n = n*lambda - 2*d_hat*sin(theta_n)`

## R05

MVP 输出两张结构化结果：

- 峰级次明细表：理论峰位、检测峰位、`sin(theta)`、逐峰 `d_n` 和误差；
- 汇总指标表：`d_mean`、`d_reg`、两类相对误差、拟合 `R^2`、Bragg 残差上限、torch 误差。

最后打印阈值检查并输出 `Validation: PASS/FAIL`。

## R06

实现策略：

- `BraggConfig` 统一管理波长、晶面间距、扫描区间、噪声和峰检测参数；
- `simulate_xrd_pattern` 负责生成合成谱线；
- `detect_peak_positions` 用 `scipy.signal.find_peaks` 提取主峰；
- `estimate_d_spacing_from_peaks` 按 Bragg 方程逐峰反演；
- `fit_bragg_linear_relation` 用线性回归做结构一致性校验；
- `evaluate_with_torch` 提供独立张量误差统计。

## R07

优点：

- 路径短，直接从谱线到 `d` 反演；
- 关键物理关系都显式写在源码里，便于审计；
- 同时有峰位法与线性回归法的双重核验。

局限：

- 未包含结构因子、仪器函数、Kalpha 双线、择优取向等实验复杂因素；
- 峰索引映射在本 MVP 中依赖“主峰按角度递增对应 `n=1,2,...`”的理想场景；
- 目标是教学级可验证流程，不是完整 Rietveld 精修。

## R08

前置知识：

- X 射线衍射几何与 `2theta` 角定义；
- Bragg 方程与峰级次概念；
- Python 数值计算基础。

依赖环境：

- Python 3.10+
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `torch`

## R09

设扫描点数为 `N`、可用峰数为 `K`。

- 合成谱线生成：`O(N*K)`（当前 `K` 很小，可视为 `O(N)`）；
- 峰检测：`O(N)`；
- 反演与拟合：`O(K)`；
- 总体时间复杂度约 `O(N)`，空间复杂度 `O(N)`。

## R10

数值稳定性与健壮性处理：

1. `arcsin` 输入做 `clip[-1,1]`，避免浮点越界；
2. 强度加入噪声后做 `clip >= 0`，防止非物理负强度；
3. 峰检测先用主阈值，不足时自动放宽一次；
4. 回归使用过原点模型，避免物理上不应出现的自由截距；
5. 最终用多指标联合阈值，而非单一指标。

## R11

默认参数（`demo.py`）：

- `lambda = 1.5406 A`
- `d_true = 3.6 A`
- 级次：请求 `n<=4`，并受物理上限自动截断
- 扫描范围：`2theta in [10, 125] deg`
- 采样：`N=5000`
- 峰宽：`sigma=0.22 deg`
- 噪声：`std=0.008`

## R12

内置验收阈值：

1. `d_mean_rel_error < 0.5%`
2. `d_reg_rel_error < 0.5%`
3. `fit_r2 > 0.999`
4. `max_abs_bragg_residual < 0.015 A`
5. `torch_rel_l1 < 0.5%`

五项全通过才输出 `Validation: PASS`，否则以非零状态退出。

## R13

保证类型说明：

- 本问题不是组合优化问题，因此无“近似比保证”；
- 本脚本为确定性流程（固定随机种子），无随机成功率语义；
- 可保证的是：在当前参数下，若实现被破坏或误差失控，会触发阈值失败并退出。

## R14

常见失效模式：

1. 峰分辨率不足（峰过宽或采样过稀）导致误检/漏检；
2. 噪声过大而阈值过低，检出伪峰；
3. 峰索引与 `n` 映射错误会直接污染 `d` 反演；
4. 单位混用（A、nm、deg/rad）会造成数量级错误；
5. `d` 取值过小导致可见衍射级次太少，拟合不稳定。

## R15

可扩展方向：

- 引入晶体结构因子和多晶面族（hkl）组合；
- 加入仪器展宽与 Kalpha 双线；
- 改为读取实验谱并做背景扣除、峰形拟合；
- 从单一 `d` 推进到晶格常数反演（如立方晶系）；
- 对接 Rietveld/全谱拟合前的快速初始化。

## R16

相关主题：

- Laue 条件与倒易空间；
- Debye-Scherrer 粉末衍射；
- 峰宽与晶粒尺寸（Scherrer 公式）；
- 晶格常数标定与应变分析；
- XRD 峰索引算法。

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-X射线物理-0100-布拉格定律_(Bragg's_Law)
uv run python demo.py
```

预期输出：

- 峰级次明细表（包含理论/检测峰位与 `d_n`）；
- 线性拟合对照与汇总指标；
- 阈值检查列表；
- 末尾 `Validation: PASS`。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `BraggConfig` 定义波长、真实 `d`、级次上限、扫描范围、峰宽、噪声与峰检参数。  
2. `simulate_xrd_pattern` 根据 `2theta_n = 2*arcsin(n*lambda/(2d))` 生成理论峰位，并叠加高斯峰构造合成强度曲线。  
3. `detect_peak_positions` 用 `find_peaks` 在曲线上找主峰；若峰数不足则做一次受控放宽，最终选最显著的 `K` 个峰。  
4. `estimate_d_spacing_from_peaks` 将检测到的 `2theta` 转 `theta`，逐峰计算 `d_n = n*lambda/(2*sin(theta_n))`，并取平均得到 `d_mean`。  
5. `fit_bragg_linear_relation` 对 `n` 与 `sin(theta)` 做过原点线性回归，得到 `d_reg`，并计算 `R^2/MAE`。  
6. `evaluate_with_torch` 用 tensor 独立计算 `d_n` 相对 L1 误差和 RMSE，作为第二实现路径的交叉校验。  
7. `main` 构建明细表与汇总表，并计算 Bragg 残差 `n*lambda - 2*d_hat*sin(theta)` 的最大绝对值。  
8. 对五个阈值逐项判定并输出 `Validation: PASS/FAIL`；任一失败则进程返回非零。  

第三方库职责是透明拆分的：

- `numpy`：向量化物理计算与数组运算；
- `scipy`：仅负责峰值检测，不替代 Bragg 反演本体；
- `pandas`：结果表结构化输出；
- `scikit-learn`：线性关系拟合与标准误差指标；
- `torch`：独立张量误差统计。
