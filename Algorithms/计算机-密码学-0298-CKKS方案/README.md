# CKKS方案

- UID: `CS-0154`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `298`
- 目标目录: `Algorithms/计算机-密码学-0298-CKKS方案`

## R01

CKKS（Cheon-Kim-Kim-Song）是面向`近似数值计算`的同态加密方案，常用于“密文上做实数/向量运算”的场景。  
它的核心特征是：
- 明文是带尺度 `scale` 的近似数值编码；
- 支持密文加法与乘法；
- 乘法后通过 `rescale` 把尺度从 `Δ^2` 拉回 `Δ`，控制数值范围；
- 解密结果是近似值，而不是严格整数精确值。

## R02

本条目的目标是提供一个`可运行、可审计、无黑盒`的 CKKS 教学 MVP：
- 实现编码/解码、密钥生成、加密、解密；
- 实现同态加法；
- 实现同态密文乘法与一次重标度（rescale）；
- 输出误差并做断言，验证近似计算闭环成立。

## R03

本实现采用“评估域（evaluation-domain）玩具化模型”来压缩复杂度，核心关系如下：
- 编码：`m = round(x * Δ) mod q`；
- 解密近似：`x_hat = center_lift(m) / Δ`；
- RLWE 风格公钥：`b = -a*s + e (mod q)`；
- 加密：
  - `c0 = b*u + e1 + m (mod q)`
  - `c1 = a*u + e2 (mod q)`
- 二元密文解密：`m' = c0 + c1*s (mod q)`；
- 密文乘法（得到三元密文）：
  - `d0 = c0*t0`
  - `d1 = c0*t1 + c1*t0`
  - `d2 = c1*t1`
- 三元解密：`m' = d0 + d1*s + d2*s^2 (mod q)`；
- 重标度：`ct / Δ`，将尺度从 `Δ^2` 回到 `Δ`。

## R04

高层流程（对应 `main`）：
1. 初始化参数 `ring_dim/q/scale/sigma/a_bound/seed`。  
2. 生成密钥对 `PublicKey(b,a)` 与 `SecretKey(s,s_square)`。  
3. 编码向量 `x,y` 得到整数明文 `mx,my`。  
4. 分别加密得到 `Enc(x), Enc(y)`。  
5. 解密 `Enc(x)` 验证基础正确性。  
6. 执行同态加法并解密，比较 `x+y`。  
7. 执行同态乘法，再 `rescale`，再解密，比较 `x*y`。  
8. 计算三类最大绝对误差并做阈值断言。

## R05

核心数据结构：
- `CKKSParams`：参数容器（`ring_dim, modulus_q, scale, sigma, a_bound, seed`）。  
- `PublicKey`：`b, a` 两个整数向量。  
- `SecretKey`：`s` 以及 `s_square = s*s`。  
- `Ciphertext`：`components`（2 或 3 个向量）与 `scale`。  

说明：为了最小可读实现，本 MVP 使用“向量逐元素环运算”表达，而非完整多项式 NTT 管线。

## R06

正确性要点：
- `decrypt(encrypt(m))` 应近似回到编码前实数；  
- 加法同态：`Dec(Enc(x)+Enc(y)) ≈ x+y`；  
- 乘法同态：`Dec(rescale(Enc(x)*Enc(y))) ≈ x*y`；  
- 乘法后密文分量数应从 2 变为 3；  
- rescale 后尺度应从 `Δ^2` 恢复到 `Δ`。  

脚本中通过误差阈值和结构断言强制检查上述性质。

## R07

复杂度（令 `n = ring_dim`）：
- 向量加法/乘法均为 `O(n)`；
- 一次加密包含常数次向量运算，总体 `O(n)`；
- 一次解密也是 `O(n)`；
- 密文乘法（2 分量 × 2 分量）包含 3 次逐元素乘法与 1 次加法，仍为 `O(n)`；
- 空间复杂度为 `O(n)`。

本 MVP 聚焦“源流程可见”，不追求性能极值。

## R08

边界与异常处理：
- `ring_dim` 必须是 2 的幂，否则抛 `ValueError`；  
- 向量长度不匹配时拒绝运算；  
- 编码输入长度超过 `ring_dim` 时拒绝；  
- `rescale` 因子必须为正；  
- 密文分量数只允许 2 或 3，其他情况直接报错。  

这些检查避免静默错误污染加密语义。

## R09

MVP 取舍与诚实声明：
- 这是`教学版 CKKS`，不是生产安全库；  
- 为了让“单模数 + 一次 rescale + 密文乘法”在短代码里稳定可复现，`a` 采用小范围采样（`a_bound`）而非真实工程中的大模均匀采样；  
- 未实现重线性化（relinearization）、模数链（modulus chain）、旋转/共轭密钥等生产机制；  
- 仍然完整展示了 CKKS 的关键算子骨架：`encode -> encrypt -> add/mul -> rescale -> decrypt`。

## R10

`demo.py` 主要函数职责：
- `encode_real / decode_real`：实数向量与模整数表示互转；  
- `keygen`：生成 RLWE 风格公私钥；  
- `encrypt / decrypt`：二元与三元密文处理；  
- `add_ciphertexts`：同尺度密文加法；  
- `mul_ciphertexts`：密文乘法并扩展为 3 分量；  
- `rescale`：按给定因子重标度并更新 `scale`；  
- `max_abs_error`：计算误差指标；  
- `main`：构造样例、执行全流程、打印结果并断言。

## R11

运行方式（无交互）：

```bash
cd Algorithms/计算机-密码学-0298-CKKS方案
uv run python demo.py
```

脚本会自动完成加解密、同态加法、同态乘法 + rescale，并输出误差统计。

## R12

输出字段说明：
- `params`：当前参数配置；  
- `x / y`：输入明文向量；  
- `dec(x)`：密文解密近似值；  
- `dec(x+y)`：同态加法结果；  
- `dec(x*y)`：同态乘法+rescale结果；  
- 三个 `max|...|`：对应三种运算的最大绝对误差；  
- `final_scale`：乘法后重标度得到的尺度；  
- `fingerprint`：结果向量摘要，便于回归比对。

## R13

内置最小测试（`main` 自动执行）：
1. 明文回归：`Dec(Enc(x))` 与 `x` 的误差阈值检查。  
2. 加法同态：`Dec(Enc(x)+Enc(y))` 与 `x+y` 比较。  
3. 乘法同态：`Dec(rescale(Enc(x)*Enc(y)))` 与 `x*y` 比较。  
4. 结构检查：乘法后密文分量数必须为 3。  
5. 尺度检查：rescale 后尺度必须回到初始 `scale`。

## R14

关键参数与数值稳定性：
- `ring_dim=8`：控制向量长度和运算成本；  
- `modulus_q=2^50`：提供模空间；  
- `scale=2^20`：编码精度与乘法后尺度增长的核心参数；  
- `sigma=1.0`：误差采样强度；  
- `a_bound=8`：教学实现中的稳定性参数。  

经验上：`scale` 越大，编码量化误差越小，但更容易接近模上界并触发数值失真。

## R15

与相关方案对比（教学视角）：
- 对比 BFV/BGV：CKKS 面向近似实数，BFV/BGV 面向精确整数。  
- 对比调用成熟库（如 SEAL/OpenFHE）：
  - 成熟库更安全、更完整、更快；
  - 本实现胜在“源码逐行可追踪”。  
- 对比纯黑盒 API：
  - 黑盒更省事；
  - 本条目更适合学习 CKKS 核心数据流。

## R16

典型应用方向：
- 隐私保护统计（均值、线性组合、近似乘法特征）；  
- 密文机器学习中的线性层/多项式近似模块；  
- 数据协作场景下的“算而不见”。  

本 MVP 仅用于算法理解与实验，不应直接用于生产密钥系统。

## R17

可扩展方向：
- 加入真正的多项式环运算与 NTT；  
- 加入重线性化（evaluation key）；  
- 加入模数链与多级 rescale；  
- 支持旋转（Galois key）实现槽位重排；  
- 增加基准测试，对比不同参数下误差与吞吐。

## R18

`demo.py` 源码级流程拆解（9 步）：
1. `main` 创建 `CKKSParams` 并调用 `ensure_power_of_two` 校验维度合法。  
2. `keygen` 采样 `s,a,e`，按 `b=-a*s+e (mod q)` 生成公钥，并预计算 `s_square`。  
3. `encode_real` 把实数向量 `x,y` 乘以 `scale` 后四舍五入并映射到模空间。  
4. `encrypt` 对 `mx,my` 分别采样 `u,e1,e2`，构造两组二元密文 `Enc(x), Enc(y)`。  
5. `decrypt` 先对 `Enc(x)` 执行 `c0+c1*s`，`decode_real` 还原近似 `x`，得到基础误差。  
6. `add_ciphertexts` 执行分量逐元素加法，`decrypt+decode` 得到 `x+y` 近似结果。  
7. `mul_ciphertexts` 计算 `d0,d1,d2` 三个分量，形成乘法后密文（尺度变为 `Δ^2`）。  
8. `rescale` 将每个分量中心提升后除以 `Δ` 并取整，尺度恢复到 `Δ`，再解密得到 `x*y` 近似值。  
9. `main` 汇总三类误差、尺度一致性、密文分量个数，并打印摘要指纹；任一检查失败即抛异常。  

以上 9 步均为本地源码显式实现，不依赖第三方同态加密黑盒函数。
