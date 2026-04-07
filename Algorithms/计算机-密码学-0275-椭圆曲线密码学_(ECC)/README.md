# 椭圆曲线密码学 (ECC)

- UID: `CS-0133`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `275`
- 目标目录: `Algorithms/计算机-密码学-0275-椭圆曲线密码学_(ECC)`

## R01

椭圆曲线密码学（ECC, Elliptic Curve Cryptography）是在有限域上的椭圆曲线群结构上构造公钥算法族。
本目录的 MVP 聚焦三件事：
- 手写可审计的椭圆曲线基础算术（点加、倍点、标量乘）；
- 基于同一套算术演示 ECDH 密钥交换；
- 基于同一套算术演示 ECDSA 数字签名与验签。

实现目标是“最小但诚实”：不调用黑盒 ECC 库的核心 API，直接展示源码级流程。

## R02

问题定义（本条目覆盖 ECC 两类典型能力）：
- 输入：
  - 曲线参数 `E(F_p): y^2 = x^3 + ax + b (mod p)`；
  - 基点 `G`、阶 `n`；
  - 私钥 `d`（签名）或 `d_A, d_B`（密钥交换）；
  - 待签名消息 `m`。
- 输出：
  - ECDH：共享点 `S = d_A * Q_B = d_B * Q_A` 与派生会话密钥；
  - ECDSA：签名 `(r, s)` 与验签布尔结果。

## R03

关键数学基础：
1. 曲线点构成加法群，无穷远点 `O` 是单位元。  
2. 点加/倍点都要在有限域 `mod p` 下进行，除法用模逆实现。  
3. 标量乘 `kP` 用 double-and-add（二进制展开）完成。  
4. ECDH 正确性来自交换律：`d_A(d_B G) = d_B(d_A G)`。  
5. ECDSA 通过 `u1*G + u2*Q` 与 `r` 的一致性完成验证。  

## R04

高层流程：
1. 加载 `secp256k1` 参数并校验群不变量。  
2. 由私钥计算公钥（标量乘）。  
3. 执行 ECDH：双方分别计算共享点并比较一致性。  
4. 对共享点 `x` 坐标做 `SHA-256` 派生会话密钥摘要。  
5. 执行 ECDSA：签名消息并验签。  
6. 对篡改消息做负向测试，验证验签失败。  
7. 对畸形公钥做负向测试，验证输入校验生效。  
8. 汇总每个案例的操作统计并输出确定性指纹。  

## R05

核心数据结构：
- `Curve`：`name, p, a, b, g, n, h`。  
- `Point`：`Tuple[int, int] | None`，`None` 表示无穷远点。  
- `OpCounter`：记录点加次数、倍点次数、模逆次数。  
- `run_case` 返回字典，`main` 用 `numpy` 汇总统计。  

## R06

正确性与安全要点：
- 私钥必须满足 `1 <= d < n`；
- 公钥必须满足：非无穷远点、在曲线上、子群检查 `nQ = O`；
- 点加覆盖特殊情形：`P+O`、`P+(-P)`、`P=P`；
- 共享点不一致立即报错；
- 验签对原消息应通过，对篡改消息应失败。

说明：该实现用于教学/审计，不包含常数时间等生产级侧信道防护。

## R07

复杂度分析（设标量位长 `L`，对 secp256k1 约为 256）：
- 单次标量乘：
  - 倍点约 `L` 次；
  - 点加平均约 `L/2` 次；
  - 每次点运算需要一次模逆与若干模乘。
- 时间复杂度：
  - 标量乘为 `O(L)` 次椭圆曲线点运算；
  - ECDH 一方计算共享点为一次标量乘；
  - ECDSA 验签涉及两次标量乘加一次点加。
- 空间复杂度：`O(1)`（不保留中间轨迹时）。

## R08

边界与异常处理：
- `mod_inv(0, m)` 抛 `ZeroDivisionError`；
- 私钥越界抛 `ValueError`；
- 非法公钥（不在曲线/子群失败）抛 `ValueError`；
- 共享点为无穷远点时报错；
- 签名中 `r=0` 或 `s=0` 自动换下一个确定性 nonce。

## R09

MVP 取舍：
- 工具栈仅 `numpy + Python 标准库`，保持最小依赖；
- 不用第三方密码黑盒函数，便于逐行审计；
- 用固定私钥与固定消息保证可复现输出；
- 目标是算法教学最小闭环，不是生产密码模块。

## R10

`demo.py` 主要函数职责：
- 参数与基础：`secp256k1`、`ensure_int`、`mod_inv`。  
- 曲线算术：`is_on_curve`、`point_neg`、`point_add`、`scalar_mult`。  
- 密钥校验：`validate_private_key`、`validate_public_key`、`derive_public_key`。  
- ECDH：`ecdh_shared_point`、`kdf_from_point_x`。  
- ECDSA：`hash_to_int`、`deterministic_nonce`、`ecdsa_sign`、`ecdsa_verify`。  
- 运行编排：`check_group_invariants`、`run_case`、`main`。  

## R11

运行方式（无需交互输入）：

```bash
cd Algorithms/计算机-密码学-0275-椭圆曲线密码学_(ECC)
uv run python demo.py
```

脚本会自动执行两组固定案例并输出统计摘要。

## R12

输出字段说明：
- `Curve` / `Group invariants: PASS`：曲线与群性质自检结果；
- 每个 `Case`：
  - `Alice/Bob pub x`：双方公钥 `x` 坐标缩写；
  - `Shared x`：ECDH 共享点 `x` 坐标缩写；
  - `ECDSA r,s`：签名核心分量缩写；
  - `Session key`：共享点派生的 `SHA-256`；
  - `Ops`：各阶段操作计数。
- `Summary`：平均操作数、共享点位长区间、确定性指纹。

## R13

内置最小测试集：
1. 群不变量测试：`nG=O`、`(n+1)G=G`、`G+(-G)=O`。  
2. ECDH 一致性测试：`d_A*Q_B == d_B*Q_A`。  
3. ECDSA 正向测试：原消息验签通过。  
4. ECDSA 负向测试：篡改消息验签失败。  
5. 输入校验测试：畸形公钥被拒绝。  

## R14

参数与工程注意：
- 本实现固定曲线 `secp256k1`（`a=0,b=7,h=1`）；
- 哈希与 KDF 均使用 `SHA-256`；
- nonce 采用“私钥+消息”确定性推导（教学简化，不是 RFC 6979 完整实现）；
- 生产环境应补充身份认证、会话上下文绑定、常数时间实现、密钥擦除等措施。

## R15

方法对比：
- 相比传统 DH/RSA，ECC 在同等安全级别下可使用更短密钥。  
- 相比直接调用密码库黑盒，本实现可读性更高但安全工程完备性更低。  
- 相比只演示单一协议（仅 ECDH 或仅 ECDSA），本 MVP 展示“同一底层曲线算术驱动两类协议”的复用关系。  

## R16

典型应用场景：
- TLS/Noise 等协议中的密钥协商；
- 区块链账户体系中的签名认证；
- 设备身份认证、固件签名验证；
- 资源受限终端对短密钥公钥算法的需求场景。

## R17

可扩展方向：
- 引入 RFC 6979 nonce 生成；
- 增加压缩公钥编码/解码（SEC1）；
- 对比 Montgomery ladder 与 double-and-add 的侧信道暴露面；
- 增加更多曲线（P-256、X25519 对照）；
- 补充基准测试与更多异常输入测试。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 调用 `secp256k1()` 载入曲线参数。  
2. `check_group_invariants` 依次验证 `nG=O`、`(n+1)G=G`、`G+(-G)=O`。  
3. `run_case` 使用 `derive_public_key`（内部走 `scalar_mult`）生成 Alice/Bob 公钥。  
4. `scalar_mult` 逐位执行 double-and-add；每次点运算由 `point_add` 完成，有限域除法由 `mod_inv` 完成。  
5. `ecdh_shared_point` 先 `validate_public_key` 再做标量乘，双方分别得到共享点并做一致性比较。  
6. `kdf_from_point_x` 取共享点 `x` 坐标并做 `SHA-256` 得到会话密钥摘要。  
7. `ecdsa_sign` 通过 `deterministic_nonce` 产生 nonce，计算 `R=kG`、`r=R_x mod n`、`s=k^{-1}(z+rd) mod n`。  
8. `ecdsa_verify` 计算 `w=s^{-1}`、`u1=zw`、`u2=rw`，再算 `X=u1G+u2Q` 并检查 `X_x mod n == r`。  
9. `main` 用 `numpy` 对案例操作计数求均值、位长范围与确定性指纹，全部断言通过后输出 `All ECC checks passed.`。  
