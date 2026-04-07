# ECDSA

- UID: `CS-0135`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `277`
- 目标目录: `Algorithms/计算机-密码学-0277-ECDSA`

## R01

`ECDSA`（Elliptic Curve Digital Signature Algorithm，椭圆曲线数字签名算法）是基于椭圆曲线离散对数难题（ECDLP）的公钥签名方案。它用于证明“消息由私钥持有者签发，且消息未被篡改”。

## R02

本题目标是给出一个可运行、可阅读的最小实现：
- 在 `secp256k1` 曲线上完成密钥对生成。
- 对消息执行签名（得到 `(r, s)`）。
- 验证原消息通过、篡改消息或篡改签名失败。

## R03

核心思想：
- 私钥是标量 `d`，公钥是曲线点 `Q = dG`。
- 签名时引入一次性随机数（本示例用可复现的确定性 nonce），将消息哈希与私钥绑定到 `(r, s)`。
- 验签时通过公开信息重构一个曲线点，检查其 `x mod n` 是否等于 `r`。

## R04

数学基础（简化版）：
- 椭圆曲线：`y^2 = x^3 + ax + b (mod p)`。
- 群运算：点加、点倍乘、标量乘（双倍-加法）。
- 模逆：`k^{-1} mod n`。
- 哈希：`z = SHA-256(m)` 转整数。
- 签名公式：
  - `R = kG, r = x_R mod n`
  - `s = k^{-1}(z + r*d) mod n`
- 验签公式：
  - `w = s^{-1} mod n`
  - `u1 = z*w mod n, u2 = r*w mod n`
  - `X = u1*G + u2*Q`
  - 接受条件：`x_X mod n == r`

## R05

输入与输出定义：
- 输入：`message: bytes`，`private_key: int`，`public_key: Point`。
- 输出：
  - `sign` 返回签名二元组 `(r, s)`。
  - `verify` 返回布尔值（是否通过）。

## R06

数据结构与参数：
- 曲线：`secp256k1`。
- 点表示：`Point = Optional[Tuple[int, int]]`，`None` 表示无穷远点。
- 域参数：`P, A, B, G=(GX,GY), N`。

## R07

实现模块划分：
- `inv_mod`：模逆。
- `is_on_curve`：点合法性检查。
- `point_add`：点加与点倍运算。
- `scalar_mult`：双倍-加法标量乘。
- `hash_to_int`：消息哈希映射到整数。
- `deterministic_nonce`：演示用确定性 nonce。
- `make_keypair / sign / verify`：签名系统主流程。

## R08

复杂度分析：
- 设标量位长为 `L`（对 secp256k1，`L≈256`）。
- `scalar_mult` 复杂度约 `O(L)` 次点运算。
- `sign` 与 `verify` 主要开销都来自 1~2 次标量乘和少量模运算，因此总体为 `O(L)` 点运算规模。

## R09

正确性要点（直观）：
- 签名里 `s = k^{-1}(z + rd)`，等价于 `k = s^{-1}(z + rd)`。
- 验签构造 `u1G + u2Q = s^{-1}zG + s^{-1}r(dG) = s^{-1}(z+rd)G = kG = R`。
- 因此验签恢复到同一个 `R`，只要签名未被篡改就满足 `x_R mod n = r`。

## R10

运行方式：
```bash
cd Algorithms/计算机-密码学-0277-ECDSA
uv run python demo.py
```

## R11

预期输出特征：
- 打印私钥、公钥、签名的截断十六进制。
- `verify(original): True`
- `verify(tampered message): False`
- `verify(tampered signature): False`

## R12

边界与异常处理：
- `k` 或 `s` 不可为 0（实现里循环重试）。
- 验签先检查 `r,s` 是否落在 `[1, n-1]`。
- 点运算中处理了无穷远点与互逆点相加场景。
- 公钥不在曲线上时，直接拒绝。

## R13

安全注意事项：
- 本实现是“教学版”，不是生产级密码库。
- 未实现常量时间防侧信道。
- nonce 生成是“可复现优先”，非 RFC 6979 的严格实现。
- 实际系统应使用成熟审计库与标准化随机源/确定性 nonce 规范。

## R14

与其他签名方案的简要比较：
- 相比 RSA，ECDSA 在同等安全级别下密钥更短、签名更小。
- 相比 EdDSA，ECDSA 实现更易踩 nonce 与参数细节坑，工程上更依赖严格规范。

## R15

MVP 设计取舍：
- 选择单文件实现，便于从“公式 -> 代码”对照学习。
- 选择 secp256k1，参数公开且常见。
- 选择确定性私钥与 nonce，保证每次运行可复现并利于回归验证。

## R16

自检清单：
- 代码可直接运行，无交互输入。
- 原消息验签成功。
- 改消息验签失败。
- 改签名验签失败。
- `README.md` 与 `demo.py` 无占位符残留。

## R17

可扩展方向：
- 用 RFC 6979 完整替换当前 nonce 生成逻辑。
- 增加 DER 编码/解码兼容。
- 增加压缩公钥、批量验签、基准测试。
- 对接 `cryptography` / `ecdsa` 库做交叉验证测试。

## R18

源码级算法流程拆解（对应 `demo.py`）：
1. 读取曲线常量 `P/A/B/G/N`，定义点类型并约定 `None` 为无穷远点。
2. `make_keypair` 生成私钥标量 `d`，并调用 `scalar_mult(d, G)` 得到公钥 `Q`。
3. `sign` 中先执行 `z = SHA256(message)`，将消息映射到大整数。
4. `sign` 调用 `deterministic_nonce` 生成 `k`，再算 `R = kG`，取 `r = x_R mod N`。
5. `sign` 计算 `s = k^{-1}(z + r*d) mod N`，并做低 `s` 规范化后输出 `(r, s)`。
6. `verify` 先做输入合法性检查：公钥在曲线上、`r,s` 处于合法区间。
7. `verify` 计算 `w=s^{-1}`, `u1=z*w`, `u2=r*w`，再算 `X = u1G + u2Q`。
8. `verify` 比较 `x_X mod N` 与 `r` 是否相等，决定签名真伪。

以上 8 步把 ECDSA 从数学公式落到具体函数调用，避免把第三方库当黑盒。
