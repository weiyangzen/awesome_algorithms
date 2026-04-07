# bcrypt

- UID: `CS-0143`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `287`
- 目标目录: `Algorithms/计算机-密码学-0287-bcrypt`

## R01

`bcrypt` 是面向口令存储的自适应哈希算法，核心设计目标是让“每次猜测口令”的成本可调且随硬件增长而提升。  
它基于 `EksBlowfish`（expensive key schedule）而不是直接对口令做单轮哈希，典型输出格式为：

`$2b$<cost>$<22-char-salt><31-char-hash>`

## R02

本条目实现一个可运行、可审阅、非黑盒的 bcrypt MVP：
- 纯 Python 实现 Blowfish 16 轮核心与 EksBlowfish 扩展；
- 实现 bcrypt 专用 base64 编解码；
- 支持 `hash` 与 `verify`（格式解析、重算校验）；
- 输出 cost 基准测试，展示工作因子带来的耗时增长。

说明：该实现用于算法理解和流程验证，不是生产级抗侧信道实现。

## R03

问题定义（密码存储视角）：
- 输入：口令 `password`、成本因子 `cost`、16 字节盐 `salt`。
- 输出：60 字符 bcrypt 字符串（`2b` 版本）。
- 验证：给定口令与现有 bcrypt 字符串，重算并做常量时间比较。

安全目标：
- 同口令不同盐得到不同哈希；
- `cost` 每 +1，理论工作量约翻倍；
- 口令数据库泄露后，离线穷举成本显著提高。

## R04

核心思想（高层）：
1. 用 Blowfish 初始常量建立状态（`P` 数组 + `S` 盒）。
2. 先执行一次“盐 + 口令”扩展。
3. 再做 `2^cost` 轮交替扩展（口令、盐）。
4. 用最终状态反复加密固定明文 `OrpheanBeholderScryDoubt`（64 轮）。
5. 取结果前 23 字节作为摘要，和盐一起编码进 bcrypt 字符串。

## R05

`demo.py` 的输入输出接口：
- `bcrypt_hash(password, cost, salt, version) -> str`
- `bcrypt_verify(password, hash_text) -> bool`
- `parse_bcrypt_hash(hash_text) -> (version, cost, salt_bytes, digest_bytes)`
- `bcrypt_raw(password_bytes, salt_bytes, cost) -> bytes`（23 字节原始摘要）

演示入口 `main()` 无交互输入，可直接运行。

## R06

关键数据结构：
- `EksBlowfishState`：
  - `p`: 18 个 32-bit 子密钥；
  - `s`: 4 个 S-Box，每个 256 个 32-bit 项。
- `BCRYPT_ALPHABET`：`./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789`
- `MAGIC_TEXT`：`OrpheanBeholderScryDoubt`

其中 `p/s` 初值来自 Blowfish 规范使用的 π 十六进制序列。

## R07

实现细节（本 MVP 的可审计点）：
- 用 `chudnovsky_pi + Decimal` 在运行时生成 Blowfish 常量，不依赖外部密码库；
- `stream_to_word` 循环读取字节流并按 32 位拼字；
- `expand_state` 与 `expand0_state` 分别对应“带盐扩展”和“零盐路径扩展”；
- `bcrypt_b64_encode/decode` 严格使用 bcrypt 自定义字符表（不是 RFC 4648 base64）。

## R08

复杂度分析（设 `c = cost`）：
- 密钥扩展主循环轮数为 `2^c`；
- 每轮包含两次 `expand0_state`，每次会触发大量 Blowfish 加密更新 `P/S`；
- 总体时间复杂度近似与 `O(2^c)` 成正比；
- 空间复杂度主要是 Blowfish 状态表，约常数级（几 KB）。

## R09

正确性检查（`demo.py` 内置）：
- 同样口令 + 同样盐 + 同样 cost，哈希结果稳定一致；
- 改动口令后，哈希显著变化；
- `bcrypt_verify` 对正确口令返回 `True`，错误口令返回 `False`；
- 不同盐下同口令得到不同哈希。

## R10

运行方式：

```bash
cd Algorithms/计算机-密码学-0287-bcrypt
uv run python demo.py
```

## R11

预期输出特征：
- 打印一条确定性 bcrypt 哈希（`$2b$05$...`）；
- 打印 `verify(correct)=True`、`verify(wrong)=False`；
- 打印两条不同盐的哈希并确认不相等；
- 打印 `cost=04/05/06` 的耗时与倍率；
- 最后一行 `all bcrypt checks passed`。

## R12

参数与边界处理：
- `salt` 必须是 16 字节；
- `cost` 在本 MVP 中限制为 `[4, 16]`（兼顾演示速度与可见增长）；
- 口令按 UTF-8 编码，并按 bcrypt 规则截断到前 72 字节后追加 `\x00`；
- 非法哈希字符串（正则不匹配）会直接抛 `ValueError`。

## R13

安全注意事项：
- 本实现是教学代码，未做常量时间内存访问、防 cache side-channel、密钥擦除等硬化；
- 生产环境应使用成熟密码库（如 `bcrypt` C/Rust 绑定）并遵循平台安全建议；
- `cost` 需要依据线上延迟预算定期上调；
- 仍需配合账户锁定、速率限制、MFA 等系统级控制。

## R14

与相关算法对比：
- 对比 `PBKDF2`：bcrypt 在口令哈希场景中历史更久，且内置专用编码格式；
- 对比 `scrypt/Argon2id`：bcrypt 主要提高计算成本，内存硬度不如后两者；
- 对比快速哈希（SHA-256 单轮）：bcrypt 明显更适合口令存储，不适合高速完整性校验。

## R15

MVP 取舍：
- 保留 bcrypt 核心算法链路（EksBlowfish + 自定义 base64 + 格式化）；
- 不引入外部 bcrypt 包，避免“一行调用黑盒”；
- 基准测试选用较低 cost（4~6）保证演示可在几秒内完成；
- 重点在“流程透明”和“可复现实验”。

## R16

自检清单：
- `README.md` 的 R01-R18 已完整填写；
- `demo.py` 无占位符残留；
- `uv run python demo.py` 可直接运行且无交互；
- 哈希/验证/盐差异/cost 增长检查均通过。

## R17

可扩展方向：
- 增加与成熟库输出的向量对照（同 salt、同 password、同 cost）；
- 增加多组统计（P50/P95）替代单次耗时；
- 支持策略参数配置（如最小 cost、密码长度策略）；
- 在同目录补充“生产实践清单”（迁移、重哈希、故障回滚）。

## R18

源码级算法流程拆解（对应 `demo.py`，9 步）：
1. `bcrypt_hash` 接收 `password/cost/salt`，先做参数校验与 UTF-8 编码。  
2. `bcrypt_raw` 对口令做 bcrypt 规则预处理：截断 72 字节并追加 `\x00`。  
3. `EksBlowfishState.fresh` 调用 `blowfish_words_from_pi`，生成并装载 Blowfish 初始 `P/S`。  
4. `expand_state(state, salt, key)` 执行“盐+口令”扩展：先异或 `P`，再通过反复 `encipher` 回填 `P/S`。  
5. 进入 `2^cost` 循环，交替执行 `expand0_state(state, key)` 与 `expand0_state(state, salt)`，放大计算成本。  
6. 构造固定 24 字节 `MAGIC_TEXT` 为 6 个 32-bit 字，执行 64 轮 Blowfish 加密链。  
7. 把 6 个 32-bit 结果拼成 24 字节并截断前 23 字节，得到 bcrypt 原始摘要。  
8. `bcrypt_b64_encode` 将 16 字节盐编码成 22 字符、23 字节摘要编码成 31 字符，拼成 `$2b$..$..` 字符串。  
9. `bcrypt_verify` 通过 `parse_bcrypt_hash` 取出版本/cost/salt 后重算，并用 `hmac.compare_digest` 做常量时间比较。
