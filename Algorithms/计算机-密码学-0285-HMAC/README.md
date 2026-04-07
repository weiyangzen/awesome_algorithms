# HMAC

- UID: `CS-0141`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `285`
- 目标目录: `Algorithms/计算机-密码学-0285-HMAC`

## R01

`HMAC`（Hash-based Message Authentication Code）是基于哈希函数构造的消息认证码，用于同时提供：
- 消息完整性（消息是否被篡改）；
- 密钥认证性（消息是否由持有共享密钥的一方生成）。

它常见于 API 签名、Webhook 校验、令牌签名链路、TLS 早期组件、以及 PBKDF2 内部 PRF。

## R02

本条目目标是给出一个可运行、可审阅、非黑盒的 HMAC 最小 MVP：
- 在 `demo.py` 手写 HMAC 核心流程（不只调用 `hmac.new` 一行）；
- 使用 RFC 4231 的 HMAC-SHA256 标准向量验证正确性；
- 与 Python 标准库 `hmac.digest` 做交叉校验；
- 保证 `uv run python demo.py` 无交互执行。

## R03

HMAC 公式：

`HMAC(K, m) = H((K0 XOR opad) || H((K0 XOR ipad) || m))`

其中：
- `H`：哈希函数（如 `SHA-256`）；
- `K0`：对密钥 `K` 规整到哈希块长后的结果；
- `ipad = 0x36` 重复 `block_size` 次；
- `opad = 0x5c` 重复 `block_size` 次；
- `||`：字节拼接。

## R04

密钥规整规则（核心细节）：
- 若 `len(K) > block_size`，先做 `K = H(K)`；
- 若 `len(K) < block_size`，右侧补零到 `block_size`；
- 若相等则直接使用。

此步骤避免不同长度密钥直接进入哈希内部时的歧义，并确保 `ipad/opad` 异或在固定长度上执行。

## R05

`demo.py` 输入输出接口：
- `hmac_manual(hash_name, key, message) -> bytes`：手写 HMAC；
- `hmac_reference(hash_name, key, message) -> bytes`：标准库参考实现；
- `run_rfc4231_sha256_vectors() -> None`：RFC 向量测试；
- `run_cross_checks() -> None`：多哈希族一致性与敏感性验证；
- `main() -> None`：统一入口。

## R06

实现约束与数据表示：
- 仅使用 Python 标准库（`hashlib`、`hmac`、`time`）；
- 输入统一按 `bytes` 处理，避免编码歧义；
- 显式执行 `key` 规整、`ipad/opad` 异或、内层哈希、外层哈希；
- 对比结果使用 `hmac.compare_digest`，避免字符串比较的时序差异。

## R07

代码结构划分：
- `_normalize_key`：完成 key 的哈希压缩与补零；
- `_xor_with_byte`：将字节串与固定字节常量逐字节异或；
- `hmac_manual`：HMAC 主流程；
- `hmac_reference`：标准库对照实现；
- `run_rfc4231_sha256_vectors`：标准向量断言；
- `run_cross_checks`：跨算法一致性和输入敏感性检查。

## R08

复杂度分析：
- 设消息长度为 `n`，哈希压缩块长为 `b`；
- HMAC 主体包含两次哈希，时间复杂度约为 `O(n)`；
- 密钥规整部分最多额外一次哈希（当 `len(K) > b`）；
- 空间复杂度为 `O(n)`（由拼接输入和摘要缓存主导）。

## R09

正确性直观说明：
- 内层哈希先把 `ipad` 和消息绑定，外层哈希再用 `opad` 重新包裹；
- 双层结构阻断了把普通哈希直接当 MAC 的扩展攻击问题；
- 若消息或密钥任一变化，内外层摘要会级联变化，输出应显著不同；
- 使用公开测试向量可验证实现是否满足规范。

## R10

运行方式（无交互）：

```bash
cd Algorithms/计算机-密码学-0285-HMAC
uv run python demo.py
```

## R11

预期输出特征：
- 先打印 RFC 4231 各测试向量是否通过；
- 再打印与 `hmac.digest` 的一致性结果；
- 输出消息扰动和密钥扰动检查结果；
- 最后打印 `all HMAC checks passed`。

## R12

边界与异常处理：
- `key` 为空在本 MVP 中视为非法输入（抛 `ValueError`）；
- `hash_name` 无效时由 `hashlib.new` 抛出异常；
- 若某向量不匹配，立即抛 `RuntimeError`；
- 若交叉验证不一致或敏感性检查失败，直接终止。

## R13

安全注意事项：
- HMAC 提供认证和完整性，不提供保密性；
- 共享密钥必须来自安全随机源并妥善管理；
- 认证时必须使用常量时间比较函数；
- 不应重复使用同一密钥执行无边界的多用途协议（建议按用途分离密钥）。

## R14

与相关方案对比：
- 对比“裸哈希校验”：HMAC 有密钥，能抵抗伪造；
- 对比数字签名：HMAC 是对称方案，速度快但不具备不可否认性；
- 对比 CMAC：CMAC 依赖分组密码（如 AES），HMAC 依赖哈希函数，部署上更通用。

## R15

MVP 取舍说明：
- 采用“手写流程 + 标准库对照”保证可读和可核验；
- 不引入第三方密码库，减少依赖噪声；
- 重点覆盖算法层正确性，不扩展到协议层重放保护、时戳窗口和密钥轮换策略。

## R16

自检清单：
- `README.md` 已填写 R01-R18；
- `demo.py` 无占位符；
- `uv run python demo.py` 可直接运行；
- RFC 4231 向量全部通过；
- 手写实现与标准库实现一致。

## R17

可扩展方向：
- 增加 `HMAC-SHA3` 或 `BLAKE2` 的对照实验；
- 增加批量基准测试（吞吐和延迟）；
- 增加“请求签名串规范化”示例（URL、Header、Body 组合）；
- 增加密钥分层派生策略（主密钥 -> 场景子密钥）。

## R18

源码级算法流程拆解（对应 `demo.py`，8 步）：
1. 读取 `hash_name/key/message`，查询哈希块长 `block_size`。  
2. 若 `key` 长于块长，先做 `key = H(key)`；若短于块长，右侧补零。  
3. 构造 `ipad` 与 `opad`（分别为 `0x36`、`0x5c` 重复 `block_size` 次）。  
4. 计算 `inner_key = key XOR ipad`，并拼接消息 `inner_key || message`。  
5. 对上一步做第一次哈希，得到 `inner_digest`。  
6. 计算 `outer_key = key XOR opad`，并拼接 `outer_key || inner_digest`。  
7. 对上一步做第二次哈希，得到最终 `tag`（HMAC 输出）。  
8. `main` 中用 RFC 4231 向量和 `hmac.digest` 交叉验证，确认流程正确。
