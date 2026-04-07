# PBKDF2

- UID: `CS-0142`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `286`
- 目标目录: `Algorithms/计算机-密码学-0286-PBKDF2`

## R01

`PBKDF2`（Password-Based Key Derivation Function 2）是经典口令密钥派生算法，定义于 PKCS #5 / RFC 8018。  
它通过“盐 + 多轮迭代 + 伪随机函数（常见为 HMAC）”把低熵口令扩展成指定长度密钥，用于口令存储或后续对称加密密钥生成。

## R02

本条目目标是实现一个可运行、可审阅、非黑盒的 PBKDF2 最小 MVP：
- 在 `demo.py` 手写 `PBKDF2-HMAC` 核心流程（而非仅调用单函数）；
- 使用 RFC 6070 标准测试向量校验正确性；
- 与 `hashlib.pbkdf2_hmac` 做交叉验证；
- 保证 `uv run python demo.py` 无交互执行。

## R03

核心公式（以块索引 `i` 表示）：

`DK = T_1 || T_2 || ... || T_l`，其中 `l = ceil(dkLen / hLen)`  
`T_i = U_1 XOR U_2 XOR ... XOR U_c`  
`U_1 = PRF(P, S || INT_32_BE(i))`  
`U_j = PRF(P, U_{j-1}), 2 <= j <= c`

其中：
- `P` 为口令（password）；
- `S` 为盐（salt）；
- `c` 为迭代次数；
- `PRF` 常用 `HMAC-SHA256` 或 `HMAC-SHA1`；
- `hLen` 为 PRF 输出字节长度。

## R04

算法参数说明（工程上最关键）：
- `hash_name`：PRF 哈希族，如 `sha256`、`sha1`；
- `password`：原始口令字节串；
- `salt`：随机且唯一的盐；
- `iterations`：迭代次数，决定计算成本；
- `dklen`：目标导出密钥长度（字节）。

参数变更策略：
- 同口令不同盐，导出结果应不同；
- 迭代次数应随硬件提升逐步上调；
- 记录中需保存 `hash/salt/iterations/dklen` 才可校验。

## R05

`demo.py` 中输入输出定义：
- `pbkdf2_hmac_manual(hash_name, password, salt, iterations, dklen) -> bytes`：手写实现；
- `pbkdf2_hmac_reference(...) -> bytes`：标准库参考实现；
- `run_rfc6070_vectors() -> None`：跑 RFC 6070 向量并断言；
- `run_cross_checks() -> None`：与标准库交叉验证并做敏感性测试；
- `main() -> None`：统一运行入口。

## R06

实现约束与数据表示：
- 全程使用 Python 标准库（`hashlib`、`hmac`、`math`、`time`）；
- 输入统一按 `bytes` 处理，避免编码歧义；
- 块索引 `i` 采用 4 字节 big-endian（`INT_32_BE(i)`）；
- 异或累积在字节层完成，严格对应公式中的 `XOR`。

## R07

代码模块划分：
- `_prf_hmac`：单次 PRF 调用（HMAC）；
- `_xor_bytes`：两个等长字节串异或；
- `pbkdf2_hmac_manual`：核心派生循环（按块 + 按迭代）；
- `pbkdf2_hmac_reference`：调用 `hashlib.pbkdf2_hmac` 作为对照；
- `run_rfc6070_vectors`：标准向量验证；
- `run_cross_checks`：一般参数下一致性与敏感性检查；
- `main`：汇总输出与失败快速退出。

## R08

复杂度分析：
- 设 `l = ceil(dkLen / hLen)`，迭代次数为 `c`；
- 每个块需要 `c` 次 PRF，总 PRF 次数约 `l * c`；
- 时间复杂度约为 `O(l * c * cost(PRF))`；
- 额外空间主要为单块缓存与输出缓冲，空间复杂度 `O(dkLen)`。

## R09

正确性要点（直观）：
- `U_1` 将 `(salt, block_index)` 绑定到每个块，避免块间冲突；
- `U_j = PRF(P, U_{j-1})` 形成迭代链，提升单次猜测成本；
- `T_i` 对所有 `U_j` 做 XOR，确保每一轮都会影响最终块结果；
- 多块拼接后截断到 `dklen`，可生成任意长度导出密钥。

## R10

运行方式（无交互）：

```bash
cd Algorithms/计算机-密码学-0286-PBKDF2
uv run python demo.py
```

## R11

预期输出特征：
- 先显示 RFC 6070 测试向量通过情况（含每条迭代次数）；
- 再显示手写实现与 `hashlib.pbkdf2_hmac` 的一致性；
- 输出口令扰动、盐扰动后的差异检查；
- 最后打印 `all PBKDF2 checks passed` 作为通过标记。

## R12

边界与异常处理：
- `iterations <= 0`、`dklen <= 0` 直接抛 `ValueError`；
- `password` 或 `salt` 为空在本 MVP 中视为非法输入；
- `_xor_bytes` 对长度不一致输入抛错，防止静默截断；
- 若任一测试向量不匹配，立即抛 `RuntimeError`。

## R13

安全注意事项：
- PBKDF2 只能“延缓”离线暴力，不会让弱口令变成强口令；
- 必须使用高质量随机盐（每条记录唯一）；
- 必须使用恒定时间比较函数（如 `hmac.compare_digest`）；
- 新系统可优先考虑内存硬算法（如 `scrypt` / `Argon2id`），PBKDF2 仍适合兼容性优先场景。

## R14

与相近方案对比：
- 对比 `bcrypt`：PBKDF2 参数和哈希族更通用，跨平台兼容性好；
- 对比 `scrypt`：PBKDF2 主要增加 CPU 成本，内存抗性较弱；
- 对比 `Argon2id`：Argon2id 在现代抗 GPU/ASIC 设计上更先进，但部署兼容面通常不如 PBKDF2。

## R15

MVP 取舍说明：
- 采用“手写核心 + 标准库对照”的双轨结构，兼顾可读性与可信验证；
- 不引入第三方密码框架，减少依赖噪声；
- 仅覆盖离线算法层，不扩展到账户锁定、速率限制、密钥轮换策略等系统策略。

## R16

自检清单：
- `README.md` 已完整填写 R01-R18；
- `demo.py` 无占位符，可直接运行；
- RFC 6070 选取向量全部通过；
- 手写实现与标准库实现一致；
- 参数/输入非法时能显式报错。

## R17

可扩展方向：
- 增加 `PBKDF2-HMAC-SHA512`、`PBKDF2-HMAC-SHA3` 对比；
- 增加参数基准测试，按目标延迟反推迭代次数；
- 加入口令记录序列化格式（算法、参数、盐、派生值）；
- 增加并发批量验证实验，评估服务端吞吐与成本。

## R18

源码级算法流程拆解（对应 `demo.py`，8 步）：
1. 接收 `hash_name/password/salt/iterations/dklen`，计算 `hLen` 并确定块数 `l = ceil(dklen / hLen)`。  
2. 对每个块索引 `i`，构造 `salt || INT_32_BE(i)`，调用 `_prf_hmac` 得到 `U_1`。  
3. 把 `U_1` 作为当前块累计值 `T` 的初始内容。  
4. 从第 2 轮到第 `c` 轮，反复执行 `U_j = PRF(P, U_{j-1})`。  
5. 每得到一轮 `U_j`，用 `_xor_bytes` 把它并入 `T`，对应公式 `T = U_1 XOR ... XOR U_c`。  
6. 块结束后把 `T` 追加到输出缓冲；如果还没达到 `dklen`，继续下一个块。  
7. 所有块完成后截断到 `dklen` 字节，返回最终导出密钥 `DK`。  
8. `main` 中使用 RFC 向量与 `hashlib.pbkdf2_hmac` 对照，验证手写流程与标准实现一致。  
