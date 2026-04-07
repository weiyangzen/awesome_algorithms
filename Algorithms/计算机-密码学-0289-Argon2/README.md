# Argon2

- UID: `CS-0145`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `289`
- 目标目录: `Algorithms/计算机-密码学-0289-Argon2`

## R01

`Argon2` 是 Password Hashing Competition (PHC) 的获胜方案，核心目标是提供“内存硬（memory-hard）”的口令哈希/密钥派生能力，提升对 GPU/ASIC 大规模并行暴力破解的成本。  
常见变体包括 `Argon2d`（偏数据相关访存）、`Argon2i`（偏数据无关访存）和 `Argon2id`（混合策略，工程上最常用）。

## R02

本条目给出一个可运行、可审阅的最小 MVP：
- 用 Python 从零实现“Argon2 风格”的内存填充与迭代混合流程；
- 输出 PHC 风格编码字符串（`$argon2id$v=19$m=...,t=...,p=...$salt$hash`）；
- 提供 `verify` 校验与自检断言，保证无交互可直接运行。

## R03

核心思想：
- 不是直接对口令做一次哈希，而是把口令、盐和参数先编码成初始摘要；
- 再在较大的内存块数组上反复做“前一块 + 参考块”的混合；
- `time_cost` 决定迭代轮数，`memory_cost_kib` 决定内存规模，`parallelism` 决定并行 lane 数；
- 最终将各 lane 末块归约为派生摘要。

## R04

本 MVP 的算法骨架（教学版 Argon2id）：
- 块大小固定为 `1024` 字节（与 Argon2 规范一致）；
- 每个 lane 先生成前两个初始块；
- 后续块按 `(prev_block, ref_block)` 做压缩扩展；
- 首轮只引用本 lane（更接近 Argon2id 前段行为），后续轮可跨 lane 引用；
- 末尾将所有 lane 的最后一块异或归约，再扩展成目标长度输出。

## R05

输入输出定义（对应 `demo.py`）：
- `argon2_mvp_hash_raw(password, salt, params) -> bytes`：输出原始哈希字节串；
- `encode_phc_string(params, salt, digest) -> str`：输出 PHC 风格文本；
- `verify_password(password, encoded) -> bool`：重算并常量时间比较；
- `main()`：固定样例演示，不需要任何交互输入。

## R06

数据结构与约束：
- 内存矩阵形态：`lanes[parallelism][lane_length]`，每个元素是 `1024-byte block`；
- 参数约束：
  - `time_cost >= 1`
  - `parallelism >= 1`
  - `memory_cost_kib >= 8 * parallelism`
  - `4 <= hash_len <= 1024`
  - `salt` 推荐至少 8 字节
- 所有计数器和长度编码均使用 little-endian `uint32`。

## R07

模块拆分：
- `_le32`：`uint32` 小端编码；
- `_xor_bytes`：等长字节异或；
- `_expand_hash`：基于 BLAKE2b 的可变长度扩展哈希；
- `_initial_hash`：参数+口令+盐的初始摘要 `H0`；
- `_reference_index`：计算参考 lane 与参考块位置；
- `_compress_block`：由 `prev/ref` 派生新块；
- `argon2_mvp_hash_raw`：整体内存填充与归约主流程；
- `encode_phc_string / parse_phc_string / verify_password`：编码、解析与验证。

## R08

复杂度分析（设 `m=memory_cost_kib`，`t=time_cost`）：
- 空间复杂度：`O(m)`（显式保留 `m` 个 1KiB 块）；
- 时间复杂度：`O(t * m)`（每轮遍历主要内存块并做压缩混合）；
- 这正是 memory-hard KDF 的关键特征：攻击者若削减内存通常会额外付出时间代价。

## R09

正确性直观：
- 输入 `(password, salt, params)` 确定后，`H0` 与后续块引用路径都是确定性的；
- 内存填充顺序固定，末块归约固定，因此输出稳定可复现；
- `verify_password` 用同一流程重算，再做常量时间比较，正确口令应得到一致摘要。

## R10

运行方式（无交互）：

```bash
cd Algorithms/计算机-密码学-0289-Argon2
uv run python demo.py
```

## R11

预期输出特征：
- 打印参数（`m/t/p/hash_len`）和运行耗时；
- 打印 PHC 风格编码（截断显示）；
- `verify(correct): True`；
- `verify(wrong): False`；
- 无异常退出（包含内置断言自检）。

## R12

边界与异常处理：
- 参数不合法会抛 `ValueError`（如 `memory_cost_kib` 太小、`hash_len` 越界）；
- PHC 字符串格式错误会在解析阶段抛 `ValueError`；
- `salt` 长度过短会被拒绝，避免弱配置误用；
- `_xor_bytes` 对长度不一致的输入显式报错，避免静默截断。

## R13

安全注意事项：
- 本实现是教学 MVP，体现 Argon2 的内存硬机制，但不是 RFC 9106 完整兼容实现；
- 生产环境应使用成熟库（如 libsodium / argon2-cffi）并结合随机盐、参数审计、版本迁移策略；
- 哈希只解决“存储口令不可逆”，不替代登录限流、多因子认证和凭据泄露监测；
- 参数应根据硬件与延迟预算定期升级，避免长期静态配置。

## R14

与相近方案对比：
- `PBKDF2`：CPU 计算硬，但内存占用低，对并行硬件不够“昂贵”；
- `bcrypt`：有一定抗并行能力，但内存维度与参数弹性较有限；
- `Argon2`：把时间成本和内存成本都显式参数化，现代口令哈希场景更主流。

## R15

MVP 设计取舍：
- 选择“可读实现优先”，不引入第三方黑盒包，便于逐行审阅；
- 压缩函数采用 BLAKE2b 扩展近似实现，而非完整 RFC 的 `G` 置换与索引细节；
- 保留 Argon2 最重要的工程语义：参数化、内存填充、lane 归约、PHC 编码与验证流程。

## R16

自检清单：
- `README.md` 与 `demo.py` 的占位符已全部清理；
- `uv run python demo.py` 可直接运行；
- 相同输入哈希结果可重复；
- 正确口令验证通过，错误口令验证失败；
- 参数校验异常路径可触发并被断言覆盖。

## R17

可扩展方向：
- 按 RFC 9106 补全真实 Argon2id 索引与压缩轮函数；
- 加入参数自动调优（按目标延迟选择 `m/t/p`）；
- 增加跨实现向量对照（与标准库输出对拍）；
- 支持批量用户凭据迁移与版本升级策略（旧参数自动重哈希）。

## R18

源码级流程拆解（`demo.py`，非第三方黑盒）：
1. 校验参数与输入长度，计算按 lane 对齐后的总内存块数。  
2. 对 `parallelism/hash_len/memory/time/version/type/password/salt` 做小端编码并 BLAKE2b，得到初始摘要 `H0`。  
3. 对每个 lane 生成前两个 1KiB 初始块（`H0 || block_id || lane_id` 经扩展哈希）。  
4. 进入 `time_cost` 轮迭代；每轮按列推进每个 lane 的剩余块。  
5. 对当前块读取 `prev_block` 前 8 字节得到 `(j1, j2)`，据此确定参考 `(ref_lane, ref_col)`。  
6. 将 `prev_block` 与 `ref_block` 异或后连同位置元数据做扩展哈希，生成新块；第 2 轮起再与旧块异或叠加。  
7. 全部填充结束后，取每个 lane 末块做异或归约，得到最终聚合块。  
8. 对聚合块做目标长度扩展，输出 raw digest，并编码为 PHC 字符串用于存储与验证。  
