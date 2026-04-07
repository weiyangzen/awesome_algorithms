# BLAKE2/BLAKE3

- UID: `CS-0140`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `284`
- 目标目录: `Algorithms/计算机-密码学-0284-BLAKE2／BLAKE3`

## R01

BLAKE2 与 BLAKE3 都是现代密码哈希算法家族，用于把任意长度输入映射为固定长度（或可扩展长度）摘要，常见场景包括完整性校验、内容寻址、去重、签名前置哈希与 MAC（带密钥模式）。

本目录 MVP 的目标是：
- 用 `hashlib` 给出 BLAKE2b/BLAKE2s 的可运行示例（普通、keyed、salt+person）。
- 用纯 Python 复现 BLAKE3 常规哈希路径（unkeyed regular hash）而不是黑箱调用。
- 对 BLAKE3 使用官方测试向量做确定性自检，确保实现可审计。

## R02

背景与定位：
- BLAKE2 来自 SHA-3 竞赛候选 BLAKE 的工程化改进版，强调“快、简单、可参数化”（digest 大小、key、salt、person）。
- BLAKE3 在 BLAKE2s 压缩函数基础上引入树哈希（Merkle tree）与 XOF（可扩展输出），对并行与大输入更友好。
- 在很多软件系统中，BLAKE3 常作为高速通用哈希；BLAKE2 常作为标准库友好的高性能替代（尤其在已有 OpenSSL/stdlib 环境）。

## R03

问题定义（本实现范围）：
- 输入：字节串 `M`（`bytes`）。
- 输出：
  - `BLAKE2b-256(M)`、`BLAKE2s-256(M)` 及其 keyed/salt/person 变体摘要；
  - `BLAKE3-256(M)` 与 `BLAKE3-XOF-512bit(M)`（这里展示 64 字节输出）。

正确性判据：
- BLAKE2 由 `hashlib` 输出，保证与库实现一致；
- BLAKE3 纯 Python 结果必须匹配官方测试向量（选取多个关键长度：`0/1/2/.../2048`）。

## R04

核心思想：
- BLAKE2：单树根哈希路径，可通过参数块支持 keyed hashing、salt、personalization；本 MVP 通过标准库直接调用。
- BLAKE3：
  1. 输入切成 `1024` 字节 chunk；
  2. chunk 内按 `64` 字节 block 压缩，打 `CHUNK_START/CHUNK_END` 标志；
  3. 每个 chunk 产出 chaining value（CV）；
  4. CV 进入二叉树归并（`PARENT` 标志）；
  5. 根节点加 `ROOT` 标志生成最终输出；
  6. 支持 XOF：根输出块计数器递增即可扩展输出长度。

## R05

关键数据结构与状态量：
- BLAKE3 常量：
  - `IV`（8 个 32-bit word）；
  - `MSG_PERMUTATION`（消息词置换）；
  - 标志位 `CHUNK_START/CHUNK_END/PARENT/ROOT`。
- `Output` 结构体（Python dataclass）：
  - `input_chaining_value`、`block_words`、`counter`、`block_len`、`flags`；
  - 提供 `chaining_value()` 和 `root_output_bytes()`。
- 运行时栈：`cv_stack`，用于按“尾零位数”规则维护未归并的子树 CV。
- 验证表：`pandas.DataFrame` 输出向量对照与雪崩统计。

## R06

正确性要点（BLAKE3 路径）：
- 压缩函数正确性：
  - 使用 7 轮 G 函数混合，轮间消息置换；
  - 结束时执行 `state[i] ^= state[i+8]` 与 `state[i+8] ^= cv[i]`。
- chunk 正确性：
  - 除最后一个 block 外都以 `block_len=64` 压缩；
  - 最后 block 通过 `CHUNK_END` 标志携带真实 `block_len`。
- 树归并正确性：
  - 每处理一个非最后 chunk，按 `total_chunks` 尾零位数归并父节点；
  - 最终从右边界回溯合并 `cv_stack`，得到根输出描述符。
- 外部验证：
  - 对官方向量长度集合逐项断言 `got == expected`，任何偏差即抛异常。

## R07

时间复杂度：
- 对输入长度 `n`：
  - BLAKE2 / BLAKE3 都是线性时间 `O(n)`；
  - BLAKE3 额外树归并节点数量是 chunk 数量的线性级别，整体仍为 `O(n)`。
- 向量校验额外开销与测试样例总长度线性相关。

## R08

空间复杂度：
- 流水压缩本体为常数级状态；
- BLAKE3 的 `cv_stack` 大小与 chunk 数的二进制位数相关，近似 `O(log n)`；
- 表格展示（`DataFrame`）属于演示输出开销，不是算法必需开销。

## R09

安全与工程性质：
- 哈希不可逆：给定摘要难以反推出原文。
- 抗碰撞依赖算法与参数选择；生产环境需遵循最新密码学实践。
- keyed 模式可作为快速 MAC 构件，但安全协议设计仍应考虑完整上下文（密钥管理、重放、上下文绑定）。
- 本目录不做抗侧信道证明，也不替代正式密码库审计。

## R10

边界与输入约束：
- `demo.py` 仅接受字节输入路径（示例中为固定字符串字节串和官方模式输入）。
- BLAKE3 输出长度 `out_len` 必须为正整数。
- 空输入支持（并被官方向量覆盖）。
- 大输入可处理，但本 MVP 主要用于教学与验证，不主打吞吐基准。

## R11

伪代码（BLAKE3 简化版）：

```text
blake3_hash(data, out_len):
  chunks = split(data, 1024), empty_input -> [b""]
  cv_stack = []

  for each non-last chunk i:
    out = chunk_output(chunk_i, counter=i)
    cv = out.chaining_value()
    merge cv into cv_stack by trailing zeros of (i+1)

  output = chunk_output(last_chunk, counter=last_index)
  while cv_stack not empty:
    output = parent_output(pop(cv_stack), output.chaining_value())

  return output.root_output_bytes(out_len)
```

## R12

本目录 MVP 实现策略：
- BLAKE2 使用标准库 `hashlib`（最小依赖、稳定可运行）。
- BLAKE3 不依赖三方包，按参考实现路径手写：
  - `_g/_round/_permute/_compress`；
  - `_chunk_output/_parent_output`；
  - `blake3_hash` 主流程（chunk 切分 + 栈归并 + ROOT 输出）。
- 结果展示使用 `pandas`，位级雪崩统计使用 `numpy`。

## R13

`demo.py` 输出字段说明：
- `BLAKE2 Showcase`：
  - `algorithm`：`BLAKE2b` / `BLAKE2s`
  - `mode`：`unkeyed` / `keyed` / `salt+person`
  - `digest_hex`：十六进制摘要
- `BLAKE3 Official Vector Verification`：
  - `input_len`：官方向量输入长度
  - `expected_32`：官方 32-byte 前缀
  - `got_32`：本实现结果
  - `pass`：是否匹配
- `Avalanche Check`：
  - `changed_bits`：1-bit 输入扰动后摘要变化位数
  - `change_ratio`：变化位占比

## R14

内置测试与校验：
- BLAKE3 官方向量校验：覆盖 `0,1,2,3,4,5,63,64,65,1023,1024,1025,2048`。
- BLAKE2 参数化演示：同一消息分别在普通/带密钥/salt+person 模式下产出不同摘要。
- 雪崩效应演示：对消息末位翻转 1 bit，统计摘要位翻转比例。
- 脚本失败策略：任一向量不匹配则抛出异常并非零退出。

## R15

BLAKE2 与 BLAKE3 对比（工程视角）：
- 共同点：都基于 ARX（加法/异或/旋转）思想，速度快、实现友好。
- BLAKE2：
  - 标准库可直接用，迁移成本低；
  - 参数化（key/salt/person）接口成熟。
- BLAKE3：
  - 原生树哈希 + XOF，适合并行与长输出；
  - 若运行时无现成库，可按参考实现重建核心流程（本目录已示例）。

## R16

适用场景：
- 文件与对象完整性校验。
- 内容寻址存储（CAS）与去重索引。
- 需要高吞吐哈希的预处理流水线。
- 需要可扩展输出（XOF）时，BLAKE3 更直接。

不适用场景：
- 需要密码协议级完整安全证明但仅想“手写实现”直接上生产；应优先采用成熟库与审计实现。

## R17

运行方式：

```bash
cd "Algorithms/计算机-密码学-0284-BLAKE2／BLAKE3"
uv run python demo.py
```

运行特性：
- 无交互输入。
- 直接打印 BLAKE2 对照表、BLAKE3 向量验证表、雪崩统计与 64-byte XOF 输出。
- 所有校验通过后打印 `All checks passed.`。

## R18

`demo.py` 源码级算法流程（9 步，非黑箱）：
1. `main` 固定输入消息，依次触发 BLAKE2 展示、BLAKE3 向量验证、雪崩统计与 XOF 输出。
2. `build_blake2_showcase` 分别调用 `hashlib.blake2b/blake2s` 的普通、keyed、salt+person 模式，展示参数如何影响摘要。
3. `verify_blake3_vectors` 按官方定义生成 `0..250` 循环字节输入，逐长度执行 `blake3_hash(..., out_len=32)` 并与官方前缀比对。
4. `blake3_hash` 将输入按 `CHUNK_LEN=1024` 切片，对非最后 chunk 计算 `chunk_output().chaining_value()`。
5. `_add_chunk_chaining_value` 根据 `total_chunks` 的尾零位数进行父节点归并：反复 `parent_cv(pop_left, new_cv)`，再把结果压回 `cv_stack`。
6. `_chunk_output` 负责 chunk 内部 block 处理：非末 block 立刻压缩更新 CV；末 block 打 `CHUNK_END`（必要时也含 `CHUNK_START`）生成 `Output`。
7. `_compress` 是核心：初始化 16-word 状态，执行 7 轮 `_round`（每轮由 `_g` 混合列/对角），轮间 `_permute` 消息词，最后做 feed-forward 异或。
8. 全部 chunk 处理后，`blake3_hash` 从右边界把 `cv_stack` 逐步并入当前 `Output`，得到根 `Output`。
9. `Output.root_output_bytes` 在 `ROOT` 标志下调用压缩函数并递增输出块计数器，拼接得到任意长度输出；本例展示 32-byte 和 64-byte 两种结果。

补充：本实现的 BLAKE3 路径是“源码级可追踪”实现，不依赖 `blake3` 第三方包黑箱调用；BLAKE2 则明确使用标准库成熟实现。
