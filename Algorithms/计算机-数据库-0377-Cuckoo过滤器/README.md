# Cuckoo过滤器

- UID: `CS-0223`
- 学科: `计算机`
- 分类: `数据库`
- 源序号: `377`
- 目标目录: `Algorithms/计算机-数据库-0377-Cuckoo过滤器`

## R01

本条目实现一个最小可运行的 Cuckoo Filter（布谷鸟过滤器）MVP，覆盖三类核心操作：
- 插入 `insert(key)`
- 查询 `contains(key)`
- 删除 `delete(key)`

实现目标是“可解释 + 可复现实验”，而不是依赖黑盒库一行调用。`demo.py` 直接给出负载率、删除后状态与误判率统计。

## R02

问题定义（本条目范围）：
- 输入：
  - 一批字符串键（`user-*`）用于插入；
  - 一批不相交字符串键（`probe-*`）用于误判率测试；
  - 参数：容量、桶大小、指纹位数、最大踢出次数。
- 输出：
  - 插入成功数量与负载率；
  - 假阴性率（应接近 0）；
  - 删除成功率与删除后残留率；
  - 假阳性率（False Positive Rate）及理论近似值。

约束：
- 无交互输入；
- 过滤器只存指纹，不存原始 key；
- 允许误判，不允许稳定性下的系统性漏报。

## R03

核心建模：

1) 两候选桶
- 每个 key 计算主桶 `i1 = h(key) mod m`
- 指纹 `fp = g(key)`, 并保证 `fp != 0`
- 备选桶 `i2 = i1 xor h(fp)`（本实现用 2 的幂桶数并通过按位与取模）

2) 查询逻辑
- 只需在 `bucket[i1]` 与 `bucket[i2]` 里查 `fp` 是否存在。

3) 插入逻辑
- 若任一候选桶有空位，直接放入；
- 若都满，则随机挑一个候选桶踢出一个已有指纹，把被踢指纹搬去它的另一个桶，最多尝试 `max_kicks` 次。

4) 删除逻辑
- 在两个候选桶中删除一个匹配指纹即可。

## R04

算法单次插入流程：
1. 计算 key 的指纹 `fp` 与桶索引 `i1`、`i2`。
2. 检查 `i1` 是否有空位，有则插入并结束。
3. 检查 `i2` 是否有空位，有则插入并结束。
4. 若都满，随机从 `i1/i2` 选一个桶开始踢出流程。
5. 在当前桶随机选槽位，与 `fp` 交换（踢出旧指纹）。
6. 令当前指纹跳到它的另一个桶：`idx = idx xor h(fp)`。
7. 若目标桶有空位则插入成功，否则继续踢出，直到达到 `max_kicks`。
8. 超过踢出上限则插入失败（过滤器接近饱和或进入循环）。

## R05

核心数据结构：
- `CuckooConfig(dataclass)`：容量与算法参数。
- `CuckooFilter`：
  - `num_buckets`：桶数量（向上取 2 的幂）；
  - `buckets: list[list[int]]`：每个桶存若干指纹整数；
  - `size`：当前已存元素计数。
- 哈希工具函数：
  - `_u64_hash`：基于 `blake2b` 的稳定 64 位哈希；
  - `_index_hash`、`_fingerprint_hash`、`_alt_index`：分别负责桶索引、指纹、备用桶映射。

## R06

正确性要点：
- 同一 key 的查询只需要访问两个确定桶（`i1`, `i2`），符合 Cuckoo Filter 设计。
- 删除只删指纹，不依赖原始 key 存储，空间开销低。
- 通过限制 `max_kicks` 防止无限循环。
- 使用稳定哈希（`blake2b`）避免 Python 运行时哈希随机化导致实验结果漂移。
- 指纹强制非零，避免“空值语义”冲突。

## R07

复杂度分析（桶大小为常数 `b`）：
- 查询：`O(b)`，常数时间。
- 删除：`O(b)`，常数时间。
- 插入：
  - 平均近似 `O(1)`；
  - 最坏 `O(max_kicks)`（连续踢出）。
- 空间复杂度：`O(m * b * f)` 位（`m` 桶数、`f` 指纹位数）。

## R08

边界与失败场景：
- 当负载率逼近上限时，插入失败概率上升（踢出链更容易循环）。
- 指纹位数过小会显著提高误判率。
- 若 `bucket_size` 太小，结构更容易拥塞；过大则降低局部缓存友好性。
- 本实现不做自动扩容，插入失败会直接返回 `False`，由上层决定是否重建更大过滤器。

## R09

MVP 取舍说明：
- 只用 `numpy`（用于 RNG 与统计）+ Python 标准库；
- 不引入第三方布谷鸟过滤器包，避免黑盒；
- 桶结构使用 `list[list[int]]`，可读性优先；
- 不做并发/锁设计，不做磁盘持久化；
- 重点是验证算法行为：插入、查询、删除、误判率。

## R10

`demo.py` 主要函数职责：
- `_u64_hash`：稳定哈希原语。
- `_next_power_of_two`：将桶数归整为 2 的幂。
- `_index_hash`：key 到主桶索引。
- `_fingerprint_hash`：key 到指纹。
- `_alt_index`：由当前桶与指纹计算备用桶。
- `CuckooFilter.insert/contains/delete`：三大操作。
- `_to_rate`：布尔序列转比例统计。
- `main`：构造实验、执行 workload、打印指标。

## R11

运行方式（无交互）：

```bash
cd /Users/wangweiyang/GitHub/awesome_algorithms/.cron/stage0_exec_repo_slot07
uv run python Algorithms/计算机-数据库-0377-Cuckoo过滤器/demo.py
```

脚本会直接输出配置、负载率、假阴性率、删除效果与误判率。

## R12

输出字段解释：
- `capacity_target`：目标容量（配置值）。
- `num_buckets, bucket_size`：桶维度参数。
- `inserted`：本轮实际插入成功数量。
- `load_factor`：`size / (num_buckets * bucket_size)`。
- `false_negative_rate`：已插入键中查询失败的比例（理想为 0）。
- `deleted_ok`：删除成功计数。
- `deleted_still_present_rate`：被删键仍返回存在的比例（理想接近 0，误判时可能非零）。
- `false_positive_rate`：未插入探针键被误判存在的比例。
- `expected_approx`：误判率理论近似 `1 - (1 - 2^-f)^(2b)`。

## R13

建议最小测试集：
- 默认参数直接运行，确认脚本能完整输出统计信息。
- 把 `fingerprint_bits` 从 12 改到 8，观察误判率显著升高。
- 把 `max_kicks` 降到 20，观察高负载时插入更早失败。
- 把 `bucket_size` 从 4 改到 2，观察负载率上限下降。
- 把插入量从 7000 提高到 9000，观察失败拐点。

## R14

关键调参建议：
- `capacity`：按预期元素数设定；
- `bucket_size`：常见取值 2/4；
- `fingerprint_bits`：误判率主控参数；
- `max_kicks`：插入成功率与时间上限之间的折中。

经验：
- 要低误判，优先增大 `fingerprint_bits`；
- 要高负载稳态，优先增大 `bucket_size` 或整体桶数。

## R15

与 Bloom Filter 对比：
- Cuckoo Filter 支持删除，Bloom Filter 原版不支持删除。
- 在低误判目标下，Cuckoo Filter 通常具有更优或可比的空间效率。
- Cuckoo Filter 插入可能失败（需重建），Bloom Filter 没有踢出失败问题。
- Bloom Filter 查询为多哈希位检查；Cuckoo Filter 查询仅看两个桶中的短指纹。

## R16

典型应用场景：
- 存储引擎中“是否可能存在”的前置判定；
- LSM/对象存储中的负查询加速；
- 去重系统中的近似成员检测；
- 网络系统中的快速黑白名单命中预判。

## R17

可扩展方向：
- 增加“重建/扩容”策略，处理插入失败；
- 支持批量插入并输出负载曲线；
- 增加分片（shard）与并发锁设计；
- 增加序列化与快照恢复；
- 对比 Bloom / XOR Filter 的同规模实验报告。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 创建 `CuckooConfig` 和 `CuckooFilter`，设定容量、桶大小、指纹位数、最大踢出次数。  
2. `insert` 时先用 `_fingerprint_hash` 得到非零指纹 `fp`，再用 `_index_hash` 得到主桶 `i1`，并用 `_alt_index` 算出备选桶 `i2`。  
3. 若 `i1` 或 `i2` 任一有空槽，直接写入指纹并更新 `size`。  
4. 若两桶都满，随机选择候选桶开始踢出：在桶内随机槽位交换指纹（新指纹入桶，旧指纹被踢出）。  
5. 被踢出的指纹根据 `_alt_index` 计算“另一个桶”，继续尝试插入；超过 `max_kicks` 则失败返回 `False`。  
6. `contains` 仅重算 `fp/i1/i2`，检查两个桶里是否有该指纹。  
7. `delete` 在两个桶中扫描并删除一个匹配指纹，成功则 `size -= 1`。  
8. `main` 对插入集、删除集、探针集分别统计假阴性率/删除残留率/假阳性率，并打印实验结果与理论近似值。  
