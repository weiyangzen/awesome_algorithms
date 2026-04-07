# MapReduce

- UID: `CS-0301`
- 学科: `计算机`
- 分类: `并行与分布式`
- 源序号: `461`
- 目标目录: `Algorithms/计算机-并行与分布式-0461-MapReduce`

## R01

MapReduce 是一种面向大规模数据处理的并行编程范式，核心思想是把任务拆成两个阶段：
- `Map`：把输入记录映射为中间键值对；
- `Reduce`：对同一 key 的中间值进行聚合。

在分布式系统中，这通常配合 `Shuffle/Sort`（按 key 重分区并分组）执行。本条目用一个可运行 Python MVP 演示其最小闭环流程与可验证正确性。

## R02

MVP 问题定义（词频统计）：
- 输入：10 条英文文本记录（固定内置语料）；
- 输出：
1. 全量词频 `word -> count`；
2. mapper 侧的原始发射条数与 combiner 压缩后条数；
3. reducer 侧接收键数与部分值数量；
4. Top-12 高频词表。

目标不是做真实集群性能测试，而是展示 MapReduce 的算法与数据流。

## R03

本实现覆盖的 MapReduce 关键环节：
1. `Split`：把文档索引分片给多个 mapper；
2. `Map`：把 token 映射为 `(word, 1)`；
3. `Combiner`：mapper 本地先聚合为 `(word, partial_count)`；
4. `Partition`：使用稳定哈希把 key 路由到 reducer；
5. `Shuffle/Sort`：按 reducer + key 分组并排序；
6. `Reduce`：对每个 key 的部分计数求和；
7. `Validate`：与顺序基线词频对比。

## R04

`demo.py` 的执行流程：
1. 准备固定语料与 `num_mappers=3`、`num_reducers=2`；
2. 调用 `run_mapreduce` 执行端到端流程；
3. 计算 combiner 压缩率；
4. 打印 mapper / reducer 统计表；
5. 打印 Top-12 高频词；
6. 输出 `All checks passed` 作为完成信号。

## R05

关键数据结构：
- `MapperStats`：记录 mapper 的文档分片、原始发射量、combiner 后发射量；
- `ReducerStats`：记录 reducer 的输入 key 数、输入部分值数量、输出 key 数；
- `MapReduceResult`：封装最终词频、统计信息与 shuffle 桶；
- `shuffle_buckets: list[dict[str, list[int]]]`：模拟网络传输后的分组结果；
- `Counter[str]`：mapper 本地计数与顺序基线计数。

## R06

正确性直觉：
- tokenizer 保证输入记录被确定性切分为 token；
- combiner 只做“同 key 局部求和”，不会改变全局和；
- partition 保证同一 key 始终进入同一个 reducer；
- reducer 对同 key 的所有 partial count 求和，等价于全局直接计数；
- 最终将 MapReduce 输出与顺序 `direct_word_count` 全量比对，作为主正确性判据。

## R07

复杂度分析（设总 token 数为 `N`，去重后 key 数为 `K`，mapper 数 `M`，reducer 数 `R`）：
- Map：`O(N)`；
- Combiner：每个 mapper 本地聚合，合计约 `O(N)`；
- Shuffle/Sort：网络级条目数约为 `K_m`（combiner 后中间规模），分桶 + 排序代价约 `O(K_m + K_m log K_m)`；
- Reduce：`O(K_m)` 到 `O(K)` 级别；
- 额外空间：中间桶存储 `O(K_m)`。

在本固定数据集里，脚本会打印 `raw` 与 `after_combiner`，可直观看到中间数据收缩。

## R08

边界与异常处理：
- `num_mappers <= 0` 或 `num_reducers <= 0`：抛 `ValueError`；
- 空文档集合：抛 `ValueError`；
- 如果某个 key 在多个 reducer 输出中重复出现：抛 `AssertionError`；
- 若 partition 一致性被破坏（key 路由不匹配 reducer）：抛 `AssertionError`；
- 若 MapReduce 结果不等于顺序基线：抛 `AssertionError`。

## R09

MVP 设计取舍：
- 不依赖 Hadoop/Spark 等重框架，保证 `uv run python demo.py` 可直接跑通；
- 使用 `numpy` 做分片与数值聚合，`pandas` 仅用于输出可读表格；
- 聚焦算法机制与可验证性，不引入真实网络、容错重试调度器、作业管理器；
- 保持单文件可读，便于教学和后续扩展到真实分布式运行时。

## R10

`demo.py` 函数职责：
- `tokenize`：规范化分词；
- `stable_partition`：稳定哈希分区；
- `split_document_shards`：输入分片；
- `map_with_combiner`：map + 本地聚合；
- `shuffle_sort`：跨 mapper 中间结果分组排序；
- `reduce_bucket`：reducer 聚合；
- `direct_word_count`：顺序基线；
- `run_mapreduce`：端到端流程 + 断言校验；
- `*_frame` 与 `result_frame`：构建输出表；
- `main`：固定实验入口与结果展示。

## R11

运行方式：

```bash
cd Algorithms/计算机-并行与分布式-0461-MapReduce
uv run python demo.py
```

脚本无交互输入，执行后直接输出统计表与验证结果。

## R12

输出说明：
- `Documents | mappers | reducers`：本次作业规模；
- `Intermediate pairs`：combiner 前后中间键值对数量和压缩比例；
- `Mapper stats`：每个 mapper 的负载与局部压缩情况；
- `Reducer stats`：每个 reducer 的聚合负载；
- `Top 12 words`：词频最高的 12 个 token；
- `All checks passed for CS-0301 (MapReduce).`：所有校验通过。

## R13

最小验证清单：
1. `final_counts == direct_word_count(documents)`；
2. 所有 key 的 `stable_partition(key, R)` 与其 reducer 一致；
3. 不允许同一 key 在多个 reducer 输出重复；
4. `combined_emit_total <= raw_emit_total`（combiner 不能放大中间规模）；
5. 脚本可重复运行且输出稳定。

## R14

固定实验参数：
- 文档数：10；
- mapper 数：3；
- reducer 数：2；
- 分词规则：正则 `[a-zA-Z]+` + 小写化；
- 分区函数：`md5(key) % num_reducers`（稳定、与 Python 随机哈希无关）；
- 结果展示：`pandas.DataFrame` 排序输出 Top-12。

## R15

与相关方案对比：
- 与单机直接 `Counter` 对比：MapReduce 增加了显式的分片、分区、shuffle、reduce 语义；
- 与 Spark/Hadoop 对比：本实现不提供分布式调度与容错基础设施，但算法流一致；
- 与 MPI 点对点通信对比：MapReduce 更偏“数据并行批处理”，MPI 更偏通用消息传递。

## R16

适用场景：
- 学习 MapReduce 工作机制；
- 在本地快速验证 key 分区与 reduce 逻辑；
- 作为迁移到真实分布式框架前的原型。

不适用场景：
- 真实 TB/PB 级离线任务吞吐评估；
- 需要集群调度、失败重试、数据本地性优化的生产场景；
- 低延迟流式场景（本示例是批处理思路）。

## R17

可扩展方向：
- 把单机模拟替换为多进程或 Ray/Dask 执行器；
- 引入倾斜 key（hot key）并实现二级聚合或采样分区；
- 增加自定义排序（例如二次排序）和复合 key；
- 增加故障注入（mapper/reducer 失败）并重试；
- 增加另一个作业示例（如倒排索引）复用同一框架骨架。

## R18

`demo.py` 源码级算法流（9 步，非黑箱）：
1. `main` 初始化固定语料、mapper/reducer 数量并调用 `run_mapreduce`。  
2. `split_document_shards` 用 `numpy.array_split` 把文档索引切成 3 个 mapper 分片。  
3. 每个分片进入 `map_with_combiner`：先分词，再发射 `(word,1)`，随后本地 `Counter` 合并为 `(word, partial_count)`。  
4. `shuffle_sort` 遍历所有 mapper 的 partial 结果，调用 `stable_partition(md5)` 把 key 路由到目标 reducer。  
5. `shuffle_sort` 在每个 reducer 桶内按 key 排序，形成确定性的 `dict[str, list[int]]` 输入。  
6. `reduce_bucket` 对每个 key 的 `list[int]` 用 `numpy.sum` 聚合，得到 reducer 局部输出。  
7. `run_mapreduce` 合并各 reducer 输出，检查 key 不重复并构建全局 `final_counts`。  
8. 同时执行三类断言：分区一致性、与 `direct_word_count` 基线相等、combiner 不放大中间规模。  
9. `main` 用 `pandas` 生成 mapper/reducer 统计表与 Top-12 词频表，最后打印通过标记。

这里没有把第三方 MapReduce 框架当黑箱，而是把分片、映射、分区、shuffle、归约、校验都显式落在源码中。
