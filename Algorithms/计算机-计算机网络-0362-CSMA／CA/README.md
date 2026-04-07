# CSMA/CA

- UID: `CS-0209`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `362`
- 目标目录: `Algorithms/计算机-计算机网络-0362-CSMA／CA`

## R01

CSMA/CA（Carrier Sense Multiple Access with Collision Avoidance）用于共享信道的分布式接入控制。核心思想是：
- 先监听（Carrier Sense），信道忙则等待；
- 信道空闲后并不立即发送，而是进入随机退避（Backoff）；
- 通过退避窗口与碰撞后指数增窗，降低再次碰撞概率。

## R02

本目录 MVP 的问题定义：
- 多个站点共享单一无线信道；
- 每个站点有待发帧队列，并可在指定时隙产生新到达；
- 信道按离散 slot 推进，站点在满足 DIFS 后倒计时并尝试发送；
- 若同 slot 多站点同时尝试则视为碰撞，否则成功发送 1 帧。

## R03

输入与输出：
- 输入：
1. `station_count`、`initial_queues`；
2. `arrival_plan[slot] -> 每站新增帧数`；
3. `cw_min/cw_max`、`difs_slots`、`tx_slots/ack_slots/collision_slots`、`max_slots`、`seed`。
- 输出：
1. 每 slot 明细日志（是否忙、谁尝试、是否碰撞、谁成功、队列状态）；
2. 每站 `success_per_station`、`attempts_per_station`、`collisions_per_station`；
3. 全局 `collision_probability`、`throughput(success/slot)`、`backlog_final`。

## R04

目标与约束：
- 目标：在共享信道中完成队列发送，并观察冲突率与公平性；
- 约束：
1. 忙信道期间不进行倒计时与发送；
2. 仅在连续空闲满足 `difs_slots` 后才允许倒计时；
3. 同一 slot 仅允许“0 个尝试 / 1 个成功 / 多个碰撞”三种互斥结果；
4. 队列长度必须保持非负。

## R05

关键状态变量：
- `queues[i]`：站点 `i` 当前待发帧数；
- `contention_windows[i]`：站点 `i` 当前竞争窗口（CW）；
- `backoff_counter[i]`：站点 `i` 当前退避计数（`-1` 表示未激活）；
- `channel_busy_remaining`：信道剩余忙时隙；
- `idle_streak`：连续空闲时隙计数（用于 DIFS）；
- `success/attempts/collisions_per_station`：每站统计量。

## R06

离散时隙迭代流程（每个 slot）：
1. 应用 `arrival_plan`，把新业务加入队列；
2. 对“有业务且未激活退避”的站点初始化随机 `backoff_counter`；
3. 若信道忙：减少 `channel_busy_remaining` 并重置 `idle_streak`；
4. 若信道空闲：累加 `idle_streak`，满足 DIFS 后让 `backoff_counter>0` 的站点倒计时；
5. 倒计时后 `backoff_counter==0` 的站点成为尝试者集合；
6. 尝试者数为 1 时发送成功（队列减 1，CW 复位，信道进入 `tx+ack` 忙期）；
7. 尝试者数大于 1 时发生碰撞（尝试者 CW 指数增大并重抽退避，信道进入碰撞忙期）；
8. 记录 slot 日志，若“无积压 + 不再到达 + 信道空闲”则提前结束。

## R07

正确性直觉：
- 安全性：忙信道不发送 + DIFS 后倒计时，避免无条件抢占信道；
- 冲突处理：并发尝试仅在计数同归零时发生，且每次碰撞后增大 CW，有助于打散后续竞争；
- 守恒性：每次成功严格发送 1 帧，最终 `sum(success_per_station)` 应与总需求一致（在仿真时长足够时）。

## R08

复杂度分析（`S` 为实际运行 slot 数，`N` 为站点数）：
- 时间复杂度：`O(S * N)`，每 slot 需要遍历站点做倒计时、判定与统计；
- 空间复杂度：`O(S + N)`，日志为 `O(S)`，状态向量为 `O(N)`。

## R09

参数影响：
- `cw_min` 越小，初期竞争更激进，碰撞概率通常更高；
- `cw_max` 越大，碰撞后退避更分散，但重传等待可能更长；
- `difs_slots` 越大，接入更保守，总吞吐可能下降；
- `tx_slots/ack_slots/collision_slots` 越大，信道忙占比上升，单位 slot 吞吐下降；
- 业务到达越突发，短时竞争越激烈。

## R10

MVP 设计取舍：
- 使用 Python + `numpy` 实现状态向量与统计，保持最小依赖；
- 使用“每次成功发送 1 帧”的离散模型，不展开物理层与速率自适应细节；
- 用确定性到达计划与固定随机种子增强可复现性；
- 不引入 RTS/CTS、隐藏终端、信道误码，聚焦 CSMA/CA 核心退避机制。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0362-CSMA／CA
uv run python demo.py
```

脚本无需交互输入，会输出汇总指标、前 20 个 slot 明细，并在通过断言后打印 `All checks passed.`。

## R12

输出指标说明：
- `success_per_station`：每站成功发送帧数；
- `attempts_per_station`：每站尝试发送次数（成功+碰撞）；
- `collisions_per_station`：每站发生碰撞次数；
- `collision_probability = total_collisions / total_attempts`：碰撞比例；
- `throughput(success/slot)`：单位 slot 成功发送量；
- `jain_fairness(success_per_station)`：发送公平性指标（越接近 1 越均衡）。

## R13

结果解读：
- 若 `backlog_final=0` 且总成功数等于总需求，说明任务被完整排空；
- `collision_probability` 偏高表示竞争激烈，可通过增大 `cw_min` 缓解；
- 公平性接近 1 说明各站长期机会接近均衡；
- 若吞吐偏低，通常是碰撞频繁或忙时隙参数过大导致。

## R14

边界与异常处理：
- `station_count<=1`、`max_slots<=0`、`cw_max<cw_min`、`difs_slots<1` 等会抛 `ValueError`；
- `initial_queues` 长度不匹配或含负数会抛 `ValueError`；
- `arrival_plan` 的 slot 为负、向量长度错误或含负值会抛 `ValueError`；
- 若 `max_slots` 不足，可能出现 `backlog_final>0`，由断言直接暴露。

## R15

与相关介质访问方法对比：
- 相对 CSMA/CD：CSMA/CA 侧重“冲突避免+碰撞后退避”，更适合无线场景；
- 相对令牌环：CSMA/CA 无中心令牌，部署简单但时延确定性较弱；
- 相对纯 ALOHA：CSMA/CA 通过载波监听与退避显著降低无谓冲突。

## R16

可扩展方向：
- 增加 RTS/CTS 与 NAV，模拟隐藏终端缓解；
- 采用分级接入参数（如 EDCA）模拟不同业务优先级；
- 加入信道误码与重传上限，分析吞吐-时延折中；
- 引入多速率和帧长分布，做更真实的 WLAN 负载评估。

## R17

最小验证清单：
- `README.md` 和 `demo.py` 已完成内容填充且不含模板占位符；
- `uv run python demo.py` 可直接运行且无需交互输入；
- 输出包含 `All checks passed.`；
- 断言覆盖至少：
1. 最终无积压（`backlog_final==0`）；
2. 总成功发送等于总需求；
3. 任一站点成功数不超过其需求；
4. 场景中确实观察到碰撞（用于验证退避路径被触发）。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `run_demo()` 设置站点数、初始队列、到达计划和 CSMA/CA 参数，创建 `CSMACASimulator`。  
2. `CSMACASimulator.__init__` 完成参数与输入校验，标准化 `arrival_plan` 为 `numpy` 向量映射。  
3. `run()` 初始化 `queues`、`contention_windows`、`backoff_counter`、统计数组与信道状态变量。  
4. 每个 slot 先注入到达流量，并为“有积压但未激活”的站点随机分配退避计数。  
5. 若信道忙，仅消耗忙时隙；若信道空闲并达到 DIFS，则对 `backoff_counter>0` 的站点倒计时。  
6. 统计 `backoff_counter==0` 的尝试集合：
   单站尝试则成功发送并复位其 CW；多站尝试则判为碰撞并对相关站点指数增窗后重抽退避。  
7. 记录 `SlotRecord`（尝试者、是否碰撞、成功站点、队列状态），并在“无积压且无后续业务且信道空闲”时提前结束。  
8. `run_demo()` 对结果做守恒和边界断言，`print_report()` 输出碰撞率、吞吐、公平性及前 20 个 slot 明细。  
