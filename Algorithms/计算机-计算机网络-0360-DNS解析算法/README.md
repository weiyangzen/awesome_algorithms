# DNS解析算法

- UID: `CS-0207`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `360`
- 目标目录: `Algorithms/计算机-计算机网络-0360-DNS解析算法`

## R01

问题定义：给定域名（如 `www.example.com`）和记录类型（如 `A`），DNS 解析算法要在分层命名体系中找到最终资源记录，或返回 `NXDOMAIN` / `SERVFAIL`。

本 MVP 聚焦“迭代解析器”核心流程：根区 -> 顶级域区 -> 权威区，外加缓存与 CNAME 跟随。

## R02

输入：
- 查询名 `qname`（字符串）
- 查询类型 `qtype`（示例使用 `A`）
- 一组内存中的权威区数据（模拟根区、`.com` 区、`example.com` 区）

输出：
- 解析状态：`NOERROR` / `NXDOMAIN` / `SERVFAIL`
- 结果记录（`ResourceRecord` 列表）
- 可追踪的解析路径（trace）

## R03

核心思想：
1. 先查本地缓存（命中则直接返回）。
2. 若未命中，从根区开始进行迭代查询。
3. 遇到 `referral`（委派）则切换到下一层权威区继续查。
4. 遇到 `CNAME` 则跳转到别名目标并重新从根区迭代。
5. 命中最终 `answer` 后写入缓存并返回。

## R04

数据结构设计：
- `ResourceRecord`：标准 DNS 资源记录（`name/rtype/value/ttl`）。
- `Delegation`：委派信息（子域、NS 主机、glue IP、TTL）。
- `Response`：权威服务器返回类型（`answer/cname/referral/nxdomain`）。
- `CacheEntry`：缓存条目（记录 + 到期时刻）。
- `ResolutionResult`：一次解析结果（状态、答案、追踪路径）。

## R05

区模型（Zone Model）：
- 根区 `.` 仅负责把 `*.com` 委派到 `.com` 权威服务器。
- `.com` 区把 `*.example.com` 委派给 `example.com` 权威服务器。
- `example.com` 区保存最终 `A` 记录与 `CNAME` 记录。

这模拟了真实 DNS 的“层级授权 + 最长后缀匹配”行为。

## R06

查询控制流（高层）：
- 当前状态由 `current_zone` 和 `current_name` 决定。
- 在 `current_zone` 上请求 `current_name/current_type`。
- 响应分支：
  - `answer`：结束。
  - `cname`：`current_name` 替换为别名目标并回到根区。
  - `referral`：切到 `next_zone`。
  - `nxdomain`：结束并返回不存在。

## R07

复杂度（单次查询）：
- 设区层级深度为 `H`（通常较小），CNAME 链长度为 `C`。
- 时间复杂度近似 `O(H + C)`（每一步是 O(1) 字典查找）。
- 空间复杂度主要为缓存，设缓存项为 `K`，则 `O(K)`。

在真实网络里，时延成本主要来自 RTT；在此 MVP 里用内存访问替代网络调用。

## R08

缓存与 TTL：
- 缓存键为 `(name, rtype)`。
- 写入时取该答案集合最小 TTL，记录 `expire_at = now + ttl`。
- 查询时若 `now >= expire_at`，条目视为过期并删除。
- 示例里重复查询 `www.example.com A` 会触发缓存命中。

## R09

CNAME 处理策略：
- `qtype != CNAME` 且命中别名时，先返回 CNAME，再跟随到目标名继续查目标类型。
- 为防循环别名，设置 `cname_hops` 上限（代码里为 6）。
- 超限返回 `SERVFAIL`，避免无限循环。

## R10

失败模式与边界：
- `NXDOMAIN`：当前权威区既无答案也无可继续委派。
- `SERVFAIL`：缺失对应权威区实现、CNAME 深度异常、或 referral 深度异常。
- 解析器对域名做标准化（小写、去尾点），避免大小写/格式差异导致误判。

## R11

`demo.py` 关键函数：
- `build_authorities`：构建根区/TLD/权威区示例数据。
- `AuthorityServer.query`：执行区内匹配与委派判断。
- `IterativeDNSResolver.resolve`：解析主流程（缓存 + referral + CNAME）。
- `_cache_lookup/_cache_store`：TTL 缓存逻辑。
- `run_demo` / `validate_results`：运行样例并自动断言正确性。

## R12

MVP 取舍：
- 使用 Python 标准库实现，不引入额外网络包。
- 不做真实 UDP/TCP DNS 报文编码，仅实现算法控制流。
- 重点是把“DNS 解析算法”的关键决策路径做成可运行、可验证、可追踪的最小闭环。

## R13

内置样例查询：
1. `www.example.com A`：正常委派并返回 A 记录。
2. `api.example.com A`：先命中 CNAME，再跳转解析 `edge.example.com A`。
3. `www.example.com A`（再次查询）：展示缓存命中。
4. `missing.example.com A`：返回 `NXDOMAIN`。

## R14

正确性直觉：
- 每一步都遵守“授权边界”：上级区只做委派，下级区给最终答案。
- 委派切换基于域后缀，确保查询逐层收敛到更具体的权威区。
- CNAME 通过“名称替换 + 重新迭代”与真实解析器一致。
- TTL 缓存只在有效期内复用答案，保证结果不会无限陈旧。

## R15

与真实 DNS 的差异（已知限制）：
- 未实现完整 DNS 报文格式、ID、flags、RCODE 等协议细节。
- 未实现 DNSSEC、EDNS0、负缓存（SOA）与重试策略。
- 未区分 UDP/TCP，也未处理真实网络超时与多 NS 负载均衡。

这些是协议工程扩展，不影响本题“DNS 解析算法流程”展示。

## R16

建议扩展测试：
1. 增加多级子域委派（如 `a.b.example.com`）。
2. 构造 CNAME 环路验证保护逻辑。
3. 增加 TTL 很短的记录并模拟过期后重新查询。
4. 扩展到 `AAAA`、`MX` 类型并验证缓存键隔离。
5. 引入多个 NS 并模拟一个 NS 故障回退。

## R17

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0360-DNS解析算法
uv run python demo.py
```

预期行为：
- 打印每个查询的状态、答案与逐跳 trace。
- 最后打印 `All assertions passed.`。

## R18

源码级算法拆解（对应 `demo.py`）：
1. `build_authorities` 构建三层区：根区 `.`、TLD 区 `com`、权威区 `example.com`，并填充 `A/CNAME/NS+glue` 数据。
2. `resolve` 开始时先做域名标准化并查 `_cache_lookup`；命中则直接返回 `NOERROR`。
3. 缓存未命中时设置 `current_zone='.'`，进入迭代循环；每轮调用 `authority.query(current_name, qtype)`。
4. `query` 内部按优先级判断：精确 `answer` -> `cname` -> `referral`（最长后缀委派）-> `nxdomain`。
5. 若得到 `referral`，解析器更新 `current_zone=response.next_zone`，继续下一轮迭代（模拟根到权威逐层下钻）。
6. 若得到 `cname`，解析器记录别名、把 `current_name` 改为目标名，并把 `current_zone` 重置为根区重新迭代。
7. 若得到 `answer`，调用 `_cache_store` 以最小 TTL 写缓存，然后返回最终结果与完整 trace。
8. 若命中 `nxdomain` 或触发 hop 上限异常（referral/CNAME 过深），返回 `NXDOMAIN` 或 `SERVFAIL`，保证流程可终止。
