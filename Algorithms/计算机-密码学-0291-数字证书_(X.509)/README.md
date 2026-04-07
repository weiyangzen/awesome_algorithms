# 数字证书 (X.509)

- UID: `CS-0147`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `291`
- 目标目录: `Algorithms/计算机-密码学-0291-数字证书_(X.509)`

## R01

X.509 数字证书是公钥基础设施（PKI）的核心数据对象，用来把“主体身份”与“公钥”绑定，并由上级颁发者（CA）签名背书。验证者无需和主体私下交换密钥，只要信任根证书并完成证书链验证，就能确认目标公钥是否可信。

本目录 MVP 聚焦“算法流程可追踪”而不是工程全量实现：
- 证书签发：构造 TBSCertificate 并做签名；
- 证书链验证：检查签名、有效期、颁发关系、CA 约束、信任锚；
- 反例测试：篡改证书、过期证书、非信任根证书。

## R02

X.509 在 TLS、代码签名、邮件签名等场景中的目标是统一回答三个问题：
- 这个公钥是谁的（Identity Binding）？
- 这个绑定是否由可信机构担保（Signature + Chain of Trust）？
- 这个绑定在当前时刻是否仍有效（Validity + Policy Constraints）？

因此，X.509 不只是“一个公钥文件”，而是包含策略字段与验证语义的带签名结构化声明。

## R03

本条目采用的最小问题定义：
- 输入：一条证书链（叶子 -> 中间 CA -> 根 CA）与信任锚集合；
- 输出：链是否通过验证，以及失败原因列表。

验证判据：
1. 每张证书都在有效期内；
2. 证书 `issuer` 与上级证书 `subject` 匹配；
3. 证书签名可被上级公钥验证；
4. 非根层级的颁发者必须是 CA；
5. `pathLen` 约束不被违反；
6. 根证书指纹在信任锚集合中。

## R04

核心思想（与真实 X.509 的抽象一致）：
1. 把待签内容（TBS）规范化序列化；
2. 对 TBS 做摘要并用颁发者私钥签名；
3. 验证时重建同一 TBS 字节流，用颁发者公钥验签；
4. 按链从叶到根逐级重复并执行策略检查；
5. 最后把根证书和本地信任锚匹配，决定是否信任整条链。

## R05

`demo.py` 的核心数据结构：
- `RSAKeyPair`：演示用 RSA 密钥 (`n, e, d`)；
- `Certificate`：证书主字段（序列号、主体、颁发者、有效期、是否 CA、路径约束、公钥、签名算法、签名值）；
- `ChainValidationResult`：验证结果（`passed` 与 `errors`）。

序列化策略：
- 对 TBS 与完整证书均使用 `json.dumps(..., sort_keys=True, separators=(",", ":"))` 生成稳定字节序列，确保签名与验签针对同一字节表示。

## R06

正确性关键点：
- 签发阶段和验证阶段必须使用同一套 TBS 序列化规则；
- 验签使用“当前证书的上级证书公钥”；
- 根证书虽可自签，但“自签成功”不等于“被信任”，仍需命中信任锚；
- `pathLen` 检查是链级约束，不是单证书字段比较。

## R07

时间复杂度（设链长度为 `L`，模长位数约 `k`）：
- 每张证书 1 次验签（大整数模幂），主导开销近似 `O(log k)` 次多精度乘法；
- 总体验证复杂度约为 `O(L * ModExp(k))`；
- 其余字段比较、时间比较、指纹集合查询均为线性或常数级附加开销。

## R08

空间复杂度：
- 证书链与错误列表按链长度线性增长，约 `O(L)`；
- 单次签名/验签仅维护常数级临时状态（不计大整数本身）；
- 演示中的 `pandas.DataFrame` 仅用于展示输出，不是验证算法必需。

## R09

安全说明（非常重要）：
- 本 MVP 用“教学生成 + 教学验签”的简化 RSA 流程（`pow(hash, d, n)`），没有实现 PKCS#1 v1.5 / PSS 填充；
- 这使其适合教学和可追踪验证，不适合生产安全场景；
- 真实系统还需要处理扩展字段（KeyUsage、EKU、SAN、NameConstraints、CRL/OCSP 等）与完整 ASN.1/DER 规则。

## R10

`demo.py` 提供的关键函数：
- `generate_rsa_keypair(bits)`：生成演示用 RSA 密钥；
- `issue_certificate(...)`：签发一张证书；
- `verify_chain(chain, trust_anchors, now)`：执行链验证；
- `build_demo_pki(now)`：构造根/中间/叶子三层示例 PKI；
- `run_validation_cases(...)`：运行正反例并生成结果表。

## R11

简化伪代码：

```text
verify_chain(chain, trust_anchors, now):
  errors = []
  for cert in chain from leaf to root:
    issuer = cert if root else next cert in chain
    check issuer/subject relation
    check now in [not_before, not_after]
    check signature(cert.tbs, issuer.public_key)
    check CA and pathLen constraints

  check fingerprint(root_cert) in trust_anchors
  return (errors is empty, errors)
```

## R12

本目录 MVP 的实现策略：
- 不依赖现成 X.509 黑盒库，直接写出“签发 + 验证”主路径；
- 用最小链路覆盖核心概念：
  1. 正常链通过；
  2. 篡改叶子字段导致验签失败；
  3. 叶子过期导致时效失败；
  4. 根不在信任锚导致最终失败；
- 输出结构化结果，便于自动化检查。

## R13

运行输出包括三部分：
- `[Certificate Chain]`：链中每张证书的摘要信息（主体、颁发者、CA 标记、有效期等）；
- `[Validation Cases]`：每个测试场景是否通过、错误条数与错误详情；
- `[Metrics]`：通过比例与信任锚数量（本例预期 `1/4` 通过）。

## R14

内置断言与校验：
- `valid_chain` 必须通过；
- `tampered_leaf_subject` 必须失败（签名不再匹配）；
- `expired_leaf` 必须失败（时间窗不合法）；
- `untrusted_root` 必须失败（根不在信任锚）；
- 任一断言失败会使脚本非零退出。

## R15

与真实工业 X.509 校验器的差异：
- 本实现覆盖“链验证骨架”，未实现完整 RFC 5280 细则；
- 未解析 DER/ASN.1，而是教学用 JSON 规范序列化；
- 未实现 CRL/OCSP、证书策略 OID、名称约束等高级策略；
- 目标是帮助理解算法流程，不替代 OpenSSL/BoringSSL/操作系统证书栈。

## R16

可扩展方向：
1. 改为标准签名流程（RSA-PSS 或 ECDSA）并引入正式编码；
2. 增加 SAN、KeyUsage、ExtendedKeyUsage 验证；
3. 增加吊销状态检查（CRL / OCSP）；
4. 支持证书路径构建（多候选中间证书搜索），而非仅验证给定链。

## R17

运行方式（无交互）：

```bash
uv run python Algorithms/计算机-密码学-0291-数字证书_(X.509)/demo.py
```

或在目录内执行：

```bash
uv run python demo.py
```

预期看到 `Status: PASS`。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `build_demo_pki` 生成三套 RSA 密钥，分别用于根、中间、叶子主体。  
2. `issue_certificate` 先构造无签名证书，再提取 `tbs_payload` 并做规范化 JSON 序列化。  
3. `rsa_sign_tbs` 对 TBS 做 SHA-256，取模后执行 `pow(hash, d, n)` 生成签名。  
4. 脚本构造 `leaf -> intermediate -> root` 链，并把根证书指纹放入信任锚集合。  
5. `verify_chain` 逐证书检查 issuer/subject 关系、有效期、CA 角色和 pathLen 约束。  
6. 同函数对每张证书重建 TBS，并用上级公钥执行 `rsa_verify_tbs_signature`。  
7. 链尾额外检查根指纹是否在信任锚中，得到最终通过/失败和错误列表。  
8. `run_validation_cases` 运行 1 个正例 + 3 个反例，`main` 用断言固定预期并打印结果。  

第三方库说明：
- `numpy` 与 `pandas` 仅用于结果统计和表格展示；
- 证书签发与验证主算法均在 `demo.py` 源码内显式实现，不依赖 X.509 黑盒库。
