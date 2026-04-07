# TLS/SSL协议

- UID: `CS-0146`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `290`
- 目标目录: `Algorithms/计算机-密码学-0290-TLS／SSL协议`

## R01

`TLS/SSL` 是“传输层安全协议”家族，用于在不可信网络上提供机密性、完整性与端点认证。  
其中 `SSL 2.0/3.0` 已淘汰，现代系统实际使用的是 `TLS 1.2/1.3`。  
本条目聚焦协议核心算法流程：握手期如何协商密钥、记录层如何保护应用数据。

## R02

本目录目标是给出一个可运行、可审计的最小 MVP，而不是黑盒调用整套 `ssl` 库：
- 用临时 DH（ephemeral Diffie-Hellman）演示会话共享秘密生成；
- 手写 TLS 1.2 风格 `PRF(HMAC-SHA256)` 推导 `master secret` 与流量密钥；
- 实现 `Finished` 校验数据推导；
- 实现“加密 + 完整性校验 + 篡改检测”的记录层。

## R03

TLS 协议可抽象为三层：
1. 握手层（Handshake）：交换随机数与密钥协商参数，导出会话密钥；
2. 记录层（Record）：把应用数据分片并做机密性/完整性保护；
3. 告警层（Alert）：在校验失败或协商失败时中止连接。

本 MVP 覆盖前两层的核心算法路径。

## R04

问题定义（单次会话）：
- 输入：客户端随机数 `ClientRandom`、服务端随机数 `ServerRandom`、双方 DH 公私钥；
- 过程：
  - 由 DH 共享秘密导出 `pre-master`；
  - 用 `PRF` 生成 `master secret`；
  - 继续扩展为双向 MAC/ENC/IV 密钥；
  - 用 `Finished` 校验握手摘要；
  - 用记录层保护应用数据。
- 输出：可用于双向传输的会话密钥与成功校验标记。

## R05

MVP 使用的密码原语：
- 密钥协商：有限域临时 DH（每次握手新私钥）；
- 哈希与 MAC：`SHA-256`、`HMAC-SHA256`；
- PRF：`TLS1.2 P_hash` 结构；
- 记录保护：教学型流加密（SHA256 生成密钥流 XOR）+ HMAC 标签。

注意：记录加密部分仅用于演示算法流，不等价于生产 TLS 套件（如 AES-GCM/ChaCha20-Poly1305）。

## R06

握手消息流（简化版）：
1. ClientHello：发送 `client_random` 与能力声明；
2. ServerHello：发送 `server_random`；
3. KeyShare：双方交换 DH 公钥；
4. 双方本地计算 DH 共享秘密；
5. 通过 PRF 导出 `master secret` 与密钥块；
6. 双方计算并校验 `Finished verify_data`；
7. 进入记录层，开始加密应用数据。

## R07

核心伪代码：

```text
shared = DH(peer_public, local_private)
master = PRF(shared, "master secret", client_random || server_random, 48)
key_block = PRF(master, "key expansion", server_random || client_random, L)
(c_mac, s_mac, c_key, s_key, c_iv, s_iv) = split(key_block)

transcript_hash = SHA256(handshake_messages)
client_finished = PRF(master, "client finished", transcript_hash, 12)
server_finished = PRF(master, "server finished", transcript_hash, 12)

record_tag = HMAC(write_mac_key, seq || len || plaintext)
ciphertext = StreamXor(write_enc_key, write_iv, seq, plaintext || record_tag)
```

## R08

复杂度（设 DH 模幂代价为 `M`，PRF 输出长度为 `n`）：
- 握手：
  - DH 共享秘密计算 2 次，主导成本约 `O(M)`；
  - PRF 与哈希均是线性字节处理，约 `O(n)`；
- 记录层：
  - 每条记录 `HMAC + XOR`，时间 `O(len(record))`；
  - 空间为线性缓冲，`O(len(record))`。

## R09

可验证安全性质（在本 MVP 范围内）：
- 双方共享秘密一致性检查；
- `Finished` 能绑定握手 transcript，避免握手内容被静默改写；
- 记录层 MAC 可检测篡改（篡改后解密校验失败）。

未覆盖或简化部分：
- 证书链验证与 PKI 信任锚；
- 真实 AEAD 套件与重放窗口控制；
- 抗侧信道、抗降级、完整握手状态机细节。

## R10

`demo.py` 关键数据结构：
- `TrafficKeys`：单方向 `mac_key/enc_key/iv`；
- `TLSSession`：保存随机数、`master_secret`、双向流量密钥、握手摘要；
- 函数分层：
  - 原语层：`hmac_sha256`、`tls12_prf`、`stream_xor_encrypt`；
  - 握手层：`perform_handshake`；
  - 记录层：`protect_record`、`unprotect_record`。

## R11

运行方式（无交互）：

```bash
cd Algorithms/计算机-密码学-0290-TLS／SSL协议
uv run python demo.py
```

脚本会自动完成多轮握手、双向应用数据传输以及篡改检测测试。

## R12

预期输出要点：
- 握手运行次数与时间统计（均值/方差）；
- `master secret` 前缀与握手摘要；
- 客户端到服务端、服务端到客户端记录长度；
- `Application data exchange: PASS`；
- `Tamper detection: True`；
- 末尾 `All TLS/SSL MVP checks passed.`。

## R13

内置验证项：
1. 双方 DH 共享秘密必须一致；
2. 客户端/服务端 `master secret` 必须一致；
3. 双方从同一握手上下文导出的 `Finished` 必须一致；
4. 双向记录解密后明文必须与发送方原文一致；
5. 人工翻转密文字节后必须触发 MAC 校验失败。

## R14

边界与异常处理：
- 公钥范围非法直接拒绝；
- XOR 输入长度不一致直接抛错；
- 密文长度小于标签长度时拒绝；
- `Finished` 或 `master secret` 不一致立刻终止；
- 任一校验失败都抛异常而非静默容错。

## R15

TLS/SSL 版本关系与工程结论：
- `SSL` 已不安全且不应启用；
- `TLS 1.2` 仍广泛部署，`TLS 1.3` 是当前推荐基线；
- 本实现更贴近 `TLS 1.2` 的“PRF + key_block + Finished”思路；
- 真实工程建议优先启用标准库 `TLS 1.3`，仅在兼容需求下保留 `TLS 1.2`。

## R16

MVP 取舍说明：
- 优先“流程可见、可验证”而非“密码学最强”；
- 不依赖第三方加密框架黑盒 API（除 Python 标准原语）；
- 保持单文件可运行，便于教学与审计；
- 明确声明非生产实现，避免误用。

## R17

可扩展方向：
1. 将记录层替换为标准 AEAD（AES-GCM 或 ChaCha20-Poly1305）；
2. 加入证书链验证与主机名校验；
3. 增加会话票据/恢复机制；
4. 扩展到 TLS 1.3 风格密钥日程（HKDF-Extract/Expand）；
5. 增加重放防护与序号窗口测试。

## R18

`demo.py` 源码级算法流程拆解（9 步）：
1. `perform_handshake` 生成双方随机数与临时 DH 密钥对。  
2. 双方各自调用 `dh_shared_secret`，得到共享秘密并做一致性比较。  
3. 调用 `derive_master_secret`，内部执行 `tls12_prf(..., "master secret", ...)` 生成 48 字节主密钥。  
4. 调用 `derive_traffic_keys`，用 `tls12_prf(..., "key expansion", ...)` 生成 key block，并切分出 `c_mac/s_mac/c_key/s_key/c_iv/s_iv`。  
5. `build_transcript` 组装握手消息字节串，`finished_verify_data` 对其哈希后再次走 PRF，生成客户端和服务端 `verify_data`。  
6. 双方交叉比较 `Finished` 值；任何不一致立即抛异常终止握手。  
7. 进入记录层时，`protect_record` 先计算 `HMAC(seq||len||plaintext)`，再把 `plaintext||tag` 用密钥流 XOR 加密。  
8. 接收方 `unprotect_record` 先解密恢复负载，再重算 HMAC 与收到标签做常数时间比较，验证完整性。  
9. `main` 在成功收发后构造篡改密文，验证 `unprotect_record` 必然拒绝，从而证明完整性检测链路有效。  
