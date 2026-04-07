# Enigma机

- UID: `CS-0158`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `303`
- 目标目录: `Algorithms/计算机-密码学-0303-Enigma机`

## R01

**问题定义**：实现一个可运行的 Enigma 机最小原型，完成明文加密与同配置解密恢复。  
**本题目标**：不用黑箱密码库，直接实现转子、反射器、插线板、步进机制。  
**输入输出**：输入字符串（允许非字母字符），输出等长密文；重置初始状态后再次处理密文应得到原文（大写形态）。

## R02

Enigma（以三转子 Enigma I 为例）可视为若干置换的串联：
- 插线板（Plugboard）：字母对换置换；
- 右/中/左转子（Rotors）：随按键步进而变化的置换；
- 反射器（Reflector）：固定互换置换，保证加密与解密同构。

单次按键路径：
`Plugboard -> Right -> Middle -> Left -> Reflector -> Left^-1 -> Middle^-1 -> Right^-1 -> Plugboard`。

## R03

可把每次按键的映射写成：

`E_t = P · R_t · M_t · L_t · U · L_t^-1 · M_t^-1 · R_t^-1 · P`

其中：
- `P` 为插线板置换；
- `R_t/M_t/L_t` 为时刻 `t` 的转子置换（包含位置与环位偏移）；
- `U` 为反射器置换，满足 `U = U^-1`；
- 因为每个组件都可逆，整体 `E_t` 也可逆，所以同配置可反向恢复。

## R04

本实现采用 Enigma 常见**双步进（double-stepping）**规则：
- 右转子每次按键都步进；
- 若右转子在其缺口位（notch），中转子本次也步进；
- 若中转子在其缺口位，中转子与左转子本次都步进。

这意味着中转子在某些相邻按键中会连续步进两次，是 Enigma 关键机械特征之一。

## R05

MVP 主流程：
1. 构造转子（接线表、缺口位、环位）、反射器、插线板。
2. 设置三转子初始窗口位置（如 `MCK`）。
3. 逐字符处理文本：字母字符触发一次步进与映射，非字母原样保留。
4. 先经插线板，再经正向转子链、反射器、反向转子链，最后再过插线板。
5. 拼接得到密文。
6. 解密时重置到同一初始位置，再处理密文即可恢复明文（大写）。

## R06

简化伪代码：

```text
process(text):
  out = []
  for ch in text:
    if ch is letter:
      step_rotors()                 # 包含双步进逻辑
      x = plugboard(ch)
      x = right.forward(x)
      x = middle.forward(x)
      x = left.forward(x)
      x = reflector(x)
      x = left.backward(x)
      x = middle.backward(x)
      x = right.backward(x)
      x = plugboard(x)
      out.append(x)
    else:
      out.append(ch)
  return ''.join(out)
```

## R07

正确性要点：
- **可逆性**：每个部件都是双射，组合仍是双射；
- **对称性**：反射器使得“同机同参再次处理”可反解；
- **状态一致性**：解密前必须恢复与加密完全一致的初始转子位置；
- **位置偏移一致性**：正向与反向编码都使用 `position/ring_setting` 的同一偏移规则。

## R08

复杂度分析（`n` 为输入长度）：
- 时间复杂度：`O(n)`，每个字母执行常数次表查和模运算；
- 空间复杂度：`O(n)`（输出缓冲）；
- 组件存储复杂度：`O(1)`（固定 26 字母映射表）。

## R09

`demo.py` 的实现结构：
- `Rotor`：维护接线映射、逆映射、位置、环位、步进与双向编码；
- `Plugboard`：验证并维护字母对换关系；
- `EnigmaMachine`：实现步进逻辑、单字符路径与整串处理；
- `letter_frequency`：用 `numpy` 做密文字母频率统计；
- `build_default_machine`：生成默认 Enigma I 配置；
- `main`：执行加密/解密自检并打印结果。

## R10

本目录默认参数（教学可复现实例）：
- 转子：`I-II-III`；
- 反射器：`Reflector B`；
- 环位：`01-01-01`（代码中为 `ring_setting=0`）；
- 初始位置：`MCK`；
- 插线板：`PO ML IU KJ NH YT GB VF RE DC`。

这些参数用于演示结构，不代表真实历史报文配置。

## R11

边界与异常处理：
- 插线板对长度不为 2、同字母互连、重复占用字母会抛异常；
- 非 `A-Z` 字符不参与步进和映射，原样输出；
- 若转子数量不是 3，会抛异常（当前 MVP 固定三转子）；
- 明文为空时可返回空串，频率向量为全零。

## R12

验证策略：
1. **对称性测试**：`decrypt(encrypt(plaintext)) == plaintext.upper()`；
2. **稳定性测试**：固定配置与初始位置可复现同一密文；
3. **统计展示**：输出密文字母 Top-5 频率，确认流程运行真实完成；
4. **自动失败机制**：对称性不成立时直接 `RuntimeError`。

## R13

预期输出包含：
- 配置摘要（起始位置）；
- 明文、密文、解密结果三行；
- Top-5 字母频率；
- `All checks passed.` 收尾标志。

这证明脚本在无交互条件下可完整执行。

## R14

历史与安全说明：
- Enigma 在二战中被广泛使用，但并非现代意义上的安全密码；
- 其弱点来自机械结构、操作流程与密钥管理问题（非单纯“数学算法强度”）；
- 本项目定位为“经典密码机理教学”，不是现代生产加密方案。

## R15

与现代密码体制差异：
- Enigma 属于机械多表代换体系；现代密码通常是分组密码/流密码 + 严格安全证明框架；
- Enigma 无现代随机化语义安全模型；
- Enigma 的安全高度依赖操作纪律，而现代方案更强调形式化安全与协议组合。

## R16

MVP 范围与限制：
- 仅覆盖三转子 Enigma I 的核心机制；
- 未实现日密钥表、消息键流程、人工操作规程等历史细节；
- 目前面向可读性与教学，不追求密码分析完整实验平台。

## R17

运行方式（仓库根目录）：

```bash
uv run python Algorithms/计算机-密码学-0303-Enigma机/demo.py
```

也可在本目录运行：

```bash
uv run python demo.py
```

无交互输入，运行后直接输出结果与自检状态。

## R18

`demo.py` 的源码级算法流（8 步）：

1. `main()` 调用 `build_default_machine()` 组装转子 `I/II/III`、反射器 `B`、插线板与默认环位。  
2. `main()` 设定起始窗口 `MCK`，调用 `machine.process(plaintext)` 进入逐字符处理。  
3. `process()` 对每个字母先调用 `_step_rotors()`：按“右转子必进 + 缺口触发中/左转子”执行双步进判定。  
4. `_encode_letter()` 先执行插线板置换 `plugboard.swap()`，把输入字母转成索引。  
5. 索引沿 `right -> middle -> left` 依次调用 `Rotor.encode_forward()`，每个转子内部应用 `position/ring_setting` 偏移后查线。  
6. 经过 `reflector` 固定互换后，沿 `left -> middle -> right` 调用 `Rotor.encode_backward()` 执行逆向映射。  
7. 输出端再次过插线板得到最终密文字母；非字母字符在 `process()` 中直接保留。  
8. `main()` 将机器重置到同一 `MCK`，再次 `process(ciphertext)` 得到解密文本，并用 `letter_frequency()`（`numpy`）统计密文频率完成结果展示与自检。
