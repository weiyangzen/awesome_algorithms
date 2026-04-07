#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "Docs/Stage0_Blueprint.md"
SOURCE_PATHS = {
    "math": ROOT / "Docs/researches/top_500_math_algorithms.md",
    "cs": ROOT / "Docs/researches/top_500_cs_algorithms.md",
    "physics": ROOT / "Docs/researches/physics_top500_algorithms.md",
}


DELIVERABLES = [
    ("R01", "算法内容", "给出核心思想、输入输出、主要步骤和关键机制。"),
    ("R02", "提出年代时间", "记录提出年份、关键论文或提出者时间点。"),
    ("R03", "提出背景", "说明它解决了什么历史问题、前人方案为何不足。"),
    ("R04", "时间复杂度分析", "给出最坏、平均、摊还或典型复杂度，并说明条件。"),
    ("R05", "空间复杂度分析", "给出额外空间、工作内存、状态规模或存储开销。"),
    ("R06", "算法示例", "给出一个可复现实例，最好包含输入、中间过程和输出。"),
    ("R07", "算法意义", "说明其理论地位、工程价值或学科影响。"),
    ("R08", "算法路径上依赖的其他算法", "列出直接依赖的先导算法、定理或数值工具。"),
    (
        "R09",
        "适用前提与边界条件",
        "说明成立假设、输入限制、适用域，以及不适用的场景。",
    ),
    (
        "R10",
        "正确性依据",
        "说明不变量、证明思路、核心定理、收敛性或最优性依据。",
    ),
    (
        "R11",
        "数值稳定性与误差传播",
        "说明误差来源、病态性、舍入误差传播，以及稳定性结论。",
    ),
    (
        "R12",
        "真实计算成本",
        "除渐近复杂度外，分析常数项、I/O、缓存、并行化、GPU 或向量化成本。",
    ),
    (
        "R13",
        "近似比/误差界/概率保证",
        "若算法为近似、随机或统计型，记录误差上界、近似比或成功概率。",
    ),
    (
        "R14",
        "鲁棒性与失效模式",
        "说明极端输入、异常参数、病态数据或错误配置下会如何失效。",
    ),
    (
        "R15",
        "工程实现要点",
        "说明数据结构选择、初始化、停止准则、调参点和常见实现陷阱。",
    ),
    (
        "R16",
        "前驱/后继算法与典型应用",
        "说明它继承了谁、后来被谁扩展，以及典型应用场景或 benchmark。",
    ),
    (
        "R17",
        "Python MVP 实现方案",
        "为该算法设计一个以 Python 为主的最小可运行演示；优先使用 numpy、scipy、pandas、scikit-learn。",
    ),
    (
        "R18",
        "源码追踪与 3-10 步算法拆解",
        "若第三方包调用可一两行完成，必须追到源码或核心实现，并把算法拆成 3-10 个合理步骤。",
    ),
]


@dataclass
class Item:
    discipline: str
    code_prefix: str
    subcategory: str
    number: int
    name: str
    description: str
    importance: str
    source: str
    original_category: str
    uid: str = ""


def clean_top_heading(text: str) -> str:
    text = re.sub(r"^\d+\.\s*", "", text).strip()
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text


def normalize_subcat(text: str) -> str:
    text = text.strip().replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", " ", text)
    return text.rstrip("：:")


def short_top_label(text: str) -> str:
    return re.sub(r"算法$", "", clean_top_heading(text)).strip()


def parse_math(path: Path) -> list[Item]:
    items: list[Item] = []
    current_section: str | None = None
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m_sec = re.match(r"^##\s+(.+)$", s)
        if m_sec:
            section = clean_top_heading(m_sec.group(1))
            if section not in {"目录", "统计概览", "附录：算法按分支索引", "总结"}:
                current_section = section
            i += 1
            continue

        m = re.match(r"^###\s+(\d+)\.\s+(.+)$", s)
        if m:
            number = int(m.group(1))
            name = m.group(2).strip()
            description = ""
            branch = ""
            importance = ""
            j = i + 1
            while j < len(lines):
                t = lines[j].strip()
                if re.match(r"^(###\s+\d+\.|##\s+)", t):
                    break
                md = re.match(r"^-\s+\*\*描述\*\*:\s*(.+)$", t)
                mb = re.match(r"^-\s+\*\*所属分支\*\*:\s*(.+)$", t)
                mi = re.match(r"^-\s+\*\*重要性\*\*:\s*(.+)$", t)
                if md:
                    description = md.group(1).strip()
                if mb:
                    branch = mb.group(1).strip()
                if mi:
                    importance = mi.group(1).strip()
                j += 1

            subcat = normalize_subcat(branch.split("/")[0] if branch else current_section or "未分类")
            original_category = " / ".join([x for x in [current_section, branch] if x])
            items.append(
                Item(
                    discipline="数学",
                    code_prefix="MATH",
                    subcategory=subcat,
                    number=number,
                    name=name,
                    description=description,
                    importance=importance,
                    source=str(path.relative_to(ROOT)),
                    original_category=original_category,
                )
            )
            i = j
            continue
        i += 1
    return items


def parse_cs(path: Path) -> list[Item]:
    items: list[Item] = []
    current_top: str | None = None
    current_sub: str | None = None
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m_top = re.match(r"^##\s+(.+)$", s)
        if m_top:
            top = clean_top_heading(m_top.group(1))
            if top not in {"目录", "附录：算法分支统计", "重要性等级说明", "算法应用领域速查"}:
                current_top = top
                current_sub = None
            i += 1
            continue

        m_sub = re.match(r"^###\s+(.+)$", s)
        if m_sub:
            current_sub = normalize_subcat(m_sub.group(1))
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+\*\*(.+?)\*\*\s*$", s)
        if m:
            number = int(m.group(1))
            name = m.group(2).strip()
            description = ""
            importance = ""
            j = i + 1
            while j < len(lines):
                t = lines[j].strip()
                if re.match(r"^(\d+\.\s+\*\*|##\s+|###\s+)", t):
                    break
                md = re.match(r"^-\s+\*\*描述\*\*:\s*(.+)$", t)
                mi = re.match(r"^-\s+\*\*重要性\*\*:\s*(.+)$", t)
                if md:
                    description = md.group(1).strip()
                if mi:
                    importance = mi.group(1).strip()
                j += 1

            if number == 201 and "Deque" in name:
                subcat = "数据结构算法"
            elif current_top == "核心基础算法" and current_sub:
                subcat = current_sub
            else:
                subcat = short_top_label(current_top or "未分类")

            original_category = " / ".join([x for x in [current_top, current_sub] if x])
            items.append(
                Item(
                    discipline="计算机",
                    code_prefix="CS",
                    subcategory=subcat,
                    number=number,
                    name=name,
                    description=description,
                    importance=importance,
                    source=str(path.relative_to(ROOT)),
                    original_category=original_category,
                )
            )
            i = j
            continue
        i += 1
    return items


def parse_physics(path: Path) -> list[Item]:
    items: list[Item] = []
    current_section: str | None = None
    current_subheading: str | None = None
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m_sec = re.match(r"^##\s+(.+)$", s)
        if m_sec:
            section = clean_top_heading(m_sec.group(1))
            if section not in {"总结", "附录：按分支分类索引"}:
                current_section = section
                current_subheading = None
            i += 1
            continue

        m_sub = re.match(r"^###\s+(.+)$", s)
        if m_sub:
            current_subheading = normalize_subcat(m_sub.group(1))
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+\*\*(.+?)\*\*\s*$", s)
        if m:
            number = int(m.group(1))
            name = m.group(2).strip()
            description = ""
            branch = ""
            importance = ""
            j = i + 1
            while j < len(lines):
                t = lines[j].strip()
                if re.match(r"^(\d+\.\s+\*\*|##\s+|###\s+)", t):
                    break
                md = re.match(r"^-\s+\*\*描述\*\*:\s*(.+)$", t)
                mb = re.match(r"^-\s+\*\*分支\*\*:\s*(.+)$", t)
                mi = re.match(r"^-\s+\*\*重要性\*\*:\s*(.+)$", t)
                if md:
                    description = md.group(1).strip()
                if mb:
                    branch = mb.group(1).strip()
                if mi:
                    importance = mi.group(1).strip()
                j += 1

            subcat = normalize_subcat(
                branch.split("/")[0] if branch else current_subheading or current_section or "未分类"
            )
            original_category = " / ".join([x for x in [current_section, current_subheading, branch] if x])
            items.append(
                Item(
                    discipline="物理",
                    code_prefix="PHYS",
                    subcategory=subcat,
                    number=number,
                    name=name,
                    description=description,
                    importance=importance,
                    source=str(path.relative_to(ROOT)),
                    original_category=original_category,
                )
            )
            i = j
            continue
        i += 1
    return items


def collect_items() -> list[Item]:
    items: list[Item] = []
    items.extend(parse_math(SOURCE_PATHS["math"]))
    items.extend(parse_cs(SOURCE_PATHS["cs"]))
    items.extend(parse_physics(SOURCE_PATHS["physics"]))

    counters: defaultdict[str, int] = defaultdict(int)
    for item in items:
        counters[item.code_prefix] += 1
        item.uid = f"{item.code_prefix}-{counters[item.code_prefix]:04d}"

    return items


def build_blueprint(items: list[Item]) -> str:
    counts = Counter(item.discipline for item in items)
    subcounts = Counter((item.discipline, item.subcategory) for item in items)

    subcategory_order: dict[str, list[str]] = defaultdict(list)
    seen_subcats: set[tuple[str, str]] = set()
    for item in items:
        key = (item.discipline, item.subcategory)
        if key not in seen_subcats:
            seen_subcats.add(key)
            subcategory_order[item.discipline].append(item.subcategory)

    lines: list[str] = []
    lines.append("# Stage0 Blueprint")
    lines.append("")
    lines.append("> 状态: Authoritative Blueprint")
    lines.append(f"> 生成日期: {date.today().isoformat()}")
    lines.append("> 作用: 本文件是 Stage 0 的唯一 requirement source；如果后续引入 execution cron，只允许从本文件派生 todo。")
    lines.append("> 结构策略: 只保留算法级 checklist；学科、子分类和研究模板都作为非-checklist 结构信息存在。")
    lines.append("> 来源: `Docs/researches/top_500_math_algorithms.md`、`Docs/researches/top_500_cs_algorithms.md`、`Docs/researches/physics_top500_algorithms.md`")
    lines.append("")
    lines.append("## 使用规则")
    lines.append("")
    lines.append("- 本蓝图严格按 `学科 -> 子分类` 两级结构组织。")
    lines.append("- 每个 `[ ]` 项对应当前源文档中的一个算法条目；跨学科同名算法、或同一学科内因上下文不同而重复出现的同名条目，不在本阶段自动合并。")
    lines.append("- 子分类、统计表、研究模板、完成标准都不是 checklist item；真正可勾选的只有算法条目本身。")
    lines.append("- 每个算法条目统一引用 `R01-R18` 研究模板，避免在 1400+ 条清单中重复展开长字段。")
    lines.append("- 每个算法后续演示默认要有 `Python MVP` 路径；首选生态是 `numpy`、`scipy`、`pandas`、`scikit-learn`。")
    lines.append("- 如果某个第三方包通过少量函数调用就能完成任务，也不能把该包当成黑箱；必须补 `R18`，把源码或核心实现路径追出来，并拆成 `3-10` 个算法步骤。")
    lines.append("")
    lines.append("## 完成标准")
    lines.append("")
    lines.append("- 只有当 `R01-R18` 全部补齐且内部表述一致时，算法条目才允许从 `[ ]` 改为 `[x]`。")
    lines.append("- `R04-R05` 对纯理论物理条目可解释为“典型计算实现或数值求解复杂度”；如果确实不存在独立算法形式，必须明确写 `N/A + 原因`。")
    lines.append("- `R11-R13` 若不适用，也必须写出 `N/A + 为什么不适用`，不能留空。")
    lines.append("- `R08` 和 `R16` 需要互相校验：前者强调依赖链，后者强调谱系位置和应用落点。")
    lines.append("- `R17` 需要给出最小可运行演示的实现边界、输入输出和依赖；原则上优先用 `numpy`、`scipy`、`pandas`、`scikit-learn` 完成。")
    lines.append("- 如果 `R17` 主要依赖外部包封装，`R18` 必须回到源码和算法本身，把关键流程拆成 `3-10` 步；只写“调用某函数”不算完成。")
    lines.append("")
    lines.append("## 推荐研究顺序")
    lines.append("")
    lines.append("- 第一层 `R01-R03`: 先把对象定义、时代位置、提出背景弄清。")
    lines.append("- 第二层 `R04-R08`: 再补复杂度、示例、意义、依赖链。")
    lines.append("- 第三层 `R09-R14`: 再补边界条件、正确性、误差、真实成本、保证、失效模式。")
    lines.append("- 第四层 `R15-R16`: 再补工程实现、谱系关系、应用和 benchmark。")
    lines.append("- 第五层 `R17-R18`: 最后给出 Python MVP 方案，并在必要时完成源码追踪和 `3-10` 步算法拆解。")
    lines.append("")
    lines.append("## 研究模板")
    lines.append("")
    lines.append("| 编码 | 字段 | 说明 |")
    lines.append("| --- | --- | --- |")
    for code, field, desc in DELIVERABLES:
        lines.append(f"| {code} | {field} | {desc} |")
    lines.append("")
    lines.append("## 源快照")
    lines.append("")
    lines.append(f"- 当前仓库快照实际可解析条目总数: `{len(items)}`")
    lines.append(f"- 数学: `{counts['数学']}`")
    lines.append(f"- 计算机: `{counts['计算机']}`")
    lines.append(f"- 物理: `{counts['物理']}`")
    lines.append("- 结构异常: `Docs/researches/top_500_cs_algorithms.md` 的目录宣称 500 项，但当前文件正文仅能解析出 410 个算法条目，编号 `111-200` 在文件正文中缺失。")
    lines.append("- 本蓝图不补造缺失算法，只收录当前仓库里实际存在且可解析的条目。")
    lines.append("")
    lines.append("## 子分类统计")
    lines.append("")
    lines.append("| 学科 | 子分类 | 条目数 |")
    lines.append("| --- | --- | ---: |")
    for discipline in ["数学", "计算机", "物理"]:
        for subcat in subcategory_order[discipline]:
            lines.append(f"| {discipline} | {subcat} | {subcounts[(discipline, subcat)]} |")
    lines.append("")
    lines.append("## 权威清单")
    lines.append("")

    for discipline in ["数学", "计算机", "物理"]:
        lines.append(f"## {discipline}")
        lines.append("")
        for subcat in subcategory_order[discipline]:
            scoped = [item for item in items if item.discipline == discipline and item.subcategory == subcat]
            scoped.sort(key=lambda item: item.number)
            lines.append(f"### {subcat} ({len(scoped)})")
            lines.append("")
            lines.append(f"- 研究模板: `R01-R18`")
            lines.append("- MVP 约束: 优先使用 `numpy`、`scipy`、`pandas`、`scikit-learn`；如需其他包，也必须保留算法解释权。")
            lines.append("- 拆解约束: 遇到黑箱包实现时，必须追源码并整理为 `3-10` 步。")
            lines.append(f"- 本子分类条目数: `{len(scoped)}`")
            lines.append("")
            for item in scoped:
                lines.append(f"- [ ] [{item.uid} | 源序号 {item.number}] {item.name}")
                lines.append(f"  - 来源: `{item.source}`")
                if item.original_category:
                    lines.append(f"  - 原始类目: `{item.original_category}`")
                if item.description:
                    lines.append(f"  - 现有摘要: {item.description}")
                if item.importance:
                    lines.append(f"  - 重要性: {item.importance}")
                lines.append("  - 完成条件: `R01-R18` 全部完成，且 `R08/R16`、`R04/R12`、`R11/R13/R14`、`R17/R18` 四组内容互相一致。")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    items = collect_items()
    OUT_PATH.write_text(build_blueprint(items), encoding="utf-8")
    print(f"wrote {OUT_PATH} with {len(items)} items")


if __name__ == "__main__":
    main()
