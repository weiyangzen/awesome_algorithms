"""van Kampen 定理算法的最小可运行 MVP。

实现目标：
1) 用群表示（generators + relators）表示 pi_1(U), pi_1(V), pi_1(U∩V)。
2) 根据 van Kampen 推前（pushout）规则构造
   pi_1(U∪V) = pi_1(U) *_pi_1(U∩V) pi_1(V)。
3) 输出结果表示，并给出 Abel 化矩阵作为可计算 sanity check。

注意：
- 这里不尝试求解一般群的词问题。
- 示例使用教学友好的小型表示（S^1、8 字形、环面、RP^2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

Letter = Tuple[str, int]
Word = Tuple[Letter, ...]


# ----------------------------
# Word-level utilities
# ----------------------------


def reduce_word(raw_letters: Sequence[Letter]) -> Word:
    """在自由群语境下做相邻同生成元指数合并与约化。"""
    stack: List[Letter] = []
    for gen, exp in raw_letters:
        if exp == 0:
            continue
        if stack and stack[-1][0] == gen:
            merged = stack[-1][1] + exp
            if merged == 0:
                stack.pop()
            else:
                stack[-1] = (gen, merged)
        else:
            stack.append((gen, exp))
    return tuple(stack)


def make_word(*letters: Letter) -> Word:
    return reduce_word(letters)


def identity_word() -> Word:
    return tuple()


def gen_word(name: str, exp: int = 1) -> Word:
    if exp == 0:
        return identity_word()
    return reduce_word(((name, exp),))


def multiply_words(*words: Word) -> Word:
    merged: List[Letter] = []
    for w in words:
        merged.extend(w)
    return reduce_word(merged)


def inverse_word(word: Word) -> Word:
    return reduce_word(tuple((g, -e) for g, e in reversed(word)))


def power_word(word: Word, k: int) -> Word:
    if k == 0:
        return identity_word()
    if k < 0:
        return power_word(inverse_word(word), -k)
    out = identity_word()
    for _ in range(k):
        out = multiply_words(out, word)
    return out


def substitute_word(word: Word, mapping: Dict[str, Word]) -> Word:
    """把一个定义在 domain 生成元上的词，替换到目标群词。"""
    out = identity_word()
    for g, e in word:
        if g not in mapping:
            raise KeyError(f"Missing image for generator '{g}'")
        out = multiply_words(out, power_word(mapping[g], e))
    return out


def word_to_str(word: Word) -> str:
    if not word:
        return "1"
    parts: List[str] = []
    for g, e in word:
        if e == 1:
            parts.append(g)
        else:
            parts.append(f"{g}^{e}")
    return " ".join(parts)


# ----------------------------
# Group presentation
# ----------------------------


@dataclass(frozen=True)
class GroupPresentation:
    generators: Tuple[str, ...]
    relators: Tuple[Word, ...]

    @classmethod
    def from_data(
        cls,
        generators: Sequence[str],
        relators: Sequence[Word],
    ) -> "GroupPresentation":
        uniq_gens: List[str] = []
        seen_gens = set()
        for g in generators:
            if g not in seen_gens:
                seen_gens.add(g)
                uniq_gens.append(g)

        cleaned_relators: List[Word] = []
        seen_relators = set()
        for rel in relators:
            rr = reduce_word(rel)
            if not rr:
                continue
            if rr not in seen_relators:
                seen_relators.add(rr)
                cleaned_relators.append(rr)

        # 基本一致性检查：关系里不能引用未知生成元
        valid = set(uniq_gens)
        for rel in cleaned_relators:
            for g, _ in rel:
                if g not in valid:
                    raise ValueError(f"Relator uses unknown generator '{g}'")

        return cls(tuple(uniq_gens), tuple(cleaned_relators))

    def pretty(self) -> str:
        g_part = ", ".join(self.generators) if self.generators else "∅"
        r_part = ", ".join(word_to_str(r) for r in self.relators) if self.relators else "∅"
        return f"< {g_part} | {r_part} >"


# ----------------------------
# van Kampen pushout
# ----------------------------


def relation_from_identification(left: Word, right: Word) -> Word:
    """把“left = right”转成 relator: left * right^{-1} = 1。"""
    return multiply_words(left, inverse_word(right))


def validate_homomorphism(
    domain: GroupPresentation,
    codomain: GroupPresentation,
    mapping: Dict[str, Word],
    name: str,
) -> None:
    domain_set = set(domain.generators)
    if set(mapping.keys()) != domain_set:
        missing = domain_set - set(mapping.keys())
        extra = set(mapping.keys()) - domain_set
        raise ValueError(f"{name} mapping mismatch, missing={missing}, extra={extra}")

    codomain_set = set(codomain.generators)
    for h, image in mapping.items():
        for g, _ in image:
            if g not in codomain_set:
                raise ValueError(
                    f"{name}: image of '{h}' uses unknown target generator '{g}'"
                )


def van_kampen_pushout(
    g_u: GroupPresentation,
    g_v: GroupPresentation,
    g_inter: GroupPresentation,
    phi_u: Dict[str, Word],
    phi_v: Dict[str, Word],
) -> Tuple[GroupPresentation, List[Word]]:
    """构造推前表示：

    pi_1(U∪V) = <G_u * G_v | rel_u, rel_v, phi_u(h)phi_v(h)^{-1} (h in generators(H))>
    """
    overlap = set(g_u.generators) & set(g_v.generators)
    if overlap:
        raise ValueError(
            "Generator names of U and V must be disjoint in this MVP, "
            f"but got overlap={sorted(overlap)}"
        )

    validate_homomorphism(g_inter, g_u, phi_u, "phi_u")
    validate_homomorphism(g_inter, g_v, phi_v, "phi_v")

    glue_relators: List[Word] = []
    for h in g_inter.generators:
        glue = relation_from_identification(phi_u[h], phi_v[h])
        if glue:
            glue_relators.append(glue)

    all_generators = list(g_u.generators) + list(g_v.generators)
    all_relators = list(g_u.relators) + list(g_v.relators) + glue_relators
    return GroupPresentation.from_data(all_generators, all_relators), glue_relators


# ----------------------------
# Algebraic sanity checks
# ----------------------------


def abelianization_matrix(pres: GroupPresentation) -> np.ndarray:
    """关系矩阵：每个 relator 对每个 generator 的总指数。"""
    m = len(pres.relators)
    n = len(pres.generators)
    mat = np.zeros((m, n), dtype=np.int64)
    index = {g: j for j, g in enumerate(pres.generators)}

    for i, rel in enumerate(pres.relators):
        for g, e in rel:
            mat[i, index[g]] += e
    return mat


def abelianization_free_rank(pres: GroupPresentation) -> Tuple[int, int]:
    mat = abelianization_matrix(pres)
    if mat.size == 0:
        return len(pres.generators), 0
    rank = int(np.linalg.matrix_rank(mat.astype(float)))
    return len(pres.generators) - rank, rank


# ----------------------------
# Demo cases
# ----------------------------


def commutator(a: str, b: str) -> Word:
    return multiply_words(gen_word(a), gen_word(b), gen_word(a, -1), gen_word(b, -1))


def build_cases() -> List[Dict[str, object]]:
    trivial = GroupPresentation.from_data([], [])
    g_a = GroupPresentation.from_data(["a"], [])
    g_b = GroupPresentation.from_data(["b"], [])
    g_ab = GroupPresentation.from_data(["a", "b"], [])
    g_t = GroupPresentation.from_data(["t"], [])

    return [
        {
            "case": "S^1 由两个可缩开弧覆盖",
            "u": trivial,
            "v": trivial,
            "inter": trivial,
            "phi_u": {},
            "phi_v": {},
            "expected": "平凡群",
            "topology_note": "U,V,U∩V 都可缩，故 pi1(U∪V)=1。",
        },
        {
            "case": "8 字形（两个圆在一点楔合）",
            "u": g_a,
            "v": g_b,
            "inter": trivial,
            "phi_u": {},
            "phi_v": {},
            "expected": "自由群 F2=<a,b>",
            "topology_note": "交集同伦等价一点，推前退化为自由积。",
        },
        {
            "case": "环面 T^2（1-骨架 + 2-胞腔）",
            "u": g_ab,
            "v": trivial,
            "inter": g_t,
            "phi_u": {"t": commutator("a", "b")},
            "phi_v": {"t": identity_word()},
            "expected": "<a,b | [a,b]>，即 Z^2",
            "topology_note": "附着映射把边界圈送到交换子，得到交换关系。",
        },
        {
            "case": "实射影平面 RP^2（1-骨架 + 2-胞腔）",
            "u": g_a,
            "v": trivial,
            "inter": g_t,
            "phi_u": {"t": power_word(gen_word("a"), 2)},
            "phi_v": {"t": identity_word()},
            "expected": "<a | a^2>，即 Z/2Z",
            "topology_note": "附着映射是 2 倍绕行，得到 a^2=1。",
        },
    ]


def run_case(case: Dict[str, object]) -> Dict[str, object]:
    name = case["case"]
    g_u = case["u"]
    g_v = case["v"]
    g_inter = case["inter"]
    phi_u = case["phi_u"]
    phi_v = case["phi_v"]

    assert isinstance(name, str)
    assert isinstance(g_u, GroupPresentation)
    assert isinstance(g_v, GroupPresentation)
    assert isinstance(g_inter, GroupPresentation)
    assert isinstance(phi_u, dict)
    assert isinstance(phi_v, dict)

    result, glue_relators = van_kampen_pushout(g_u, g_v, g_inter, phi_u, phi_v)
    free_rank, relation_rank = abelianization_free_rank(result)
    mat = abelianization_matrix(result)

    print(f"\n=== {name} ===")
    print(f"拓扑说明: {case['topology_note']}")
    print(f"U 的表示: {g_u.pretty()}")
    print(f"V 的表示: {g_v.pretty()}")
    print(f"U∩V 的表示: {g_inter.pretty()}")

    if glue_relators:
        print("粘合关系 (phi_u(h)=phi_v(h) -> relator):")
        for i, rel in enumerate(glue_relators, start=1):
            print(f"  g{i}: {word_to_str(rel)} = 1")
    else:
        print("粘合关系: 无（交集群生成元为空）")

    print(f"推导得到 pi1(U∪V) 表示: {result.pretty()}")
    print(f"期望结论: {case['expected']}")

    if mat.size:
        df_mat = pd.DataFrame(
            mat,
            columns=list(result.generators),
            index=[f"r{i+1}" for i in range(mat.shape[0])],
        )
        print("Abel 化关系矩阵（行=关系，列=生成元指数和）:")
        print(df_mat.to_string())
    else:
        print("Abel 化关系矩阵: 空矩阵")

    print(
        "Abel 化自由秩(近似检查): "
        f"rank(H1_free)={free_rank}, rank(rel_matrix)={relation_rank}"
    )

    return {
        "case": name,
        "presentation": result.pretty(),
        "n_generators": len(result.generators),
        "n_relators": len(result.relators),
        "abel_free_rank": free_rank,
        "expected": case["expected"],
    }


def main() -> None:
    rows: List[Dict[str, object]] = []
    for case in build_cases():
        rows.append(run_case(case))

    summary = pd.DataFrame(rows)
    print("\n=== 汇总表 ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
