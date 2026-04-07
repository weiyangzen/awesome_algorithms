"""Minimal runnable MVP for Beatty sequence and complementarity check."""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal, getcontext


def decimal_floor(value: Decimal) -> int:
    """Return floor(value) using Decimal rounding mode."""
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def beatty_prefix(alpha: Decimal, n: int) -> list[int]:
    """Generate first n terms of Beatty sequence floor(k*alpha), k=1..n."""
    if alpha <= Decimal(1):
        raise ValueError("alpha must be > 1")
    if n < 0:
        raise ValueError("n must be non-negative")

    return [decimal_floor(Decimal(k) * alpha) for k in range(1, n + 1)]


def complementary_beta(alpha: Decimal) -> Decimal:
    """Compute beta = alpha/(alpha-1), the Beatty complementary parameter."""
    if alpha <= Decimal(1):
        raise ValueError("alpha must be > 1")
    return alpha / (alpha - Decimal(1))


def beatty_upto(alpha: Decimal, limit: int) -> list[int]:
    """Generate all Beatty terms <= limit."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if alpha <= Decimal(1):
        raise ValueError("alpha must be > 1")

    est = decimal_floor(Decimal(limit) / alpha) + 8
    est = max(est, 8)
    seq = beatty_prefix(alpha, est)

    while seq and seq[-1] < limit:
        est *= 2
        seq = beatty_prefix(alpha, est)

    return [x for x in seq if x <= limit]


def verify_complementarity(alpha: Decimal, limit: int) -> dict[str, object]:
    """Verify disjoint+covering property on [1, limit] for alpha and beta."""
    beta = complementary_beta(alpha)
    a_terms = beatty_upto(alpha, limit)
    b_terms = beatty_upto(beta, limit)

    sa = set(a_terms)
    sb = set(b_terms)
    overlap = sorted(sa & sb)
    universe = set(range(1, limit + 1))
    missing = sorted(universe - (sa | sb))

    return {
        "alpha": alpha,
        "beta": beta,
        "limit": limit,
        "a_terms": a_terms,
        "b_terms": b_terms,
        "is_disjoint": len(overlap) == 0,
        "is_covering": len(missing) == 0,
        "overlap_preview": overlap[:10],
        "missing_preview": missing[:10],
    }


def run_demo_cases() -> None:
    """Run deterministic non-interactive demo cases."""
    getcontext().prec = 80

    sqrt2 = Decimal(2).sqrt()
    phi = (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)

    cases: list[tuple[str, Decimal, int]] = [
        ("sqrt(2)", sqrt2, 500),
        ("golden_ratio_phi", phi, 800),
    ]

    total = 0
    passed = 0

    print("=== Beatty Sequence Demo ===")
    for name, alpha, limit in cases:
        report = verify_complementarity(alpha, limit)
        beta = report["beta"]
        a_terms = report["a_terms"]
        b_terms = report["b_terms"]
        is_disjoint = bool(report["is_disjoint"])
        is_covering = bool(report["is_covering"])
        ok = is_disjoint and is_covering

        total += 1
        passed += int(ok)

        print(f"\nCase: {name}")
        print(f"alpha ≈ {alpha}")
        print(f"beta  ≈ {beta}")
        print(f"A first 15: {a_terms[:15]}")
        print(f"B first 15: {b_terms[:15]}")
        print(
            "check[1..{limit}] -> disjoint={disjoint}, covering={covering}, "
            "missing={missing}, overlap={overlap} | {status}".format(
                limit=limit,
                disjoint=is_disjoint,
                covering=is_covering,
                missing=report["missing_preview"],
                overlap=report["overlap_preview"],
                status="PASS" if ok else "FAIL",
            )
        )

    print(f"\nSummary: {passed}/{total} cases passed.")
    if passed != total:
        raise RuntimeError("Beatty complementarity demo failed on at least one case")


def main() -> None:
    run_demo_cases()


if __name__ == "__main__":
    main()
