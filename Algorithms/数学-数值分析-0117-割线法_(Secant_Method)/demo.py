"""Minimal runnable MVP for Secant Method (MATH-0117)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, List, Optional


@dataclass
class IterationRecord:
    iteration: int
    x_prev: float
    x_curr: float
    x_next: float
    f_prev: float
    f_curr: float
    f_next: float
    step: float


@dataclass
class SecantResult:
    root: float
    f_root: float
    iterations: int
    converged: bool
    reason: str
    history: List[IterationRecord]


def secant_method(
    func: Callable[[float], float],
    x0: float,
    x1: float,
    tol_x: float = 1e-12,
    tol_f: float = 1e-12,
    max_iter: int = 80,
    tiny: float = 1e-15,
) -> SecantResult:
    """Find a root of f(x)=0 via secant iterations.

    Stopping criteria:
    - residual: |f(x_k)| <= tol_f
    - relative step: |x_k - x_{k-1}| <= tol_x * (1 + |x_k|)
    """
    history: List[IterationRecord] = []

    f0 = func(x0)
    f1 = func(x1)
    if not (math.isfinite(f0) and math.isfinite(f1)):
        return SecantResult(
            root=x1,
            f_root=f1,
            iterations=0,
            converged=False,
            reason="non-finite initial function value",
            history=history,
        )

    for k in range(1, max_iter + 1):
        denom = f1 - f0
        if abs(denom) <= tiny:
            return SecantResult(
                root=x1,
                f_root=f1,
                iterations=k - 1,
                converged=False,
                reason="denominator too small (f_k - f_{k-1} ~ 0)",
                history=history,
            )

        x2 = x1 - f1 * (x1 - x0) / denom
        f2 = func(x2)

        record = IterationRecord(
            iteration=k,
            x_prev=x0,
            x_curr=x1,
            x_next=x2,
            f_prev=f0,
            f_curr=f1,
            f_next=f2,
            step=abs(x2 - x1),
        )
        history.append(record)

        if not math.isfinite(f2):
            return SecantResult(
                root=x2,
                f_root=f2,
                iterations=k,
                converged=False,
                reason="non-finite function value encountered",
                history=history,
            )

        if abs(f2) <= tol_f:
            return SecantResult(
                root=x2,
                f_root=f2,
                iterations=k,
                converged=True,
                reason="residual tolerance reached",
                history=history,
            )

        if record.step <= tol_x * (1.0 + abs(x2)):
            return SecantResult(
                root=x2,
                f_root=f2,
                iterations=k,
                converged=True,
                reason="step tolerance reached",
                history=history,
            )

        x0, x1 = x1, x2
        f0, f1 = f1, f2

    return SecantResult(
        root=x1,
        f_root=f1,
        iterations=max_iter,
        converged=False,
        reason="max iterations reached",
        history=history,
    )


def estimate_order(result: SecantResult, true_root: Optional[float]) -> Optional[float]:
    """Estimate local convergence order p from error triples.

    p_k = log(e_{k+1}/e_k) / log(e_k/e_{k-1}), where e_k = |x_k - r|.
    """
    if true_root is None or len(result.history) < 3:
        return None

    errors = [abs(rec.x_next - true_root) for rec in result.history]
    p_values: List[float] = []

    for i in range(2, len(errors)):
        e_im1 = errors[i - 2]
        e_i = errors[i - 1]
        e_ip1 = errors[i]
        if e_im1 <= 0.0 or e_i <= 0.0 or e_ip1 <= 0.0:
            continue
        denom = math.log(e_i / e_im1)
        if denom == 0.0:
            continue
        p = math.log(e_ip1 / e_i) / denom
        if math.isfinite(p):
            p_values.append(p)

    if not p_values:
        return None

    tail = p_values[-3:]
    return sum(tail) / len(tail)


def run_case(
    name: str,
    func: Callable[[float], float],
    x0: float,
    x1: float,
    true_root: Optional[float] = None,
    max_iter: int = 80,
) -> None:
    result = secant_method(func=func, x0=x0, x1=x1, max_iter=max_iter)

    print(f"[Case] {name}")
    print(f"  init: x0={x0:.12g}, x1={x1:.12g}")
    print(
        "  status: "
        f"converged={result.converged}, iterations={result.iterations}, reason={result.reason}"
    )
    print(f"  root ~= {result.root:.16f}")
    print(f"  |f(root)| = {abs(result.f_root):.3e}")

    if true_root is not None:
        abs_err = abs(result.root - true_root)
        order = estimate_order(result, true_root)
        print(f"  |root - true_root| = {abs_err:.3e}")
        if order is not None:
            print(f"  estimated local order p ~= {order:.4f}")

    if result.history:
        show_n = min(3, len(result.history))
        print("  first iterations:")
        for rec in result.history[:show_n]:
            print(
                "    "
                f"k={rec.iteration:>2d}: "
                f"x_next={rec.x_next:.16f}, "
                f"f_next={rec.f_next:.3e}, "
                f"step={rec.step:.3e}"
            )

        last = result.history[-1]
        print(
            "  last iteration: "
            f"k={last.iteration}, x_next={last.x_next:.16f}, "
            f"f_next={last.f_next:.3e}"
        )

    print("-" * 72)


def main() -> None:
    print("Secant Method MVP (MATH-0117)")
    print("=" * 72)

    run_case(
        name="f(x)=x^2-2",
        func=lambda x: x * x - 2.0,
        x0=1.0,
        x1=2.0,
        true_root=math.sqrt(2.0),
    )

    run_case(
        name="f(x)=cos(x)-x",
        func=lambda x: math.cos(x) - x,
        x0=0.0,
        x1=1.0,
        true_root=0.7390851332151607,
    )

    run_case(
        name="f(x)=x^3-x-2",
        func=lambda x: x * x * x - x - 2.0,
        x0=1.0,
        x1=2.0,
        true_root=1.5213797068045676,
    )

    run_case(
        name="f(x)=(x-1)^2 (multiple root, typically slower)",
        func=lambda x: (x - 1.0) ** 2,
        x0=0.8,
        x1=2.0,
        true_root=1.0,
        max_iter=120,
    )

    run_case(
        name="f(x)=(x-1)^2 with symmetric seeds (expect denominator failure)",
        func=lambda x: (x - 1.0) ** 2,
        x0=0.0,
        x1=2.0,
        true_root=1.0,
    )

    print("Done.")


if __name__ == "__main__":
    main()
