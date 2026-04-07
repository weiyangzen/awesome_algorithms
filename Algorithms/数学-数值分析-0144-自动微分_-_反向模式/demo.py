"""自动微分（反向模式）最小可运行示例。

实现要点：
- 纯 Python 微型计算图 Value 节点；
- 显式局部导数与拓扑逆序反传；
- 与解析梯度、PyTorch 梯度做数值对照。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - 环境可能无 PyTorch
    torch = None


class Value:
    """标量计算图节点。"""

    def __init__(
        self,
        data: float,
        children: Iterable["Value"] = (),
        op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(children)
        self._op = op
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self._op}')"

    @staticmethod
    def _to_value(other: float | "Value") -> "Value":
        return other if isinstance(other, Value) else Value(float(other))

    def __add__(self, other: float | "Value") -> "Value":
        other = self._to_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

    def __neg__(self) -> "Value":
        out = Value(-self.data, (self,), "neg")

        def _backward() -> None:
            self.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-self._to_value(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        return self._to_value(other) + (-self)

    def __mul__(self, other: float | "Value") -> "Value":
        other = self._to_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float | "Value") -> "Value":
        return self * other

    def pow(self, exponent: float) -> "Value":
        out = Value(self.data ** exponent, (self,), f"pow({exponent})")

        def _backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1.0)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: float | "Value") -> "Value":
        other = self._to_value(other)
        return self * other.pow(-1.0)

    def __rtruediv__(self, other: float | "Value") -> "Value":
        return self._to_value(other) * self.pow(-1.0)

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def sin(self) -> "Value":
        out = Value(math.sin(self.data), (self,), "sin")

        def _backward() -> None:
            self.grad += math.cos(self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """对标量输出节点执行反向传播。"""
        topo = []
        visited = set()

        def build(v: "Value") -> None:
            if v in visited:
                return
            visited.add(v)
            for parent in v._prev:
                build(parent)
            topo.append(v)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


@dataclass
class GradientReport:
    point: Tuple[float, float, float]
    value_engine: float
    value_analytic: float
    grads_engine: Dict[str, float]
    grads_analytic: Dict[str, float]
    max_abs_grad_error: float
    torch_grads: Optional[Dict[str, float]]
    torch_max_abs_grad_error: Optional[float]


def target_function_value(x: Value, y: Value, z: Value) -> Value:
    """示例目标函数: f(x,y,z) = ((x*y)+sin(z))*exp(x) + y^3*z"""
    return ((x * y) + z.sin()) * x.exp() + y.pow(3.0) * z


def target_function_float(x: float, y: float, z: float) -> float:
    return ((x * y) + math.sin(z)) * math.exp(x) + (y**3) * z


def analytic_gradients(x: float, y: float, z: float) -> Dict[str, float]:
    ex = math.exp(x)
    return {
        "x": ex * (x * y + math.sin(z) + y),
        "y": x * ex + 3.0 * (y**2) * z,
        "z": math.cos(z) * ex + y**3,
    }


def evaluate_with_engine(x: float, y: float, z: float) -> Tuple[float, Dict[str, float]]:
    xv = Value(x, label="x")
    yv = Value(y, label="y")
    zv = Value(z, label="z")

    out = target_function_value(xv, yv, zv)
    out.backward()

    grads = {"x": xv.grad, "y": yv.grad, "z": zv.grad}
    return out.data, grads


def try_evaluate_with_torch(
    x: float, y: float, z: float
) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    if torch is None:
        return None, None

    xt = torch.tensor(float(x), dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(float(y), dtype=torch.float64, requires_grad=True)
    zt = torch.tensor(float(z), dtype=torch.float64, requires_grad=True)

    out = ((xt * yt) + torch.sin(zt)) * torch.exp(xt) + (yt**3) * zt
    out.backward()

    grads = {
        "x": float(xt.grad.item()),
        "y": float(yt.grad.item()),
        "z": float(zt.grad.item()),
    }
    return float(out.item()), grads


def summarize_point(point: Tuple[float, float, float]) -> GradientReport:
    x, y, z = point
    value_engine, grads_engine = evaluate_with_engine(x, y, z)
    value_analytic = target_function_float(x, y, z)
    grads_analytic = analytic_gradients(x, y, z)

    per_var_error = {
        k: abs(grads_engine[k] - grads_analytic[k]) for k in grads_engine.keys()
    }
    max_abs_grad_error = max(per_var_error.values())

    torch_value, torch_grads = try_evaluate_with_torch(x, y, z)
    torch_max_abs_grad_error = None
    if torch_grads is not None:
        torch_max_abs_grad_error = max(
            abs(grads_engine[k] - torch_grads[k]) for k in grads_engine.keys()
        )
        # 函数值也做一次一致性检查（仅展示，不中断流程）
        _ = abs(value_engine - float(torch_value))

    return GradientReport(
        point=point,
        value_engine=value_engine,
        value_analytic=value_analytic,
        grads_engine=grads_engine,
        grads_analytic=grads_analytic,
        max_abs_grad_error=max_abs_grad_error,
        torch_grads=torch_grads,
        torch_max_abs_grad_error=torch_max_abs_grad_error,
    )


def run_gradient_checks(num_cases: int = 8, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    max_error = 0.0
    for _ in range(num_cases):
        x, y, z = rng.uniform(-1.5, 1.5, size=3)
        _, grads_engine = evaluate_with_engine(float(x), float(y), float(z))
        grads_analytic = analytic_gradients(float(x), float(y), float(z))
        current = max(abs(grads_engine[k] - grads_analytic[k]) for k in grads_engine)
        max_error = max(max_error, current)
    return max_error


def fmt_grads(grads: Dict[str, float]) -> str:
    ordered = [f"{k}={grads[k]: .10f}" for k in ("x", "y", "z")]
    return ", ".join(ordered)


def main() -> None:
    point = (1.2, -0.7, 0.9)
    report = summarize_point(point)

    print("=== 自动微分 - 反向模式 MVP ===")
    print(f"point = (x={point[0]}, y={point[1]}, z={point[2]})")
    print(f"f(engine)   = {report.value_engine:.12f}")
    print(f"f(analytic) = {report.value_analytic:.12f}")
    print(f"grad_engine   : {fmt_grads(report.grads_engine)}")
    print(f"grad_analytic : {fmt_grads(report.grads_analytic)}")
    print(f"max_abs_grad_error(analytic) = {report.max_abs_grad_error:.3e}")

    if report.torch_grads is not None:
        print(f"grad_torch    : {fmt_grads(report.torch_grads)}")
        print(
            "max_abs_grad_error(torch)    = "
            f"{report.torch_max_abs_grad_error:.3e}"
        )
    else:
        print("grad_torch    : <PyTorch not available, skip>" )

    batch_error = run_gradient_checks(num_cases=8, seed=7)
    print(f"batch_check_max_error(8 cases) = {batch_error:.3e}")


if __name__ == "__main__":
    main()
