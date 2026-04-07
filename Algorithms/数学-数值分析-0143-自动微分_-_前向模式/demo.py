"""Minimal runnable MVP for automatic differentiation (forward mode)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, List, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None


@dataclass(frozen=True)
class Dual:
    """Dual number a + bε with ε^2 = 0 for forward-mode AD."""

    real: float
    dual: float

    def __add__(self, other: float | Dual) -> Dual:
        o = to_dual(other)
        return Dual(self.real + o.real, self.dual + o.dual)

    def __radd__(self, other: float | Dual) -> Dual:
        return self.__add__(other)

    def __sub__(self, other: float | Dual) -> Dual:
        o = to_dual(other)
        return Dual(self.real - o.real, self.dual - o.dual)

    def __rsub__(self, other: float | Dual) -> Dual:
        o = to_dual(other)
        return Dual(o.real - self.real, o.dual - self.dual)

    def __mul__(self, other: float | Dual) -> Dual:
        o = to_dual(other)
        return Dual(
            self.real * o.real,
            self.real * o.dual + self.dual * o.real,
        )

    def __rmul__(self, other: float | Dual) -> Dual:
        return self.__mul__(other)

    def __truediv__(self, other: float | Dual) -> Dual:
        o = to_dual(other)
        if o.real == 0.0:
            raise ZeroDivisionError("division by zero in Dual")
        real = self.real / o.real
        dual = (self.dual * o.real - self.real * o.dual) / (o.real * o.real)
        return Dual(real, dual)

    def __rtruediv__(self, other: float | Dual) -> Dual:
        o = to_dual(other)
        return o.__truediv__(self)

    def __neg__(self) -> Dual:
        return Dual(-self.real, -self.dual)

    def __pow__(self, power: float) -> Dual:
        # d(x^p) = p*x^(p-1)*dx, where p is constant.
        if self.real <= 0.0 and abs(power - round(power)) > 1e-12:
            raise ValueError("non-integer power on non-positive base is undefined")
        real = self.real**power
        dual = power * (self.real ** (power - 1.0)) * self.dual
        return Dual(real, dual)


def to_dual(x: float | Dual) -> Dual:
    """Promote scalar to Dual with zero tangent."""
    if isinstance(x, Dual):
        return x
    return Dual(float(x), 0.0)


def d_sin(x: float | Dual) -> Dual:
    z = to_dual(x)
    return Dual(math.sin(z.real), math.cos(z.real) * z.dual)


def d_cos(x: float | Dual) -> Dual:
    z = to_dual(x)
    return Dual(math.cos(z.real), -math.sin(z.real) * z.dual)


def d_exp(x: float | Dual) -> Dual:
    z = to_dual(x)
    ev = math.exp(z.real)
    return Dual(ev, ev * z.dual)


def d_log1p(x: float | Dual) -> Dual:
    z = to_dual(x)
    if z.real <= -1.0:
        raise ValueError("log1p domain error")
    return Dual(math.log1p(z.real), z.dual / (1.0 + z.real))


def d_sigmoid(x: float | Dual) -> Dual:
    z = to_dual(x)
    # sigmoid(z) = 1/(1+exp(-z))
    return 1.0 / (1.0 + d_exp(-z))


def scalar_objective(xs: Sequence[float | Dual]) -> float | Dual:
    """f(x1,x2,x3)=sin(x1*x2)+x1*sigmoid(x3)+log(1+x2^2)."""
    if len(xs) != 3:
        raise ValueError("scalar_objective expects 3 variables")
    x1, x2, x3 = xs
    return d_sin(x1 * x2) + x1 * d_sigmoid(x3) + d_log1p(x2 * x2)


def vector_objective(xs: Sequence[float | Dual]) -> List[float | Dual]:
    """g(x,y)=[x*y+sin(x), x^2+cos(y)] for Jacobian demo."""
    if len(xs) != 2:
        raise ValueError("vector_objective expects 2 variables")
    x, y = xs
    return [x * y + d_sin(x), x * x + d_cos(y)]


def forward_directional_derivative(
    f: Callable[[Sequence[float | Dual]], float | Dual],
    x: np.ndarray,
    v: np.ndarray,
) -> tuple[float, float]:
    """Compute f(x) and directional derivative df(x)[v] via one forward pass."""
    if x.shape != v.shape:
        raise ValueError("x and v must have the same shape")
    dual_inputs = [Dual(float(xi), float(vi)) for xi, vi in zip(x, v)]
    out = f(dual_inputs)
    out_dual = to_dual(out)
    return out_dual.real, out_dual.dual


def forward_gradient(
    f: Callable[[Sequence[float | Dual]], float | Dual],
    x: np.ndarray,
) -> np.ndarray:
    """Compute gradient of scalar f by seeding basis vectors."""
    n = x.size
    grad = np.zeros(n, dtype=float)
    for i in range(n):
        seed = np.zeros(n, dtype=float)
        seed[i] = 1.0
        _, directional = forward_directional_derivative(f, x, seed)
        grad[i] = directional
    return grad


def forward_jacobian(
    g: Callable[[Sequence[float | Dual]], Sequence[float | Dual]],
    x: np.ndarray,
) -> np.ndarray:
    """Compute Jacobian of vector function g via repeated forward seeding."""
    n = x.size
    base = g([float(v) for v in x])
    m = len(base)
    j = np.zeros((m, n), dtype=float)

    for col in range(n):
        dual_inputs: List[Dual] = []
        for i in range(n):
            dual_inputs.append(Dual(float(x[i]), 1.0 if i == col else 0.0))
        out = g(dual_inputs)
        for row in range(m):
            j[row, col] = to_dual(out[row]).dual
    return j


def finite_difference_grad(
    f_scalar: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-6,
) -> np.ndarray:
    """Central finite-difference gradient for validation."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        grad[i] = (f_scalar(xp) - f_scalar(xm)) / (2.0 * h)
    return grad


def torch_gradient_reference(x: np.ndarray) -> np.ndarray | None:
    """Use PyTorch autograd as a reference (optional dependency)."""
    if torch is None:
        return None

    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = (
        torch.sin(xt[0] * xt[1])
        + xt[0] * torch.sigmoid(xt[2])
        + torch.log1p(xt[1] * xt[1])
    )
    y.backward()
    return xt.grad.detach().cpu().numpy()


def scalar_objective_numpy(x: np.ndarray) -> float:
    return float(
        np.sin(x[0] * x[1]) + x[0] / (1.0 + np.exp(-x[2])) + np.log1p(x[1] * x[1])
    )


def main() -> None:
    print("Forward-Mode Automatic Differentiation (Dual Number) Demo")
    print("=" * 88)

    x = np.array([1.2, -0.7, 0.3], dtype=float)
    v = np.array([0.5, -1.0, 2.0], dtype=float)

    f_val, f_dir = forward_directional_derivative(scalar_objective, x, v)
    grad = forward_gradient(scalar_objective, x)
    fd_grad = finite_difference_grad(scalar_objective_numpy, x)
    torch_grad = torch_gradient_reference(x)

    print("Scalar objective: f(x1,x2,x3)=sin(x1*x2)+x1*sigmoid(x3)+log(1+x2^2)")
    print(f"x = {x}")
    print(f"f(x) = {f_val:.12f}")
    print(f"direction v = {v}")
    print(f"forward directional derivative df(x)[v] = {f_dir:.12f}")
    print(f"gradient by forward AD      = {grad}")
    print(f"gradient by finite diff     = {fd_grad}")
    print(f"||grad_ad - grad_fd||_inf   = {np.max(np.abs(grad - fd_grad)):.3e}")

    if torch_grad is None:
        print("PyTorch reference           = <skipped: torch not available>")
    else:
        print(f"gradient by PyTorch         = {torch_grad}")
        print(f"||grad_ad - grad_torch||_inf= {np.max(np.abs(grad - torch_grad)):.3e}")

    print("-" * 88)

    xy = np.array([0.8, -0.4], dtype=float)
    jac = forward_jacobian(vector_objective, xy)
    jac_exact = np.array(
        [
            [xy[1] + math.cos(xy[0]), xy[0]],
            [2.0 * xy[0], -math.sin(xy[1])],
        ],
        dtype=float,
    )

    print("Vector objective: g(x,y)=[x*y+sin(x), x^2+cos(y)]")
    print(f"point = {xy}")
    print("Jacobian by forward AD:")
    print(jac)
    print("Analytic Jacobian:")
    print(jac_exact)
    print(f"||J_ad - J_exact||_inf = {np.max(np.abs(jac - jac_exact)):.3e}")

    assert np.max(np.abs(grad - fd_grad)) < 1e-5
    if torch_grad is not None:
        assert np.max(np.abs(grad - torch_grad)) < 1e-10
    assert np.max(np.abs(jac - jac_exact)) < 1e-12

    print("=" * 88)
    print("All checks passed.")


if __name__ == "__main__":
    main()
