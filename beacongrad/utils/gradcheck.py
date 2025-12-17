"""
Numeric gradcheck (finite differences) for BeaconGrad.

This verifies that autograd's analytic gradients match numeric gradients:
  d f / d x_i ~= (f(x_i + eps) - f(x_i - eps)) / (2*eps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

from beacongrad.tensor import Tensor


@dataclass(frozen=True)
class GradcheckFailure(AssertionError):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


def _as_float64_leaf(t: Tensor) -> Tensor:
    if not isinstance(t, Tensor):
        raise TypeError(f"gradcheck inputs must be Tensor, got {type(t)}")
    if not t.requires_grad:
        raise ValueError("gradcheck inputs must have requires_grad=True")
    # Convert in-place so callers keep references.
    if t.data.dtype != np.float64:
        t.data = t.data.astype(np.float64, copy=False)
    t.dtype = np.float64
    return t


def _to_scalar_float(y: Tensor) -> float:
    if not isinstance(y, Tensor):
        raise TypeError(f"f must return a Tensor, got {type(y)}")
    if y.size != 1:
        raise ValueError(
            f"f must return a scalar Tensor (size==1), got shape={y.shape}"
        )
    return float(np.asarray(y.data).reshape(()))


def _relative_error(a: float, n: float) -> float:
    denom = max(1.0, abs(a), abs(n))
    return abs(a - n) / denom


def gradcheck(
    f: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> bool:
    """
    Finite-difference gradient check.

    Args:
        f: Function returning a scalar Tensor.
        inputs: List of Tensor with requires_grad=True.
        eps: Finite difference step.
        rtol: Relative tolerance (per-element).
        atol: Absolute tolerance (per-element).

    Returns:
        True if all gradients match within tolerance.

    Raises:
        GradcheckFailure if any element fails.
    """
    if not isinstance(inputs, (list, tuple)) or len(inputs) == 0:
        raise ValueError("inputs must be a non-empty list/tuple of Tensor")

    leaves: List[Tensor] = [_as_float64_leaf(t) for t in inputs]

    # Analytic gradients
    y = f(*leaves)
    _ = _to_scalar_float(y)
    y.backward()

    analytic_grads: List[np.ndarray] = []
    for t in leaves:
        if t.grad is None:
            raise GradcheckFailure("Input has grad=None after backward().")
        analytic_grads.append(np.array(t.grad, dtype=np.float64, copy=True))

    # Numeric gradients (per element)
    for input_idx, (t, a_grad) in enumerate(zip(leaves, analytic_grads)):
        it = np.ndindex(t.data.shape) if t.data.shape != () else [()]
        for idx in it:
            orig = float(t.data[idx])

            t.data[idx] = orig + eps
            f_plus = _to_scalar_float(f(*leaves))

            t.data[idx] = orig - eps
            f_minus = _to_scalar_float(f(*leaves))

            t.data[idx] = orig

            n_grad = (f_plus - f_minus) / (2.0 * eps)
            a = float(a_grad[idx])

            rel = _relative_error(a, n_grad)
            abs_err = abs(a - n_grad)
            if abs_err > atol and rel > rtol:
                raise GradcheckFailure(
                    "gradcheck failed:\n"
                    f"- input: {input_idx}\n"
                    f"- index: {idx}\n"
                    f"- analytic: {a:.18e}\n"
                    f"- numeric:  {n_grad:.18e}\n"
                    f"- abs_err:  {abs_err:.18e} (atol={atol})\n"
                    f"- rel_err:  {rel:.18e} (rtol={rtol})"
                )

    return True


