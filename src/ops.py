"""
Functional operations for tensors (similar to torch.nn.functional).

All operations here are differentiable and work with autograd.
"""

import numpy as np
from typing import Union, Tuple
from .tensor import Tensor, unbroadcast, EPS


# ==================== Arithmetic Operations ====================


def add(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise addition with broadcasting."""
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)

    out = Tensor(
        a.data + b.data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="add",
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += unbroadcast(out.grad, a.shape)
        if b.requires_grad:
            b.grad += unbroadcast(out.grad, b.shape)

    out._backward = _backward
    return out


def sub(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise subtraction with broadcasting."""
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)

    out = Tensor(
        a.data - b.data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="sub",
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += unbroadcast(out.grad, a.shape)
        if b.requires_grad:
            b.grad -= unbroadcast(out.grad, b.shape)

    out._backward = _backward
    return out


def mul(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise multiplication with broadcasting."""
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)

    out = Tensor(
        a.data * b.data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="mul",
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += unbroadcast(b.data * out.grad, a.shape)
        if b.requires_grad:
            b.grad += unbroadcast(a.data * out.grad, b.shape)

    out._backward = _backward
    return out


def div(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise division with broadcasting."""
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)

    out = Tensor(
        a.data / b.data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="div",
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += unbroadcast((1.0 / b.data) * out.grad, a.shape)
        if b.requires_grad:
            b.grad += unbroadcast((-a.data / (b.data**2)) * out.grad, b.shape)

    out._backward = _backward
    return out


def pow(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise power with broadcasting."""
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)

    out = Tensor(
        a.data**b.data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="pow",
    )

    def _backward():
        if out.grad is None:
            return
        # d(x**y)/dx = y * x^(y-1)
        if a.requires_grad:
            grad_a = b.data * (a.data ** (b.data - 1)) * out.grad
            a.grad += unbroadcast(grad_a, a.shape)

        # d(x**y)/dy = x**y * log(x)
        if b.requires_grad:
            safe_log = np.log(np.clip(a.data, EPS, None))
            grad_b = out.data * safe_log * out.grad
            b.grad += unbroadcast(grad_b, b.shape)

    out._backward = _backward
    return out


def neg(a: Tensor) -> Tensor:
    """Unary negation."""
    out = Tensor(-a.data, requires_grad=a.requires_grad, _children=(a,), _op="neg")

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad -= out.grad

    out._backward = _backward
    return out


def matmul(a: Union[Tensor, np.ndarray], b: Union[Tensor, np.ndarray]) -> Tensor:
    """Matrix multiplication with batching support."""
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)

    out = Tensor(
        np.matmul(a.data, b.data),
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="matmul",
    )

    def _backward():
        if out.grad is None:
            return

        # dL/dA = dL/dOut @ B^T
        if a.requires_grad:
            dA = np.matmul(out.grad, np.swapaxes(b.data, -1, -2))
            a.grad += unbroadcast(dA, a.shape)

        # dL/dB = A^T @ dL/dOut
        if b.requires_grad:
            dB = np.matmul(np.swapaxes(a.data, -1, -2), out.grad)
            b.grad += unbroadcast(dB, b.shape)

    out._backward = _backward
    return out


# ==================== Reduction Operations ====================


def sum(a: Tensor, axis=None, keepdims=False) -> Tensor:
    """Sum elements over given axis."""
    out = Tensor(
        a.data.sum(axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="sum",
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            # Broadcast gradient back to input shape
            grad = out.grad
            if not keepdims and axis is not None:
                # Add back reduced dimensions
                if isinstance(axis, int):
                    grad = np.expand_dims(grad, axis=axis)
                else:
                    for ax in sorted(axis):
                        grad = np.expand_dims(grad, axis=ax)
            # Broadcast to input shape
            a.grad += np.broadcast_to(grad, a.shape)

    out._backward = _backward
    return out


def mean(a: Tensor, axis=None, keepdims=False) -> Tensor:
    """Mean of elements over given axis."""
    out = Tensor(
        a.data.mean(axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="mean",
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            # Gradient is distributed evenly
            grad = out.grad
            if not keepdims and axis is not None:
                if isinstance(axis, int):
                    grad = np.expand_dims(grad, axis=axis)
                else:
                    for ax in sorted(axis):
                        grad = np.expand_dims(grad, axis=ax)

            # Divide by number of elements that were reduced
            if axis is None:
                n = a.size
            elif isinstance(axis, int):
                n = a.shape[axis]
            else:
                n = np.prod([a.shape[ax] for ax in axis])

            grad = grad / n
            a.grad += np.broadcast_to(grad, a.shape)

    out._backward = _backward
    return out


# ==================== Shape Operations ====================


def reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshape tensor."""
    out = Tensor(
        a.data.reshape(shape),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="reshape",
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += out.grad.reshape(a.shape)

    out._backward = _backward
    return out


def transpose(a: Tensor, axes=None) -> Tensor:
    """Transpose tensor."""
    if axes is None:
        # Default: reverse all axes (or swap last two for 2D)
        axes = tuple(range(a.ndim)[::-1])

    out = Tensor(
        np.transpose(a.data, axes),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="transpose",
    )

    # Compute inverse permutation
    inv_axes = np.argsort(axes)

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += np.transpose(out.grad, inv_axes)

    out._backward = _backward
    return out


# ==================== Activation Functions ====================


def relu(a: Tensor) -> Tensor:
    """ReLU activation: max(0, x)."""
    out = Tensor(
        np.maximum(0, a.data), requires_grad=a.requires_grad, _children=(a,), _op="relu"
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += (a.data > 0) * out.grad

    out._backward = _backward
    return out


def leaky_relu(a: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Leaky ReLU activation."""
    out = Tensor(
        np.where(a.data > 0, a.data, negative_slope * a.data),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="leaky_relu",
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            grad = np.where(a.data > 0, 1.0, negative_slope) * out.grad
            a.grad += grad

    out._backward = _backward
    return out


def sigmoid(a: Tensor) -> Tensor:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    # Use a dtype-dependent safe exponent range to avoid float32 overflow.
    x = a.data
    finfo = np.finfo(x.dtype)
    max_log = float(np.log(finfo.max) - 2.0)  # margin for safety
    x_clip = np.clip(x, -max_log, max_log)
    sig = 1 / (1 + np.exp(-x_clip))
    out = Tensor(sig, requires_grad=a.requires_grad, _children=(a,), _op="sigmoid")

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += sig * (1 - sig) * out.grad

    out._backward = _backward
    return out


def tanh(a: Tensor) -> Tensor:
    """Tanh activation."""
    tanh_val = np.tanh(a.data)
    out = Tensor(tanh_val, requires_grad=a.requires_grad, _children=(a,), _op="tanh")

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += (1 - tanh_val**2) * out.grad

    out._backward = _backward
    return out


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Softmax activation along specified axis."""
    # Numerical stability: subtract max
    x_max = a.data.max(axis=axis, keepdims=True)
    exp_x = np.exp(a.data - x_max)
    softmax_val = exp_x / exp_x.sum(axis=axis, keepdims=True)

    out = Tensor(
        softmax_val, requires_grad=a.requires_grad, _children=(a,), _op="softmax"
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            # Jacobian of softmax: s_i * (δ_ij - s_j)
            # Efficient computation: grad = softmax * (grad_out - (grad_out * softmax).sum())
            s = softmax_val
            grad_out = out.grad
            sum_term = (grad_out * s).sum(axis=axis, keepdims=True)
            a.grad += s * (grad_out - sum_term)

    out._backward = _backward
    return out


def log_softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Log-softmax activation (numerically stable)."""
    x_max = a.data.max(axis=axis, keepdims=True)
    shifted = a.data - x_max
    log_sum_exp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    log_softmax_val = shifted - log_sum_exp

    out = Tensor(
        log_softmax_val,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="log_softmax",
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            # grad = grad_out - softmax * grad_out.sum()
            softmax_val = np.exp(log_softmax_val)
            sum_term = out.grad.sum(axis=axis, keepdims=True)
            a.grad += (out.grad - softmax_val * sum_term)

    out._backward = _backward
    return out


# ==================== Loss Functions ====================


def mse_loss(pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """Mean squared error loss."""
    target = target if isinstance(target, Tensor) else Tensor(target)
    diff = pred - target
    return (diff * diff).mean()


def binary_cross_entropy(pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """Binary cross-entropy loss (expects pred in [0, 1])."""
    target = target if isinstance(target, Tensor) else Tensor(target)
    # Numerical stability: clip, but keep gradient path to `pred`.
    pred_clipped = clip(pred, EPS, 1 - EPS)
    return -(
        target * pred_clipped.log() + (1 - target) * (1 - pred_clipped).log()
    ).mean()


def cross_entropy(
    pred: Tensor, target: Union[Tensor, np.ndarray, np.ndarray]
) -> Tensor:
    """
    Cross-entropy loss (combines log_softmax and negative log-likelihood).

    Args:
        pred: Logits of shape (batch_size, num_classes)
        target: Class indices of shape (batch_size,) or one-hot vectors
    """
    log_probs = log_softmax(pred, axis=-1)

    if isinstance(target, np.ndarray):
        target_data = target
    elif isinstance(target, Tensor):
        target_data = target.data
    else:
        target_data = np.array(target)

    # If target is class indices
    if target_data.ndim == 1 or (target_data.ndim == 2 and target_data.shape[1] == 1):
        target_data = target_data.flatten().astype(int)
        # Gather log probabilities at target indices
        batch_size = log_probs.shape[0]
        log_probs_selected = log_probs.data[np.arange(batch_size), target_data]
        loss = -log_probs_selected.mean()

        out = Tensor(
            loss, requires_grad=pred.requires_grad, _children=(log_probs,), _op="nll"
        )

        def _backward():
            if out.grad is None:
                return
            if pred.requires_grad:
                grad = np.exp(log_probs.data)  # softmax
                grad[np.arange(batch_size), target_data] -= 1
                grad = grad / batch_size * out.grad
                log_probs.grad += grad

        out._backward = _backward
        return out
    else:
        # Target is one-hot or probability distribution
        target_tensor = target if isinstance(target, Tensor) else Tensor(target)
        return -(target_tensor * log_probs).sum(axis=-1).mean()


# ==================== Helper Functions ====================

def clip(a: Tensor, min_value: float, max_value: float) -> Tensor:
    """Element-wise clip with straight-through gradient inside bounds."""
    clipped = np.clip(a.data, min_value, max_value)
    out = Tensor(
        clipped, requires_grad=a.requires_grad, _children=(a,), _op="clip"
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            # Undefined at the exact boundaries; we choose 0 gradient there.
            mask = (a.data > min_value) & (a.data < max_value)
            a.grad += mask * out.grad

    out._backward = _backward
    return out


def exp(a: Tensor) -> Tensor:
    """Element-wise exponential."""
    # NOTE:
    # - For float32, exp(500) overflows to inf (and gradients become NaN).
    # - Clip to a dtype-dependent safe range based on log(max_float).
    x = a.data
    finfo = np.finfo(x.dtype)
    max_log = float(np.log(finfo.max) - 2.0)  # margin for safety
    exp_val = np.exp(np.clip(x, -max_log, max_log))
    out = Tensor(exp_val, requires_grad=a.requires_grad, _children=(a,), _op="exp")

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += exp_val * out.grad

    out._backward = _backward
    return out


def log(a: Tensor) -> Tensor:
    """Element-wise natural logarithm."""
    out = Tensor(
        np.log(np.clip(a.data, EPS, None)),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="log",
    )

    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            a.grad += (1.0 / np.clip(a.data, EPS, None)) * out.grad

    out._backward = _backward
    return out

# Add .exp and .log to Tensor class
Tensor.exp = lambda self: exp(self)
Tensor.log = lambda self: log(self)