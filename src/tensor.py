"""Core Tensor class with automatic differentiation support."""

import numpy as np
from typing import Union, Optional, Tuple


EPS = 1e-12


def unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce gradient back to original shape after broadcasting."""
    # Remove leading axes added by broadcasting
    if grad.ndim > len(shape):
        axes = tuple(range(grad.ndim - len(shape)))
        grad = grad.sum(axis=axes)

    # Sum over broadcasted dimensions (where original had size 1)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor:
    """
    N-dimensional array with automatic differentiation.

    Similar to torch.Tensor, supports:
    - Arithmetic operations with autograd
    - Broadcasting semantics
    - Efficient NumPy backend
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        dtype=np.float32,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        self.data = np.asarray(data, dtype=dtype)
        self.dtype = dtype
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.size = self.data.size
        self.requires_grad = requires_grad

        # Gradient (initialized on first backward pass)
        self.grad: Optional[np.ndarray] = None

        # Computation graph
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        grad_str = f", grad={self.grad}" if self.grad is not None else ""
        return f"Tensor({self.data}, requires_grad={self.requires_grad}{grad_str})"

    def __str__(self) -> str:
        return f"Tensor({self.data})"

    # ==================== Arithmetic Operations ====================

    def __add__(self, other):
        """Element-wise addition."""
        from . import ops

        return ops.add(self, other)

    def __radd__(self, other):
        """Reverse addition."""
        from . import ops

        return ops.add(other, self)

    def __sub__(self, other):
        """Element-wise subtraction."""
        from . import ops

        return ops.sub(self, other)

    def __rsub__(self, other):
        """Reverse subtraction."""
        from . import ops

        return ops.sub(other, self)

    def __mul__(self, other):
        """Element-wise multiplication."""
        from . import ops

        return ops.mul(self, other)

    def __rmul__(self, other):
        """Reverse multiplication."""
        from . import ops

        return ops.mul(other, self)

    def __truediv__(self, other):
        """Element-wise division."""
        from . import ops

        return ops.div(self, other)

    def __rtruediv__(self, other):
        """Reverse division."""
        from . import ops

        return ops.div(other, self)

    def __pow__(self, other):
        """Element-wise power."""
        from . import ops

        return ops.pow(self, other)

    def __rpow__(self, other):
        """Reverse power."""
        from . import ops

        return ops.pow(other, self)

    def __matmul__(self, other):
        """Matrix multiplication."""
        from . import ops

        return ops.matmul(self, other)

    def __rmatmul__(self, other):
        """Reverse matrix multiplication."""
        from . import ops

        return ops.matmul(other, self)

    def __neg__(self):
        """Unary negation."""
        from . import ops

        return ops.neg(self)
    
    def __log__(self):
        """Natural logarithm."""
        from . import ops

        return ops.log(self)

    def __exp__(self):
        """Exponential function."""
        from . import ops

        return ops.exp(self)

    # ==================== Tensor Operations ====================

    def sum(self, axis=None, keepdims=False):
        """Sum elements over a given axis."""
        from . import ops

        return ops.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        """Mean of elements over a given axis."""
        from . import ops

        return ops.mean(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape):
        """Reshape tensor."""
        from . import ops

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return ops.reshape(self, shape)

    def transpose(self, axes=None):
        """Transpose tensor."""
        from . import ops

        return ops.transpose(self, axes=axes)

    @property
    def T(self):
        """Transpose (last two dimensions)."""
        from . import ops

        return ops.transpose(self)

    # ==================== Activation Functions ====================

    def relu(self):
        """ReLU activation."""
        from . import ops

        return ops.relu(self)

    def sigmoid(self):
        """Sigmoid activation."""
        from . import ops

        return ops.sigmoid(self)

    def tanh(self):
        """Tanh activation."""
        from . import ops

        return ops.tanh(self)

    def softmax(self, axis=-1):
        """Softmax activation."""
        from . import ops

        return ops.softmax(self, axis=axis)

    # ==================== Autograd ====================

    def backward(self, grad: Optional[Union["Tensor", np.ndarray]] = None):
        """
        Compute gradients via reverse-mode autodiff.

        Args:
            grad: Gradient of scalar output w.r.t. this tensor.
                  If None, this tensor must be scalar (loss).
        """
        # Build topological order
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Initialize gradients
        for node in topo:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)

        # Seed gradient
        if grad is None:
            if self.size != 1:
                raise RuntimeError(
                    "grad must be specified for non-scalar outputs. "
                    "Use .backward(grad=...) or ensure output is scalar."
                )
            self.grad = np.ones_like(self.data)
        else:
            if isinstance(grad, Tensor):
                grad = grad.data
            grad = np.asarray(grad, dtype=self.dtype)
            if grad.shape != self.shape:
                grad = np.broadcast_to(grad, self.shape).astype(self.dtype)
            self.grad = grad

        # Backward pass
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Zero out gradients."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # ==================== Utility Methods ====================

    def item(self):
        """Get scalar value (tensor must contain single element)."""
        if self.size != 1:
            raise ValueError(
                "only one element tensors can be converted to Python scalars"
            )
        return self.data.item()

    def numpy(self) -> np.ndarray:
        """Get underlying numpy array."""
        return self.data

    def detach(self) -> "Tensor":
        """Create a new tensor detached from computation graph."""
        return Tensor(self.data.copy(), requires_grad=False, dtype=self.dtype)

    def clone(self) -> "Tensor":
        """Clone tensor (keeps requires_grad)."""
        return Tensor(
            self.data.copy(), requires_grad=self.requires_grad, dtype=self.dtype
        )


def tensor(data, requires_grad: bool = False, dtype=np.float32) -> Tensor:
    """Create a tensor (convenience function)."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad: bool = False, dtype=np.float32) -> Tensor:
    """Create tensor filled with zeros."""
    return Tensor(
        np.zeros(shape, dtype=dtype), requires_grad=requires_grad, dtype=dtype
    )


def ones(*shape, requires_grad: bool = False, dtype=np.float32) -> Tensor:
    """Create tensor filled with ones."""
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad, dtype=dtype)


def randn(*shape, requires_grad: bool = False, dtype=np.float32) -> Tensor:
    """Create tensor with random normal values."""
    return Tensor(
        np.random.randn(*shape).astype(dtype), requires_grad=requires_grad, dtype=dtype
    )


def rand(*shape, requires_grad: bool = False, dtype=np.float32) -> Tensor:
    """Create tensor with random uniform values [0, 1)."""
    return Tensor(
        np.random.rand(*shape).astype(dtype), requires_grad=requires_grad, dtype=dtype
    )
