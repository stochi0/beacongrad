import numpy as np

EPS = 1e-12  # small value to stabilize log in pow derivative


def unbroadcast(grad, shape):
    """
    Reduce gradient `grad` back to `shape` after broadcasting.
    Equivalent to PyTorch's sum_to_shape.
    """

    # 1. Remove leading axes added by broadcasting
    if grad.ndim > len(shape):
        # Reduce all extra leading dims at once
        axes = tuple(range(grad.ndim - len(shape)))
        grad = grad.sum(axis=axes)

    # 2. For axes where the original shape had 1 (broadcasted dim),
    #    sum over that axis.
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, label="", dtype=np.float32):
        self.data = np.asarray(data, dtype=dtype)
        self.shape = self.data.shape
        self.requires_grad = requires_grad

        # .grad is None until a backward pass initializes it.
        # This avoids leaking gradients for constants and matches
        # common autodiff semantics.
        self.grad = None

        # DAG connectivity and local backward function
        self._prev = set()
        self._backward = lambda: None

        self.label = label

    def __repr__(self):
        return (
            f"Tensor(data={self.data}, shape={self.shape}, "
            f"requires_grad={self.requires_grad}, grad={self.grad}, label={self.label})"
        )

    # ----------------- ops -----------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad),
                     label=f"({self.label}+{other.label})")
        out._prev = [self, other]

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=(self.requires_grad or other.requires_grad),
                     label=f"({self.label}-{other.label})")
        out._prev = [self, other]

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad -= unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad),
                     label=f"({self.label}*{other.label})")
        out._prev = [self, other]

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += unbroadcast(other.data * out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=(self.requires_grad or other.requires_grad),
                     label=f"({self.label}/{other.label})")
        out._prev = [self, other]

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += unbroadcast((1.0 / other.data) * out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast((-self.data / (other.data ** 2)) * out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        x ** y
        Note: derivative wrt y uses log(x). If x contains non-positive values,
        the derivative is not defined; we clip log's argument to EPS to avoid NaNs,
        but mathematically the operation assumes x > 0 when differentiating w.r.t y.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data, requires_grad=(self.requires_grad or other.requires_grad),
                     label=f"({self.label}**{other.label})")
        out._prev = [self, other]

        def _backward():
            if out.grad is None:
                return
            # d(x**y)/dx = y * x^(y-1)
            if self.requires_grad:
                self.grad += unbroadcast(other.data * (self.data ** (other.data - 1)) * out.grad, self.shape)

            # d(x**y)/dy = x**y * log(x)
            if other.requires_grad:
                # clip to avoid log(0); still, mathematically x must be > 0
                safe_log = np.log(np.clip(self.data, EPS, None))
                other.grad += unbroadcast(out.data * safe_log * out.grad, other.shape)

        out._backward = _backward
        return out

    # ----------------- autograd -----------------

    def backward(self, grad=None):
        """
        Reverse-mode autodiff.
          - If grad is None, output must be scalar and we seed with 1.
          - If grad is provided, it must be a numpy array (or broadcastable) matching output shape.

        This initializes `.grad` for all tensors in the graph that `requires_grad`.
        """
        # ---- build topo ----
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)

        # ---- reset gradients only for nodes that require grad ----
        for node in topo:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)
            else:
                node.grad = None

        # ---- seed gradient ----
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar outputs (call backward(grad=...))")
            self.grad = np.ones_like(self.data)
        else:
            # accept numpy array or Tensor
            if isinstance(grad, Tensor):
                grad = grad.data
            grad = np.asarray(grad, dtype=np.float32)
            if grad.shape != self.data.shape:
                # try broadcasting the supplied grad to the output shape
                try:
                    grad = np.broadcast_to(grad, self.data.shape).astype(np.float32)
                except Exception:
                    raise RuntimeError("Provided grad is not broadcastable to output shape")
            self.grad = grad

        # ---- backward pass ----
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """
        Convenience: zero `.grad` for all tensors in the connected graph.
        Use this before optimization steps to avoid accidental accumulation.
        """
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        for node in topo:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)
            else:
                node.grad = None


# ----------------- quick smoke test -----------------
if __name__ == "__main__":
    x = Tensor([[1.0], [2.0], [3.0]], requires_grad=True, label="x")
    y = Tensor([4.0, 5.0, 6.0, 7.0], requires_grad=True, label="y")
    print(f"x: shape={x.shape}, data=\n{x.data}\n")
    print(f"y: shape={y.shape}, data=\n{y.data}\n")

    z = x * y
    print(f"z = x * y:\n  shape={z.shape}\n  data=\n{z.data}\n")

    z.backward(np.ones_like(z.data))

    print("After backward:")
    print(f"z.grad:\n{z.grad}\n")
    print(f"x.grad:\n{x.grad}\n")
    print(f"y.grad:\n{y.grad}\n")
