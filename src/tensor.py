import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, label=''):
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._prev = set()
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.requires_grad}, grad={self.grad}, label={self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, label=f"({self.label} + {other.label})")
        out._prev = {self, other}
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, label=f"({self.label} * {other.label})")
        out._prev = {self, other}
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data, requires_grad=self.requires_grad or other.requires_grad, label=f"({self.label} ** {other.label})")
        out._prev = {self, other}
        def _backward():
            if self.requires_grad:
                self.grad += other.data * self.data ** (other.data - 1) * out.grad
            if other.requires_grad:
                other.grad += self.data ** other.data * np.log(self.data) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad, label=f"({self.label} / {other.label})")
        out._prev = {self, other}
        def _backward():
            if self.requires_grad:
                self.grad += 1 / other.data * out.grad
            if other.requires_grad:
                other.grad += -self.data / other.data**2 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad, label=f"({self.label} - {other.label})")
        out._prev = {self, other}
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad
        out._backward = _backward
        return out

    def backward(self):
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        visited = set()
        topo = []
        build_topo(self)
        for node in topo:
            node.grad = np.zeros_like(node.data)
        self.grad = np.ones_like(self.data) if self.requires_grad else None
        for node in reversed(topo):
            node._backward()

if __name__ == "__main__":
    x = Tensor([1,2,3], requires_grad=True, label='x')
    y = Tensor([4,5,6], requires_grad=True, label='y')
    z = x + y
    z.backward()
    print("z=x+y=", z)
    print("x.grad=", x.grad)
    print("y.grad=", y.grad)

    z = x * y
    z.backward()
    print("z=x*y=", z)
    print("x.grad=", x.grad)
    print("y.grad=", y.grad)