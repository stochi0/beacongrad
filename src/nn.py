import numpy as np

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._op = _op
        self._children = set(_children)
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, _children={self._children}, _op={self._op}, label={self.label}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+', f"({self.label} + {other.label})")
        def _backward():
            self.grad = 1.0*out.grad
            other.grad = 1.0*out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*', f"({self.label} * {other.label})")
        def _backward():
            self.grad = other.data*out.grad
            other.grad = self.data*out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        out =  Value(self.data / other.data, (self, other), '/', f"({self.label} / {other.label})")
        def _backward():
            self.grad = 1.0/other.data*out.grad
            other.grad = -1.0/other.data**2*out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Value(self.data ** other.data, (self, other), '**', f"({self.label} ** {other.label})")
        def _backward():
            self.grad = other.data*self.data**(other.data-1)*out.grad
            other.grad = self.data**other.data*np.log(self.data)*out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()