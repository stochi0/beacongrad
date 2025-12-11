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

    def tanh(self):
        out = Value(np.tanh(self.data), (self,), 'tanh', f"tanh({self.label})")
        def _backward():
            self.grad = (1 - np.tanh(self.data)**2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        out = Value(1 / (1 + np.exp(-self.data)), (self,), 'sigmoid', f"sigmoid({self.label})")
        def _backward():
            self.grad = (1 - 1 / (1 + np.exp(-self.data))) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu', f"relu({self.label})")
        def _backward():
            self.grad = (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def leaky_relu(self):
        out = Value(max(0.01*self.data, self.data), (self,), 'leaky_relu', f"leaky_relu({self.label})")
        def _backward():
            self.grad = (out.data > 0) * out.grad + (out.data <= 0) * 0.01 * out.grad
        out._backward = _backward
        return out
    
    def softmax(self):
        out = Value(np.exp(self.data) / np.sum(np.exp(self.data)), (self,), 'softmax', f"softmax({self.label})")
        def _backward():
            self.grad = (out.data - self.data) * out.grad
        out._backward = _backward
        return out

    def log_softmax(self):
        out = Value(np.log(np.exp(self.data) / np.sum(np.exp(self.data))), (self,), 'log_softmax', f"log_softmax({self.label})")
        def _backward():
            self.grad = (out.data - self.data) * out.grad
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