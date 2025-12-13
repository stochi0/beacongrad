import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, label=''):
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self.label = label
        self.backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.requires_grad}, grad={self.grad}, label={self.label})"