"""
Neural network modules and layers (similar to torch.nn).

Provides building blocks for creating neural networks with automatic
parameter management and gradient computation.
"""

import numpy as np
from typing import List
from .tensor import Tensor, randn, zeros


class Module:
    """
    Base class for all neural network modules.

    Your models should subclass this class.
    """

    def __init__(self):
        self._parameters = []
        self._modules = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Define the forward pass. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")

    def parameters(self) -> List[Tensor]:
        """Return list of all parameters in this module and submodules."""
        params = list(self._parameters)
        for module in self._modules:
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        """Zero all gradients."""
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        """Set module to training mode."""
        self.training = True
        for module in self._modules:
            module.train()

    def eval(self):
        """Set module to evaluation mode."""
        self.training = False
        for module in self._modules:
            module.eval()

class Linear(Module):
    """
    Linear (fully-connected) layer: y = xW^T + b

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with Xavier/Glorot initialization
        k = np.sqrt(1.0 / in_features)
        self.weight = randn(out_features, in_features, requires_grad=True) * k

        if bias:
            self.bias = zeros(out_features, requires_grad=True)
        else:
            self.bias = None

        self._parameters = [self.weight] + ([self.bias] if bias else [])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
