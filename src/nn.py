"""
Neural network modules and layers (similar to torch.nn).

Provides building blocks for creating neural networks with automatic
parameter management and gradient computation.
"""

from typing import List
from .tensor import Tensor


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
