"""
Optimization algorithms (similar to torch.optim).

Provides various optimizers for training neural networks.
"""

import numpy as np
from typing import List
from .tensor import Tensor


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, parameters: List[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """Zero all gradients."""
        for p in self.parameters:
            if p.requires_grad:
                p.grad = np.zeros_like(p.data)

    def step(self):
        """Perform a single optimization step. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement step()")
        