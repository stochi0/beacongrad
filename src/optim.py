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


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Args:
        parameters: List of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        weight_decay: L2 regularization factor (default: 0)
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Initialize velocity for momentum
        self.velocities = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        """Perform parameter update."""
        for i, p in enumerate(self.parameters):
            if not p.requires_grad or p.grad is None:
                continue

            grad = p.grad

            # Add weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # Update velocity and parameter
            if self.momentum != 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                p.data -= self.lr * self.velocities[i]
            else:
                p.data -= self.lr * grad


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Args:
        parameters: List of parameters to optimize
        lr: Learning rate
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: L2 regularization factor (default: 0)
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates
        self.m = [np.zeros_like(p.data) for p in parameters]  # First moment
        self.v = [np.zeros_like(p.data) for p in parameters]  # Second moment
        self.t = 0  # Timestep

    def step(self):
        """Perform parameter update."""
        self.t += 1

        for i, p in enumerate(self.parameters):
            if not p.requires_grad or p.grad is None:
                continue

            grad = p.grad

            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    Args:
        parameters: List of parameters to optimize
        lr: Learning rate
        alpha: Smoothing constant (default: 0.99)
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: L2 regularization factor (default: 0)
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize squared gradient moving average
        self.square_avg = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        """Perform parameter update."""
        for i, p in enumerate(self.parameters):
            if not p.requires_grad or p.grad is None:
                continue

            grad = p.grad

            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # Update squared gradient moving average
            self.square_avg[i] = self.alpha * self.square_avg[i] + (1 - self.alpha) * (
                grad**2
            )

            # Update parameters
            avg = np.sqrt(self.square_avg[i]) + self.eps
            p.data -= self.lr * grad / avg
