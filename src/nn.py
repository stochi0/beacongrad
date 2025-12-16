"""
Neural network modules and layers (similar to torch.nn).

Provides building blocks for creating neural networks with automatic
parameter management and gradient computation.
"""

import numpy as np
from typing import List
from .tensor import Tensor, randn, zeros, ones


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

class MLP(Module):
    """
    Multi-Layer Perceptron (fully-connected neural network).

    Args:
        input_size: Size of input features
        hidden_sizes: List of hidden layer sizes
        output_size: Size of output
        activation: Activation function ('relu', 'tanh', 'sigmoid')
        dropout: Dropout probability (0 means no dropout)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(Linear(prev_size, hidden_size))

            # Add activation
            if activation == "relu":
                layers.append(ReLU())
            elif activation == "tanh":
                layers.append(Tanh())
            elif activation == "sigmoid":
                layers.append(Sigmoid())

            # Add dropout
            if dropout > 0:
                layers.append(Dropout(dropout))

            prev_size = hidden_size

        # Output layer (no activation)
        layers.append(Linear(prev_size, output_size))

        self.network = Sequential(*layers)
        self._modules = [self.network]

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def __repr__(self):
        return repr(self.network)

class ReLU(Module):
    """ReLU activation function."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation function."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Tanh activation function."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def __repr__(self):
        return "Tanh()"

class Softmax(Module):
    """Softmax activation function."""

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)

    def __repr__(self):
        return f"Softmax(axis={self.axis})"


class Sequential(Module):
    """
    Sequential container of modules.

    Modules are applied in the order they are passed.
    """

    def __init__(self, *modules):
        super().__init__()
        self._modules = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules:
            x = module(x)
        return x

    def __repr__(self):
        module_str = ",\n  ".join(repr(m) for m in self._modules)
        return f"Sequential(\n  {module_str}\n)"

class Embedding(Module):
    """
    Embedding layer for discrete tokens.

    Args:
        num_embeddings: Size of vocabulary
        embedding_dim: Dimension of embeddings
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize embeddings
        self.weight = randn(num_embeddings, embedding_dim, requires_grad=True) * 0.01
        self._parameters = [self.weight]

    def forward(self, indices: np.ndarray) -> Tensor:
        """
        Look up embeddings for given indices.

        Args:
            indices: Array of token indices

        Returns:
            Embeddings for the given indices
        """
        # Note: indices are not differentiable; gradients flow to `self.weight`.
        idx = np.asarray(indices, dtype=np.int64)

        out = Tensor(
            self.weight.data[idx],
            requires_grad=self.weight.requires_grad,
            _children=(self.weight,),
            _op="embedding",
        )

        def _backward():
            if out.grad is None:
                return
            if not self.weight.requires_grad:
                return

            # Accumulate dL/dW for the rows that were selected.
            # Handles repeated indices correctly via np.add.at.
            grad_out = out.grad
            idx_flat = idx.reshape(-1)
            grad_flat = grad_out.reshape(-1, self.embedding_dim)
            np.add.at(self.weight.grad, idx_flat, grad_flat)

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"

class Dropout(Module):
    """
    Dropout layer for regularization.

    During training, randomly zeroes elements with probability p.
    During evaluation, performs identity operation.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0:
            mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            return x * Tensor(mask, requires_grad=False)
        return x

    def __repr__(self):
        return f"Dropout(p={self.p})"

class BatchNorm1d(Module):
    """
    Batch normalization for 1D inputs.

    Normalizes input across the batch dimension.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Learnable parameters
        self.gamma = ones(num_features, requires_grad=True)
        self.beta = zeros(num_features, requires_grad=True)

        # Running statistics (not trained)
        self.running_mean = zeros(num_features)
        self.running_var = ones(num_features)

        self._parameters = [self.gamma, self.beta]

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(axis=0)
            batch_var = ((x - batch_mean) ** 2).mean(axis=0)

            # Update running statistics
            self.running_mean.data = (
                1 - self.momentum
            ) * self.running_mean.data + self.momentum * batch_mean.data
            self.running_var.data = (
                1 - self.momentum
            ) * self.running_var.data + self.momentum * batch_var.data

            # Normalize
            x_norm = (x - batch_mean) / ((batch_var + self.eps) ** 0.5)
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / ((self.running_var + self.eps) ** 0.5)

        # Scale and shift
        return self.gamma * x_norm + self.beta

    def __repr__(self):
        return f"BatchNorm1d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"

# ==================== Loss Functions as Modules ====================

class MSELoss(Module):
    """Mean squared error loss."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        from . import ops

        return ops.mse_loss(pred, target)

    def __repr__(self):
        return "MSELoss()"

class CrossEntropyLoss(Module):
    """Cross-entropy loss (combines softmax and negative log-likelihood)."""

    def forward(self, pred: Tensor, target) -> Tensor:
        from . import ops

        return ops.cross_entropy(pred, target)

    def __repr__(self):
        return "CrossEntropyLoss()"

class BCELoss(Module):
    """Binary cross-entropy loss."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        from . import ops

        return ops.binary_cross_entropy(pred, target)

    def __repr__(self):
        return "BCELoss()"
