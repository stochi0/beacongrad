"""
automatic differentiation library.

Main components:
- Tensor: Core tensor class with autograd
- nn: Neural network modules
- ops: Functional operations
- optim: Optimization algorithms
"""

__version__ = "0.0.1"

# Core tensor functionality
from .tensor import (
    Tensor,
    tensor,
    zeros,
    ones,
    randn,
    rand,
)

# Import submodules
from . import nn
from . import ops
from . import optim

# Commonly used items
from .nn import (
    Module,
    Linear,
    MLP,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Sequential,
    Dropout,
    BatchNorm1d,
    Embedding,
    MSELoss,
    CrossEntropyLoss,
    BCELoss,
)

from .optim import (
    SGD,
    Adam,
    RMSprop,
    AdaGrad,
)

# Functional operations (most commonly used)
from .ops import (
    # Activations
    relu,
    sigmoid,
    tanh,
    softmax,
    log_softmax,
    # Loss functions
    mse_loss,
    cross_entropy,
    binary_cross_entropy,
    # Math operations
    matmul,
    sum,
    mean,
    exp,
    log,
)

__all__ = [
    # Core
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "rand",
    # Modules
    "nn",
    "ops",
    "optim",
    # Neural network components
    "Module",
    "Linear",
    "MLP",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Sequential",
    "Dropout",
    "BatchNorm1d",
    "Embedding",
    # Loss functions
    "MSELoss",
    "CrossEntropyLoss",
    "BCELoss",
    "mse_loss",
    "cross_entropy",
    "binary_cross_entropy",
    # Optimizers
    "SGD",
    "Adam",
    "RMSprop",
    "AdaGrad",
    # Operations
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "matmul",
    "sum",
    "mean",
    "exp",
    "log",
]
