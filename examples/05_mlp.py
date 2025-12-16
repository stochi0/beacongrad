import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import Tensor, Module, Linear, ReLU, Tanh, Sigmoid, Sequential, Dropout
import numpy as np

print("=" * 50)
print("MLP")
print("=" * 50)

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 5).astype(np.float32)

true_weights = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = X @ true_weights + 0.1 * np.random.randn(100).astype(np.float32)

# Create tensors
X_tensor = Tensor(X)
y_tensor = Tensor(y.reshape(-1, 1))

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
