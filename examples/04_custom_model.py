"""Create a custom neural network model."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src import Tensor, Module, Linear, ReLU, Dropout, Adam, MSELoss


class CustomResidualBlock(Module):
    """Custom residual block with skip connection."""

    def __init__(self, features):
        super().__init__()
        self.linear1 = Linear(features, features)
        self.linear2 = Linear(features, features)
        self.relu = ReLU()
        self.dropout = Dropout(0.1)

        # Register submodules
        self._modules = [self.linear1, self.linear2, self.relu, self.dropout]

    def forward(self, x):
        # Residual connection: out = x + f(x)
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out

    def __repr__(self):
        return f"CustomResidualBlock(features={self.linear1.in_features})"


class ResNet(Module):
    """Simple ResNet-style architecture."""

    def __init__(self, input_size, hidden_size, output_size, num_blocks=2):
        super().__init__()
        self.input_layer = Linear(input_size, hidden_size)
        self.blocks = [CustomResidualBlock(hidden_size) for _ in range(num_blocks)]
        self.output_layer = Linear(hidden_size, output_size)
        self.relu = ReLU()

        self._modules = (
            [self.input_layer] + self.blocks + [self.output_layer, self.relu]
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)

        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)

        x = self.output_layer(x)
        return x

    def __repr__(self):
        blocks_str = "\n  ".join(repr(b) for b in self.blocks)
        return (
            f"ResNet(\n  {self.input_layer},\n  {blocks_str},\n  {self.output_layer}\n)"
        )


if __name__ == "__main__":
    print("=" * 50)
    print("Custom Model: ResNet")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    y = np.sin(X.sum(axis=1, keepdims=True)) + 0.1 * np.random.randn(200, 1).astype(
        np.float32
    )

    X_tensor = Tensor(X)
    y_tensor = Tensor(y)

    # Create custom model
    model = ResNet(input_size=10, hidden_size=32, output_size=1, num_blocks=3)

    print(f"Model:\n{model}")
    print(f"\nNumber of parameters: {sum(p.size for p in model.parameters())}")

    # Train
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 100
    for epoch in range(epochs):
        # Forward
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    print("\nTraining complete!")
