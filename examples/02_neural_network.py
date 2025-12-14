"""Train a simple neural network on synthetic data."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src import Tensor, Linear, ReLU, Sequential, MSELoss, SGD

print("=" * 50)
print("Neural Network Training")
print("=" * 50)

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 5).astype(np.float32)
# y = X @ [1, 2, 3, 4, 5] + noise
true_weights = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = X @ true_weights + 0.1 * np.random.randn(100).astype(np.float32)

# Create tensors
X_tensor = Tensor(X)
y_tensor = Tensor(y.reshape(-1, 1))

print(f"Training data: X shape={X.shape}, y shape={y.shape}")

# Build model
model = Sequential(Linear(5, 10), ReLU(), Linear(10, 1))

print(f"\nModel:\n{model}")

# Loss and optimizer
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

print("\nTraining complete!")

# Test predictions
with_preds = model(X_tensor)
final_loss = criterion(with_preds, y_tensor)
print(f"Final loss: {final_loss.item():.6f}")
