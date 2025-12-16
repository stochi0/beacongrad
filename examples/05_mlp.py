import sys
import os
from typing import List
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import Tensor, Module, Linear, ReLU, Tanh, Sigmoid, Sequential, Dropout, MSELoss, SGD, MLP
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

def regression_accuracy(y_pred: Tensor, y_true: Tensor, tolerance: float = 0.1):
    """
    Compute regression "accuracy" as the percentage of predictions within
    a given tolerance of the true value.
    """
    pred_np = y_pred.data.flatten()
    true_np = y_true.data.flatten()
    # Within tolerance from true
    correct = np.abs(pred_np - true_np) < tolerance
    return correct.mean()

model = MLP(input_size=5, hidden_sizes=[10, 10], output_size=1, activation="relu", dropout=0.1)

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

    # Compute "accuracy" (percentage within tolerance = 0.1)
    accuracy = regression_accuracy(pred, y_tensor, tolerance=0.1)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Accuracy: {accuracy:.4f}")

print("\nTraining complete!")

# Test predictions
with_preds = model(X_tensor)
final_loss = criterion(with_preds, y_tensor)
final_acc = regression_accuracy(with_preds, y_tensor, tolerance=0.1)
print(f"Final loss: {final_loss.item():.6f}, Final Accuracy: {final_acc:.4f}")