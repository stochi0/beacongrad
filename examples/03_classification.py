"""Train a classifier on synthetic data."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src import Tensor, MLP, CrossEntropyLoss, Adam

print("=" * 50)
print("Classification with MLP")
print("=" * 50)

# Generate synthetic classification data (3 classes)
np.random.seed(42)
n_samples = 300
n_features = 4
n_classes = 3

# Create clusters for each class
X_list = []
y_list = []
for i in range(n_classes):
    X_class = np.random.randn(n_samples // n_classes, n_features) + i * 3
    y_class = np.full(n_samples // n_classes, i)
    X_list.append(X_class)
    y_list.append(y_class)

X = np.vstack(X_list).astype(np.float32)
y = np.hstack(y_list).astype(np.int64)

# Shuffle data
indices = np.random.permutation(n_samples)
X = X[indices]
y = y[indices]

# Create tensors
X_tensor = Tensor(X)

print(f"Data: X shape={X.shape}, y shape={y.shape}")
print(f"Classes: {np.unique(y)}")

# Build model
model = MLP(
    input_size=n_features,
    hidden_sizes=[16, 8],
    output_size=n_classes,
    activation="relu",
    dropout=0.1,
)

print(f"\nModel:\n{model}")
print(f"Number of parameters: {sum(p.size for p in model.parameters())}")

# Loss and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
epochs = 50
for epoch in range(epochs):
    # Forward pass
    logits = model(X_tensor)
    loss = criterion(logits, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute accuracy
    predictions = np.argmax(logits.data, axis=1)
    accuracy = (predictions == y).mean()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
        )

# Final evaluation
model.eval()
logits = model(X_tensor)
predictions = np.argmax(logits.data, axis=1)
accuracy = (predictions == y).mean()

print(f"\nFinal Accuracy: {accuracy:.4f}")

# Show some predictions
print("\nSample predictions (first 10):")
for i in range(10):
    print(f"  True: {y[i]}, Predicted: {predictions[i]}")
