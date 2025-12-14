"""Basic tensor operations and autograd."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import tensor

print("=" * 50)
print("Basic Tensor Operations")
print("=" * 50)

# Create tensors
x = tensor([1.0, 2.0, 3.0], requires_grad=True)
y = tensor([4.0, 5.0, 6.0], requires_grad=True)

print(f"x: {x}")
print(f"y: {y}")

# Arithmetic operations
z = x + y
print(f"\nx + y = {z}")

z = x * y
print(f"x * y = {z}")

# Backward pass
z = (x * y).sum()
print(f"\n(x * y).sum() = {z}")
z.backward()

print(f"x.grad = {x.grad}")  # dy/dx = y
print(f"y.grad = {y.grad}")  # dy/dy = x

print("\n" + "=" * 50)
print("Broadcasting Example")
print("=" * 50)

# Broadcasting
a = tensor([[1.0], [2.0], [3.0]], requires_grad=True)
b = tensor([4.0, 5.0, 6.0, 7.0], requires_grad=True)

print(f"a shape: {a.shape}")
print(f"b shape: {b.shape}")

c = a * b  # Broadcasting: (3, 1) * (4,) -> (3, 4)
print(f"a * b shape: {c.shape}")
print(f"a * b:\n{c.data}")

# Backward through broadcast
loss = c.sum()
loss.backward()

print(f"\na.grad:\n{a.grad}")
print(f"b.grad:\n{b.grad}")
