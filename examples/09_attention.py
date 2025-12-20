"""Attention mechanism examples using beacongrad."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from beacongrad import Tensor, Attention, Linear, MSELoss, SGD, Adam

print("=" * 60)
print("Attention Mechanism Examples")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# ==================== Example 1: Single-Head Self-Attention ====================
print("\n" + "=" * 60)
print("Example 1: Single-Head Self-Attention")
print("=" * 60)

# Create a simple self-attention layer
embed_dim = 64
seq_len = 10
batch_size = 2

# Create random input: (batch_size, seq_len, embed_dim)
x = Tensor(np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32))

# Create single-head attention
attn = Attention(embed_dim=embed_dim, num_heads=1, dropout=0.0)

print(f"\nInput shape: {x.shape}")
print(f"Attention module: {attn}")

# Forward pass
out = attn(x)

print(f"Output shape: {out.shape}")
print(f"Input shape preserved: {x.shape == out.shape}")

# Test backward pass
loss = out.sum()
loss.backward()

print(f"Loss: {loss.item():.6f}")
print("Gradients computed successfully!")

# ==================== Example 2: Multi-Head Attention ====================
print("\n" + "=" * 60)
print("Example 2: Multi-Head Attention")
print("=" * 60)

# Create multi-head attention (8 heads)
num_heads = 8
multi_head_attn = Attention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)

print(f"\nMulti-head attention: {multi_head_attn}")
print(f"Number of heads: {num_heads}")
print(f"Head dimension: {embed_dim // num_heads}")

# Forward pass
out_multi = multi_head_attn(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out_multi.shape}")

# Test backward pass
loss_multi = out_multi.sum()
loss_multi.backward()

print(f"Loss: {loss_multi.item():.6f}")
print("Multi-head attention gradients computed successfully!")

# ==================== Example 3: Cross-Attention ====================
print("\n" + "=" * 60)
print("Example 3: Cross-Attention (Different Q, K, V)")
print("=" * 60)

# Create query, key, and value with different sequence lengths
query = Tensor(np.random.randn(batch_size, 5, embed_dim).astype(np.float32))
key = Tensor(np.random.randn(batch_size, 10, embed_dim).astype(np.float32))
value = Tensor(np.random.randn(batch_size, 10, embed_dim).astype(np.float32))

print(f"\nQuery shape: {query.shape}")
print(f"Key shape: {key.shape}")
print(f"Value shape: {value.shape}")

# Forward pass with cross-attention
out_cross = multi_head_attn(query, key=key, value=value)

print(f"Output shape: {out_cross.shape}")
print("Note: Output sequence length matches query sequence length")

# ==================== Example 4: Attention with Masking ====================
print("\n" + "=" * 60)
print("Example 4: Attention with Masking")
print("=" * 60)

# Create a causal mask (lower triangular) for autoregressive models
seq_len_q = 5
seq_len_k = 5
mask = np.tril(np.ones((seq_len_q, seq_len_k), dtype=np.float32))
mask_tensor = Tensor(mask)

print(f"\nMask shape: {mask.shape}")
print("Mask (1 = attend, 0 = mask):")
print(mask)

# Create input
x_masked = Tensor(np.random.randn(batch_size, seq_len_q, embed_dim).astype(np.float32))

# Forward pass with mask
out_masked = multi_head_attn(x_masked, mask=mask_tensor)

print(f"Input shape: {x_masked.shape}")
print(f"Output shape: {out_masked.shape}")
print("Masked attention applied successfully!")

# ==================== Example 5: Training a Simple Model with Attention ====================
print("\n" + "=" * 60)
print("Example 5: Training a Simple Model with Attention")
print("=" * 60)

# Create a simple model: Attention + Linear layer
class AttentionModel:
    def __init__(self, embed_dim, num_heads=4, output_dim=1):
        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
        self.linear = Linear(embed_dim, output_dim)
    
    def __call__(self, x):
        # Apply attention
        attn_out = self.attention(x)
        # Pool over sequence dimension (mean pooling)
        pooled = attn_out.mean(axis=1)  # (batch_size, embed_dim)
        # Final linear layer
        out = self.linear(pooled)
        return out
    
    def parameters(self):
        return self.attention.parameters() + self.linear.parameters()
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

# Create model
model = AttentionModel(embed_dim=embed_dim, num_heads=4, output_dim=1)

# Generate synthetic data
n_samples = 50
X_train = Tensor(np.random.randn(n_samples, seq_len, embed_dim).astype(np.float32))
# Simple target: sum of first dimension of embeddings
y_train = Tensor((X_train.data[:, :, 0].sum(axis=1) * 0.1).reshape(-1, 1).astype(np.float32))

print(f"\nTraining data shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")

# Loss and optimizer
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 20
print("\nTraining...")
for epoch in range(epochs):
    # Forward pass
    pred = model(X_train)
    loss = criterion(pred, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

print("\nTraining complete!")

# Test on new data
X_test = Tensor(np.random.randn(5, seq_len, embed_dim).astype(np.float32))
y_test = Tensor((X_test.data[:, :, 0].sum(axis=1) * 0.1).reshape(-1, 1).astype(np.float32))

model.attention.eval()  # Set to eval mode (disables dropout)
pred_test = model(X_test)
test_loss = criterion(pred_test, y_test)

print(f"\nTest Loss: {test_loss.item():.6f}")
print(f"Sample predictions: {pred_test.data[:3].flatten()}")
print(f"Sample targets: {y_test.data[:3].flatten()}")

# ==================== Example 6: Visualizing Attention Weights ====================
print("\n" + "=" * 60)
print("Example 6: Visualizing Attention Patterns")
print("=" * 60)

# Create a simple attention layer to inspect weights
simple_attn = Attention(embed_dim=32, num_heads=1, dropout=0.0)

# Create input with clear pattern
x_pattern = Tensor(np.random.randn(1, 5, 32).astype(np.float32))

# Forward pass
out_pattern = simple_attn(x_pattern)

# To visualize attention weights, we'd need to extract them from the forward pass
# For now, we'll just show that the mechanism works
print(f"Input shape: {x_pattern.shape}")
print(f"Output shape: {out_pattern.shape}")
print("Attention mechanism applied successfully!")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
