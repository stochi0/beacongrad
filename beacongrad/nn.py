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
        self.training = True

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


class Attention(Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        embed_dim: Dimension of embeddings (d_model)
        num_heads: Number of attention heads (for multi-head attention, default=1)
        dropout: Dropout probability (default=0.0)
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(d_k)
        
        # For multi-head attention, we'll project Q, K, V
        if num_heads > 1:
            self.q_proj = Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = Linear(embed_dim, embed_dim, bias=False)
            self.out_proj = Linear(embed_dim, embed_dim, bias=False)
            self._modules = [self.q_proj, self.k_proj, self.v_proj, self.out_proj]
        else:
            # Single head attention - no projections needed if Q, K, V are provided
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None
        
        if dropout > 0:
            self.dropout = Dropout(dropout)
            self._modules.append(self.dropout)
        else:
            self.dropout = None

    def forward(
        self, 
        query: Tensor, 
        key: Tensor = None, 
        value: Tensor = None,
        mask: Tensor = None
    ) -> Tensor:
        """
        Forward pass of attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, embed_dim)
            key: Key tensor of shape (batch_size, seq_len_k, embed_dim). 
                 If None, uses query.
            value: Value tensor of shape (batch_size, seq_len_v, embed_dim).
                   If None, uses query.
            mask: Optional mask tensor of shape (batch_size, seq_len_q, seq_len_k)
                  or (seq_len_q, seq_len_k). Values should be 0 for masked positions.
        
        Returns:
            Output tensor of shape (batch_size, seq_len_q, embed_dim)
        """
        # If key/value not provided, use query (self-attention)
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Multi-head attention: project Q, K, V
        if self.num_heads > 1:
            Q = self.q_proj(query)  # (batch_size, seq_len_q, embed_dim)
            K = self.k_proj(key)    # (batch_size, seq_len_k, embed_dim)
            V = self.v_proj(value)  # (batch_size, seq_len_v, embed_dim)
            
            # Reshape for multi-head: (batch_size, seq_len, num_heads, head_dim)
            Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
            K = K.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)
            V = V.reshape(batch_size, value.shape[1], self.num_heads, self.head_dim)
            
            # Transpose to (batch_size, num_heads, seq_len, head_dim)
            Q = Q.transpose(axes=(0, 2, 1, 3))
            K = K.transpose(axes=(0, 2, 1, 3))
            V = V.transpose(axes=(0, 2, 1, 3))
        else:
            Q = query
            K = key
            V = value
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # For multi-head: (batch_size, num_heads, seq_len_q, head_dim) @ 
        #                 (batch_size, num_heads, head_dim, seq_len_k)
        # For single-head: (batch_size, seq_len_q, embed_dim) @ 
        #                  (batch_size, embed_dim, seq_len_k)
        if self.num_heads > 1:
            # Q: (batch_size, num_heads, seq_len_q, head_dim)
            # K: (batch_size, num_heads, seq_len_k, head_dim)
            # Need to transpose K: (batch_size, num_heads, head_dim, seq_len_k)
            K_T = K.transpose(axes=(0, 1, 3, 2))
            scores = Q @ K_T * self.scale  # (batch_size, num_heads, seq_len_q, seq_len_k)
        else:
            # Single head: (batch_size, seq_len_q, embed_dim) @ (batch_size, embed_dim, seq_len_k)
            scores = Q @ K.transpose(axes=(0, 2, 1)) * self.scale
            # scores: (batch_size, seq_len_q, seq_len_k)
        
        # Apply mask if provided
        if mask is not None:
            # Mask should be 0 for positions to mask, 1 for valid positions
            # Convert to large negative values for masked positions
            if mask.ndim == 2:
                # Broadcast mask: (seq_len_q, seq_len_k) -> (batch_size, seq_len_q, seq_len_k)
                mask = Tensor(mask.data[None, :, :], requires_grad=False)
            elif mask.ndim == 3:
                # (batch_size, seq_len_q, seq_len_k)
                mask = mask
            else:
                raise ValueError(f"Mask must be 2D or 3D, got {mask.ndim}D")
            
            # Apply mask: set masked positions to large negative value
            scores = scores + (1.0 - mask) * (-1e9)
        
        # Apply softmax to get attention weights
        if self.num_heads > 1:
            attn_weights = scores.softmax(axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        else:
            attn_weights = scores.softmax(axis=-1)  # (batch_size, seq_len_q, seq_len_k)
        
        # Apply dropout
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: attn_weights @ V
        if self.num_heads > 1:
            # attn_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
            # V: (batch_size, num_heads, seq_len_v, head_dim)
            out = attn_weights @ V  # (batch_size, num_heads, seq_len_q, head_dim)
            
            # Concatenate heads: transpose and reshape
            out = out.transpose(axes=(0, 2, 1, 3))  # (batch_size, seq_len_q, num_heads, head_dim)
            out = out.reshape(batch_size, seq_len_q, self.embed_dim)
            
            # Output projection
            out = self.out_proj(out)
        else:
            # Single head: (batch_size, seq_len_q, seq_len_k) @ (batch_size, seq_len_v, embed_dim)
            out = attn_weights @ V  # (batch_size, seq_len_q, embed_dim)
        
        return out

    def __repr__(self):
        return f"Attention(embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout.p if self.dropout else 0.0})"
