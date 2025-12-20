import numpy as np

from beacongrad.tensor import Tensor
from beacongrad.utils import gradcheck
from beacongrad import ops
from beacongrad.nn import Attention


def test_scalar_elementwise():
    x = Tensor(np.random.randn(3, 2), requires_grad=True, dtype=np.float64)
    y = Tensor(np.random.randn(3, 2), requires_grad=True, dtype=np.float64)

    gradcheck(lambda x, y: (x + y).sum(), [x, y])
    gradcheck(lambda x, y: (x - y).sum(), [x, y])
    gradcheck(lambda x, y: (x * y).sum(), [x, y])
    gradcheck(lambda x, y: (x / (y + 1.5)).sum(), [x, y])  # avoid div by ~0

    # pow: positive base only
    base = Tensor(np.abs(np.random.randn(4, 3)) + 1.0, requires_grad=True, dtype=np.float64)
    exp = Tensor(np.random.randn(4, 3), requires_grad=True, dtype=np.float64)
    gradcheck(lambda a, b: (a**b).sum(), [base, exp])

    # activations (avoid relu kink at 0)
    z = Tensor(np.random.randn(5, 4) + 0.1, requires_grad=True, dtype=np.float64)
    gradcheck(lambda z: z.relu().sum(), [z])
    gradcheck(lambda z: z.tanh().sum(), [z])
    gradcheck(lambda z: z.sigmoid().sum(), [z])


def test_broadcasting():
    x = Tensor(np.random.randn(2, 3), requires_grad=True, dtype=np.float64)
    b = Tensor(np.random.randn(3), requires_grad=True, dtype=np.float64)
    gradcheck(lambda x, b: (x + b).sum(), [x, b])
    gradcheck(lambda x, b: (x * b).sum(), [x, b])


def test_sum_reshape_transpose():
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True, dtype=np.float64)
    gradcheck(lambda x: x.sum(axis=1).sum(), [x])
    gradcheck(lambda x: x.reshape(6, 4).sum(), [x])
    gradcheck(lambda x: x.transpose((2, 0, 1)).sum(), [x])


def test_matmul_grad():
    x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
    w = Tensor(np.random.randn(4, 5), requires_grad=True, dtype=np.float64)

    def f(x, w):
        return (x @ w).sum()

    gradcheck(f, [x, w])

    # batched
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True, dtype=np.float64)
    b = Tensor(np.random.randn(2, 4, 5), requires_grad=True, dtype=np.float64)
    gradcheck(lambda a, b: (a @ b).sum(), [a, b])


def test_cross_entropy():
    # small batch
    logits = Tensor(np.random.randn(4, 3), requires_grad=True, dtype=np.float64)
    target = np.array([0, 2, 1, 0], dtype=np.int64)
    gradcheck(lambda l: ops.cross_entropy(l, target), [logits])


def test_attention():
    embed_dim = 8
    batch_size = 2
    seq_len = 4
    
    # Single-head self-attention - test input
    query = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True, dtype=np.float64)
    attn_single = Attention(embed_dim=embed_dim, num_heads=1, dropout=0.0)
    gradcheck(lambda q: attn_single(q).sum(), [query])
    
    # Multi-head self-attention - test input
    query_multi = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True, dtype=np.float64)
    attn_multi = Attention(embed_dim=embed_dim, num_heads=2, dropout=0.0)
    gradcheck(lambda q: attn_multi(q).sum(), [query_multi])
    
    # Multi-head attention - test module parameters
    # Convert weights to float64 for gradcheck
    query_param = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=False, dtype=np.float64)
    
    # Test Q projection weight
    q_weight = attn_multi.q_proj.weight
    q_weight.data = q_weight.data.astype(np.float64, copy=False)
    q_weight.dtype = np.float64
    gradcheck(lambda w: attn_multi(query_param).sum(), [q_weight])
    
    # Test K projection weight
    k_weight = attn_multi.k_proj.weight
    k_weight.data = k_weight.data.astype(np.float64, copy=False)
    k_weight.dtype = np.float64
    gradcheck(lambda w: attn_multi(query_param).sum(), [k_weight])
    
    # Test V projection weight
    v_weight = attn_multi.v_proj.weight
    v_weight.data = v_weight.data.astype(np.float64, copy=False)
    v_weight.dtype = np.float64
    gradcheck(lambda w: attn_multi(query_param).sum(), [v_weight])
    
    # Test output projection weight
    out_weight = attn_multi.out_proj.weight
    out_weight.data = out_weight.data.astype(np.float64, copy=False)
    out_weight.dtype = np.float64
    gradcheck(lambda w: attn_multi(query_param).sum(), [out_weight])
    
    # Cross-attention with different Q, K, V
    query_cross = Tensor(np.random.randn(batch_size, 3, embed_dim), requires_grad=True, dtype=np.float64)
    key_cross = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True, dtype=np.float64)
    value_cross = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True, dtype=np.float64)
    
    attn_cross = Attention(embed_dim=embed_dim, num_heads=1, dropout=0.0)
    gradcheck(lambda q, k, v: attn_cross(q, key=k, value=v).sum(), [query_cross, key_cross, value_cross])


if __name__ == "__main__":
    np.random.seed(0)
    test_scalar_elementwise()
    test_broadcasting()
    test_sum_reshape_transpose()
    test_matmul_grad()
    test_cross_entropy()
    test_attention()
    print("All gradchecks passed.")


