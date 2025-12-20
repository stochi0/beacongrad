"""
BeaconGrad vs PyTorch parity checks.

This compares BeaconGrad forward + backward against PyTorch for a small set of models:
- Linear
- 2-layer MLP

Run:
  uv run python examples/compare_with_pytorch.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from beacongrad.tensor import Tensor
from beacongrad import ops
from beacongrad.nn import Attention


def _max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def _max_err(a: np.ndarray, b: np.ndarray) -> float:
    return _max_abs(np.asarray(a) - np.asarray(b))


def _print_row(name: str, fwd: float, grad: float):
    print(f"{name:18s}  forward={fwd:.3e}  grad={grad:.3e}")


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "PyTorch is required for parity checks.\n"
            "Install it with uv:\n"
            "  uv pip install torch\n"
            f"Import error: {e}"
        )


def parity_linear():
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    dtype = torch.float64

    x_np = np.random.randn(7, 4).astype(np.float64)
    w_np = np.random.randn(3, 4).astype(np.float64)
    b_np = np.random.randn(3).astype(np.float64)

    # BeaconGrad
    x_bg = Tensor(x_np, requires_grad=True, dtype=np.float64)
    w_bg = Tensor(w_np, requires_grad=True, dtype=np.float64)
    b_bg = Tensor(b_np, requires_grad=True, dtype=np.float64)
    y_bg = (x_bg @ w_bg.T) + b_bg
    loss_bg = ops.mse_loss(y_bg, Tensor(np.zeros_like(y_bg.data), dtype=np.float64))
    loss_bg.backward()

    # PyTorch
    x_th = torch.from_numpy(x_np).to(dtype).requires_grad_(True)
    w_th = torch.from_numpy(w_np).to(dtype).requires_grad_(True)
    b_th = torch.from_numpy(b_np).to(dtype).requires_grad_(True)
    y_th = (x_th @ w_th.T) + b_th
    loss_th = torch.mean((y_th - torch.zeros_like(y_th)) ** 2)
    loss_th.backward()

    fwd_err = _max_err(y_bg.data, y_th.detach().cpu().numpy())
    dx_err = _max_err(x_bg.grad, x_th.grad.detach().cpu().numpy())
    dw_err = _max_err(w_bg.grad, w_th.grad.detach().cpu().numpy())
    db_err = _max_err(b_bg.grad, b_th.grad.detach().cpu().numpy())
    grad_err = max(dx_err, dw_err, db_err)

    print("\nLinear parity")
    print(f"  forward max error: {fwd_err:.3e}")
    print(f"  dx max error:      {dx_err:.3e}")
    print(f"  dW max error:      {dw_err:.3e}")
    print(f"  db max error:      {db_err:.3e}")

    return fwd_err, grad_err


def parity_mlp_2layer():
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    dtype = torch.float64

    bsz, din, dh, dout = 8, 5, 16, 3
    x_np = np.random.randn(bsz, din).astype(np.float64)
    w1_np = np.random.randn(dh, din).astype(np.float64)
    b1_np = np.random.randn(dh).astype(np.float64)
    w2_np = np.random.randn(dout, dh).astype(np.float64)
    b2_np = np.random.randn(dout).astype(np.float64)

    # BeaconGrad
    x_bg = Tensor(x_np, requires_grad=True, dtype=np.float64)
    w1_bg = Tensor(w1_np, requires_grad=True, dtype=np.float64)
    b1_bg = Tensor(b1_np, requires_grad=True, dtype=np.float64)
    w2_bg = Tensor(w2_np, requires_grad=True, dtype=np.float64)
    b2_bg = Tensor(b2_np, requires_grad=True, dtype=np.float64)

    h_bg = (x_bg @ w1_bg.T) + b1_bg
    h_bg = h_bg.relu()
    y_bg = (h_bg @ w2_bg.T) + b2_bg
    loss_bg = ops.mse_loss(y_bg, Tensor(np.zeros_like(y_bg.data), dtype=np.float64))
    loss_bg.backward()

    # PyTorch
    x_th = torch.from_numpy(x_np).to(dtype).requires_grad_(True)
    w1_th = torch.from_numpy(w1_np).to(dtype).requires_grad_(True)
    b1_th = torch.from_numpy(b1_np).to(dtype).requires_grad_(True)
    w2_th = torch.from_numpy(w2_np).to(dtype).requires_grad_(True)
    b2_th = torch.from_numpy(b2_np).to(dtype).requires_grad_(True)

    h_th = (x_th @ w1_th.T) + b1_th
    h_th = torch.relu(h_th)
    y_th = (h_th @ w2_th.T) + b2_th
    loss_th = torch.mean((y_th - torch.zeros_like(y_th)) ** 2)
    loss_th.backward()

    fwd_err = _max_err(y_bg.data, y_th.detach().cpu().numpy())
    errs = [
        _max_err(x_bg.grad, x_th.grad.detach().cpu().numpy()),
        _max_err(w1_bg.grad, w1_th.grad.detach().cpu().numpy()),
        _max_err(b1_bg.grad, b1_th.grad.detach().cpu().numpy()),
        _max_err(w2_bg.grad, w2_th.grad.detach().cpu().numpy()),
        _max_err(b2_bg.grad, b2_th.grad.detach().cpu().numpy()),
    ]
    grad_err = float(max(errs))

    print("\nMLP (2-layer) parity")
    print(f"  forward max error: {fwd_err:.3e}")
    print(f"  grad max error:    {grad_err:.3e}")

    return fwd_err, grad_err


def _torch_attention_single_head(query_th, *, scale: float):
    # query_th: (batch, seq, embed)
    # Self-attention with Q=K=V=query
    scores = (query_th @ query_th.transpose(-1, -2)) * scale  # (batch, seq, seq)
    attn = scores.softmax(dim=-1)
    out = attn @ query_th  # (batch, seq, embed)
    return out


def _torch_attention_multi_head(
    query_th,
    *,
    wq_th,
    wk_th,
    wv_th,
    wo_th,
    num_heads: int,
    head_dim: int,
    scale: float,
):
    # Mirrors beacongrad.nn.Attention forward for num_heads > 1 (dropout=0, no bias)
    # query_th: (batch, seq, embed)
    Q = query_th @ wq_th.T
    K = query_th @ wk_th.T
    V = query_th @ wv_th.T

    bsz, seq_len, embed_dim = query_th.shape
    Q = Q.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)  # (b, h, s, d)
    K = K.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

    scores = (Q @ K.transpose(-1, -2)) * scale  # (b, h, s, s)
    attn = scores.softmax(dim=-1)
    out = attn @ V  # (b, h, s, d)

    out = out.transpose(1, 2).reshape(bsz, seq_len, embed_dim)  # (b, s, embed)
    out = out @ wo_th.T
    return out


def parity_attention():
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    dtype = torch.float64

    print("\nAttention parity")

    # -------------------------
    # Single-head self-attention
    # -------------------------
    bsz, seq_len, embed_dim = 2, 5, 8
    x_np = np.random.randn(bsz, seq_len, embed_dim).astype(np.float64)

    attn_bg = Attention(embed_dim=embed_dim, num_heads=1, dropout=0.0)
    x_bg = Tensor(x_np, requires_grad=True, dtype=np.float64)
    y_bg = attn_bg(x_bg)
    loss_bg = ops.mse_loss(y_bg, Tensor(np.zeros_like(y_bg.data), dtype=np.float64))
    loss_bg.backward()

    x_th = torch.from_numpy(x_np).to(dtype).requires_grad_(True)
    y_th = _torch_attention_single_head(x_th, scale=float(attn_bg.scale))
    loss_th = torch.mean((y_th - torch.zeros_like(y_th)) ** 2)
    loss_th.backward()

    fwd_err_sh = _max_err(y_bg.data, y_th.detach().cpu().numpy())
    dx_err_sh = _max_err(x_bg.grad, x_th.grad.detach().cpu().numpy())
    grad_err_sh = float(dx_err_sh)
    _print_row("Attention (1h)", fwd_err_sh, grad_err_sh)

    # -------------------------
    # Multi-head self-attention (includes projection weights)
    # -------------------------
    bsz, seq_len, embed_dim, num_heads = 2, 4, 8, 2
    head_dim = embed_dim // num_heads
    x_np = np.random.randn(bsz, seq_len, embed_dim).astype(np.float64)

    wq_np = np.random.randn(embed_dim, embed_dim).astype(np.float64)
    wk_np = np.random.randn(embed_dim, embed_dim).astype(np.float64)
    wv_np = np.random.randn(embed_dim, embed_dim).astype(np.float64)
    wo_np = np.random.randn(embed_dim, embed_dim).astype(np.float64)

    attn_bg = Attention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    # Overwrite weights so Torch + BeaconGrad use identical parameters
    attn_bg.q_proj.weight.data = wq_np
    attn_bg.k_proj.weight.data = wk_np
    attn_bg.v_proj.weight.data = wv_np
    attn_bg.out_proj.weight.data = wo_np
    for p in [attn_bg.q_proj.weight, attn_bg.k_proj.weight, attn_bg.v_proj.weight, attn_bg.out_proj.weight]:
        p.data = p.data.astype(np.float64, copy=False)
        p.dtype = np.float64

    x_bg = Tensor(x_np, requires_grad=True, dtype=np.float64)
    y_bg = attn_bg(x_bg)
    loss_bg = ops.mse_loss(y_bg, Tensor(np.zeros_like(y_bg.data), dtype=np.float64))
    loss_bg.backward()

    x_th = torch.from_numpy(x_np).to(dtype).requires_grad_(True)
    wq_th = torch.from_numpy(wq_np).to(dtype).requires_grad_(True)
    wk_th = torch.from_numpy(wk_np).to(dtype).requires_grad_(True)
    wv_th = torch.from_numpy(wv_np).to(dtype).requires_grad_(True)
    wo_th = torch.from_numpy(wo_np).to(dtype).requires_grad_(True)

    y_th = _torch_attention_multi_head(
        x_th,
        wq_th=wq_th,
        wk_th=wk_th,
        wv_th=wv_th,
        wo_th=wo_th,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=float(attn_bg.scale),
    )
    loss_th = torch.mean((y_th - torch.zeros_like(y_th)) ** 2)
    loss_th.backward()

    fwd_err_mh = _max_err(y_bg.data, y_th.detach().cpu().numpy())
    errs = [
        _max_err(x_bg.grad, x_th.grad.detach().cpu().numpy()),
        _max_err(attn_bg.q_proj.weight.grad, wq_th.grad.detach().cpu().numpy()),
        _max_err(attn_bg.k_proj.weight.grad, wk_th.grad.detach().cpu().numpy()),
        _max_err(attn_bg.v_proj.weight.grad, wv_th.grad.detach().cpu().numpy()),
        _max_err(attn_bg.out_proj.weight.grad, wo_th.grad.detach().cpu().numpy()),
    ]
    grad_err_mh = float(max(errs))
    _print_row("Attention (2h)", fwd_err_mh, grad_err_mh)

    fwd_err = float(max(fwd_err_sh, fwd_err_mh))
    grad_err = float(max(grad_err_sh, grad_err_mh))
    return fwd_err, grad_err


if __name__ == "__main__":
    _require_torch()

    print("=" * 60)
    print("BeaconGrad vs PyTorch parity checks (float64)")
    print("=" * 60)

    rows = []

    fwd, grad = parity_linear()
    rows.append(("Linear", fwd, grad))

    fwd, grad = parity_mlp_2layer()
    rows.append(("MLP", fwd, grad))

    fwd, grad = parity_attention()
    rows.append(("Attention", fwd, grad))

    print("\nSummary")
    print("Model\tForward max error\tGrad max error")
    for name, fwd, grad in rows:
        print(f"{name}\t{fwd:.3e}\t{grad:.3e}")

    # Tight checks (as requested). Print-first, then assert.
    for name, fwd, grad in rows:
        if fwd >= 1e-6 or grad >= 1e-6:
            raise SystemExit(f"Parity check failed for {name}: fwd={fwd:.3e}, grad={grad:.3e}")

