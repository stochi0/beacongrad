## BeaconGrad

A tiny, fun NumPy autograd engine you can read in one sitting — **tensor-first**, with **broadcasting-aware** ops.

### What’s included
- **Autograd**: `Tensor` (n‑D arrays) with reverse-mode autodiff
- **Ops**: broadcasting-aware arithmetic, matmul, reductions, activations, basic losses
- **nn**: `Module`, `Linear`, `Sequential`, `MLP`, activations, dropout, batchnorm, embedding
- **optim**: `SGD`, `Adam`, `RMSprop`, `AdaGrad`

### Install
This project uses **uv**.

```bash
uv sync
```

### Install as a package (editable)
If you want `import beacongrad` to work from anywhere (and to hack on it locally), install it in editable mode:

```bash
uv sync
uv pip install -e .
uv run python -c "import beacongrad; print(beacongrad.__version__)"
```

### Build (wheel + sdist)

```bash
uv build
```

### Quick usage

```python
import numpy as np
from beacongrad import Tensor, Sequential, Linear, ReLU, MSELoss, Adam

# Synthetic regression data
np.random.seed(0)
X = Tensor(np.random.randn(256, 10).astype(np.float32))
y = Tensor(np.random.randn(256, 1).astype(np.float32))

model = Sequential(
    Linear(10, 32),
    ReLU(),
    Linear(32, 1),
)

loss_fn = MSELoss()
opt = Adam(model.parameters(), lr=1e-2)

for epoch in range(100):
    pred = model(X)
    loss = loss_fn(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch + 1) % 10 == 0:
        print(f"epoch={epoch+1} loss={loss.item():.4f}")
```

### Run examples

```bash
uv run python examples/01_basic_tensor.py
uv run python examples/02_neural_network.py
uv run python examples/03_classification.py
uv run python examples/04_custom_model.py
uv run python examples/05_mlp.py
```

### Notes
- `graphviz` is included as a dependency for future graph visualization (it’s not required for the core engine today).
