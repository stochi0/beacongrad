# BeaconGrad

NumPy autograd and a neural-network library with a PyTorch-style API.
BeaconGrad builds a computation graph from tensor operations and propagates
gradients through it using reverse-mode automatic differentiation.

Includes tensor operations, neural-network layers, optimizers, gradient checks,
and graph visualization. This is an autodiff study implementation; NumPy operations
are not optimized for large-scale training.

## Installation

Use Python 3.9 or later and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/stochi0/beacongrad.git
cd beacongrad
uv sync --locked
```

For an editable package installation, run `uv pip install -e .` after syncing.

## A first gradient

```python
from beacongrad import Tensor

x = Tensor([2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()

print(loss.item())  # 13.0
print(x.grad)      # [4. 6.]
```

For a complete training loop:

```sh
uv run python examples/02_neural_network.py
```

## Gradient and parity checks

Check derivatives against finite differences:

```sh
uv run python examples/08_gradcheck.py
```

Compare forward outputs and gradients against PyTorch for linear layers, an MLP,
and attention. PyTorch is an optional dependency for this comparison:

```sh
uv pip install torch
uv run python examples/compare_with_pytorch.py
```

The script reports maximum absolute output and gradient errors for its test
cases. These checks cover the included examples rather than the entire API.
The retained comparison plot can be regenerated with
`uv run python examples/10_parity_plot.py`.

![Maximum absolute output and gradient errors in the retained PyTorch comparison](assets/parity_comparison.png)

See [examples/](examples/) for training, attention, custom modules, and graph
visualization. Graph rendering also requires the system Graphviz executable.
The retained [parity plot](assets/parity_comparison.png) and
[graph example](assets/example_06_graph_simple.svg) remain available.

## Development and references

```sh
uv build
```

[micrograd](https://github.com/karpathy/micrograd) is a useful starting point for
scalar autodiff. BeaconGrad extends that learning exercise to tensor operations.
The unused mascot asset retains its [OpenMoji attribution](assets/ATTRIBUTION.md).

## Citation

<!-- BEGIN generated citations -->

```bibtex
@software{bodadeprojectbeacongrad,
  author = {Bodade, Ayush},
  title = {{BeaconGrad}},
  url = {https://github.com/stochi0/beacongrad},
  note = {Software implementation; cite the original research separately. Specify the commit or version used.}
}
```

<!-- END generated citations -->
