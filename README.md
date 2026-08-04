# fracmem

[![PyPI](https://img.shields.io/pypi/v/fracmem.svg)](https://pypi.org/project/fracmem/)
[![Python](https://img.shields.io/pypi/pyversions/fracmem.svg)](https://pypi.org/project/fracmem/)
[![Tests](https://github.com/edith-lang/fracmem/actions/workflows/tests.yml/badge.svg)](https://github.com/edith-lang/fracmem/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Compressed fractional-order derivative filters for embedded and real-time systems, with a **certified offline error bound** and **data-refined weights**.

Fractional-order derivatives (used in fractional-order PID control, viscoelastic material models, battery impedance models, and more) are expensive to compute directly: they depend on the *entire* signal history, so a naive implementation's cost grows without bound as the system runs. `fracmem` compresses that unbounded history into a fixed-size recursive filter — constant memory, constant compute per sample, forever — using a Sum-of-Exponentials (SOE) decomposition with decay rates derived from an exact mathematical identity, refined with a small amount of representative training data.

## Contents

- [Install](#install)
- [Quick example](#quick-example)
- [Examples](#examples)
- [Features](#features)
- [Why this exists](#why-this-exists)
- [What it's good for](#what-its-good-for)
- [Citing / background](#citing--background)
- [License](#license)

## Install

### From PyPI (recommended)

```bash
pip install fracmem

# with the optional PyTorch / GPU backend
pip install fracmem[torch]

# with the optional JAX backend
pip install fracmem[jax]

# both, plus test dependencies
pip install fracmem[torch,jax,dev]
```

### Directly from GitHub

```bash
pip install git+https://github.com/edith-lang/fracmem.git
```

### From a local clone (editable / development install)

```bash
git clone https://github.com/edith-lang/fracmem.git
cd fracmem
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

`fracmem` itself only depends on `numpy` and `scipy` — `torch` and `jax` are optional extras, needed only for their respective backends.

## Quick example

```python
import numpy as np
from fracmem import CompressedFractionalFilter

ALPHA = 0.5   # fractional order (e.g. 0.5 = "half derivative")
H = 0.01      # sample time (seconds)
L = 32        # exact local window length (samples)
P = 16        # number of compressed tail modes

# A handful of representative training signals -- whatever your filter
# will actually see in deployment. 8-32 signals is typically enough.
train_signals = [np.random.randn(3000).cumsum() * 0.01 for _ in range(8)]

f = CompressedFractionalFilter(alpha=ALPHA, h=H, L=L, p=P)
f.fit(train_signals, j_max=10_000)   # offline, done once

# Deploy: O(L+p) multiply-adds per sample, independent of signal length
new_signal = np.random.randn(50_000).cumsum() * 0.01
y_hat = f.predict(new_signal)
```

That's it — `f.predict` costs the same fixed amount of work per sample whether `new_signal` is 50 samples or 50 million.

## Examples

The [`examples/`](examples/) directory has a standalone, runnable script for every feature below — clone the repo and run them directly:

```bash
git clone https://github.com/edith-lang/fracmem.git
cd fracmem
pip install -e ".[torch,jax,dev]"
python examples/basic-usage.py
```

| Script | What it shows |
|---|---|
| [`basic-usage.py`](examples/basic-usage.py) | Fit, deploy, and check accuracy + speed against the exact reference. Start here. |
| [`streaming.py`](examples/streaming.py) | One-sample-at-a-time `.step()`, checked against batch `.predict()`. |
| [`definitions.py`](examples/definitions.py) | GL/RL vs Caputo on a signal with a nonzero initial value. |
| [`auto-params.py`](examples/auto-params.py) | `CompressedFractionalFilter.auto(...)` picks `(L, p)` from a target RMSE. |
| [`adaptive-soe.py`](examples/adaptive-soe.py) | The low-level SOE tail construction growing `p` until a tolerance is met. |
| [`save-and-load.py`](examples/save-and-load.py) | Persist a fitted filter to disk and reload it elsewhere. |
| [`torch-backend.py`](examples/torch-backend.py) | `TorchFractionalLayer`, a differentiable, GPU-ready `nn.Module`. |
| [`jax-backend.py`](examples/jax-backend.py) | `jit`/`grad`/`vmap`-compatible functional predict. |
| [`embedded-export.py`](examples/embedded-export.py) | Export a fitted filter as a standalone MicroPython file, then verify it under CPython. |

The [`ros2_ws/src/fracmem_ros2`](ros2_ws/src/fracmem_ros2) package is a fuller worked example for streaming a live topic through a fitted filter — see [ROS2](#ros2) below.

## Features

- **Derivative definitions**: Grünwald-Letnikov / Riemann-Liouville (the default) and Caputo.
- **Streaming**: an `O(L+p)`-per-sample `.step()` API for online use, alongside the batch `.predict()`.
- **Adaptive SOE**: automatically grow the number of exponential modes until a target error tolerance is met, instead of hand-picking `p`.
- **Automatic parameter selection**: `CompressedFractionalFilter.auto(...)` grid-searches `(L, p)` for the cheapest deployed filter meeting a target RMSE.
- **GPU / PyTorch**: `fracmem.backends.torch_backend.TorchFractionalLayer` — a differentiable `nn.Module`, runs on CUDA/MPS via `.to(device)`.
- **JAX**: `fracmem.backends.jax_backend` — `jit`-, `grad`-, and `vmap`-compatible functional predict, using `lax.scan` for the recurrence.
- **ROS2**: a standalone `ament_python` package ([`ros2_ws/src/fracmem_ros2`](ros2_ws/src/fracmem_ros2)) that streams a topic through a fitted filter.
- **ESP32 / MicroPython**: `fracmem.embedded` exports a fitted filter as a dependency-free, single-file MicroPython runtime — no numpy/scipy on the device, fixed RAM.

### Derivative definitions

```python
from fracmem import CompressedFractionalFilter

f_rl = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16, definition="rl")       # == "gl", the default
f_caputo = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16, definition="caputo")
```

GL and RL are the same discretization for a causal signal (`x(t)=0, t<0`); Caputo is computed as RL applied to the signal shifted by its own first sample (`D^alpha_C x = D^alpha_RL [x - x(0)]`), which removes the `t^-alpha` singularity a nonzero initial value would otherwise introduce — see `kernel.caputo_derivative`.

### Streaming

```python
f.fit(train_signals)
f.reset_stream()
for x_k in live_signal:
    y_k = f.step(x_k)   # one sample in, one derivative estimate out
```

Numerically identical to calling `.predict()` on the same signal in one batch; this is what the ROS2 node and the embedded runtime are checked against in tests.

### Automatic parameter selection

```python
f = CompressedFractionalFilter.auto(alpha=0.5, h=0.01, tol=1e-3)
print(f.L, f.p, f.auto_rmse_)
```

Searches `(L, p)` cheapest-first and returns the fitted filter for the first configuration meeting `tol` (RMSE against a held-out synthetic — or your own, via `train_signals=...` — signal), falling back to the lowest-RMSE configuration tried.

### Adaptive SOE

```python
from fracmem import gl_weights, adaptive_soe_tail_kernel

w = gl_weights(alpha, j_max + 1)
lam, c, p_used, err = adaptive_soe_tail_kernel(alpha, L, w, j_max, tol=1e-4)
```

Grows `p` (doubling from `p_min`) until the tail kernel's worst-case relative error against the exact GL tail is `<= tol`, instead of a fixed `p` chosen up front.

### GPU / PyTorch

```bash
pip install fracmem[torch]
```

```python
import torch
from fracmem.backends.torch_backend import TorchFractionalLayer

layer = TorchFractionalLayer(f, learnable_c=True).to("cuda")   # or "mps", or leave on CPU
y_hat = layer(x)   # x: (T,) or (batch, T)
```

There's no separate GPU code path — moving the module with `.to(device)` runs every op, including the mode recurrence, on-device. `lambda` (the SOE decay rates) stays a fixed, non-learnable buffer; set `learnable_c=True` to fine-tune the linear readout end-to-end by backprop instead of the ridge-regression fit.

### JAX

```bash
pip install fracmem[jax]
```

```python
from fracmem.backends.jax_backend import predict, jit_predict

y_hat = jit_predict(x, f.lam, f.c, w, f.alpha, f.h, f.L)
grad_wrt_c = jax.grad(lambda c: predict(x, f.lam, c, w, f.alpha, f.h, f.L).sum())(f.c)
```

### ROS2

A standalone `ament_python` package at [`ros2_ws/src/fracmem_ros2`](ros2_ws/src/fracmem_ros2). Fit and save a filter offline, then point the node at it:

```python
f.fit(train_signals)
f.save("filter.npz")
```

```bash
pip install fracmem   # into the ROS2 Python environment
colcon build --packages-select fracmem_ros2
source install/setup.bash
ros2 run fracmem_ros2 derivative_node --ros-args -p filter_path:=/path/to/filter.npz
```

Subscribes `std_msgs/Float64` on `fracmem/input`, publishes the streamed derivative on `fracmem/output` (topic names configurable via the `input_topic`/`output_topic` parameters).

### ESP32 / MicroPython

The deployed filter (`lambda`, `c`, and the first `L` GL weights) is just arrays and multiply-adds — no numpy/scipy needed to *run* it, only to *fit* it. `fracmem.embedded` bakes a fitted filter into a standalone, dependency-free `.py` file (stdlib `array` module only) that runs unmodified under MicroPython:

```python
from fracmem.embedded import export_micropython

f.fit(train_signals)
export_micropython(f, "device.py")   # or generate_source(f) for the string
```

```bash
mpremote cp device.py :main.py   # flash to an ESP32 running MicroPython
```

On the device:

```python
from device import filt

while True:
    y_k = filt.step(read_sensor())
```

`O(L+p)` multiply-adds per sample, `L + 2p` float32 persistent state allocated once — fixed RAM regardless of runtime, the same guarantee `fracmem` gives on desktop.

## Why this exists

Given a fractional order $\alpha$, the Grünwald–Letnikov derivative is

$$D^\alpha x(t_k) \approx h^{-\alpha}\sum_{j=0}^{k} w_j\, x_{k-j}, \qquad w_j \sim \frac{j^{-\alpha-1}}{|\Gamma(-\alpha)|}$$

The weights decay as a **power law** — slowly, compared to the geometric decay of an ordinary linear system — so truncating the sum naively costs real accuracy unless kept impractically long. `fracmem` instead:

1. Splits the sum into an **exact** local window (the first `L` samples, computed directly, no approximation) and a **tail** (everything beyond `L`).
2. Approximates the tail's power-law kernel by a short sum of exponentials, using an exact Gamma-function integral identity discretized by quadrature — this gives `p` decay rates that are provably close to optimal, computed from pure mathematics, no data required.
3. Refines the linear readout weights for those `p` modes using a small amount of representative training data, via **cross-validated ridge regression** — a well-posed, convex problem, unlike trying to also re-fit the decay rates.

The result deploys as `p` independent one-line recursions (`m_i[k] = lambda_i * m_i[k-1] + x[k]`) — `O(L+p)` cost per sample and `O(p)` persistent memory, regardless of runtime.

Full derivations, proofs, and a line-by-line code walkthrough are in [`docs/`](docs/) — `theory.pdf` covers the mathematics from first principles (including a formal, offline-computable worst-case error bound), and `code_walkthrough.pdf` explains every function in the library.

## What it's good for

- Fractional-order PID / FOPID controllers on microcontrollers or single-board computers (developed and validated on a Raspberry Pi).
- Any system needing a bounded-memory approximation of a fractional-order operator with a **provable**, not just empirically-observed, worst-case error guarantee.
- Viscoelastic material models, battery impedance models, and other domains where fractional-order dynamics arise but embedded deployment constraints rule out unbounded-history computation.

## Citing / background

The Sum-of-Exponentials approach implemented here follows the classical construction of Jiang, Zhang, Zhang, Zhang and related work (Lubich–Schädle, Baffet–Hesthaven); this library adds a certified offline error bound and a cross-validated data-refinement step on top of the classical construction.

## License

MIT — see [LICENSE](LICENSE).
