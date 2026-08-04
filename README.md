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

`fracmem`'s only dependencies are `numpy` and `scipy` — deliberately lightweight, since the whole point is to be embeddable.

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
pip install -e ".[dev]"
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
| [`embedded-export.py`](examples/embedded-export.py) | Export a fitted filter as a standalone MicroPython file, then verify it under CPython — flash-ready. |

## Features

- **Derivative definitions**: Grünwald-Letnikov / Riemann-Liouville (the default) and Caputo.
- **Streaming**: an `O(L+p)`-per-sample `.step()` API for online use, alongside the batch `.predict()`.
- **Adaptive SOE**: automatically grow the number of exponential modes until a target error tolerance is met, instead of hand-picking `p`.
- **Automatic parameter selection**: `CompressedFractionalFilter.auto(...)` grid-searches `(L, p)` for the cheapest deployed filter meeting a target RMSE.
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

Numerically identical to calling `.predict()` on the same signal in one batch; this is what the embedded runtime is checked against in tests.

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
