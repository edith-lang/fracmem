# fracmem

[![Tests](https://github.com/edith-lang/fracmem/actions/workflows/tests.yml/badge.svg)](https://github.com/edith-lang/fracmem/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

fracmem turns a fractional order derivative, which normally needs the entire signal history to compute, into a fixed size recursive filter: constant compute and constant memory per sample, forever. It is built for fractional order PID control, viscoelastic material models, battery impedance models, and any other setting where fractional order dynamics meet an embedded or real time budget.

## Install

```bash
pip install git+https://github.com/edith-lang/fracmem.git
```

Or from a local clone, for development:

```bash
git clone https://github.com/edith-lang/fracmem.git
cd fracmem
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

The only dependencies are `numpy` and `scipy`, kept deliberately light since the whole point is to be embeddable.

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/FracmemGuide.pdf`](docs/FracmemGuide.pdf) | Everything in one place, plain language: the problem, why the method works (with a short real proof), a hand-checkable example, how to use the library, and real results. Start here. |
| [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) | Five-test accuracy/speed sweep against a hand-verified GPU exact reference, fully reproducible. |

## The idea

The Grunwald Letnikov definition of a fractional derivative of order alpha is a weighted sum over the entire past of the signal:

```
D^alpha x(t_k) ≈ h^(-alpha) * sum_{j=0}^{k} w_j * x[k-j]
```

The weights `w_j` decay as a power law, `w_j ~ j^(-alpha-1)`. A power law decays far slower than the geometric decay of an ordinary linear filter, so truncating the sum at a fixed lag throws away real accuracy, and computing it exactly costs a little more with every new sample, an unbounded cost that rules out embedded or real time use.

fracmem handles this in three steps.

**Split the sum in two.** The most recent `L` samples are an exact local window, computed directly with no approximation at all. Everything older than that is the tail.

**Replace the tail with a short sum of exponentials.** An exponential mode decays geometrically, so it can be tracked with a single number and updated with one multiply and one add per sample: `m_i[k] = lambda_i * m_i[k-1] + x[k]`. The decay rates `lambda_i` are not fit to data. They come from an exact Gamma function integral identity for the power law kernel, discretized by quadrature, giving `p` rates that are provably close to the best possible choice for a given tolerance before a single training signal is seen.

**Refine the readout weights with a little data.** The `p` exponential modes, plus the exact local window, are combined into a derivative estimate through linear weights `c`. Those weights are fit with a handful of representative training signals using cross validated ridge regression, a small, convex, well posed least squares problem, quite unlike the poorly posed problem of trying to refit the decay rates themselves.

The deployed filter is `p` independent one line recursions plus the `L` term local window: `O(L+p)` compute and `O(p)` memory per sample, independent of signal length. Because the decay rates come from a proven identity rather than a data fit, the tail's worst case error against the exact kernel can also be bounded and computed offline, before the filter ever sees a signal.

The full derivation, kept simple, is in [`docs/FracmemGuide.pdf`](docs/FracmemGuide.pdf).

### Derivative definitions

Grunwald Letnikov and Riemann Liouville are the same discretization for a causal signal. Caputo is computed as Riemann Liouville applied to the signal after subtracting its own first sample, `D^alpha_Caputo x = D^alpha_RL [x - x(0)]`, which removes the singularity a nonzero initial value would otherwise introduce.

```python
f_rl = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16, definition="rl")   # default
f_caputo = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16, definition="caputo")
```

## Quick example

```python
import numpy as np
from fracmem import CompressedFractionalFilter

ALPHA = 0.5   # fractional order, e.g. 0.5 is a "half derivative"
H = 0.01      # sample time in seconds
L = 32        # exact local window length, in samples
P = 16        # number of compressed tail modes

train_signals = [np.random.randn(3000).cumsum() * 0.01 for _ in range(8)]

f = CompressedFractionalFilter(alpha=ALPHA, h=H, L=L, p=P)
f.fit(train_signals, j_max=10_000)   # done once, offline

new_signal = np.random.randn(50_000).cumsum() * 0.01
y_hat = f.predict(new_signal)        # O(L+p) per sample, whether the signal is 50 samples or 50 million
```

Streaming, one sample at a time, is numerically identical to `predict` on the same signal in one batch:

```python
f.reset_stream()
for x_k in live_signal:
    y_k = f.step(x_k)
```

## Results

Measured with `alpha=0.5`, `L=32`, `p=16`, filter fit on 8 training signals of 3000 samples each. Reproduce with `python examples/basic-usage.py`.

| Signal length | fracmem predict | exact reference (unbounded memory) |
|---:|---:|---:|
| 5,000 samples | 8.5 ms | 2.8 ms |
| 20,000 samples | 34.8 ms | 51.1 ms |
| 80,000 samples | 169.5 ms | 710.2 ms |

fracmem's cost per sample is fixed, so its total time grows linearly with signal length. The exact reference recomputes the full weighted sum for every new sample, so its cost grows quadratically. The crossover, where fracmem becomes the faster option, happens well before 20,000 samples here, and the gap widens without bound as the signal gets longer.

On a 50,000 sample test signal, well beyond the length of anything in the training set, fracmem tracks the exact reference to 1.15% relative RMSE.

Accuracy against deployed cost is a direct trade, controlled by the tolerance passed to `CompressedFractionalFilter.auto`:

| target RMSE | L | p | multiply adds per sample | achieved RMSE |
|---:|---:|---:|---:|---:|
| 1e-2 | 8 | 4 | 12 | 5.65e-3 |
| 1e-3 | 8 | 8 | 16 | 2.27e-4 |
| 1e-4 | 8 | 16 | 24 | 8.65e-7 |

```python
f = CompressedFractionalFilter.auto(alpha=0.5, h=0.01, tol=1e-3)
print(f.L, f.p, f.auto_rmse_)
```

Reproduce with `python examples/auto-params.py`.

For a large-scale, GPU-verified benchmark -- five real tests sweeping training-signal
length against a hand-verified, non-FFT exact reference on a 5,000,000-sample signal,
including a genuine float32-vs-float64 drift finding in the compiled C export -- see
[`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md).

## Features

- **Derivative definitions**: Grunwald Letnikov / Riemann Liouville (the default) and Caputo.
- **Streaming**: an `O(L+p)` per sample `.step()` call for online use, alongside batch `.predict()`.
- **Adaptive tail construction**: grow the number of exponential modes automatically until a target error tolerance is met, instead of hand picking `p`.
- **Automatic parameter selection**: `CompressedFractionalFilter.auto(...)` searches `(L, p)` for the cheapest filter meeting a target RMSE.
- **ESP32 / MicroPython export**: `fracmem.embedded` exports a fitted filter as a single, standalone MicroPython file with no external dependencies and fixed RAM use.

```python
from fracmem.embedded import export_micropython

f.fit(train_signals)
export_micropython(f, "device.py")
```

```bash
mpremote cp device.py :main.py   # flash to an ESP32 running MicroPython
```

See [`examples/`](examples/) for a runnable script covering each feature above, including streaming, adaptive tail construction, automatic parameters, saving and loading a fitted filter, and the embedded export.

## Background

The sum of exponentials approach here follows the classical construction of Jiang, Zhang, Zhang and Zhang, and related work by Lubich and Schadle, and by Baffet and Hesthaven. fracmem adds a certified offline error bound and a cross validated data refinement step on top of that classical construction.

## License

MIT, see [LICENSE](LICENSE).
