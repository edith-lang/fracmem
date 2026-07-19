# fracmem

Compressed fractional-order derivative filters for embedded and real-time systems, with a **certified offline error bound** and **data-refined weights**.

Fractional-order derivatives (used in fractional-order PID control, viscoelastic material models, battery impedance models, and more) are expensive to compute directly: they depend on the *entire* signal history, so a naive implementation's cost grows without bound as the system runs. `fracmem` compresses that unbounded history into a fixed-size recursive filter — constant memory, constant compute per sample, forever — using a Sum-of-Exponentials (SOE) decomposition with decay rates derived from an exact mathematical identity, refined with a small amount of representative training data.

## Install

```bash
pip install fracmem
```

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
