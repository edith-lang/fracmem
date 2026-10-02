# fracmem

[![Tests](https://github.com/edith-lang/fracmem/actions/workflows/tests.yml/badge.svg)](https://github.com/edith-lang/fracmem/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fractional-order derivatives as a fixed-size recursive filter: constant compute and memory per sample, so they run on a microcontroller.

An exact fractional derivative needs the entire signal history. fracmem fits a small filter once in Python, then deploys it as plain C.

## Install

```bash
pip install fracmem
```

Requires `numpy` and `scipy`. Deploying needs only a C compiler.

## Usage

```python
import numpy as np
from fracmem import CompressedFractionalFilter

train = [np.random.randn(3000).cumsum() * 0.01 for _ in range(8)]

f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16)
f.fit(train, j_max=10_000)       # once, offline
y = f.predict(signal)            # batch

f.reset_stream()
y_k = f.step(x_k)                # or one sample at a time
```

`definition="rl"` (default, same as Grünwald–Letnikov) or `"caputo"`.

## How it works

The derivative is a weighted sum over all past samples, with power-law weights `w_j ~ j^(-alpha-1)`.

- The latest `L` samples are computed exactly.
- Older samples (the tail) are replaced by `p` exponential modes, each updated with one multiply-add per sample. Decay rates come from a Gamma-function integral identity, not from data.
- A few training signals fit the readout weights by cross-validated ridge regression.

Cost is `O(L+p)` compute and `O(p)` memory per sample.

## Embedded C

```python
from fracmem.embedded import export_c
export_c(f, "device_filter.c")
```

```c
#include "fracmemfilter.h"
filtSetup();
float y = fracmemStep(&filt, x_k);
```

No heap allocation. See [`examples/`](examples/) for the full fit, export, compile and verify round trip.

## Background

Builds on the sum-of-exponentials construction of Jiang, Zhang, Zhang and Zhang, and related work by Lubich and Schädle, and by Baffet and Hesthaven.

## License

MIT
