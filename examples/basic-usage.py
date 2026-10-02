"""
Fit a compressed fractional-derivative filter, deploy it on a signal much
longer than anything it trained on, and check it against the exact
(unbounded-memory) reference computation.

Run:
    python examples/basic-usage.py
"""
import time

import numpy as np

from fracmem import CompressedFractionalFilter
from fracmem.kernel import full_gl_derivative, gl_weights

ALPHA = 0.5   # fractional order ("half derivative")
H = 0.01      # sample time (seconds)
L = 32        # exact local window length (samples)
P = 16        # number of compressed tail modes


def main():
    rng = np.random.default_rng(0)

    # A handful of representative training signals -- whatever the filter
    # will actually see in deployment. 8-32 signals is typically enough.
    train_signals = [np.cumsum(rng.standard_normal(3000)) * 0.01 for _ in range(8)]

    filt = CompressedFractionalFilter(alpha=ALPHA, h=H, L=L, p=P)
    filt.fit(train_signals, j_max=10_000)
    print(f"fitted: L={filt.L}, p={filt.p}, ridge reg={filt.reg_used:.1e}")

    # Deploy on a signal 15x longer than anything used in training.
    test_signal = np.cumsum(rng.standard_normal(50_000)) * 0.01

    t0 = time.perf_counter()
    y_hat = filt.predict(test_signal)
    compressed_time = time.perf_counter() - t0

    # Exact reference: unbounded-memory direct convolution against every
    # GL weight -- what fracmem exists to avoid computing directly.
    w = gl_weights(ALPHA, len(test_signal) + 1)
    t0 = time.perf_counter()
    y_exact = full_gl_derivative(test_signal, ALPHA, H, w)
    exact_time = time.perf_counter() - t0

    rmse = float(np.sqrt(np.mean((y_hat - y_exact) ** 2)))
    scale = float(np.sqrt(np.mean(y_exact ** 2)))

    print(f"\nsignal length:      {len(test_signal):,} samples")
    print(f"compressed predict:  {compressed_time * 1000:.2f} ms  (O(L+p) per sample, fixed)")
    print(f"exact reference:     {exact_time * 1000:.2f} ms  (O(n) per sample, grows forever)")
    print(f"RMSE vs exact:       {rmse:.6f}  (signal RMS: {scale:.6f}, {100 * rmse / scale:.3f}% relative)")


if __name__ == "__main__":
    main()
