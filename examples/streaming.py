"""
Feed a filter one sample at a time with .step() -- the same API a live
sensor loop would use -- and confirm it's numerically identical to
calling .predict() on the whole signal at once.

Run:
    python examples/streaming.py
"""
import numpy as np

from fracmem import CompressedFractionalFilter


def main():
    rng = np.random.default_rng(1)
    train_signals = [np.cumsum(rng.standard_normal(2000)) * 0.01 for _ in range(6)]

    filt = CompressedFractionalFilter(alpha=0.7, h=0.01, L=24, p=12)
    filt.fit(train_signals)

    live_signal = np.cumsum(rng.standard_normal(1000)) * 0.01

    # Batch: everything available up front.
    y_batch = filt.predict(live_signal)

    # Streaming: one sample in, one estimate out -- fixed memory, no
    # buffering of past samples beyond the filter's own O(L+p) state.
    filt.reset_stream()
    y_stream = np.array([filt.step(float(x_k)) for x_k in live_signal])

    max_diff = float(np.max(np.abs(y_batch - y_stream)))
    print(f"samples streamed:        {len(live_signal)}")
    print(f"max |batch - stream|:    {max_diff:.2e}  (should be ~0, both are the same math)")
    print(f"batch and stream agree:  {np.allclose(y_batch, y_stream)}")


if __name__ == "__main__":
    main()
