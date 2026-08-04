"""
Fit once, save the deployed filter, load it back later (a fresh process,
a different machine, an embedded build step) and confirm predictions are
bit-for-bit identical.

Run:
    python examples/save-and-load.py
"""
import tempfile
from pathlib import Path

import numpy as np

from fracmem import CompressedFractionalFilter


def main():
    rng = np.random.default_rng(3)
    train_signals = [np.cumsum(rng.standard_normal(2000)) * 0.01 for _ in range(6)]

    filt = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16)
    filt.fit(train_signals)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "filter.npz"
        filt.save(str(path))
        print(f"saved fitted filter to {path}  ({path.stat().st_size} bytes)")

        reloaded = CompressedFractionalFilter.load(str(path))

        test_signal = np.cumsum(rng.standard_normal(1000)) * 0.01
        y_original = filt.predict(test_signal)
        y_reloaded = reloaded.predict(test_signal)

        print(f"alpha/h/L/p match:  {(reloaded.alpha, reloaded.h, reloaded.L, reloaded.p) == (filt.alpha, filt.h, filt.L, filt.p)}")
        print(f"predictions match:  {np.array_equal(y_original, y_reloaded)}")


if __name__ == "__main__":
    main()
