"""
Fit a filter on the desktop, export it as a standalone, dependency-free
MicroPython file, then run the generated file's own filter class under
plain CPython to confirm it reproduces the desktop predictions -- this
is the exact code path meant to be copied onto an ESP32.

Run:
    python examples/embedded-export.py
"""
import runpy
import tempfile
from pathlib import Path

import numpy as np

from fracmem import CompressedFractionalFilter
from fracmem.embedded import export_micropython


def main():
    rng = np.random.default_rng(6)
    train_signals = [np.cumsum(rng.standard_normal(1500)) * 0.01 for _ in range(6)]

    filt = CompressedFractionalFilter(alpha=0.5, h=0.01, L=16, p=8)
    filt.fit(train_signals, j_max=1500)

    with tempfile.TemporaryDirectory() as tmp:
        device_path = Path(tmp) / "device.py"
        export_micropython(filt, str(device_path))
        print(f"exported {device_path.stat().st_size} bytes to {device_path.name}")
        print("(stdlib `array` module only -- no numpy/scipy, ready for `mpremote cp device.py :main.py`)")

        # Run the generated file itself and use the `filt` object it defines,
        # exactly like MicroPython would after importing it on-device.
        module = runpy.run_path(str(device_path))
        device_filter = module["filt"]

        test_signal = np.cumsum(rng.standard_normal(300)) * 0.01
        y_desktop = filt.predict(test_signal)
        y_device = np.array([device_filter.step(float(v)) for v in test_signal])

        max_diff = float(np.max(np.abs(y_desktop - y_device)))
        print(f"\nmax |desktop - device| over {len(test_signal)} samples: {max_diff:.2e}"
              "  (float32 rounding only)")


if __name__ == "__main__":
    main()
