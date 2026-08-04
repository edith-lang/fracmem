# Examples

Each file is a standalone, runnable script -- clone the repo, install `fracmem`, and run:

```bash
pip install -e ".[dev]"
python examples/basic-usage.py
```

| Script | What it shows |
|---|---|
| [`basic-usage.py`](basic-usage.py) | Fit, deploy, and check accuracy + speed against the exact (unbounded-memory) reference. Start here. |
| [`streaming.py`](streaming.py) | One-sample-at-a-time `.step()` API, checked against batch `.predict()`. |
| [`definitions.py`](definitions.py) | GL/RL vs Caputo derivative definitions on a signal with a nonzero initial value. |
| [`auto-params.py`](auto-params.py) | `CompressedFractionalFilter.auto(...)` picks `(L, p)` for you from a target RMSE. |
| [`adaptive-soe.py`](adaptive-soe.py) | The low-level SOE tail construction: mode count grows until a tolerance is met, no data required. |
| [`save-and-load.py`](save-and-load.py) | Persist a fitted filter to disk and reload it elsewhere. |
| [`embedded-export.py`](embedded-export.py) | Export a fitted filter as a standalone MicroPython file for an ESP32, then verify it under CPython -- flash-ready. |

Every script only needs the base install (`pip install fracmem`).
