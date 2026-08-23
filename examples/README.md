# Examples

Each file is a standalone, runnable script -- clone the repo, install `fracmem`, and run:

```bash
pip install -e ".[dev]"
python examples/basic-usage.py
```

| Script | What it shows |
|---|---|
| [`basic-usage.py`](basic-usage.py) | Fit, deploy, and check accuracy + speed against the exact (unbounded-memory) reference. Start here. |
| [`adaptive-soe.py`](adaptive-soe.py) | The low-level SOE tail construction: mode count grows until a tolerance is met, no data required. |
| [`auto-params.py`](auto-params.py) | `CompressedFractionalFilter.auto(...)` picks `(L, p)` for you from a target RMSE. |
| [`embedded-export-c.py`](embedded-export-c.py) | Fit, export to plain C, compile, run, and verify against Python -- the full round trip to a real embedded device. |

Every script only needs the base install (`pip install fracmem`); `embedded-export-c.py` additionally needs a C compiler (`gcc` or `cc`) on `PATH`.
