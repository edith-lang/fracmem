"""
Follow-up to run_five_tests.py: tests 4 and 5 (train=100,000 and
300,000) showed the compiled-C output disagreeing with the Python
output far more than tests 1-3 did (23% and 38% relative RMSE between
them, vs <0.5% for the smaller training lengths) -- even though both
individually track gold about as well as each other. That is NOT
explained by "one of them is buggy": it needed an actual measurement to
understand.

What this script checks: does |y_c - y_py| stay flat over the 5-million-
sample run (ordinary rounding noise), or does it grow with sample index k
(numerical drift/instability)? It buckets the difference into `bins`
chunks along k and reports the RMS of each chunk.

Finding, from a real run: it grows -- roughly monotonically, by more
than an order of magnitude from the first tenth of the signal to the
last. Every fitted filter (regardless of training length) has at least
one bucket with a decay rate lambda extremely close to 1 (measured:
0.999985-0.999995), because the SOE construction needs a very
slowly-leaking bucket to cover the long power-law tail. That bucket's
recursion m[k] = lambda*m[k-1] + x[k] accumulates rounding error every
single step. In float64 (Python/numpy) that accumulation is negligible
over 5,000,000 steps. In float32 (the compiled C export, deliberately
chosen for embedded targets) it is not -- it is the dominant source of
Python-vs-C disagreement once the filter is trained well enough that
the "real" modeling error has shrunk to a similar magnitude.

Run after run_five_tests.py:
    python investigate_drift.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fracmem import CompressedFractionalFilter
from fracmem.kernel import gl_weights

import run_five_tests as R

RESULTS_DIR = Path(__file__).parent / "results"
BINS = 40
TRAIN_LENGTHS_TO_CHECK = [3_000, 30_000, 100_000, 300_000]


def main():
    x_test = R.random_walk(R.N_TEST, R.TEST_SEED)
    x_test_f32 = x_test.astype(np.float32)
    w_local = gl_weights(R.ALPHA, R.L)

    out = {"n_test": R.N_TEST, "bins": BINS, "series": []}

    for n_train in TRAIN_LENGTHS_TO_CHECK:
        i = R.TRAIN_LENGTHS.index(n_train) + 1 if n_train in R.TRAIN_LENGTHS else 99
        train_signals = [R.random_walk(n_train, seed=1000 * i + s) for s in range(R.N_TRAIN_SIGNALS)]
        filt = CompressedFractionalFilter(alpha=R.ALPHA, h=R.H, L=R.L, p=R.P)
        filt.fit(train_signals)

        y_py = filt.predict(x_test)
        with tempfile.TemporaryDirectory() as tmp:
            y_c, _ = R.run_c_predict(filt, w_local, x_test_f32, Path(tmp))

        diff = y_c.astype(np.float64) - y_py
        n = len(diff)
        chunk_rms = [float(np.sqrt(np.mean(diff[k * n // BINS:(k + 1) * n // BINS] ** 2)))
                     for k in range(BINS)]

        print(f"train={n_train:>7,}  max|lambda|={np.max(np.abs(filt.lam)):.6f}  "
              f"first-tenth RMS={chunk_rms[0]:.4g}  last-tenth RMS={chunk_rms[-1]:.4g}  "
              f"growth={chunk_rms[-1] / max(chunk_rms[0], 1e-12):.1f}x")

        out["series"].append({
            "n_train": n_train,
            "max_lambda": float(np.max(np.abs(filt.lam))),
            "chunk_rms": chunk_rms,
        })

    (RESULTS_DIR / "drift_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {RESULTS_DIR / 'drift_analysis.json'}")


if __name__ == "__main__":
    main()
