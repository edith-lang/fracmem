"""
The five-test sweep: how does fracmem's accuracy/speed/fit-cost
trade-off actually move as you change ONE thing -- how long the
training signals are -- while everything else (test signal, alpha, L,
p, seed) stays fixed?

Round 1/Round 2 of the earlier benchmark (see SoeVsBruteForce.pdf) only
gave two points on this curve (50,000 and 500,000-sample training
signals). This script fills in five points, log-spaced from a very
short training signal to a fairly long one, all measured against the
SAME real GPU-computed exact answer, on the SAME real machine.

What happens, once, for the whole sweep:
  1. Generate ONE test signal (a 5,000,000-sample random walk).
  2. Compute its EXACT fractional derivative on the GPU (honest_gpu_
     reference.gpu_exact_gl_derivative -- no FFT shortcut, verified
     against fracmem's own CPU reference first). This is "gold".

What happens, once per test (5 times):
  3. Generate 8 training signals of the given length.
  4. filt.fit(...) them -- timed.
  5. filt.predict(test_signal) in pure Python -- timed.
  6. Export the fitted filter to C, compile it against fracmemfilter.c,
     run it over the test signal in bulk (binary I/O, so file parsing
     doesn't pollute the measurement) -- timed.
  7. Compare both the Python and the C output against gold: relative
     RMSE, over the WHOLE test signal.

Everything reported is a real, measured number from this run -- fit
time, predict time, compile+run time, and the achieved accuracy. See
BENCHMARKS.md for the write-up and SoeVsBruteForce.tex for the slide
version of the results.

Run with the project's own fracmem installed, plus torch (any CUDA
build) and matplotlib:
    python run_five_tests.py
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fracmem import CompressedFractionalFilter
from fracmem.kernel import gl_weights
from fracmem.embedded import export_c

from honest_gpu_reference import gpu_exact_gl_derivative, verify_against_numpy

C_DIR = Path(__file__).parent.parent / "src" / "fracmem" / "embedded" / "c"
BULK_C = Path(__file__).parent / "bulk_predict.c"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALPHA = 0.5
H = 1.0
L = 32
P = 16
N_TEST = 5_000_000
N_TRAIN_SIGNALS = 8
TRAIN_LENGTHS = [3_000, 10_000, 30_000, 100_000, 300_000]
TEST_SEED = 42


def random_walk(n, seed):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)).astype(np.float64) * 0.01


def relative_rmse(y, gold):
    y = np.asarray(y, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y - gold) ** 2)))
    denom = float(np.sqrt(np.mean(gold ** 2)))
    return rmse / denom if denom > 0 else float("nan")


def run_c_predict(filt, w_local, x_test_f32, workdir):
    device_c = workdir / "device_filter.c"
    export_c(filt, str(device_c), w=w_local, instance_name="filt")

    exe = workdir / "bulk_predict"
    cc = "cc"
    subprocess.run(
        [cc, "-O2", "-o", str(exe), str(BULK_C), str(device_c), str(C_DIR / "fracmemfilter.c"),
         "-I", str(C_DIR)],
        check=True, capture_output=True,
    )

    in_bin = workdir / "in.f32"
    out_bin = workdir / "out.f32"
    x_test_f32.tofile(in_bin)

    result = subprocess.run([str(exe), str(in_bin), str(out_bin)], check=True,
                             capture_output=True, text=True)
    c_elapsed = float(result.stdout.strip())
    y_c = np.fromfile(out_bin, dtype=np.float32)
    return y_c, c_elapsed


def main():
    print("Step 0: verifying the GPU exact reference against fracmem's own CPU reference...")
    check = verify_against_numpy(alpha=ALPHA, h=H, n=2000, block=256)
    print(f"  rel_rmse={check['rel_rmse']:.3e}  max_rel_diff={check['max_rel_diff']:.3e}")
    assert check["rel_rmse"] < 1e-4, "GPU reference does not match fracmem's own math -- stopping."

    print(f"\nStep 1: generating the test signal (n={N_TEST:,}, seed={TEST_SEED})...")
    x_test = random_walk(N_TEST, TEST_SEED)
    x_test_f32 = x_test.astype(np.float32)

    print("Step 2: computing the exact ('gold') derivative on the GPU...")
    dev = torch.device("cuda")
    w_full_np = gl_weights(ALPHA, N_TEST).astype(np.float32)
    x_t = torch.tensor(x_test_f32, device=dev)
    w_t = torch.tensor(w_full_np, device=dev)
    torch.cuda.synchronize()
    t0 = time.time()
    gold_t = gpu_exact_gl_derivative(x_t, w_t, H, ALPHA, block=8192)
    torch.cuda.synchronize()
    gold_time = time.time() - t0
    gold = gold_t.cpu().numpy().astype(np.float64)
    print(f"  gold computed in {gold_time:.2f}s  (n={N_TEST:,}, this is the 'standard/exact' method)")

    w_local = gl_weights(ALPHA, L)

    results = {
        "config": {"alpha": ALPHA, "h": H, "L": L, "p": P, "n_test": N_TEST,
                   "n_train_signals": N_TRAIN_SIGNALS, "test_seed": TEST_SEED,
                   "gpu": torch.cuda.get_device_name(0)},
        "gold_time_s": gold_time,
        "tests": [],
    }

    for i, n_train in enumerate(TRAIN_LENGTHS, start=1):
        print(f"\n=== Test {i}/5: train_length={n_train:,} "
              f"(test/train ratio = {N_TEST / n_train:.1f}x) ===")

        train_signals = [random_walk(n_train, seed=1000 * i + s) for s in range(N_TRAIN_SIGNALS)]

        filt = CompressedFractionalFilter(alpha=ALPHA, h=H, L=L, p=P)
        t0 = time.time()
        filt.fit(train_signals)
        fit_time = time.time() - t0
        print(f"  fit:      {fit_time:8.2f}s")

        t0 = time.time()
        y_py = filt.predict(x_test)
        predict_py_time = time.time() - t0
        print(f"  predict (Python): {predict_py_time:8.2f}s")

        with tempfile.TemporaryDirectory() as tmp:
            y_c, predict_c_time = run_c_predict(filt, w_local, x_test_f32, Path(tmp))
        print(f"  predict (C):      {predict_c_time:8.4f}s")

        err_py = relative_rmse(y_py, gold)
        err_c = relative_rmse(y_c, gold)
        py_vs_c = relative_rmse(y_c, y_py)
        print(f"  relative RMSE vs gold -- Python: {err_py * 100:6.2f}%   C: {err_c * 100:6.2f}%"
              f"   (Python vs C agree to {py_vs_c:.2e})")

        results["tests"].append({
            "n_train": n_train,
            "ratio_test_over_train": N_TEST / n_train,
            "fit_time_s": fit_time,
            "predict_py_time_s": predict_py_time,
            "predict_c_time_s": predict_c_time,
            "rel_rmse_py_vs_gold": err_py,
            "rel_rmse_c_vs_gold": err_c,
            "rel_rmse_py_vs_c": py_vs_c,
            "speedup_c_vs_gold": gold_time / predict_c_time,
            "speedup_py_vs_gold": gold_time / predict_py_time,
        })

    out_path = RESULTS_DIR / "five_tests.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
