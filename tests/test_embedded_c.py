"""
Compiles the plain-C runtime (fracmemfilter.c) with a small test driver
and checks it produces the same numbers as CompressedFractionalFilter's
own Python .predict().

Skips (instead of failing) if no C compiler is available.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fracmem import CompressedFractionalFilter, gl_weights
from fracmem.embedded import export_c

CC = shutil.which("cc") or shutil.which("gcc")
C_DIR = Path(__file__).parent.parent / "src" / "fracmem" / "embedded" / "c"
TEST_C_DIR = Path(__file__).parent / "c"

pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler found")


def _make_signal(n, seed, offset=0.0):
    rng = np.random.default_rng(seed)
    return offset + np.cumsum(rng.standard_normal(n)) * 0.01


def _compile_testmain(tmp_path):
    exe = tmp_path / "testmain"
    subprocess.run(
        [CC, "-O2", "-o", str(exe),
         str(TEST_C_DIR / "testmain.c"), str(C_DIR / "fracmemfilter.c"),
         "-I", str(C_DIR)],
        check=True,
    )
    return exe


def _run_c_filter(exe, L, p, h_pow, caputo, w, lam, c, signal):
    stdin_lines = [f"{L} {p} {h_pow!r} {int(caputo)}"]
    stdin_lines += [repr(float(v)) for v in w]
    stdin_lines += [repr(float(v)) for v in lam]
    stdin_lines += [repr(float(v)) for v in c]
    stdin_lines += [str(len(signal))]
    stdin_lines += [repr(float(v)) for v in signal]
    result = subprocess.run(
        [str(exe)], input="\n".join(stdin_lines), capture_output=True, text=True, check=True,
    )
    return np.array([float(line) for line in result.stdout.split()])


@pytest.mark.parametrize("definition", ["gl", "caputo"])
def test_c_filter_matches_python_predict(tmp_path, definition):
    alpha, h, L, p = 0.5, 0.01, 16, 8
    train = [_make_signal(1500, s, offset=1.0) for s in range(4)]
    f = CompressedFractionalFilter(alpha=alpha, h=h, L=L, p=p, definition=definition)
    f.fit(train, j_max=1500)
    w = gl_weights(alpha, L)

    exe = _compile_testmain(tmp_path)
    sig = _make_signal(300, 99, offset=1.0)
    c_out = _run_c_filter(exe, L, p, h ** -alpha, definition == "caputo", w, f.lam, f.c, sig)
    py_out = f.predict(sig)

    np.testing.assert_allclose(c_out, py_out, atol=1e-3)  # float32 rounding


def test_export_c_generates_compilable_file(tmp_path):
    train = [_make_signal(1500, s) for s in range(4)]
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=16, p=8)
    f.fit(train, j_max=1500)

    device_c = tmp_path / "device_filter.c"
    export_c(f, str(device_c), instance_name="deployed")
    src = device_c.read_text()
    assert '#include "fracmemfilter.h"' in src
    assert "fracmemInit(&deployed" in src

    main_c = tmp_path / "main.c"
    main_c.write_text(
        '#include "fracmemfilter.h"\n'
        "extern FracmemFilter deployed;\n"
        "void deployedSetup(void);\n"
        "int main(void) {\n"
        "    deployedSetup();\n"
        "    float y = fracmemStep(&deployed, 1.0f);\n"
        "    return (y == y) ? 0 : 1;\n"  # NaN check: only 0 (success) if y is a real number
        "}\n"
    )
    exe = tmp_path / "device_main"
    subprocess.run(
        [CC, "-O2", "-o", str(exe), str(main_c), str(device_c), str(C_DIR / "fracmemfilter.c"),
         "-I", str(C_DIR)],
        check=True,
    )
    subprocess.run([str(exe)], check=True)
