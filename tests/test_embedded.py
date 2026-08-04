"""
Runs the MicroPython-targeted runtime under plain CPython -- it only
uses the stdlib `array` module, so it works unmodified in both.
"""
import numpy as np
import pytest

from fracmem import CompressedFractionalFilter, gl_weights
from fracmem.embedded import MicroFractionalFilter, generate_source, export_micropython


def _make_signal(n, seed, offset=0.0):
    rng = np.random.default_rng(seed)
    return offset + np.cumsum(rng.standard_normal(n)) * 0.01


@pytest.mark.parametrize("definition", ["gl", "caputo"])
def test_micro_filter_matches_numpy_predict_to_float32_precision(definition):
    alpha, h, L, p = 0.5, 0.01, 16, 8
    train = [_make_signal(1500, s, offset=1.0) for s in range(4)]
    f = CompressedFractionalFilter(alpha=alpha, h=h, L=L, p=p, definition=definition)
    f.fit(train, j_max=1500)
    w = gl_weights(alpha, L)

    mf = MicroFractionalFilter(alpha, h, L, list(w), list(f.lam), list(f.c),
                                caputo=(definition == "caputo"))

    sig = _make_signal(300, 99, offset=1.0)
    batch = f.predict(sig)
    micro = np.array([mf.step(float(v)) for v in sig])

    np.testing.assert_allclose(micro, batch, atol=1e-3)  # float32 rounding


def test_micro_filter_fixed_memory_no_growth():
    lam, c, w = [0.9, 0.5], [0.1, 0.2], [1.0, -0.5, -0.1]
    mf = MicroFractionalFilter(0.5, 0.01, 3, w, lam, c)
    sizes_before = (len(mf.buf), len(mf.m))
    for i in range(10000):
        mf.step(float(i % 7))
    sizes_after = (len(mf.buf), len(mf.m))
    assert sizes_before == sizes_after == (3, 2)


def test_export_micropython_generates_standalone_runnable_file(tmp_path):
    train = [_make_signal(1500, s) for s in range(4)]
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=16, p=8)
    f.fit(train, j_max=1500)

    src = generate_source(f, instance_name="deployed")
    assert "import numpy" not in src
    assert "import scipy" not in src
    assert "import fracmem" not in src
    assert "class MicroFractionalFilter" in src

    path = tmp_path / "device.py"
    export_micropython(f, str(path), instance_name="deployed")

    ns = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    sig = _make_signal(100, 42)
    out = np.array([ns["deployed"].step(float(v)) for v in sig])
    expected = f.predict(sig)
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_export_requires_fitted_filter():
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=8, p=4)
    with pytest.raises(RuntimeError):
        generate_source(f)
