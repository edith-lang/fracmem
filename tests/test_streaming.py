import numpy as np
import pytest

from fracmem import CompressedFractionalFilter


def _make_signal(n, seed, offset=0.0):
    rng = np.random.default_rng(seed)
    return offset + np.cumsum(rng.standard_normal(n)) * 0.01


@pytest.mark.parametrize("definition", ["gl", "caputo"])
def test_step_matches_batch_predict(definition):
    alpha, h, L, p = 0.5, 0.01, 16, 8
    train = [_make_signal(1500, s, offset=1.0) for s in range(4)]
    f = CompressedFractionalFilter(alpha=alpha, h=h, L=L, p=p, definition=definition)
    f.fit(train, j_max=1500)

    sig = _make_signal(300, 99, offset=1.0)
    batch = f.predict(sig)

    f.reset_stream()
    stream = np.array([f.step(float(v)) for v in sig])

    np.testing.assert_allclose(stream, batch, atol=1e-9)


def test_step_requires_fit():
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=8, p=4)
    with pytest.raises(RuntimeError):
        f.step(1.0)


def test_reset_stream_actually_resets():
    train = [_make_signal(800, s) for s in range(3)]
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=8, p=4)
    f.fit(train, j_max=800)

    sig = _make_signal(50, 7)
    f.reset_stream()
    first_pass = [f.step(float(v)) for v in sig]
    f.reset_stream()
    second_pass = [f.step(float(v)) for v in sig]
    np.testing.assert_allclose(first_pass, second_pass)
