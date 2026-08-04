import numpy as np
import pytest

jax = pytest.importorskip("jax")

from fracmem import CompressedFractionalFilter, gl_weights
from fracmem.backends.jax_backend import predict as jax_predict, jit_predict


def _make_signal(n, seed, offset=0.0):
    rng = np.random.default_rng(seed)
    return offset + np.cumsum(rng.standard_normal(n)) * 0.01


@pytest.mark.parametrize("definition", ["gl", "caputo"])
def test_jax_predict_matches_numpy_predict(definition):
    alpha, h, L, p = 0.5, 0.01, 16, 8
    train = [_make_signal(1500, s, offset=1.0) for s in range(4)]
    f = CompressedFractionalFilter(alpha=alpha, h=h, L=L, p=p, definition=definition)
    f.fit(train, j_max=1500)
    w = gl_weights(alpha, L)

    sig = _make_signal(300, 99, offset=1.0)
    batch = f.predict(sig)
    out = np.asarray(jax_predict(sig, f.lam, f.c, w, alpha, h, L, definition=definition))

    np.testing.assert_allclose(out, batch, atol=1e-3)


def test_jit_predict_matches_eager():
    alpha, h, L, p = 0.5, 0.01, 8, 4
    train = [_make_signal(1000, s) for s in range(3)]
    f = CompressedFractionalFilter(alpha=alpha, h=h, L=L, p=p)
    f.fit(train, j_max=1000)
    w = gl_weights(alpha, L)

    sig = _make_signal(200, 7)
    eager = np.asarray(jax_predict(sig, f.lam, f.c, w, alpha, h, L))
    jitted = np.asarray(jit_predict(sig, f.lam, f.c, w, alpha, h, L))
    np.testing.assert_allclose(eager, jitted, atol=1e-5)


def test_jax_predict_differentiable_wrt_c():
    alpha, h, L, p = 0.5, 0.01, 8, 4
    train = [_make_signal(1000, s) for s in range(3)]
    f = CompressedFractionalFilter(alpha=alpha, h=h, L=L, p=p)
    f.fit(train, j_max=1000)
    w = gl_weights(alpha, L)
    sig = _make_signal(200, 7)

    def loss(c):
        return jax_predict(sig, f.lam, c, w, alpha, h, L).sum()

    grad = jax.grad(loss)(jax.numpy.asarray(f.c, dtype=jax.numpy.float32))
    assert np.all(np.isfinite(np.asarray(grad)))
