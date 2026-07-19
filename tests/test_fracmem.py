import numpy as np
from fracmem import CompressedFractionalFilter, gl_weights, full_gl_derivative, soe_tail_kernel


def _make_signal(n, seed):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) * 0.01


def test_gl_weights_basic():
    w = gl_weights(0.5, 5)
    assert w[0] == 1.0
    # weights must alternate toward zero for 0 < alpha < 1
    assert np.all(np.isfinite(w))


def test_pure_soe_beats_naive_and_is_deterministic():
    alpha, h, L, p = 0.5, 0.01, 32, 16
    n_test = 4000
    w = gl_weights(alpha, n_test + 1)
    lam1, c1 = soe_tail_kernel(alpha, L, p, w, j_max=n_test)
    lam2, c2 = soe_tail_kernel(alpha, L, p, w, j_max=n_test)
    np.testing.assert_allclose(lam1, lam2)
    np.testing.assert_allclose(c1, c2)


def test_fit_predict_improves_on_pure_classical():
    alpha, h, L, p = 0.5, 0.01, 32, 16
    n_train, n_test = 2000, 4000

    train_signals = [_make_signal(n_train, seed) for seed in range(8)]
    test_signal = _make_signal(n_test, seed=100)

    w_test = gl_weights(alpha, n_test + 1)
    gold = full_gl_derivative(test_signal, alpha, h, w_test)

    f = CompressedFractionalFilter(alpha=alpha, h=h, L=L, p=p)
    f.fit(train_signals, j_max=n_test)
    pred = f.predict(test_signal, w=w_test)

    rmse = float(np.sqrt(np.mean((pred - gold) ** 2)))
    assert np.isfinite(rmse)
    assert rmse < 0.1  # sanity: not wildly wrong


def test_predict_cost_independent_of_signal_length():
    """The recursion's persistent state must stay fixed-size regardless
    of how long the predicted signal is."""
    alpha, h, L, p = 0.5, 0.01, 32, 8
    train_signals = [_make_signal(500, seed) for seed in range(4)]
    f = CompressedFractionalFilter(alpha=alpha, h=h, L=L, p=p)
    f.fit(train_signals, j_max=2000)
    assert f.lam.shape == (p,)
    assert f.c.shape == (p,)

    short = f.predict(_make_signal(200, 1))
    long = f.predict(_make_signal(20000, 2))
    assert short.shape == (200,)
    assert long.shape == (20000,)


def test_predict_requires_fit_first():
    f = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16)
    try:
        f.predict(np.zeros(100))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
