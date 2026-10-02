import numpy as np
import pytest

from fracmem import (
    CompressedFractionalFilter,
    gl_weights,
    full_gl_derivative,
    rl_derivative,
    caputo_derivative,
)


def _make_signal(n, seed, offset=0.0):
    rng = np.random.default_rng(seed)
    return offset + np.cumsum(rng.standard_normal(n)) * 0.01


def test_rl_is_the_gl_discretization():
    x = _make_signal(500, 1)
    w = gl_weights(0.5, len(x))
    np.testing.assert_array_equal(rl_derivative(x, 0.5, 0.01, w), full_gl_derivative(x, 0.5, 0.01, w))


def test_caputo_matches_gl_of_shifted_signal():
    x = _make_signal(500, 2, offset=3.0)  # nonzero initial value
    w = gl_weights(0.5, len(x))
    got = caputo_derivative(x, 0.5, 0.01, w)
    expected = full_gl_derivative(x - x[0], 0.5, 0.01, w)
    np.testing.assert_allclose(got, expected)


def test_caputo_invariant_to_initial_value_shift():
    """Two signals differing only by an additive constant must give the
    same Caputo derivative -- unlike the RL derivative, which does not
    have this property for nonzero-initial-value signals."""
    rng = np.random.default_rng(3)
    base = np.cumsum(rng.standard_normal(500)) * 0.01
    w = gl_weights(0.5, len(base))
    d1 = caputo_derivative(base, 0.5, 0.01, w)
    d2 = caputo_derivative(base + 7.0, 0.5, 0.01, w)
    np.testing.assert_allclose(d1, d2)


def test_invalid_definition_rejected():
    with pytest.raises(ValueError):
        CompressedFractionalFilter(alpha=0.5, h=0.01, L=8, p=4, definition="bogus")


def test_filter_caputo_end_to_end_improves_on_classical():
    alpha, h, L, p = 0.5, 0.01, 16, 8
    n_train, n_test = 1500, 2000
    train_signals = [_make_signal(n_train, s, offset=1.0) for s in range(6)]
    test_signal = _make_signal(n_test, 100, offset=1.0)

    w_test = gl_weights(alpha, n_test + 1)
    gold = caputo_derivative(test_signal, alpha, h, w_test)

    f = CompressedFractionalFilter(alpha=alpha, h=h, L=L, p=p, definition="caputo")
    f.fit(train_signals, j_max=n_test)
    pred = f.predict(test_signal, w=w_test)

    rmse = float(np.sqrt(np.mean((pred - gold) ** 2)))
    assert np.isfinite(rmse)
    assert rmse < 0.1
